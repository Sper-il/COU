"""
src/models/smoke_test.py
Kiểm thử nhanh module mô hình trên **dữ liệu Jena thật** (chưa huấn luyện đầy đủ).

Mục đích: xác nhận 3 kiến trúc trong `architecture.py` chạy đúng đầu-cuối trên
tập dữ liệu thật — đúng shape, đúng đơn vị, sốc được gradient, hàm mất mát
giảm khi học, và độ trễ suy luận nằm trong ngưỡng 100 ms/mẫu.

Đây là bước kiểm chứng **trước khi** huấn luyện đầy đủ (thuộc phần Bước 5),
nên chỉ dùng một lô nhỏ và vài bước tối ưu hóa, chạy được trong vài phút.

Cách chạy::

    python -m src.models.smoke_test                    # cả 3 kiến trúc
    python -m src.models.smoke_test --arch lstm_s2s    # chỉ LSTM cơ sở
    python -m src.models.smoke_test --steps 10 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.dataset import build_dataloaders, get_scaler
from src.data.split import load_config
from src.evaluation.metrics import (
    compute_persistence_baseline,
    compute_seasonal_naive,
    inverse_transform_target,
)
from src.models.architecture import (
    ARCH_REGISTRY,
    build_model,
    count_parameters,
    estimated_checkpoint_mb,
)
from src.models.loss import mae_loss, mse_loss

ROOT = Path(__file__).resolve().parents[2]

# Tham số mô hình lấy từ configs/model_config.yaml
ARCH_TO_SECTION = {
    "lstm_s2s": "baseline",
    "lstm_attention": "attention",
    "mini_transformer": "transformer",
}


def _model_kwargs(section: dict) -> dict:
    """Trích tham số truyền cho `build_model` từ một khối của model_config."""
    common = {
        "input_dim": int(section.get("input_dim", 15)),
        "num_layers": int(section.get("num_layers", 2)),
        "dropout": float(section.get("dropout", 0.1)),
        "pred_len": int(section.get("pred_len", 144)),
    }
    if "d_model" in section:
        common.update({
            "d_model": int(section["d_model"]),
            "nhead": int(section.get("nhead", 4)),
        })
    else:
        common.update({
            "hidden_dim": int(section.get("hidden_dim", 128)),
        })
    return common


def _iter_batches(loader, limit: int):
    """Lấy tối đa `limit` lô đầu tiên từ loader (giữ nguyên thứ tự)."""
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        x, y = batch["input"], batch["target"]
        yield x, y


def _check_batch(x: torch.Tensor, y: torch.Tensor, pred_len: int) -> None:
    """Kiểm tra shape của lô thật: x (B,144,15), y (B,144)."""
    if x.dim() != 3 or x.size(1) != 144 or x.size(2) != 15:
        raise AssertionError(
            f"Shape lô vào sai: {tuple(x.shape)} (mong đợi (B, 144, 15))"
        )
    if y.dim() != 2 or y.size(1) != pred_len:
        raise AssertionError(
            f"Shape lô đích sai: {tuple(y.shape)} (mong đợi (B, {pred_len}))"
        )


def smoke_one_arch(
    arch: str,
    model_kwargs: dict,
    loaders: dict,
    scaler,
    steps: int = 5,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Chạy kiểm thử nhanh cho một kiến trúc, trả về dict kết quả."""
    result: dict = {"arch": arch}

    torch.manual_seed(42)
    np.random.seed(42)
    model = build_model(arch, **model_kwargs).to(device)
    result["n_params"] = count_parameters(model)
    result["ckpt_mb"] = round(estimated_checkpoint_mb(model), 3)

    # ---- 1. Lấy một lô thật từ tập huấn luyện ---------------------------- #
    batch = next(iter(loaders["train"]))
    if isinstance(batch, dict):
        x, y = batch["input"], batch["target"]
    else:
        x, y = batch[0], batch[1]
    _check_batch(x, y, model_kwargs["pred_len"])
    x = x[:batch_size].to(device)
    y = y[:batch_size].to(device)
    result["batch_shape"] = f"{tuple(x.shape)} -> {tuple(y.shape)}"

    # ---- 2. Suy luận tự hồi tiếp (chế độ dùng thật khi đánh giá) ---------- #
    model.eval()
    with torch.no_grad():
        pred = model(x, y_in=None)
    if tuple(pred.shape) != tuple(y.shape):
        raise AssertionError(
            f"[{arch}] Suy luận trả về {tuple(pred.shape)}, mong đợi {tuple(y.shape)}"
        )
    if not torch.isfinite(pred).all():
        raise AssertionError(f"[{arch}] Dự báo chứa NaN/Inf")

    # ---- 3. Chuyển về °C gốc và so với đường cơ sở ----------------------- #
    preds_c, targets_c = inverse_transform_target(
        pred.cpu().numpy(), y.cpu().numpy(), scaler
    )
    inputs_c = x.cpu().numpy()
    result["mae_c"] = round(float(np.mean(np.abs(preds_c - targets_c))), 4)
    result["persistence_mae_c"] = round(
        compute_persistence_baseline(inputs_c, targets_c, scaler), 4
    )
    result["seasonal_naive_mae_c"] = round(
        compute_seasonal_naive(inputs_c, targets_c, scaler), 4
    )
    lo, hi = -40.0, 50.0
    result["out_of_bounds"] = int(
        np.sum((preds_c < lo) | (preds_c > hi))
    )

    # ---- 4. Sốc được gradient + hàm mất mát giảm khi học ---------------- #
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, y_in=y, teacher_forcing_ratio=0.5)
        loss = mse_loss(out, y)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    result["loss_first"] = round(losses[0], 5)
    result["loss_last"] = round(losses[-1], 5)
    result["loss_decreased"] = losses[-1] < losses[0]
    result["grad_norm_mean"] = round(grad_norm, 4)
    result["mae_train_z"] = round(float(mae_loss(out, y).detach()), 5)

    # ---- 5. Độ trễ suy luận 1 mẫu/ngày (ngưỡng ≤ 100 ms) ------------------ #
    model.eval()
    sample = x[:1]
    with torch.no_grad():
        model(sample, y_in=None)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10):
            model(sample, y_in=None)
        latency_ms = (time.perf_counter() - t0) / 10 * 1000
    result["latency_ms"] = round(latency_ms, 2)
    result["latency_ok"] = latency_ms <= 100.0

    # ---- 6. Kiểm tra trên tập kiểm thử (chỉ 1 lô) ------------------------ #
    te_batch = next(iter(loaders["test"]))
    if isinstance(te_batch, dict):
        x_te, y_te = te_batch["input"], te_batch["target"]
    else:
        x_te, y_te = te_batch[0], te_batch[1]
    model.eval()
    with torch.no_grad():
        pred_te = model(x_te, y_in=None)
    if tuple(pred_te.shape) != tuple(y_te.shape):
        raise AssertionError(
            f"[{arch}] Tập kiểm thử: {tuple(pred_te.shape)} != {tuple(y_te.shape)}"
        )
    result["test_shape"] = f"{tuple(x_te.shape)} -> {tuple(pred_te.shape)}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiểm thử nhanh module mô hình trên dữ liệu Jena thật"
    )
    parser.add_argument(
        "--arch", default="all",
        choices=["all", "lstm_s2s", "lstm_attention", "mini_transformer"],
        help="Kiến trúc cần kiểm thử (mặc định: cả 3)",
    )
    parser.add_argument("--steps", type=int, default=5,
                        help="Số bước tối ưu hóa kiểm tra gradient (mặc định 5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Số mẫu trong lô kiểm thử (mặc định 8)")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-ckpt", default=None,
                        help="Đường dẫn lưu checkpoint sau khi kiểm thử")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- dữ liệu thật ---------------------------------------------------- #
    data_cfg = load_config(ROOT / args.data_config)
    loaders = build_dataloaders(data_cfg)
    scaler = get_scaler()
    print(f"[*] Dữ liệu: train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")
    print(f"[*] Thiết bị: {device}\n")

    # ---- cấu hình mô hình ------------------------------------------------ #
    model_cfg = load_config(ROOT / "configs" / "model_config.yaml")
    archs = (
        list(ARCH_TO_SECTION)
        if args.arch == "all"
        else [args.arch]
    )

    results = []
    for arch in archs:
        section = model_cfg[ARCH_TO_SECTION[arch]]
        kwargs = _model_kwargs(section)
        print(f"==> Kiểm thử {arch} ({section['type']}) "
              f"{kwargs} | steps={args.steps} batch={args.batch_size}")
        res = smoke_one_arch(
            arch, kwargs, loaders, scaler,
            steps=args.steps, batch_size=args.batch_size, device=device,
        )
        results.append(res)

        # Lưu checkpoint nếu được yêu cầu
        if args.save_ckpt:
            from src.training.trainer import save_checkpoint
            torch.manual_seed(42)
            model = build_model(arch, **kwargs)
            path = save_checkpoint(
                args.save_ckpt, model,
                {"arch": arch, "arch_kwargs": kwargs, "model_name": "smoke"},
                epoch=0, val_loss=float("nan"), teacher_forcing=0.0,
            )
            res["checkpoint"] = path

        print(f"    shape lô      : {res['batch_shape']}")
        print(f"    tham số       : {res['n_params']:,} "
              f"| checkpoint ~{res['ckpt_mb']} MB")
        print(f"    MAE (°C gốc)  : {res['mae_c']} | persistence "
              f"{res['persistence_mae_c']} | seasonal-naive "
              f"{res['seasonal_naive_mae_c']}")
        print(f"    loss MSE      : {res['loss_first']} -> {res['loss_last']} "
              f"({'giảm OK' if res['loss_decreased'] else 'KHÔNG giảm!'})")
        print(f"    độ trễ suy luận: {res['latency_ms']} ms/mẫu "
              f"({'OK' if res['latency_ok'] else 'QUÁ 100 ms!'})")
        print(f"    ngoài biên vật lý: {res['out_of_bounds']} giá trị")
        print(f"    tập kiểm thử : {res['test_shape']}\n")

    # ---- tổng kết -------------------------------------------------------- #
    print("=" * 74)
    print("TỔNG KẾT KIỂM THỬ MÔ HÌNH (dữ liệu Jena thật)")
    print("=" * 74)
    header = f"{'Kiến trúc':<18}{'Shape':<22}{'MAE °C':>9}{'Loss ↓':>8}{'ms/mẫu':>10}"
    print(header)
    print("-" * 74)
    for r in results:
        ok = "OK" if r["loss_decreased"] and r["latency_ok"] else "LỖI"
        print(f"{r['arch']:<18}{r['batch_shape']:<22}{r['mae_c']:>9}"
              f"{('Có' if r['loss_decreased'] else 'Không'):>8}"
              f"{r['latency_ms']:>10}  [{ok}]")
    print("=" * 74)
    print(f"[OK] Hoàn tất kiểm thử {len(results)} kiến trúc.")
    print("Lưu ý: đây là kiểm thử nhanh (vài bước tối ưu hóa), KHÔNG phải "
          "huấn luyện đầy đủ.\n"
          "Bước tiếp theo (Bước 5 — phần huấn luyện):\n"
          "  python -m src.training --arch baseline --name jena_lstm_v1")


if __name__ == "__main__":
    main()
