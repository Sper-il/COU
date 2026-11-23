"""
src/evaluation/run_eval.py
Chạy đánh giá đầu-cuối: nạp checkpoint -> đo chỉ số trên tập kiểm thử ->
sinh đồ thị -> ghi `metrics.json` -> dựng `report.html` ngoại tuyến.

Cách chạy::

    python -m src.evaluation.run_eval --ckpt models/checkpoints/jena_lstm_v1.pt
    python -m src.evaluation.run_eval --ckpt <ckpt> --run-dir results/runs/<ts>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.dataset import build_dataloaders, get_scaler
from src.data.split import load_config
from src.evaluation import plots as plot_lib
from src.evaluation.metrics import evaluate_model, save_metrics
from src.evaluation.visualize_html import build_report
from src.training.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def _unpack(batch):
    if isinstance(batch, dict):
        return batch["input"], batch["target"]
    return batch[0], batch[1]


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Dự báo toàn bộ tập, trả về (inputs, targets, preds) đều ở °C gốc."""
    from src.evaluation.metrics import inverse_transform_target

    model.eval()
    xs, ys, ps = [], [], []
    for batch in loader:
        x, y = _unpack(batch)
        x, y = x.to(device), y.to(device)
        out = model(x, y_in=None)
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
        ps.append(out.cpu().numpy())

    inputs = np.concatenate(xs, 0)
    targets = np.concatenate(ys, 0)
    preds = np.concatenate(ps, 0)
    return inputs, targets, preds


def measure_inference_latency(model, sample_x, device, repeats: int = 20) -> float:
    """Đo độ trễ suy luận 1 mẫu/ngày trên CPU (mục tiêu ≤ 100 ms)."""
    model.eval()
    x = sample_x.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(3):  # warm-up
            model(x, y_in=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x, y_in=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / repeats * 1000.0  # ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint Seq2Seq Jena")
    parser.add_argument("--ckpt", required=True, help="Đường dẫn checkpoint .pt")
    parser.add_argument("--run-dir", default=None, help="Thư mục lượt chạy (mặc định tạo mới)")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true", help="Bỏ qua sinh đồ thị")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (ROOT / "results" / "runs" / timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Thiết bị: {device}")
    print(f"[*] Checkpoint: {ckpt_path}")

    # ---- dữ liệu -------------------------------------------------------- #
    data_cfg = load_config(ROOT / args.data_config)
    loaders = build_dataloaders(data_cfg)
    loader = loaders[args.split]
    scaler = get_scaler()

    # ---- mô hình + chỉ số ------------------------------------------------ #
    model, payload = load_checkpoint(ckpt_path, device)
    model_name = payload.get("model_config", {}).get("model_name", ckpt_path.stem)

    t0 = time.perf_counter()
    metrics = evaluate_model(model, loader, device, scaler)
    metrics["model_name"] = model_name
    metrics["split"] = args.split
    metrics["checkpoint"] = str(ckpt_path)
    metrics["epoch"] = int(payload.get("epoch", -1))
    metrics["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    eval_time = time.perf_counter() - t0

    # ---- độ trễ suy luận -------------------------------------------------- #
    # Chỉ tạo MỘT iterator cho loader (trước đây gọi next(iter(loader)) ba
    # lần, mỗi lần dựng lại iterator và lấy một lô khác nhau).
    first_batch = next(iter(loader))
    if isinstance(first_batch, dict):
        sample_x = first_batch["input"][0]
    else:
        sample_x = first_batch[0][0]
    latency_ms = measure_inference_latency(model, sample_x, device)
    metrics["inference_latency_ms"] = round(latency_ms, 2)
    metrics["eval_time_s"] = round(eval_time, 2)

    print(f"[*] MAE = {metrics['mae']:.4f} °C | RMSE = {metrics['rmse']:.4f} °C | "
          f"persistence = {metrics['persistence_mae']:.4f} °C")
    print(f"[*] MAE@1={metrics['mae_h1']:.4f} @24={metrics['mae_h24']:.4f} "
          f"@72={metrics['mae_h72']:.4f} @144={metrics['mae_h144']:.4f} °C")
    print(f"[*] Độ trễ suy luận: {latency_ms:.1f} ms/mẫu")

    # ---- đồ thị ----------------------------------------------------------- #
    if not args.no_plots:
        inputs, targets, preds = collect_predictions(model, loader, device)
        from src.evaluation.metrics import inverse_transform_target
        preds_c, targets_c = inverse_transform_target(preds, targets, scaler)

        # ánh xạ mẫu -> ngày (dùng ngày của bước đầu tiên trong cửa sổ dự báo)
        df = loaders["_meta"]["df"]
        starts = np.asarray(loaders["_meta"]["splits"][args.split])
        dates = []
        for s in starts[: len(targets_c)]:
            idx = int(s) + 144  # bước đầu tiên của ngày được dự báo
            dates.append(df["dt"].iloc[idx].strftime("%Y-%m-%d %H:%M"))

        # lịch sử hàm mất mát (nếu có trong thư mục lượt chạy huấn luyện)
        hist_path = run_dir / "history.json"
        if not hist_path.exists():
            # thử tìm history gần nhất trong results/runs
            candidates = sorted((ROOT / "results" / "runs").glob("*/history.json"))
            if candidates:
                hist_path = candidates[-1]
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            plot_lib.plot_loss_curve(history, plots_dir / "loss.png")
            print(f"[OK] Đồ thị hàm mất mát: {plots_dir / 'loss.png'}")

        plot_lib.plot_mae_horizon(metrics, metrics, plots_dir / "mae_horizon.png")

        # mẫu tốt nhất / xấu nhất theo MAE toàn tầm
        per_sample_mae = np.mean(np.abs(preds_c - targets_c), axis=1)
        best_i = int(np.argmin(per_sample_mae))
        worst_i = int(np.argmax(per_sample_mae))

        plot_lib.plot_prediction(
            targets_c[best_i], preds_c[best_i],
            dates[best_i] if best_i < len(dates) else f"mẫu #{best_i}",
            "Dự báo tốt nhất (MAE thấp nhất)", plots_dir / "best_prediction.png",
        )
        plot_lib.plot_prediction(
            targets_c[worst_i], preds_c[worst_i],
            dates[worst_i] if worst_i < len(dates) else f"mẫu #{worst_i}",
            "Dự báo xấu nhất (MAE cao nhất)", plots_dir / "worst_prediction.png",
        )
        plot_lib.plot_worst_case(
            targets_c[worst_i], preds_c[worst_i],
            dates[worst_i] if worst_i < len(dates) else f"mẫu #{worst_i}",
            plots_dir / "worst_case.png",
        )
        metrics["best_sample_index"] = best_i
        metrics["worst_sample_index"] = worst_i
        metrics["best_sample_date"] = dates[best_i] if best_i < len(dates) else None
        metrics["worst_sample_date"] = dates[worst_i] if worst_i < len(dates) else None
        print(f"[OK] Đồ thị mẫu tốt/xấu + trường hợp tệ nhất: {plots_dir}")

    # ---- lưu kết quả ------------------------------------------------------ #
    save_metrics(metrics, run_dir / "metrics.json")

    # copy cấu hình huấn luyện vào lượt chạy để report.html hiển thị
    for name in ("train_config.yaml", "config.yaml"):
        src = ROOT / "configs" / name
        if src.exists():
            shutil.copy2(src, run_dir / name)
            break

    # ---- báo cáo HTML ------------------------------------------------------ #
    report = build_report(str(run_dir))
    print(f"[OK] Báo cáo HTML: {report}")

    # ---- tóm tắt ngưỡng nghiệm thu ---------------------------------------- #
    print("\n" + "=" * 70)
    print("TÓM TẮT NGƯỠNG NGHIỆM THU (tập 2016, tầm 144)")
    print("=" * 70)
    for key, label, thr in (
        ("mae", "MAE toàn tầm", 1.5),
        ("rmse", "RMSE toàn tầm", 2.0),
        ("mae_h24", "MAE@24", 1.2),
        ("mae_h144", "MAE@144", 2.2),
    ):
        val = metrics[key]
        mark = "ĐẠT" if val <= thr else "KHÔNG ĐẠT"
        print(f"  {label:<16} {val:7.4f} °C  (≤ {thr})  -> {mark}")
    beats = metrics["mae"] < metrics["persistence_mae"]
    print(f"  {'Persistence':<16} {metrics['persistence_mae']:7.4f} °C  "
          f"-> {'ĐẠT (đã vượt)' if beats else 'KHÔNG ĐẠT'}")
    seasonal = metrics.get("seasonal_naive_mae")
    if seasonal is not None:
        beats_s = metrics["mae"] < seasonal
        print(f"  {'Seasonal-naive':<16} {seasonal:7.4f} °C  "
              f"-> {'ĐẠT (đã vượt)' if beats_s else 'KHÔNG ĐẠT'}")
    print(f"  {'Độ trễ suy luận':<16} {latency_ms:7.1f} ms   (≤ 100)  -> "
          f"{'ĐẠT' if latency_ms <= 100 else 'KHÔNG ĐẠT'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
