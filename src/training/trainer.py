"""Vòng huấn luyện Seq2Seq cho bài toán dự báo nhiệt độ Jena.

Cung cấp ``train(config) -> best_ckpt_path`` theo đúng `docs/api.md` mục 3:

* Adam (``lr = 1e-3``, ``weight_decay = 1e-5``), lô 64, tối đa 30 epoch.
* Teacher forcing giảm dần: ``0,5 × 0,98`` mỗi epoch.
* Cắt chuẩn gradient 1,0.
* Dừng sớm theo hàm mất mát kiểm định (patience 5).
* Hạt giống 42 cho tính tái lập.
* Checkpoint lưu ``state_dict`` + ``model_config`` + ``epoch`` + ``val_loss``
  + ``teacher_forcing`` để mô-đun đánh giá nạp lại được trên máy khác.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.dataset import build_dataloaders
from src.data.split import load_config
from src.models.architecture import build_model, count_parameters, estimated_checkpoint_mb
from src.models.loss import mae_loss, mse_loss
from src.training.scheduler import build_scheduler

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Cố định hạt giống cho python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    """Chọn thiết bị: ``auto`` dùng cuda nếu có, ngược lại cpu."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Bóc (X, y) từ batch của `JenaWindowDataset` (dict) hoặc tuple/list."""
    if isinstance(batch, dict):
        return batch["input"], batch["target"]
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    raise TypeError(f"Batch không hợp lệ: {type(batch)}")


# --------------------------------------------------------------------------- #
# Cấu hình huấn luyện
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Cấu hình huấn luyện đã chuẩn hóa (đọc từ `configs/train_config.yaml`)."""

    arch: str = "lstm_s2s"
    version: str = "v1"
    model_name: str = "jena_lstm_v1"
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    teacher_forcing: float = 0.5
    teacher_forcing_decay: float = 0.98
    teacher_forcing_min: float = 0.0
    # Số epoch để đưa teacher forcing từ giá trị khởi đầu xuống 0.
    # Lịch giảm dần về 0 là bắt buộc: bộ giải mã phải học được chế độ tự hồi tiếp
    # đúng như lúc suy luận, nếu không sẽ hỏng do exposure bias (sai số tích lũy).
    teacher_forcing_decay_epochs: int = 10
    teacher_forcing_warmup_epochs: int = 1
    clip: float = 1.0
    early_stop: int = 5
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    # siêu tham số mô hình
    input_dim: int = 15
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    d_model: int = 64
    nhead: int = 4
    pred_len: int = 144
    # đường dẫn
    data_config: str = "configs/data_config.yaml"
    run_dir: str | None = None
    ckpt_dir: str = "models/checkpoints"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        train_config_path: str | Path,
        model_config_path: str | Path | None = None,
        arch: str | None = None,
        model_name: str | None = None,
        overrides: dict | None = None,
    ) -> "TrainConfig":
        """Đọc `configs/train_config.yaml` (+ `model_config.yaml`) thành `TrainConfig`."""
        train_cfg = load_config(train_config_path)
        cfg = cls()
        for key in (
            "batch_size", "epochs", "lr", "optimizer", "weight_decay",
            "teacher_forcing", "teacher_forcing_decay", "teacher_forcing_min",
            "teacher_forcing_decay_epochs", "teacher_forcing_warmup_epochs",
            "clip", "early_stop", "seed",
        ):
            if key in train_cfg:
                setattr(cfg, key, train_cfg[key])
        if "num_workers" in train_cfg:
            cfg.num_workers = int(train_cfg["num_workers"])

        # siêu tham số mô hình theo kiến trúc
        arch_key = arch or "baseline"
        if model_config_path and Path(model_config_path).exists():
            model_cfg = load_config(model_config_path)
            section = model_cfg.get(arch_key, {})
            if section:
                for key in ("input_dim", "hidden_dim", "num_layers", "dropout",
                            "d_model", "nhead"):
                    if key in section:
                        setattr(cfg, key, section[key])
                if "pred_len" in section:
                    cfg.pred_len = int(section["pred_len"])
                arch_type = section.get("type")
                if arch_type:
                    cfg.arch = arch_type
        if arch:
            # tên kiến trúc rút gọn -> khớp registry
            mapping = {
                "baseline": "lstm_s2s",
                "attention": "lstm_attention",
                "transformer": "mini_transformer",
            }
            cfg.arch = mapping.get(arch, arch)

        if model_name:
            cfg.model_name = model_name
        if overrides:
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        return cfg

    def model_kwargs(self) -> dict:
        """Tham số truyền cho `build_model` theo kiến trúc."""
        if self.arch in ("mini_transformer", "transformer"):
            return {
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "pred_len": self.pred_len,
            }
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "pred_len": self.pred_len,
        }


# --------------------------------------------------------------------------- #
# Vòng huấn luyện
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    teacher_forcing_ratio: float,
    clip: float,
) -> dict:
    """Một epoch huấn luyện. Trả về {loss, mae} (đơn vị chuẩn hóa)."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)

        # mục tiêu dịch phải 1 bước: teacher_input[t] = y[t-1]
        # (hàm forward tự ghép ký hiệu bắt đầu = T cuối ngày hôm qua)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, y_in=y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_mae += float(mae_loss(pred, y).item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> dict:
    """Đánh giá kiểm định theo đúng chế độ suy luận (tự hồi tiếp 144 bước).

    Đánh giá bằng chế độ tự hồi tiếp là điều kiện cần để checkpoint được chọn
    đúng: hàm mất mát kiểm định phải phản ánh đúng lỗi suy luận thật (teacher
    forcing trong huấn luyện che mất sai số tích lũu ở nửa sau ngày).
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        pred = model(x, y_in=None)
        total_loss += float(criterion(pred, y).item())
        total_mae += float(mae_loss(pred, y).item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict,
    epoch: int,
    val_loss: float,
    teacher_forcing: float,
    extra: dict | None = None,
) -> str:
    """Lưu checkpoint gồm state_dict + cấu hình để nạp lại được ở máy khác."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "teacher_forcing": float(teacher_forcing),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return str(path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Nạp checkpoint -> (model, payload). Tự dựng lại mô hình từ `model_config`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    mc = payload["model_config"]
    model = build_model(mc["arch"], **mc.get("arch_kwargs", {}))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def train(config: TrainConfig | dict | None = None, **overrides):
    """Huấn luyện một kiến trúc và trả về đường dẫn checkpoint tốt nhất.

    Args:
        config: `TrainConfig` hoặc dict cấu hình. Nếu None thì đọc mặc định
            trong `configs/`.
        **overrides: ghi đè thuộc tính của `TrainConfig` (vd `epochs=2` để thử nhanh).

    Returns:
        Đường dẫn (str) tới checkpoint có hàm mất mát kiểm định tốt nhất.
    """
    if config is None:
        config = TrainConfig.from_files(
            ROOT / "configs" / "train_config.yaml",
            ROOT / "configs" / "model_config.yaml",
        )
    elif isinstance(config, dict):
        merged = dict(config)
        merged.update(overrides)
        config = TrainConfig(**merged)
        overrides = {}
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    set_seed(config.seed)
    device = get_device(config.device)

    # ---- thư mục lượt chạy -------------------------------------------- #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.run_dir) if config.run_dir else (ROOT / "results" / "runs" / timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- dữ liệu ------------------------------------------------------- #
    data_cfg = load_config(ROOT / config.data_config)
    data_cfg["batch_size"] = config.batch_size
    data_cfg["num_workers"] = config.num_workers
    data_cfg["seed"] = config.seed
    loaders = build_dataloaders(data_cfg)

    # ---- mô hình -------------------------------------------------------- #
    model = build_model(config.arch, **config.model_kwargs()).to(device)
    n_params = count_parameters(model)
    ckpt_mb = estimated_checkpoint_mb(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimizer, config.extra)
    criterion = mse_loss

    model_config = {
        "arch": config.arch,
        "arch_kwargs": config.model_kwargs(),
        "model_name": config.model_name,
        "version": config.version,
        "input_dim": config.input_dim,
        "pred_len": config.pred_len,
        "data_config": config.data_config,
    }

    # ---- log ------------------------------------------------------------ #
    log_path = run_dir / "logs.txt"
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    best_ckpt = str(ckpt_dir / f"{config.model_name}.pt")
    patience_left = config.early_stop
    tf_ratio = config.teacher_forcing

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("=" * 78)
    log(f"Bắt đầu huấn luyện: {config.arch} ({config.model_name})")
    log(f"Thiết bị: {device} | hạt giống: {config.seed}")
    log(f"Số tham số: {n_params:,} | checkpoint ước tính: {ckpt_mb:.2f} MB")
    log(f"Tập: train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")
    log(f"Siêu tham số: {config.model_kwargs()}")
    log(f"Train: epochs={config.epochs} batch={config.batch_size} lr={config.lr} "
        f"clip={config.clip} early_stop={config.early_stop}")
    log(f"Teacher forcing: {config.teacher_forcing} x {config.teacher_forcing_decay}/epoch")
    log("=" * 78)

    for epoch in range(1, config.epochs + 1):
        # Lịch teacher forcing: giữ ở mức khởi đầu trong epoch đầu để mô hình
        # ổn định, sau đó giảm TUYẾN TÍNH về 0 trong `teacher_forcing_decay_epochs`
        # epoch. Khi tỉ lệ về 0, mô hình buộc phải dựa vào dự báo của chính nó
        # (đúng chế độ suy luận) nên không còn "sốc" khi chạy tự do.
        if epoch <= config.teacher_forcing_warmup_epochs:
            tf_ratio = config.teacher_forcing
        else:
            span = max(config.teacher_forcing_decay_epochs, 1)
            progress = min((epoch - config.teacher_forcing_warmup_epochs) / span, 1.0)
            tf_ratio = config.teacher_forcing * (1.0 - progress)
            tf_ratio = max(tf_ratio, config.teacher_forcing_min)

        t0 = time.time()
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, tf_ratio, config.clip
        )
        train_time = time.time() - t0

        t1 = time.time()
        val_stats = validate(model, loaders["val"], criterion, device)
        val_time = time.time() - t1

        scheduler.step(val_stats["loss"])

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mae": train_stats["mae"],
            "val_loss": val_stats["loss"],
            "val_mae": val_stats["mae"],
            "teacher_forcing": tf_ratio,
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_s": train_time,
            "val_time_s": val_time,
        })

        improved = val_stats["loss"] < best_val
        log(
            f"Epoch {epoch:03d} | train loss {train_stats['loss']:.5f} "
            f"mae {train_stats['mae']:.5f} | val loss {val_stats['loss']:.5f} "
            f"mae {val_stats['mae']:.5f} | tf {tf_ratio:.4f} "
            f"| lr {optimizer.param_groups[0]['lr']:.2e} "
            f"| {train_time:.1f}s + {val_time:.1f}s"
            + ("  <-- tốt nhất" if improved else "")
        )

        if improved:
            best_val = val_stats["loss"]
            best_epoch = epoch
            save_checkpoint(
                best_ckpt, model, model_config, epoch, best_val, tf_ratio,
                extra={"arch": config.arch, "model_name": config.model_name},
            )
            patience_left = config.early_stop
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"DỪNG SỚM: val loss không cải thiện trong {config.early_stop} epoch.")
                break

    # ---- kết thúc -------------------------------------------------------- #
    elapsed = sum(h["train_time_s"] + h["val_time_s"] for h in history)
    log("=" * 78)
    log(f"Huấn luyện kết thúc sau {len(history)} epoch, tổng {elapsed / 60:.1f} phút")
    log(f"Checkpoint tốt nhất: {best_ckpt} (epoch {best_epoch}, val loss {best_val:.5f})")

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(run_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "arch": config.arch,
                "model_name": config.model_name,
                "model_config": model_config,
                "n_parameters": n_params,
                "checkpoint_mb": round(ckpt_mb, 3),
                "epochs_run": len(history),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "device": str(device),
                "total_minutes": round(elapsed / 60, 2),
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return best_ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Seq2Seq Jena")
    parser.add_argument(
        "--arch", default="baseline",
        choices=["baseline", "attention", "transformer"],
        help="Kiến trúc: baseline (LSTM S2S), attention (Bahdanau), transformer",
    )
    parser.add_argument("--name", default=None, help="Tên checkpoint, vd jena_lstm_v1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    default_names = {
        "baseline": "jena_lstm_v1",
        "attention": "jena_attn_v1",
        "transformer": "jena_trans_v1",
    }
    overrides = {"device": args.device, "data_config": args.data_config}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.run_dir:
        overrides["run_dir"] = args.run_dir

    cfg = TrainConfig.from_files(
        ROOT / "configs" / "train_config.yaml",
        ROOT / "configs" / "model_config.yaml",
        arch=args.arch,
        model_name=args.name or default_names[args.arch],
    )
    ckpt = train(cfg, **overrides)
    print(f"\n[OK] Checkpoint tốt nhất: {ckpt}")


if __name__ == "__main__":
    main()
