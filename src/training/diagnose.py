"""Chẩn đoán nhanh: mô hình LSTM S2S có vượt được đường cơ sở persistence không?

Kịch bản dùng để ra quyết định thiết kế trước khi chạy huấn luyện dài:
- Huấn luyện trên tập con (2009–2010), đánh giá trên 2011.
- Teacher forcing giảm tuyến tính về 0 (giống hẳn `trainer.py`).
- In MAE theo tầm 1/24/72/144 cùng persistence và seasonal-naive.

Vì sao cần seasonal-naive: cửa sổ đầu vào (24 giờ quá khứ) và cửa sổ đích
(24 giờ tương lai) lệch nhau đúng 144 bước, nên ``y[t]`` xảy ra **cùng giờ
của hôm qua** với ``x[t]``. Do đó ``y[t] ≈ x[t]`` là một đường cơ sở rất mạnh
và là mức sàn thực tế mà mô hình phải vượt.

Chạy:  python -m src.training.diagnose [epochs]
"""

from __future__ import annotations

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
    inverse_transform_target,
)
from src.models.architecture import LSTMS2S
from src.models.loss import mse_loss

ROOT = Path(__file__).resolve().parents[2]


def seasonal_naive_mae(inputs: np.ndarray, targets_c: np.ndarray, scaler) -> float:
    """MAE của ``y[t] = x[t]`` — cùng giờ của hôm qua (mục tiêu ở °C)."""
    temp = inputs[:, :, 1]
    temp_c = scaler["target_scaler"].inverse_transform(temp.reshape(-1, 1)).reshape(temp.shape)
    return float(np.mean(np.abs(temp_c - targets_c)))


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    decay_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    data_cfg = load_config(ROOT / "configs" / "data_config.yaml")
    data_cfg["split"] = {"train": [2009, 2010], "val": [2011], "test": [2012]}
    loaders = build_dataloaders(data_cfg)
    scaler = get_scaler()

    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMS2S().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_train = len(loaders["train"].dataset)
    print(f"Tập chẩn đoán: train={n_train} val={len(loaders['val'].dataset)} "
          f"| epochs={epochs} | tf về 0 sau {decay_epochs} epoch")

    for epoch in range(1, epochs + 1):
        if epoch <= 1:
            tf = 0.5
        else:
            prog = min((epoch - 1) / max(decay_epochs, 1), 1.0)
            tf = 0.5 * (1.0 - prog)
        model.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        for batch in loaders["train"]:
            x, y = batch["input"], batch["target"]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, y_in=y, teacher_forcing_ratio=tf)
            loss = mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += float(loss.item())
            nb += 1
        print(f"  Epoch {epoch}: train loss {tot / nb:.5f} | tf {tf:.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)

    # đánh giá ở chế độ tự hồi tiếp
    model.eval()
    xs, ys, ps = [], [], []
    with torch.no_grad():
        for batch in loaders["val"]:
            x, y = batch["input"], batch["target"]
            out = model(x, y_in=None)
            xs.append(x.numpy())
            ys.append(y.numpy())
            ps.append(out.numpy())
    inputs = np.concatenate(xs, 0)
    targets = np.concatenate(ys, 0)
    preds = np.concatenate(ps, 0)

    preds_c, targets_c = inverse_transform_target(preds, targets, scaler)
    mae = float(np.mean(np.abs(preds_c - targets_c)))
    pers = compute_persistence_baseline(inputs, targets_c, scaler)
    seas = seasonal_naive_mae(inputs, targets_c, scaler)

    print("\n" + "=" * 64)
    print(f"MAE toàn tầm mô hình      : {mae:.4f} °C")
    print(f"MAE toàn tầm persistence  : {pers:.4f} °C")
    print(f"MAE toàn tầm seasonal-naive: {seas:.4f} °C")
    print(f"=> so với persistence: {'VƯỢT' if mae < pers else 'CHƯA VƯỢT'}"
          f" | so với seasonal: {'VƯỢT' if mae < seas else 'CHƯA VƯỢT'}")
    print("-" * 64)
    for h in (1, 24, 72, 144):
        m = float(np.mean(np.abs(preds_c[:, h - 1] - targets_c[:, h - 1])))
        p = compute_persistence_baseline(inputs, targets_c[:, :h], scaler)
        print(f"  MAE@{h:<4} mô hình {m:.4f} | persistence {p:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
