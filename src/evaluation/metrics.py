"""
src/evaluation/metrics.py
Module đánh giá mô hình Seq2Seq cho dự báo nhiệt độ Jena Climate (144 bước).
Tính toán MAE, RMSE toàn tầm, MAE tại các mốc cụ thể (1, 24, 72, 144),
so sánh với Persistence baseline, kiểm tra giới hạn vật lý (-40 đến 50 °C),
và xuất kết quả ra file JSON.

Lưu ý về đơn vị: mô hình xuất giá trị **đã chuẩn hóa** (StandardScaler fit trên
tập huấn luyện). Mọi chỉ số báo cáo ở đây đều được chuyển ngược về **°C gốc**
trước khi tính, nhờ vậy con số so sánh được với ngưỡng nghiệm thu và với mốc
persistence 2,806 °C đo từ EDA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

try:
    from src.data.dataset import get_scaler
except ImportError:  # pragma: no cover - khi chạy ngoài gói src
    get_scaler = None

# Ngưỡng nghiệm thu (docs/description.md mục 6, tập kiểm thử 2016, tầm 144)
THRESHOLDS = {
    "mae": 1.5,
    "rmse": 2.0,
    "mae_h24": 1.2,
    "mae_h144": 2.2,
}

# Biên vật lý hợp lý cho nhiệt độ (docs/define.md mục 5)
PHYSICAL_BOUNDS = (-40.0, 50.0)

# Chỉ số theo tầm dự báo (bước 10 phút): 1, 24, 72, 144
HORIZONS = (1, 24, 72, 144)


def _target_scaler(scaler):
    """Trả về StandardScaler của biến mục tiêu T (degC) từ nhiều dạng lưu trữ."""
    if scaler is None:
        return None
    # scaler.pkl lưu dict {"feature_scaler", "target_scaler", ...}
    if isinstance(scaler, dict):
        return scaler.get("target_scaler")
    return scaler


def inverse_transform_target(
    preds: np.ndarray, targets: np.ndarray, scaler=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chuyển dự báo và mục tiêu đã chuẩn hóa về đơn vị gốc (°C).

    Args:
        preds: Mảng dự đoán, shape (N, 144) hoặc (N, 144, 1)
        targets: Mảng thực tế, shape (N, 144) hoặc (N, 144, 1)
        scaler: StandardScaler (hoặc dict chứa "target_scaler") đã fit trên tập train

    Returns:
        preds_degC, targets_degC ở dạng (N, 144)
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    if targets.ndim == 3:
        targets = targets.squeeze(-1)

    tgt_scaler = _target_scaler(scaler)
    if tgt_scaler is not None:
        # scaler đã fit trên cột T dạng (N, 1) -> reshape để khớp
        preds = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        targets = tgt_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

    return preds, targets


def _last_step_temp_celsius(
    x_inputs: np.ndarray, scaler=None, temp_col_idx: int = 1
) -> np.ndarray:
    """Nhiệt độ bước cuối của ngày quá khứ, quy về °C gốc.

    `x_inputs` là dữ liệu **đã chuẩn hóa**, còn mục tiêu so sánh là °C gốc,
    nên phải chuyển ngược trước khi tính persistence. Bộ chuẩn hóa biến mục
    tiêu và cột `T (degC)` trong bộ đặc trưng được fit trên cùng một cột dữ
    liệu nên có cùng tham số — dùng chung `target_scaler` là đúng.
    """
    x_inputs = np.asarray(x_inputs, dtype=float)
    idx = temp_col_idx if x_inputs.shape[-1] > temp_col_idx else 0
    last = x_inputs[:, -1, idx]  # (N,)

    tgt_scaler = _target_scaler(scaler)
    if tgt_scaler is not None:
        last = tgt_scaler.inverse_transform(last.reshape(-1, 1)).reshape(-1)
    return last


def compute_seasonal_naive(
    x_inputs: np.ndarray, targets: np.ndarray, scaler=None, temp_col_idx: int = 1
) -> float:
    """MAE của đường cơ sở seasonal-naive: ``y[t] = x[t]``.

    Vì ``input_len == pred_len == 144`` và hai cửa sổ lệch nhau đúng 24 giờ,
    ``y[t]`` xảy ra **cùng giờ của hôm qua** với ``x[t]``. Đây là một đường cơ
    sở rất mạnh, thường mạnh hơn persistence (trên 2011: 2,74 °C so với
    3,21 °C), nên mô hình phải vượt cả hai.
    """
    x_inputs = np.asarray(x_inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    idx = temp_col_idx if x_inputs.shape[-1] > temp_col_idx else 0
    same_clock = x_inputs[:, :, idx]  # (N, S)

    tgt_scaler = _target_scaler(scaler)
    if tgt_scaler is not None:
        same_clock = tgt_scaler.inverse_transform(
            same_clock.reshape(-1, 1)
        ).reshape(same_clock.shape)

    n = min(same_clock.shape[1], targets.shape[1])
    return float(np.mean(np.abs(same_clock[:, :n] - targets[:, :n])))


def compute_seasonal_naive_horizons(
    x_inputs: np.ndarray, targets: np.ndarray, scaler=None, temp_col_idx: int = 1
) -> Dict[str, float]:
    """MAE của seasonal-naive tại các tầm 1/24/72/144 (targets ở °C)."""
    targets = np.asarray(targets, dtype=float)
    x_inputs = np.asarray(x_inputs, dtype=float)
    idx = temp_col_idx if x_inputs.shape[-1] > temp_col_idx else 0
    same_clock = x_inputs[:, :, idx]  # (N, S)

    tgt_scaler = _target_scaler(scaler)
    if tgt_scaler is not None:
        same_clock = tgt_scaler.inverse_transform(
            same_clock.reshape(-1, 1)
        ).reshape(same_clock.shape)

    out: Dict[str, float] = {}
    for h in HORIZONS:
        n = min(h, same_clock.shape[1], targets.shape[1])
        out[f"seasonal_naive_mae_h{h}"] = float(
            np.mean(np.abs(same_clock[:, :n] - targets[:, :n]))
        )
    return out


def compute_persistence_baseline(
    x_inputs: np.ndarray, targets: np.ndarray, scaler=None, temp_col_idx: int = 1
) -> float:
    """
    Tính MAE của đường cơ sở persistence (lặp lại giá trị nhiệt độ cuối cùng
    của ngày hôm qua cho cả 144 bước dự báo ngày tiếp theo).

    Args:
        x_inputs: Mảng đầu vào shape (N, 144, F) — **đã chuẩn hóa**.
        targets: Mảng thực tế shape (N, 144) — **đơn vị gốc (°C)**.
        scaler: Bộ chuẩn hóa biến mục tiêu, dùng để chuyển nhiệt độ cuối ngày
            hôm qua về °C. Bắt buộc để so sánh cùng đơn vị với `targets`.

    Returns:
        persistence_mae: Giá trị MAE trung bình của persistence baseline.
    """
    targets = np.asarray(targets, dtype=float)
    last_known_temp = _last_step_temp_celsius(x_inputs, scaler, temp_col_idx)

    persistence_preds = np.tile(last_known_temp[:, np.newaxis], (1, targets.shape[1]))
    return float(np.mean(np.abs(persistence_preds - targets)))


def compute_persistence_horizons(
    x_inputs: np.ndarray, targets: np.ndarray, scaler=None, temp_col_idx: int = 1
) -> Dict[str, float]:
    """MAE của persistence tại các tầm 1/24/72/144 (targets ở °C)."""
    targets = np.asarray(targets, dtype=float)
    last_known_temp = _last_step_temp_celsius(x_inputs, scaler, temp_col_idx)
    preds = np.tile(last_known_temp[:, np.newaxis], (1, targets.shape[1]))
    out: Dict[str, float] = {}
    for h in HORIZONS:
        out[f"persistence_mae_h{h}"] = float(np.mean(np.abs(preds[:, :h] - targets[:, :h])))
    return out


def _unpack_batch(batch):
    """Bóc (X, y) từ batch dict của JenaWindowDataset hoặc tuple/list."""
    if isinstance(batch, dict):
        return batch["input"], batch["target"]
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    raise TypeError(f"Batch không hợp lệ: {type(batch)}")


def _predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Suy luận tuỳ lớp mô hình (Seq2Seq cần y_in=None, Transformer không cần)."""
    try:
        return model(x, y_in=None)
    except TypeError:
        return model(x)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    scaler=None,
) -> Dict[str, Any]:
    """
    Đánh giá mô hình trên DataLoader, tính toán đầy đủ các chỉ số MAE, RMSE
    theo tầm và so sánh với Persistence baseline.

    Args:
        model: Mô hình PyTorch đã huấn luyện
        data_loader: DataLoader của tập kiểm định hoặc kiểm thử
        device: Thiết bị chạy (cpu / cuda)
        scaler: StandardScaler của biến mục tiêu (hoặc dict từ `get_scaler()`)

    Returns:
        Dict chứa các chỉ số đánh giá và trạng thái đạt ngưỡng nghiệm thu.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    preds_list, targets_list, inputs_list = [], [], []
    for batch in data_loader:
        x, y = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        out = _predict(model, x)
        preds_list.append(out.detach().cpu().numpy())
        targets_list.append(y.detach().cpu().numpy())
        inputs_list.append(x.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    inputs = np.concatenate(inputs_list, axis=0)

    # Chuyển ngược về đơn vị gốc (°C)
    preds_degC, targets_degC = inverse_transform_target(preds, targets, scaler)

    # Kiểm tra biên vật lý (-40..50 °C)
    lo, hi = PHYSICAL_BOUNDS
    out_of_bounds = int(np.sum((preds_degC < lo) | (preds_degC > hi)))
    if out_of_bounds > 0:
        print(
            f"[Cảnh báo] Phát hiện {out_of_bounds} giá trị dự báo vượt ngưỡng "
            f"vật lý ({lo:.0f}..{hi:.0f} °C)!"
        )

    # Chỉ số chính
    mae_full = float(np.mean(np.abs(preds_degC - targets_degC)))
    rmse_full = float(np.sqrt(np.mean((preds_degC - targets_degC) ** 2)))

    # MAE/RMSE tại các mốc: h1->index 0, h24->23, h72->71, h144->143
    horizon_metrics: Dict[str, float] = {}
    for h in HORIZONS:
        idx = h - 1
        err = preds_degC[:, idx] - targets_degC[:, idx]
        horizon_metrics[f"mae_h{h}"] = float(np.mean(np.abs(err)))
        horizon_metrics[f"rmse_h{h}"] = float(np.sqrt(np.mean(err ** 2)))

    # Persistence baseline trên cùng tập (cùng đơn vị °C)
    persistence_mae = compute_persistence_baseline(inputs, targets_degC, scaler)
    persistence = compute_persistence_horizons(inputs, targets_degC, scaler)
    # Seasonal-naive: y[t] = x[t] (cùng giờ hôm qua) — mốc sàn thực tế
    seasonal_mae = compute_seasonal_naive(inputs, targets_degC, scaler)
    seasonal = compute_seasonal_naive_horizons(inputs, targets_degC, scaler)

    mae = mae_full
    rmse = rmse_full
    metrics: Dict[str, Any] = {
        "mae": mae,
        "rmse": rmse,
        "mae_full": mae,
        "rmse_full": rmse,
        **horizon_metrics,
        "persistence_mae": persistence_mae,
        **persistence,
        "seasonal_naive_mae": seasonal_mae,
        **seasonal,
        "out_of_bounds_count": out_of_bounds,
        "n_samples": int(preds.shape[0]),
    }

    # Đánh giá ngưỡng nghiệm thu
    metrics["thresholds"] = dict(THRESHOLDS)
    metrics["status"] = {
        "mae_passed": mae <= THRESHOLDS["mae"],
        "rmse_passed": rmse <= THRESHOLDS["rmse"],
        "mae_h24_passed": metrics["mae_h24"] <= THRESHOLDS["mae_h24"],
        "mae_h144_passed": metrics["mae_h144"] <= THRESHOLDS["mae_h144"],
        "beats_persistence": mae < persistence_mae,
        "beats_seasonal_naive": mae < seasonal_mae,
    }
    metrics["all_passed"] = all(metrics["status"].values())
    return metrics


def evaluate(
    ckpt: str | Path,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    scaler=None,
) -> Dict[str, Any]:
    """
    Nạp checkpoint rồi đánh giá — chữ ký theo `docs/api.md` mục 4.

    Args:
        ckpt: Đường dẫn tới file checkpoint `.pt`.
        loader: DataLoader tập kiểm định/kiểm thử.
        device: cpu hoặc cuda.
        scaler: Bộ chuẩn hóa; nếu None thì tự nạp từ `data/processed/scaler.pkl`.

    Returns:
        Dict chỉ số (xem `evaluate_model`).
    """
    from src.training.trainer import load_checkpoint

    device = torch.device(device)
    model, payload = load_checkpoint(ckpt, device)

    if scaler is None and get_scaler is not None:
        try:
            scaler = get_scaler()
        except Exception:
            scaler = None

    metrics = evaluate_model(model, loader, device, scaler)
    metrics["checkpoint"] = str(ckpt)
    metrics["epoch"] = int(payload.get("epoch", -1))
    metrics["model_config"] = payload.get("model_config", {})
    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Lưu điển kèm metrics ra file JSON ngoại tuyến."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"[Info] Đã lưu metrics tại: {output_path}")
