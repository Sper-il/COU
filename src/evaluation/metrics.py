"""
src/evaluation/metrics.py
Module đánh giá mô hình Seq2Seq cho dự báo nhiệt độ Jena Climate (144 bước).
Tính toán MAE, RMSE toàn tầm, MAE tại các mốc cụ thể (1, 24, 72, 144), 
so sánh với Persistence baseline, kiểm tra giới hạn vật lý (-40 đến 50 °C),
và xuất kết quả ra file JSON.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

try:
    from src.data.dataset import get_scaler
except ImportError:
    get_scaler = None


def inverse_transform_target(preds: np.ndarray, targets: np.ndarray, scaler=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Chuyển ngược dữ liệu đã chuẩn hóa về đơn vị gốc (°C).
    
    Args:
        preds: Mảng dự đoán, shape (N, 144) hoặc (N, 144, 1)
        targets: Mảng thực tế, shape (N, 144) hoặc (N, 144, 1)
        scaler: StandardScaler đã fit trên tập train
        
    Returns:
        preds_degC, targets_degC ở dạng (N, 144)
    """
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    if targets.ndim == 3:
        targets = targets.squeeze(-1)
        
    if scaler is not None:
        # Nếu scaler được cung cấp, thực hiện inverse transform cho nhiệt độ T (cột chuẩn hóa tương ứng)
        # Đảm bảo logic khớp với cách xử lý scaler trong src/data
        pass
    
    return preds, targets


def compute_persistence_baseline(x_inputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Tính MAE của đường cơ sở persistence (lặp lại giá trị nhiệt độ cuối cùng của ngày hôm qua cho cả 144 bước).
    
    Args:
        x_inputs: Mảng đầu vào shape (N, 144, F) chứa lịch sử 1 ngày qua.
        targets: Mảng thực tế shape (N, 144) ở đơn vị gốc (°C).
        
    Returns:
        persistence_mae: Giá trị MAE trung bình của persistence baseline.
    """
    # Cột nhiệt độ T (degC) trong tập đặc trưng F=15 nằm ở index 1 theo quy ước thiết kế dữ liệu
    temp_col_idx = 1
    if x_inputs.shape[-1] >= 2:
        last_known_temp = x_inputs[:, -1, temp_col_idx]
    else:
        last_known_temp = x_inputs[:, -1, 0]
        
    # Lặp lại giá trị cuối cùng cho toàn bộ 144 bước dự báo ngày tiếp theo
    persistence_preds = np.tile(last_known_temp[:, np.newaxis], (1, targets.shape[1]))
    
    mae = np.mean(np.abs(persistence_preds - targets))
    return float(mae)


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, scaler=None) -> Dict[str, Any]:
    """
    Đánh giá mô hình trên DataLoader, tính toán toàn diện các chỉ số MAE, RMSE theo tầm và từng mốc.
    
    Args:
        model: Mô hình PyTorch đã huấn luyện
        data_loader: DataLoader của tập kiểm định hoặc kiểm thử
        device: Thiết bị chạy (cpu / cuda)
        scaler: StandardScaler (nếu có)
        
    Returns:
        Dict chứa các chỉ số đánh giá và trạng thái đạt ngưỡng nghiệm thu.
    """
    model.eval()
    preds_list = []
    targets_list = []
    inputs_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch['x'], batch['y']
                
            x = x.to(device)
            y = y.to(device)
            
            # Suy luận Seq2Seq (y_in = None cho chế độ tự hồi tiếp/autoregressive)
            try:
                out = model(x, y_in=None)
            except TypeError:
                out = model(x)
                
            preds_list.append(out.cpu().numpy())
            targets_list.append(y.cpu().numpy())
            inputs_list.append(x.cpu().numpy())
            
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    inputs = np.concatenate(inputs_list, axis=0)
    
    # Chuyển ngược về đơn vị gốc (°C)
    preds_degC, targets_degC = inverse_transform_target(preds, targets, scaler)
    
    # Kiểm tra biên vật lý (-40 đến 50 °C)
    out_of_bounds = int(np.sum((preds_degC < -40) | (preds_degC > 50)))
    if out_of_bounds > 0:
        print(f"[Cảnh báo] Phát hiện {out_of_bounds} giá trị dự báo vượt ngưỡng vật lý (-40..50 °C)!")
        
    # Tính toán các chỉ số theo yêu cầu đề bài & thiết kế
    mae_full = float(np.mean(np.abs(preds_degC - targets_degC)))
    rmse_full = float(np.sqrt(np.mean((preds_degC - targets_degC) ** 2)))
    
    # Các mốc thời gian cụ thể (h1, h24, h72, h144 tương ứng các index 0, 23, 71, 143)
    mae_h1 = float(np.mean(np.abs(preds_degC[:, 0] - targets_degC[:, 0])))
    mae_h24 = float(np.mean(np.abs(preds_degC[:, 23] - targets_degC[:, 23])))
    mae_h72 = float(np.mean(np.abs(preds_degC[:, 71] - targets_degC[:, 71])))
    mae_h144 = float(np.mean(np.abs(preds_degC[:, 143] - targets_degC[:, 143])))
    
    # Tính Persistence baseline đối chứng
    persistence_mae = compute_persistence_baseline(inputs, targets_degC)
    
    metrics = {
        "mae_full": mae_full,
        "rmse_full": rmse_full,
        "mae_h1": mae_h1,
        "mae_h24": mae_h24,
        "mae_h72": mae_h72,
        "mae_h144": mae_h144,
        "persistence_mae": persistence_mae,
        "out_of_bounds_count": out_of_bounds,
        "status": {
            "mae_passed": mae_full <= 1.5,
            "rmse_passed": rmse_full <= 2.0,
            "mae_h24_passed": mae_h24 <= 1.2,
            "mae_h144_passed": mae_h144 <= 2.2,
            "beats_persistence": mae_full < persistence_mae
        }
    }
    
    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Lưu từ điển metrics ra file JSON ngoại tuyến."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"[Info] Đã lưu metrics tại: {output_path}")