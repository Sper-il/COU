# === 4. Cấu hình chung cho cả chế độ sweep và chế độ train chính ===

import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Thư mục lưu kết quả (bản clone mới chưa có)
Path("results").mkdir(exist_ok=True)

# `display` có sẵn trong Colab/Jupyter; thêm bản dự phòng cho môi trường khác.
try:
    from IPython.display import display  # noqa: F401
except Exception:  # noqa: BLE001
    def display(*args, **kwargs):
        """In ra console khi không có IPython."""
        for obj in args:
            print(obj)

from src.data.dataset import build_dataloaders, get_scaler
from src.data.split import load_config
from src.models.architecture import build_model, count_parameters
from src.models.loss import mae_loss, mse_loss
from src.training.scheduler import build_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False

# Ngưỡng nghiệm thu lấy từ docs/description.md mục 6
THRESHOLDS = {"mae": 1.5, "rmse": 2.0, "mae_h24": 1.2, "mae_h144": 2.2}
HORIZONS = (1, 24, 72, 144)

# Mốc đường cơ sở đo từ EDA (results/eda/eda_summary.json) — dùng để tham chiếu
PERSISTENCE_2016 = 2.8058
SEASONAL_2016 = 2.4747


def set_seed(seed: int = SEED) -> None:
    """Cố định hạt giống cho python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_config="configs/data_config.yaml"):
    """Dựng 3 DataLoader train/val/test theo cấu hình trong repo."""
    cfg = load_config(data_config)
    cfg["batch_size"] = 64
    cfg["num_workers"] = 2
    cfg["seed"] = SEED
    return build_dataloaders(cfg)


print("Thiết bị:", DEVICE)
print("Torch:", torch.__version__)
