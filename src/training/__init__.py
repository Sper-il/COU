"""Module huấn luyện Seq2Seq cho dự báo nhiệt độ Jena.

Xuất `train` (vòng huấn luyện), `TrainConfig` (cấu hình) và `build_scheduler`
(lịch giảm tốc độ học) để các mô-đun khác import từ đây.
"""

from src.training.scheduler import build_scheduler
from src.training.trainer import (
    TrainConfig,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train,
    train_one_epoch,
    validate,
)

__all__ = [
    "TrainConfig",
    "build_scheduler",
    "get_device",
    "load_checkpoint",
    "save_checkpoint",
    "set_seed",
    "train",
    "train_one_epoch",
    "validate",
]
