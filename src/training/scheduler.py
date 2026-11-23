"""Lịch giảm tốc độ học cho vòng huấn luyện Seq2Seq.

Yêu cầu (`docs/api.md` mục 3): ``build_scheduler(optimizer, config) -> LRScheduler``
giảm tốc độ học khi hàm mất mát kiểm định bão hòa.

Cài đặt dùng ``ReduceLROnPlateau``: theo dõi hàm mất mát kiểm định, khi loss
không cải thiện trong ``patience`` epoch thì nhân tố điều chỉnh (``factor``).
Điều này khớp trực tiếp với lịch dừng sớm: cùng một tín hiệu "bão hòa".
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Dựng ``ReduceLROnPlateau`` theo cấu hình huấn luyện.

    Args:
        optimizer: Optimizer đã cấu hình (Adam theo `configs/train_config.yaml`).
        config: Dict cấu hình huấn luyện. Khoá đọc được:
            ``lr_factor`` (mặc định 0,5), ``lr_patience`` (mặc định 3),
            ``min_lr`` (mặc định 1e-6).

    Returns:
        ``torch.optim.lr_scheduler.ReduceLROnPlateau``
    """
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_factor", 0.5)),
        patience=int(config.get("lr_patience", 3)),
        min_lr=float(config.get("min_lr", 1e-6)),
    )
