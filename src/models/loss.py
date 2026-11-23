"""Hàm mất mát cho bài toán hồi quy chuỗi-tới-chuỗi.

Hàm mất mát huấn luyện là MSE (Mean Squared Error) giữa dự báo và mục tiêu,
đúng theo `docs/define.md` mục 3. Cả hai đầu vào đều ở không gian đã chuẩn hóa
(StandardScaler fit trên tập huấn luyện), nên giá trị loss là "đơn vị chuẩn hóa".
"""

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE trung bình trên toàn bộ (B, 144).

    Args:
        pred: Dự báo, shape (B, pred_len), float32.
        target: Mục tiêu thật, shape (B, pred_len), float32.

    Returns:
        Scalar tensor (0-dim) chứa giá trị MSE.

    Raises:
        ValueError: Nếu shape của hai đầu vào không khớp nhau.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    return F.mse_loss(pred, target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAE trên toàn bộ (B, 144) — dùng khi theo dõi trong log huấn luyện."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    return F.l1_loss(pred, target)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE chỉ tính trên các vị trí có `mask == 1` (tuỳ chọn).

    Hiện bản chạy tối thiểu không dùng mask (mọi bước đều hợp lệ), nhưng hàm
    được giữ lại để mở rộng đa ngày {144, 288, 432} mà không phải viết lại.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    if mask is None:
        return mse_loss(pred, target)
    mask = mask.to(pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom
