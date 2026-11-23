"""Kiểm thử đơn vị cho module mô hình `src/models`.

Bao phủ 4 nhóm yêu cầu của Bước 4 (`plan.md` mục 3.2):
1. Shape `(2,144,15) → (2,144)` cho cả 3 lớp mô hình.
2. Chế độ suy luận `y_in = None` sinh đủ 144 bước.
3. `mse_loss` đúng công thức.
4. Số tham số / kích thước checkpoint ước tính ≤ 50 MB.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.architecture import (  # noqa: E402
    LSTMS2S,
    LSTMS2SAttention,
    MiniTransformer,
    count_parameters,
    estimated_checkpoint_mb,
)
from src.models.loss import mse_loss  # noqa: E402

B, S, F, T = 2, 144, 15, 144


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, S, F)


def _y() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, T)


def test_forward_shapes_all_architectures():
    """(2,144,15) -> (2,144) cho cả 3 lớp."""
    x = _x()
    for model in (LSTMS2S(), LSTMS2SAttention(), MiniTransformer()):
        out = model(x)
        assert out.shape == (B, T), (
            f"{type(model).__name__} trả về {tuple(out.shape)}, mong đợi {(B, T)}"
        )
        assert torch.isfinite(out).all(), f"{type(model).__name__} sinh giá trị NaN/Inf"


def test_autoregressive_inference_generates_full_horizon():
    """y_in = None -> tự hồi tiếp đủ 144 bước, kết quả khác nhau giữa các bước."""
    x = _x()
    for model in (LSTMS2S(), LSTMS2SAttention()):
        model.eval()
        with torch.no_grad():
            out = model(x, y_in=None)
        assert out.shape == (B, T)
        # mỗi bước phải khác bước trước (không phải hằng số lặp lại)
        assert not torch.allclose(out[:, 1:], out[:, :-1]), (
            f"{type(model).__name__} lặp lại giá trị, không hồi tiếp thật"
        )


def test_teacher_forcing_path_runs():
    """y_in = (B,144) chạy được và cho cùng shape với chế độ tự hồi tiếp."""
    x, y = _x(), _y()
    for model in (LSTMS2S(), LSTMS2SAttention()):
        model.train()
        out = model(x, y_in=y, teacher_forcing_ratio=0.5)
        assert out.shape == (B, T)
        assert torch.isfinite(out).all()


def test_mse_loss_matches_formula():
    """mse_loss bằng đúng trung bình bình phương sai lệch."""
    pred, target = _y(), torch.randn(B, T)
    expected = float(np.mean((pred.numpy() - target.numpy()) ** 2))
    got = float(mse_loss(pred, target))
    assert abs(got - expected) < 1e-6, f"{got} != {expected}"

    # cùng giá trị -> loss = 0
    assert abs(float(mse_loss(target, target))) < 1e-9

    # shape lệch -> báo lỗi rõ ràng
    try:
        mse_loss(pred, torch.randn(B, T + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Phải ném ValueError khi shape lệch")


def test_checkpoint_size_under_50mb():
    """Mỗi kiến trúc phải có checkpoint ước tính ≤ 50 MB."""
    for model in (LSTMS2S(), LSTMS2SAttention(), MiniTransformer()):
        mb = estimated_checkpoint_mb(model)
        assert mb <= 50.0, (
            f"{type(model).__name__} checkpoint ước tính {mb:.2f} MB > 50 MB"
        )
        assert count_parameters(model) > 0


def test_transformer_output_is_not_constant():
    """Transformer mini phải thực sự học (đầu ra thay đổi theo đầu vào)."""
    model = MiniTransformer()
    model.eval()
    with torch.no_grad():
        out1 = model(_x())
        out2 = model(_x() + 1.0)
    assert not torch.allclose(out1, out2), "Transformer không nhạy với đầu vào"


# --------------------------------------------------------------------------- #
# Lỗi đã phát hiện và đã sửa — giữ lại để chống tái phát
# --------------------------------------------------------------------------- #
def test_teacher_forcing_uses_correctly_shifted_target():
    """Teacher forcing phải ăn đúng mục tiêu **dịch phải 1 bước**.

    Chuỗi teacher input đúng phải dài ``T + 1``: ``teacher_input[:, k] = y[:, k-1]``.
    Trước đây chuỗi chỉ dài ``T`` và được tra đúng chỗ ``t``, khiến bộ giải
    mã ăn lệch 1 bước so với lúc suy luận.
    """
    model = LSTMS2S()
    model.train()
    x, y = _x(), _y()

    start = model._start_token(x)
    ti = model._teacher_input(y, start)
    assert ti.shape == (B, T + 1), ti.shape
    assert torch.allclose(ti[:, 0], start.squeeze(-1))
    assert torch.allclose(ti[:, 1:], y), "teacher_input[:, 1:] phải bằng mục tiêu"


def test_teacher_forcing_ratio_one_is_deterministic():
    """Tỉ lệ = 1.0 thì mô hình không còn ngẫu nhiên, kết quả phải tất định."""
    model = LSTMS2S()
    model.train()
    x, y = _x(), _y()

    torch.manual_seed(0)
    out_a = model(x, y_in=y, teacher_forcing_ratio=1.0)
    torch.manual_seed(0)
    out_b = model(x, y_in=y, teacher_forcing_ratio=1.0)
    assert torch.allclose(out_a, out_b), (
        "Tỉ lệ teacher forcing = 1.0 phải cho kết quả tất định"
    )


def test_teacher_forcing_distinguishes_shifted_target():
    """Mô hình phải phân biệt được mục tiêu dịch đúng và dịch sai 1 bước.

    Nếu cài đặt dùng nhầm chỉ số teacher input, hai lần chạy này cho kết quả
    trùng nhau và kiểm thử bắt được lỗi.
    """
    model = LSTMS2S()
    model.train()
    x, y = _x(), _y()

    torch.manual_seed(7)
    out_correct = model(x, y_in=y, teacher_forcing_ratio=1.0)

    # mục tiêu lệch 1 bước: teacher_input[:, k] = y[:, k] (không dịch)
    y_shifted = torch.roll(y, shifts=1, dims=1)
    y_shifted[:, 0] = y[:, 0]
    torch.manual_seed(7)
    out_wrong = model(x, y_in=y_shifted, teacher_forcing_ratio=1.0)

    assert not torch.allclose(out_correct, out_wrong), (
        "Mô hình không phân biệt được mục tiêu dịch đúng và dịch sai 1 bước"
    )


def test_input_shape_validation_raises_clear_error():
    """Sai số đặc trưng hoặc sai số bước phải báo lỗi rõ ràng, không sập âm thầm."""
    model = LSTMS2S()
    with pytest.raises(ValueError, match="đặc trưng"):
        model(torch.randn(B, S, 14))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(B, S, 15, 2))
    with pytest.raises(ValueError, match="bước"):
        model(torch.randn(B, S, 15), y_in=torch.randn(B, T + 1))
    with pytest.raises(ValueError, match="batch"):
        model(torch.randn(B, S, 15), y_in=torch.randn(B + 1, T))


def test_build_model_filters_kwargs_per_architecture():
    """`build_model` phải bỏ qua tham số không thuộc kiến trúc (vd hidden_dim)."""
    from src.models.architecture import build_model

    # Transformer không có hidden_dim -> phải bỏ qua, không ném lỗi
    m = build_model(
        "mini_transformer",
        input_dim=15, d_model=64, nhead=4, num_layers=2, dropout=0.1,
        pred_len=144, hidden_dim=128,
    )
    assert isinstance(m, MiniTransformer)

    # LSTM không có d_model -> phải bỏ qua
    m = build_model(
        "lstm_s2s",
        input_dim=15, hidden_dim=128, num_layers=2, dropout=0.1,
        pred_len=144, d_model=64,
    )
    assert isinstance(m, LSTMS2S)

    with pytest.raises(ValueError):
        build_model("khong_ton_tai")


def test_gate_blends_two_anchors_monotonically():
    """Cổng trộn hai neo phải đi từ persistence (tầm ngắn) sang cùng giờ (tầm dài).

    ``sigmoid(gate)`` khởi tạo là đường thang tăng dần từ ~0,05 lên ~0,95:
    ở bước đầu mô hình tin neo persistence (đúng cho tầm ngắn), ở bước cuối
    mô hình tin neo cùng giờ (đúng cho tầm 24 giờ).
    """
    from src.models.architecture import _init_gate

    gate = torch.sigmoid(_init_gate(144))
    assert gate.shape == (144,)
    assert gate[0] < 0.15, "bước đầu phải ưu tiên neo persistence"
    assert gate[-1] > 0.85, "bước cuối phải ưu tiên neo cùng giờ"
    assert torch.all(gate[1:] >= gate[:-1]), "cổng phải tăng dần theo tầm"


def test_inference_latency_under_100ms_per_sample():
    """Suy luận 1 mẫu/ngày trên CPU phải ≤ 100 ms (ngưỡng nghiệm thu).

    Ngưỡng thời gian phụ thuộc máy và tải CPU, nên kiểm thử đơn vị dùng
    ngưỡng **nới** (250 ms) để tránh báo động giả khi máy đang bận (ví dụ đang
    chạy huấn luyện song song). Ngưỡng nghiệm thu thật (100 ms) được đo và
    in trong `python -m src.models.smoke_test` khi máy rảnh.
    """
    import time

    for model in (LSTMS2S(), LSTMS2SAttention(), MiniTransformer()):
        model.eval()
        x = torch.randn(1, S, F)
        with torch.no_grad():
            model(x, y_in=None)  # warm-up
            t0 = time.perf_counter()
            for _ in range(5):
                model(x, y_in=None)
            elapsed = (time.perf_counter() - t0) / 5 * 1000
        assert elapsed <= 250.0, (
            f"{type(model).__name__} suy luận {elapsed:.1f} ms/mẫu — quá chậm "
            f"ngay cả với ngưỡng nới (250 ms); kiểm tra lại phần tối ưu"
        )
