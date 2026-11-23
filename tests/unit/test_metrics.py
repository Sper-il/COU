"""Kiểm thử đơn vị cho module đánh giá `src/evaluation/metrics.py`.

Trọng tâm: bảo đảm chỉ số được báo cáo đúng **đơn vị °C gốc** (không phải
đơn vị chuẩn hóa) và đường cơ sở persistence được tính đúng khi so sánh.

Vì sao quan trọng: bộ đánh giá phải khớp với mốc persistence 2,806 °C đo từ
EDA trên tập 2016. Nếu inverse-transform hoặc persistence bị lẫn đơn vị,
con số sẽ ra sai nhưng vẫn "chạy được" — đây là loại lỗi nguy hiểm nhất.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    HORIZONS,
    PHYSICAL_BOUNDS,
    compute_persistence_baseline,
    compute_persistence_horizons,
    evaluate_model,
    inverse_transform_target,
)


class _FakeScaler:
    """StandardScaler giả lập: mean=0, scale=1 là đơn vị gốc."""

    class _S:
        mean_ = np.array([0.0])
        scale_ = np.array([1.0])

        def inverse_transform(self, x):
            return np.asarray(x, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)

    def transform(self, x):
        return np.asarray(x, dtype=float)


def test_inverse_transform_actually_transforms():
    """Không được bỏ trống: đầu ra phải đổi theo scaler."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.array([[10.0], [20.0], [30.0]]))
    preds = np.array([[0.0, 1.0, 2.0]])
    targets = np.array([[0.0, 1.0, 2.0]])
    p, t = inverse_transform_target(preds, targets, scaler)
    assert not np.allclose(p, preds), "inverse_transform_target không biến đổi gì"
    assert p.shape == (1, 3)
    assert np.allclose(p, t)


def test_inverse_transform_accepts_dict_scaler():
    """scaler.pkl lưu dict {'feature_scaler', 'target_scaler'} -> phải dùng target_scaler."""
    from sklearn.preprocessing import StandardScaler

    ts = StandardScaler().fit(np.array([[10.0], [20.0], [30.0]]))
    preds = np.array([[0.0, 1.0, 2.0]])
    p, _ = inverse_transform_target(preds, preds.copy(), {"target_scaler": ts})
    assert not np.allclose(p, preds)


def test_persistence_matches_manual_computation_in_celsius():
    """Persistence lặp giá trị T cuối ngày hôm qua, tính trên °C gốc."""
    # 2 mẫu, 4 bước: cuối ngày hôm qua = 10 °C và 20 °C
    x = np.zeros((2, 4, 15))
    x[0, -1, 1] = 10.0
    x[1, -1, 1] = 20.0
    targets = np.array([[11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0]])

    mae = compute_persistence_baseline(x, targets, scaler=None)
    expected = np.mean(np.abs(np.array([10, 10, 10, 10]) - targets[0]))
    expected = (expected + np.mean(np.abs(np.array([20, 20, 20, 20]) - targets[1]))) / 2
    assert abs(mae - expected) < 1e-9


def test_persistence_handles_scaled_inputs_correctly():
    """Đầu vào chuẩn hóa + scaler -> kết quả phải bằng tính tay trên °C."""
    from sklearn.preprocessing import StandardScaler

    raw = np.array([[10.0], [20.0], [30.0], [40.0]])
    ts = StandardScaler().fit(raw)
    # mẫu: cuối ngày hôm qua = 40 °C (đã chuẩn hóa)
    x = np.zeros((1, 4, 15))
    x[0, -1, 1] = ts.transform([[40.0]])[0, 0]
    targets = np.array([[41.0, 42.0, 43.0, 44.0]])

    mae = compute_persistence_baseline(x, targets, scaler=ts)
    expected = np.mean(np.abs(np.array([40.0, 40.0, 40.0, 40.0]) - targets[0]))
    assert abs(mae - expected) < 1e-6, f"{mae} != {expected}"


def test_persistence_horizons_keys():
    x = np.zeros((3, 144, 15))
    x[:, -1, 1] = 5.0
    targets = np.zeros((3, 144))
    out = compute_persistence_horizons(x, targets, scaler=None)
    for h in HORIZONS:
        assert f"persistence_mae_h{h}" in out


def test_evaluate_model_on_dummy_loader_end_to_end():
    """Chạy đánh giá trên loader giả: kiểm tra khóa đầu ra và ngưỡng."""
    import torch

    class DummyModel(torch.nn.Module):
        """Trả về hằng số 0 (đơn vị chuẩn hóa) -> dự báo = giá trị trung bình."""

        def forward(self, x, y_in=None):
            return x[:, -1, 1:2].expand(-1, 144).clone()

    n, S, F = 4, 144, 15
    x = torch.randn(n, S, F)
    y = torch.randn(n, 144)
    loader = [(x, y)]

    metrics = evaluate_model(DummyModel(), loader, device="cpu", scaler=None)
    for key in ("mae", "rmse", "mae_h1", "mae_h24", "mae_h72", "mae_h144",
                "persistence_mae", "status"):
        assert key in metrics, f"Thiếu khoá '{key}' trong metrics"

    # Khoá "status" gồm đủ 5 tiêu chí nghiệm thu cốt lõi + 1 tiêu chí bổ sung
    # (beats_seasonal_naive) do metrics.py bổ sung cho đường cơ sở cùng giờ.
    assert {
        "mae_passed", "rmse_passed", "mae_h24_passed", "mae_h144_passed",
        "beats_persistence",
    }.issubset(set(metrics["status"])), (
        f"Thiếu tiêu chí nghiệm thu: {metrics['status']}"
    )
    assert metrics["n_samples"] == n
    assert isinstance(metrics["all_passed"], bool)


def test_evaluate_model_handles_dict_batches():
    """Loader trả dict {'input','target'} (đúng JenaWindowDataset) -> không KeyError."""
    import torch

    class DummyModel(torch.nn.Module):
        def forward(self, x, y_in=None):
            return torch.zeros(x.size(0), 144)

    x = torch.randn(3, 144, 15)
    y = torch.randn(3, 144)
    loader = [{"input": x, "target": y}]

    metrics = evaluate_model(DummyModel(), loader, device="cpu", scaler=None)
    assert metrics["n_samples"] == 3


def test_physical_bounds_are_sane():
    lo, hi = PHYSICAL_BOUNDS
    assert lo < 0 < hi < 100
