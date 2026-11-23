"""
src/evaluation/plots.py
Sinh các đồ thị cho báo cáo `report.html` (tầm 144, tập kiểm thử 2016).

Toàn bộ hình được lưu dưới dạng PNG trong `results/runs/<ts>/plots/` để
`visualize_html.py` nhúng base64, bảo đảm báo cáo mở ngoại tuyến 100%.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không cần màn hình, chạy được trên server/CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Bảng màu thống nhất cho cả báo cáo
C_TRAIN = "#1e3a8a"
C_VAL = "#b91c1c"
C_PRED = "#0f766e"
C_TRUE = "#111827"
C_GRID = "#e2e8f0"


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_loss_curve(history: list[dict], out_path: str | Path) -> str:
    """Đường hàm mất mát huấn luyện / kiểm định theo epoch."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    ax.plot(epochs, train_loss, marker="o", markersize=4, color=C_TRAIN,
            label="Huấn luyện (train loss)")
    ax.plot(epochs, val_loss, marker="s", markersize=4, color=C_VAL,
            label="Kiểm định (val loss, tự hồi tiếp 144 bước)")

    best_idx = int(np.argmin(val_loss))
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], s=110, color="#15803d",
               zorder=5, label=f"Checkpoint tốt nhất (epoch {epochs[best_idx]})")

    _style_axis(ax, "Hàm mất mát MSE qua các epoch", "Epoch", "MSE (đơn vị chuẩn hóa)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_mae_horizon(metrics: dict, persistence: dict | None = None,
                      out_path: str | Path = "mae_horizon.png") -> str:
    """MAE của mô hình và các đường cơ sở tại các tầm 1/24/72/144.

    Vẽ cả hai đường cơ sở:
      * **Persistence** — lặp giá trị cuối ngày hôm qua.
      * **Seasonal-naive** — neo cùng giờ hôm qua (``y[t] = x[t]``).
    """
    horizons = [1, 24, 72, 144]
    mae_model = [metrics.get(f"mae_h{h}", float("nan")) for h in horizons]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    ax.plot(horizons, mae_model, marker="o", markersize=6, linewidth=2,
            color=C_PRED, label="Seq2Seq (LSTM)")
    if persistence:
        mae_pers = [persistence.get(f"persistence_mae_h{h}", float("nan")) for h in horizons]
        ax.plot(horizons, mae_pers, marker="s", markersize=6, linewidth=2,
                linestyle="--", color="#a16207", label="Persistence (đường cơ sở)")
    mae_seas = [metrics.get(f"seasonal_naive_mae_h{h}", float("nan")) for h in horizons]
    if any(v == v for v in mae_seas):  # có ít nhất một giá trị hữu hạn
        ax.plot(horizons, mae_seas, marker="^", markersize=6, linewidth=1.8,
                linestyle=":", color="#7c3aed",
                label="Seasonal-naive (cùng giờ hôm qua)")

    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}\n({'10 phút' if h == 1 else f'{h / 6:.0f} giờ'})" for h in horizons])
    _style_axis(ax, "Sai số MAE theo tầm dự báo (tập kiểm thử 2016)",
                "Tầm dự báo (bước 10 phút)", "MAE (°C)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_prediction(true_c: np.ndarray, pred_c: np.ndarray, date_label: str,
                    title: str, out_path: str | Path) -> str:
    """So sánh chuỗi thực tế và dự báo trong 1 ngày (144 bước)."""
    steps = np.arange(len(true_c))
    hours = steps / 6.0

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=130)
    ax.plot(hours, true_c, linewidth=2.0, color=C_TRUE, label="Thực tế (°C)")
    ax.plot(hours, pred_c, linewidth=2.0, color=C_PRED, label="Dự báo Seq2Seq (°C)")
    ax.fill_between(hours, true_c, pred_c, color=C_PRED, alpha=0.12)

    _style_axis(ax, f"{title} — {date_label}", "Giờ trong ngày (bước 10 phút)",
                "Nhiệt độ (°C)")
    ax.set_xlim(0, len(true_c) / 6.0)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_worst_case(true_c: np.ndarray, pred_c: np.ndarray, date_label: str,
                     out_path: str | Path) -> str:
    """Trường hợp tệ nhất: thêm biểu đồ sai số theo bước và thống kê."""
    err = np.abs(true_c - pred_c)
    steps = np.arange(len(err))
    hours = steps / 6.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=130,
                                   gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(hours, err, linewidth=1.8, color="#b91c1c")
    ax1.fill_between(hours, err, color="#b91c1c", alpha=0.15)
    ax1.set_xticks(range(0, 25, 4))
    _style_axis(ax1, f"Sai số tuyệt đối theo bước — trường hợp tệ nhất ({date_label})",
                "Giờ trong ngày", "|Dự báo - Thực tế| (°C)")

    ax2.bar([1, 24, 72, 144],
            [err[0], err[23], err[71], err[143]],
            color=["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"])
    ax2.set_xticks([1, 24, 72, 144])
    _style_axis(ax2, "Sai số tại các mốc", "Tầm", "°C")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
