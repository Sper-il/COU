"""Phân tích khám phá (EDA) Jena — chốt F=15 và cửa sổ 144→144.
Chạy: python notebooks/2026-11-03_eda.py
Đầu ra: results/eda/*.png + eda_summary.json
"""
import json
import sys
from pathlib import Path

# Ép stdout sang UTF-8 để chạy được trên Windows (tránh lỗi cp932)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "data_config.yaml", encoding="utf-8"))
OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

CSV = ROOT / CFG["data_path"]
print(f"Đọc {CSV} ...")
df = pd.read_csv(CSV)
df["dt"] = pd.to_datetime(df[CFG["datetime_col"]], format=CFG["datetime_format"])
df = df.sort_values("dt").reset_index(drop=True)
feats = CFG["feature_cols_original"]
target = CFG["target_col"]

# 1. Tổng quan + mã thiếu -9999
summary = {}
summary["shape"] = list(df.shape)
summary["time_min"] = str(df["dt"].min())
summary["time_max"] = str(df["dt"].max())
summary["nan_per_col"] = {c: int(df[c].isna().sum()) for c in df.columns if c != "dt"}
sent = CFG["sentinel_value"]
summary["sentinel_minus9999"] = {c: int((df[c] == sent).sum()) for c in feats}
summary["target_describe"] = df[target].describe().to_dict()
df["year"] = df["dt"].dt.year
summary["year_counts"] = {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()}
gaps = df["dt"].diff().dropna()
summary["freq_mode_min"] = float(gaps.dt.total_seconds().div(60).mode().iloc[0])
summary["max_gap"] = str(gaps.max())
summary["corr_with_T"] = df[feats].corr(numeric_only=True)[target].sort_values(ascending=False).to_dict()

# Xử lý mã thiếu: thay -9999 ở wv bằng NaN rồi nội suy tuyến tính
for c in ["wv (m/s)", "max. wv (m/s)"]:
    df[c] = df[c].replace(sent, np.nan)
    df[c] = df[c].interpolate(limit_direction="both")

# 2. Hình 1: chuỗi nhiệt độ 2 tuần mẫu + chu kỳ ngày
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
sample = df.iloc[10000:10000 + 144 * 14]
ax[0].plot(sample["dt"], sample[target], linewidth=0.9)
ax[0].set_title("Chuỗi nhiệt độ T (degC) — 14 ngày mẫu")
ax[0].set_ylabel("°C")
ax[0].tick_params(axis="x", rotation=20)
hourly = df.groupby(df["dt"].dt.hour)[target].agg(["mean", "std"])
ax[1].plot(hourly.index, hourly["mean"], marker="o", label="Trung bình theo giờ")
ax[1].fill_between(hourly.index, hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=0.25, label="±1 độ lệch chuẩn")
ax[1].set_title("Nhiệt độ trung bình theo giờ trong ngày")
ax[1].set_xlabel("Giờ trong ngày"); ax[1].set_ylabel("°C"); ax[1].legend(); ax[1].set_xticks(range(0, 24))
fig.tight_layout(); fig.savefig(OUT / "01_nhietdo_chuoi_va_chukyngay.png", dpi=130); plt.close(fig)

# 3. Hình 2: phân phối 14 biến (hist)
fig, axes = plt.subplots(4, 4, figsize=(14, 10))
axes = axes.ravel()
for i, c in enumerate(feats):
    axes[i].hist(df[c].dropna(), bins=60, alpha=0.8)
    axes[i].set_title(c, fontsize=9)
    axes[i].tick_params(labelsize=7)
for j in range(len(feats), len(axes)):
    axes[j].axis("off")
fig.suptitle("Phân phối 14 biến Jena")
fig.tight_layout(); fig.savefig(OUT / "02_phanphoi_14bien.png", dpi=130); plt.close(fig)

# 4. Hình 3: ma trận tương quan
corr = df[feats].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(feats, fontsize=8)
ax.set_title("Ma trận tương quan Pearson 14 biến")
fig.colorbar(im, ax=ax, label="Pearson r")
fig.tight_layout(); fig.savefig(OUT / "03_tuongquan.png", dpi=130); plt.close(fig)

# 5. Hình 4: chứng minh tách wd thành sin/cos
wd = df["wd (deg)"].to_numpy(dtype=float)
theta = np.deg2rad(wd)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(wd, bins=72)
ax[0].set_title("wd (deg) gốc")
ax[0].set_xlabel("độ")
ax[1].scatter(theta[:5000], np.sin(theta[:5000]), s=2)
ax[1].set_title("wd_sin = sin(wd)")
ax[2].scatter(theta[:5000], np.cos(theta[:5000]), s=2, color="darkorange")
ax[2].set_title("wd_cos = cos(wd)")
fig.suptitle("Hướng gió wd và dạng mã hoá sin/cos (F = 15)")
fig.tight_layout(); fig.savefig(OUT / "04_gio_sincos.png", dpi=130); plt.close(fig)

# 6. Hình 5: khoảng trống thời gian + nhiệt độ theo tháng
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
monthly = df.groupby(df["dt"].dt.month)[target].mean()
ax[0].bar(monthly.index, monthly.values)
ax[0].set_title("Nhiệt độ trung bình theo tháng")
ax[0].set_xlabel("Tháng"); ax[0].set_ylabel("°C"); ax[0].set_xticks(range(1, 13))
top_gaps = gaps.nlargest(30).dt.total_seconds().div(3600)
ax[1].hist(gaps.dt.total_seconds().div(60).clip(0, 60), bins=40)
ax[1].set_title("Phân phối khoảng cách mẫu (phút, cắt 60 phút)")
ax[1].set_xlabel("phút")
fig.tight_layout(); fig.savefig(OUT / "05_thang_va_khoangtrong.png", dpi=130); plt.close(fig)

# 7. Hình 6: ví dụ cửa sổ 144->144
fig, ax = plt.subplots(figsize=(12, 3.8))
s = 50000
x = df[target].iloc[s:s + 144]
y = df[target].iloc[s + 144:s + 288]
ax.plot(range(144), x.values, label="Đầu vào: 1 ngày quá khứ (144 bước)")
ax.plot(range(144, 288), y.values, label="Mục tiêu: 1 ngày tiếp theo (144 bước)")
ax.axvline(144, color="red", linestyle="--", label="Ranh giới dự báo")
ax.set_title(f"Ví dụ cửa sổ ngày-tới-ngày từ {df['dt'].iloc[s]}")
ax.set_xlabel("Bước 10 phút"); ax.set_ylabel("°C"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "06_cuaso_144_144.png", dpi=130); plt.close(fig)

# 8. Baseline persistence trên 2016
t2016 = df.loc[df["year"] == 2016, target].to_numpy()
n = len(t2016) // 144 * 144
tt = t2016[:n].reshape(-1, 144)
prev_last = np.concatenate([[tt[0, 0]], tt[:-1, -1]])
pred = np.repeat(prev_last[:, None], 144, axis=1)
mae_full = float(np.abs(tt - pred).mean())
mae_h = {}
for h in [1, 24, 72, 144]:
    mae_h[f"MAE@{h}"] = float(np.abs(tt[:, :h] - pred[:, :h]).mean())
summary["persistence_2016"] = {"n_days": int(tt.shape[0]), "MAE_full": mae_full, **mae_h}
summary["n_windows_train_s12"] = int((len(df) - 288) // 12 + 1)
summary["n_windows_eval_s144"] = int((len(df) - 288) // 144 + 1)
summary["decision_F"] = "F = 15 (14 biến, wd gốc thay bằng wd_sin/wd_cos)"
summary["plots"] = sorted([p.name for p in OUT.glob("*.png")])

with open(OUT / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("MAE persistence 2016:", round(mae_full, 3), mae_h)
print(f"Xong EDA — {len(summary['plots'])} hình + eda_summary.json tại {OUT}")
