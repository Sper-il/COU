# Giao diện module (API nội bộ) — `jena-seq2seq`

> Đầu vào/đầu ra từng module kèm shape và định dạng. Nguồn đúng duy nhất cho shape là `configs/data_config.yaml` (`input_len = 144`, `pred_len = 144`, `F = 15`). Các hàm `src/data`, `src/models`, `src/training` là đặc tả để cài đặt ở giai đoạn Build; riêng `visualize_eda_html` đã cài đặt xong.

## 1. Dữ liệu — `src/data/`

```python
# src/data/dataset.py
build_dataloaders(config) -> {"train": DataLoader, "val": DataLoader, "test": DataLoader}
#   mỗi batch nhả (X, y) với X: (B,144,15) float32 đã chuẩn hóa, y: (B,144) float32 đã chuẩn hóa
#   B = 64, num_workers = 0, seed = 42

get_scaler() -> StandardScaler
#   đã fit CHỈ trên tập train; dùng inverse-transform cho y khi đánh giá về °C

# src/data/transforms.py
build_train_transforms() -> Compose  # nội suy -9999 + tách wd thành sin/cos + chuẩn hóa
build_val_transforms() -> Compose    # giống train nhưng KHÔNG fit lại scaler

# src/data/split.py
make_splits(config) -> {"train_idx": [...], "val_idx": [...], "test_idx": [...]}
#   chia theo năm 2009–2014/2015/2016, loại cửa sổ cắt qua gap 3 ngày 10/2016, lưu vào data/splits/
```

Chạy kiểm tra nhanh (sau khi cài đặt): `python -m src.data --config configs/data_config.yaml` in shape 1 batch.

## 2. Mô hình — `src/models/`

```python
# src/models/architecture.py
class LSTMS2S(nn.Module):            # cơ sở mức Must, đúng Model trong đề
    def forward(x: (B,144,15), y_in: (B,144) | None) -> (B,144): ...

class LSTMS2SAttention(nn.Module):   # mở rộng Attention Bahdanau
    def forward(x: (B,144,15), y_in: (B,144) | None) -> (B,144): ...

class MiniTransformer(nn.Module):    # đối sánh Transformer mini (d_model=64, nhead=4, 2 lớp)
    def forward(x: (B,144,15)) -> (B,144): ...

# src/models/loss.py
mse_loss(pred: (B,144), target: (B,144)) -> scalar  # MSE khi train
```

Lúc inference `y_in = None` (decoder tự hồi tiếp 144 bước); lúc train `y_in` là target dịch phải 1 bước (teacher forcing).

## 3. Huấn luyện — `src/training/`

```python
# src/training/trainer.py
train(config) -> best_ckpt_path: str
#   Adam (lr=1e-3, weight_decay=1e-5), tối đa 30 epoch, teacher_forcing=0,5 ×0,98/epoch,
#   clip=1,0, early stopping patience=5 theo MSE val tầm 144, seed=42

# src/training/scheduler.py
build_scheduler(optimizer, config) -> LRScheduler  # giảm lr khi loss val bão hòa
```

## 4. Đánh giá và trực quan — `src/evaluation/`

```python
# src/evaluation/metrics.py
evaluate(ckpt: str, loader: DataLoader) -> {
  "mae": float, "rmse": float,           # toàn tầm 144, đơn vị °C gốc
  "mae_h1": float, "mae_h24": float, "mae_h72": float, "mae_h144": float,
  "persistence_mae": float, "persistence_mae_h1/h24/h72/h144": float, ...}

# src/evaluation/visualize_html.py (giai đoạn Eval)
build_report(run_dir: str) -> str
#   đọc results/runs/<ts>/{metrics.json, config.yaml, plots/*.png} → render report.html offline

# src/evaluation/visualize_eda_html.py ✅ ĐÃ CÀI ĐẶT
#   Chạy: python -m src.evaluation.visualize_eda_html
#   Đọc: results/eda/eda_summary.json + results/eda/*.png
#   Ghi: results/eda/eda_report.html (CSS nội tuyến, ảnh base64, không CDN, ~638 KB)
```

## 5. Demo — `src/demo.py` (mức Could)

```python
# Chạy: python -m src.demo --date 2016-06-15
# Nạp checkpoint tốt nhất, dự báo 1 ngày mẫu: (144,15) → (144,) kèm hình, latency live ≤ 3 s end-to-end
```
