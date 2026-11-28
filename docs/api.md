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
    def forward(x: (B,144,15), y_in: (B,144) | None,
                teacher_forcing_ratio: float = 0.0) -> (B,144): ...

class LSTMS2SAttention(nn.Module):   # mở rộng Attention Bahdanau
    def forward(x: (B,144,15), y_in: (B,144) | None,
                teacher_forcing_ratio: float = 0.0) -> (B,144): ...

class MiniTransformer(nn.Module):    # đối chiếu Transformer mini (d_model=64, nhead=4, 2 lớp)
    def forward(x: (B,144,15), y_in: (B,144) | None,
                teacher_forcing_ratio: float = 0.0) -> (B,144): ...

# src/models/loss.py
mse_loss(pred: (B,144), target: (B,144)) -> scalar  # MSE khi train
mae_loss(pred: (B,144), target: (B,144)) -> scalar  # MAE để theo dõi trong log

# src/models/architecture.py — factory
build_model(arch: str, **kwargs) -> nn.Module
#   arch: lstm_s2s | lstm_attention (bahdanau) | mini_transformer (transformer)
#   Tham số không thuộc kiến trúc bị bỏ qua (vd hidden_dim với Transformer).
```

Lúc inference `y_in = None` (decoder tự hồi tiếp 144 bước); lúc train `y_in` là target dịch phải 1 bước (teacher forcing).

Cả 3 kiến trúc dùng chung dạng đầu ra **neo theo cùng giờ + phần dư**:
`y_t = w_t*x[t] + (1-w_t)*x[143] + delta_t`, với `w_t = sigmoid(gate_t)` học được
(`gate` khởi tạo thang tăng dần từ ~0,05 lên ~0,95 theo tầm dự báo).

Kiểm thử nhanh trên dữ liệu Jena thật (1 lệnh, vài phút):

```bash
python -m src.models.smoke_test                      # cả 3 kiến trúc
python -m src.models.smoke_test --arch lstm_s2s      # chỉ một kiến trúc
```

In ra: shape lô, số tham số, MAE (°C gốc) so với persistence và seasonal-naive,
loss MSE trước/sau, độ trễ suy luận (ms/mẫu), số giá trị ngoài biên vật lý.

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

## 6. Notebook huấn luyện trên Google Colab

> Tệp: `notebooks/colab_train.ipynb` (sinh bằng `python notebooks/build_colab_notebook.py`).
> Dùng để huấn luyện trên GPU khi máy cá nhân chỉ có CPU.

```text
notebooks/
├── colab_train.ipynb      # notebook Colab: sweep 15 epoch → train 30 epoch → đánh giá
├── build_colab_notebook.py  # script sinh notebook từ các ô trong _nbcells/
└── _nbcells/              # nguồn từng ô (md = markdown, py = code) + ORDER.txt
```

Các ô chính của notebook (21 ô, chạy theo thứ tự từ trên xuống):

| # | Nội dung | Kết quả |
|---|----------|---------|
| 1 | Cài thư viện (không cài lại `torch` để giữ CUDA) | In GPU đang dùng |
| 2 | Clone repo `Sper-il/COU` | Mã nguồn trong `/content/COU` |
| 3 | Tải dữ liệu Jena | `data/raw/.../jena_climate_2009_2016.csv` |
| 4 | Cấu hình chung: thiết bị, ngưỡng nghiệm thu, mốc EDA | `DEVICE`, `THRESHOLDS` |
| 5 | Kiểm tra chia tập + shape lô | `(64,144,15) → (64,144)` |
| 6 | Đường cơ sở persistence và seasonal-naive | Mốc sàn cần vượt |
| 7 | Hàm `train_loop` (teacher forcing về 0, cắt chuẩn, dừng sớm) | `history` chi tiết từng epoch |
| 8 | `evaluate_model` + `print_scoreboard` | Chỉ số °C gốc kèm đạt/không đạt |
| 9 | `plot_training_curves` — 6 biểu đồ theo dõi train | Loss, MAE, teacher forcing, lr, thời gian, overfit |
| 10 | `plot_config_comparison` + `plot_learning_curves_grid` | Bảng xếp hạng cấu hình |
| 11–13 | **Sweep 9 cấu hình × 15 epoch** trên tập kiểm định 2015 | `results/sweep_results.csv` |
| 14 | Phân tích sweep, chọn cấu hình tốt nhất | `BEST_NAME` |
| 15 | **Train 30 epoch** với cấu hình tốt nhất | `final_model` |
| 16 | Quan sát chi tiết + diễn giải tự động | Nhận xét quá khớp / TF về 0 / dừng sớm |
| 17 | Đánh giá tập kiểm thử 2016 + mẫu tốt/xấu | `results/final_metrics.json` |
| 18 | So sánh 3 kiến trúc (bảng bắt buộc của đề) | `results/compare_architectures.csv` |
| 19 | Đo độ trễ suy luận (ngưỡng ≤ 100 ms/mẫu) | `results/latency.csv` |
| 20 | Tổng kết và đối chiếu ngưỡng nghiệm thu | Bảng kết luận |
| 21 | Lưu kết quả lên Google Drive | `MyDrive/jena-seq2seq/` |

Kiểm tra các ô chạy được (dùng tập con, 1–2 epoch) trước khi đẩy lên Colab:

```bash
python notebooks/_verify_nb.py
```

Quy tắc khi chọn cấu hình: **chỉ chọn trên tập kiểm định 2015**, tập kiểm thử 2016
giữ lại để đánh giá một lần cuối — tránh rò rỉ thông tin qua việc tinh chỉnh.
