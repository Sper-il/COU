# Thiết kế hệ thống — Dự án Seq2Seq Jena

> Giai đoạn 2 sau định nghĩa yêu cầu. Mọi quyết định quan trọng phải ghi thêm vào `docs/progress_status.md` mục nhật ký quyết định. Thiết kế này khớp 9 dòng đề: hồi quy, Jena, dự báo thời tiết, chuỗi-tới-chuỗi, ngày tiếp theo, Encoder–Decoder LSTM, nhiều bước, attention, Transformer.

## 1. Kiến trúc tổng thể — Hệ thống gồm gì

- **Mẫu kiến trúc (pattern):** quy trình học máy theo lô cổ điển, gồm 4 tầng `Dữ liệu → Mô hình → Huấn luyện → Đánh giá`, không máy chủ, không phát trực tuyến.
- **Mô tả 2 câu:** tầng dữ liệu biến CSV Jena thô thành các cửa sổ ngày-tới-ngày `(144, 15) → (144,)` đã chuẩn hóa; tầng mô hình/huấn luyện học ánh xạ chuỗi-tới-chuỗi cho 1 ngày tiếp theo rồi sinh điểm kiểm tra, tầng đánh giá biến điểm kiểm tra thành `metrics.json` kèm `report.html` ngoại tuyến.

```text
data/raw/jena_*.csv  (chu kỳ 10 phút, 14 biến + Date Time)
        │  src/data (đọc, làm sạch, chuẩn hóa, cửa sổ 144→144, chia theo năm)
        v
data/processed + data/splits  ──►  DataLoader (B,144,15)
        │  src/models (lstm encoder-decoder / kèm attention / transformer mini)
        v
src/training (teacher forcing giảm dần, dừng sớm) ──► models/checkpoints + trained
        │  src/evaluation (chuyển ngược tỉ lệ, MAE/RMSE tầm 1/24/72/144, persistence)
        v
results/runs/<thời gian>/{metrics.json, plots/, report.html}
```

## 2. Các mô-đun — Danh sách mô-đun và trách nhiệm

| Mô-đun | Thư mục | Trách nhiệm | Đầu vào → Đầu ra |
|--------|---------|-------------|------------------|
| Dữ liệu | `src/data/` (`dataset.py`, `transforms.py`, `split.py`) | Đọc CSV Jena, nội suy mã thiếu `-9999` ở 2 cột gió, tách `wd` thành sin/cos (chốt `F = 15`), khớp bộ chuẩn hóa trên tập huấn luyện, tạo cửa sổ 144→144, chia theo năm 2009–2014/2015/2016 | CSV thô → `DataLoader` theo lô `(B,144,15)`, `(B,144)` |
| Mô hình | `src/models/` (`architecture.py`, `loss.py`) | 3 kiến trúc đúng đề: (a) LSTM Encoder–Decoder cơ sở, (b) kèm Attention Bahdanau, (c) Transformer mini; hàm mất mát MSE cho hồi quy | `(B,144,15)` → `(B,144)` |
| Huấn luyện | `src/training/` (`trainer.py`, `scheduler.py`) | Vòng huấn luyện/kiểm định, teacher forcing giảm dần (cần cho giải mã dài 144), dừng sớm, điểm kiểm tra, nhật ký | DataLoader kèm cấu hình → checkpoint tốt nhất |
| Đánh giá | `src/evaluation/` (`metrics.py`, `visualize.py`, `visualize_html.py`) | Chuyển ngược 144 giá trị về °C, MAE/RMSE toàn tầm và tại tầm 1/24/72/144, đường cơ sở persistence lặp giá trị cuối, dựng `report.html` | Checkpoint kèm bộ đọc kiểm thử → `metrics.json` kèm `report.html` |
| Demo | `src/demo.py` | Nạp checkpoint tốt nhất, dự báo một ngày mẫu 144 bước, vẽ nhanh | Một mẫu `(144,15)` → `(144,)` kèm hình |

## 3. Cấu trúc thư mục — Tổ chức theo chuẩn

```text
e:/COU/
├── data/{raw, processed, splits}      # bỏ qua git trừ mẫu nhỏ
├── src/{data, models, training, evaluation}
├── models/{checkpoints, trained, configs}
├── configs/{data_config.yaml, model_config.yaml, train_config.yaml}
├── notebooks/                         # phân tích khám phá đặt tên YYYY-MM-DD_*.ipynb
├── tests/{unit, integration}
├── docs/{description.md, define.md, design.md (= tệp này),
│         architecture.md, data-flow.md, api.md, progress_status.md, AI_Usage.md}
├── results/runs/<thời gian>/{metrics.json, plots/, report.html}
└── README.md
```

Việc dựng đủ thư mục kèm `.gitignore` và `requirements.txt` là mốc tuần 1.

## 4. Luồng dữ liệu — Dữ liệu đi từ đâu đến đâu (kèm kích thước)

1. **Thô:** `jena_climate_2009_2016.csv` (khoảng 420 nghìn dòng, 14 cột số kèm `Date Time`, đúng tập Jena trong đề).
2. **Làm sạch:** sắp xếp theo thời gian, nội suy tuyến tính mã thiếu `-9999` ở 2 cột gió, tách `wd → (wd_sin, wd_cos)` (chốt `F = 15` từ EDA, bỏ `wd` gốc), chuẩn hóa.
3. **Chia theo năm:** huấn luyện 2009–2014, kiểm định 2015, kiểm thử 2016 (không xáo trộn liên tập, chống rò rỉ).
4. **Cửa sổ ngày-tới-ngày:** `X (144,15)` của 1 ngày quá khứ → `y (144,)` của 1 ngày tiếp theo; bước trượt huấn luyện 12 (2 giờ), đánh giá 144 (1 ngày, không chồng lấn).
5. **Lô:** `(64,144,15) → (64,144)` đưa vào mô hình; bộ giải mã lúc huấn luyện nhận `y` dịch phải một bước với teacher forcing khởi đầu 0,5 rồi giảm dần.
6. **Đầu ra:** véc-tơ `(144,)` đã chuẩn hóa → chuyển ngược về °C → chỉ số tại tầm 1/24/72/144 kèm đồ thị theo ngày.

Tần suất: theo lô ngoại tuyến; không thời gian thực.

## 5. Giao diện — Đầu vào, đầu ra từng mô-đun

```python
# src/data/dataset.py
build_dataloaders(config) -> {train, val, test}  # mỗi bộ đọc nhả (X:(B,144,F), y:(B,144))
get_scaler() -> StandardScaler                    # đã khớp trên huấn luyện, dùng chuyển ngược cho y

# src/models/architecture.py
class LSTMS2S(nn.Module):            # cơ sở mức Must, đúng Model trong đề
    forward(x:(B,144,15), y_in:(B,144) | None) -> (B,144)
class LSTMS2SAttention(nn.Module):   # mở rộng attention trong đề
    forward(x:(B,144,15), y_in:(B,144) | None) -> (B,144)
class MiniTransformer(nn.Module):    # đối sánh transformer trong đề
    forward(x:(B,144,15)) -> (B,144)

# src/training/trainer.py
train(config) -> best_ckpt_path      # dừng sớm theo MSE kiểm định tầm 144

# src/evaluation/metrics.py
evaluate(ckpt, loader) -> {mae, rmse, mae_h1, mae_h24, mae_h72, mae_h144, persistence_mae, ...}
```

Chi tiết đầy đủ xem `docs/api.md`.

## 6. Cột mốc — Mốc theo mô-đun (đề xuất 4 tuần, có thể nén)

| Mốc | Việc | Kết quả nghiệm thu |
|-----|------|--------------------|
| Tuần 1: Dựng khung và phân tích khám phá | Dựng thư mục, cấu hình 144→144, sổ tay phân tích, chốt F=15 (tách gió sin/cos) | `notebooks/*_eda.ipynb` kèm bảng thống kê NaN/phân phối |
| Tuần 2: Dữ liệu | Cài đặt `src/data`, kiểm thử đơn vị về kích thước `(144,15)/(144,)`, rò rỉ bộ chuẩn hóa, chia thời gian | `DataLoader` chạy được, kiểm thử đạt |
| Tuần 3: Mô hình và huấn luyện | Chạy đầu-cuối LSTM cơ sở tầm 144, sau đó kèm Attention và Transformer | 3 checkpoint kèm `metrics.json` kiểm định |
| Tuần 4: Đánh giá, trực quan và tài liệu | `report.html` theo ngày, demo một lệnh một ngày mẫu, README kèm AI_Usage và tiến độ | Nháy đúp `report.html` mở ngoại tuyến, số khớp `metrics.json` |

## 7. Rủi ro kỹ thuật — Rủi ro và cách giảm thiểu (chú ý giải mã dài 144)

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Rò rỉ do xáo trộn/chuẩn hóa sai | Trung bình | Cao (chỉ số ảo) | Chia theo năm trước khi chuẩn hóa; bộ chuẩn hóa chỉ khớp trên huấn luyện; kiểm thử đảm bảo `scaler.fit` không gọi trên kiểm định/kiểm thử | Kiểm thử đơn vị về rò rỉ |
| Sai số tích lũy ở nửa sau ngày dự báo (MAE@144 vọt) | Cao | Trung bình | Teacher forcing giảm dần kèm Attention; báo cáo chỉ số tại 1/24/72/144 thay vì giấu | Đồ thị MAE theo tầm |
| Huấn luyện chậm trên CPU vì chuỗi dài 144 (mỗi epoch quá 20 phút) | Trung bình | Trung bình | Lô 64, hidden 128, bước trượt đánh giá 144; lấy mẫu theo giờ ở mức Could; dùng GPU nếu có | Đo thời gian epoch đầu |
| Quá khớp (hàm mất mát kiểm định tăng sau vài epoch) | Trung bình | Trung bình | Dừng sớm kiên nhẫn 5, dropout 0,1, weight decay | Đường hàm mất mát huấn luyện/kiểm định |
| Hướng gió dạng góc gây nhiễu | Thấp | Thấp | Tách sin/cos; thử có/không để đối sánh | Phân tích khám phá kèm thử nghiệm |

## 8. Cấu hình chốt cho bản chạy tối thiểu (nguồn đúng duy nhất, chép vào `configs/` khi viết mã)

```yaml
data: { input_len: 144, pred_len: 144, stride_train: 12, stride_eval: 144,
        split: { train: 2009-2014, val: 2015, test: 2016 }, scaler: standard }
model_baseline: { type: lstm_s2s, hidden: 128, layers: 2, dropout: 0.1 }
model_attention: { type: bahdanau, hidden: 128, layers: 2 }
model_transformer: { d_model: 64, nhead: 4, layers: 2, dropout: 0.1 }
train: { batch: 64, epochs: 30, lr: 1e-3, optimizer: adam,
         teacher_forcing: 0.5, clip: 1.0, early_stop: 5, seed: 42 }
eval: { horizons: [1, 24, 72, 144], baseline: persistence }
```
