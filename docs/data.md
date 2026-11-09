# Quy ước dữ liệu — `jena-seq2seq`

> Chuẩn làm việc với dữ liệu: từ raw đến splits sẵn sàng cho training, tránh leakage và mất vết. Số liệu EDA dưới đây là số thật từ `results/eda/eda_summary.json`.

## 1. Nguồn dữ liệu

- Tệp duy nhất: `data/raw/jena_climate_2009_2016.csv/jena_climate_2009_2016.csv` (~43 MB, 420.551 dòng, 01-01-2009 → 01-01-2017, chu kỳ 10 phút, trạm Jena). Đúng tập Jena trong đề.
- 14 cột đặc trưng số + 1 cột `Date Time` (định dạng `%d.%m.%Y %H:%M:%S`, ví dụ `13.01.2009 00:00:00`). Biến mục tiêu duy nhất: `T (degC)`.
- Giấy phép: bộ dữ liệu Jena Climate công khai cho nghiên cứu; giữ file gốc bất biến, không sửa trực tiếp.

## 2. Mô tả thống kê (từ EDA 2026-11-03)

- `T (degC)`: trung bình 9,45 °C, độ lệch chuẩn 8,42 °C, biên −23,01…37,28 °C. Chu kỳ ngày rõ (thấp nhất ~5 giờ, cao nhất ~15 giờ), mùa vụ rõ (tháng 1 ~ −0,1 °C, tháng 7 ~ 19,3 °C).
- Thiếu `NaN`: 0 mọi cột. Mã thiếu `-9999` của Jena: `wv` 18 dòng, `max. wv` 20 dòng, `T` 0 dòng → thay bằng NaN rồi nội suy tuyến tính 2 chiều, không xóa dòng.
- Tương quan với `T`: `Tpot` 0,997, `rho` −0,963, `VPmax` 0,951, cụm ẩm 0,76–0,90, `rh` −0,57, nhóm gió/áp suất ≈ 0 (giữ lại vì tín hiệu phi tuyến).
- Khoảng trống lớn nhất 3 ngày 02:20 (25–28/10/2016, trong tập test) — loại cửa sổ cắt qua đoạn này.
- `wd (deg)` trong [0, 360], là biến vòng tròn → tách thành `wd_sin/wd_cos`, chốt `F = 15`.

## 3. Quy ước thư mục

- `data/raw/` bất biến (không sửa, không ghi thêm).
- `data/processed/` chứa dữ liệu đã làm sạch + `scaler.pkl` (StandardScaler chỉ fit trên train).
- `data/splits/` chứa chỉ mục cửa sổ train/val/test (seed 42, tái tạo từ `split.py`).
- Tất cả `processed/`, `splits/` đều gitignore; chỉ commit `configs/*.yaml` và `results/eda/eda_summary.json`.

## 4. Tiền xử lý (đúng thứ tự)

1. Sắp xếp theo `dt`, thay `-9999` ở `wv`/`max. wv` bằng NaN → `interpolate(limit_direction="both")`.
2. Tách `wd → (wd_sin, wd_cos)`, bỏ cột `wd` gốc (F = 14 → 15).
3. Chia theo năm trước khi chuẩn hóa: train 2009–2014 (75,1%), val 2015 (12,5%), test 2016 (12,4%).
4. `StandardScaler` chỉ `fit` trên train (cả đặc trưng và target); val/test chỉ `transform`.
5. Cửa sổ trượt ngày-tới-ngày: `input_len = 144`, `pred_len = 144`, `stride_train = 12` (~35.022 mẫu), `stride_eval = 144` (~2.919 mẫu, không chồng lấn). Loại cửa sổ chứa gap dài 10/2016.

## 5. Tăng cường (augmentation)

- Bản tối thiểu: không augmentation chuỗi (giữ nguyên để đối sánh công bằng 3 mô hình).
- Mức Could: nhiễu Gauss nhỏ trên train, không áp dụng cho val/test.

## 6. Chia tập (split)

- Chia theo thời gian (theo năm), không xáo trộn liên tập, seed 42, file split lưu ở `data/splits/`.
- Mỗi lần chạy lại từ `configs/data_config.yaml` phải cho cùng tập (sai số hạt giống cho phép ở khâu shuffle trong epoch, không ở khâu split).

## 7. Dataset / DataLoader

- Class `src/data/dataset.py` trả về `dict {"input": X (B,144,15), "target": y (B,144)}`, `batch_size = 64`, `num_workers = 0`, `pin_memory = False` trên CPU.
- Kiểm tra: `python -m src.data --config configs/data_config.yaml` in shape 1 batch đúng `(64,144,15) → (64,144)`.
- Hậu xử lý khi đánh giá: inverse-transform 144 giá trị về °C, chặn biên −40…50 °C để phát hiện lỗi.
