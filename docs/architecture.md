# Kiến trúc hệ thống — `jena-seq2seq`

> Mẫu kiến trúc: quy trình học máy theo lô cổ điển, 4 tầng `Dữ liệu → Mô hình → Huấn luyện → Đánh giá`. Không máy chủ, không phát trực tuyến thời gian thực. Mọi quyết định khớp 9 dòng đề (hồi quy, Jena, dự báo thời tiết, Seq2Seq, upcoming days, Encoder–Decoder LSTM, multi-step, attention, Transformer).

## 1. Sơ đồ tổng thể

```text
data/raw/jena_*.csv (chu kỳ 10 phút, 14 biến + Date Time, ~420.551 dòng)
        │  src/data/dataset.py, transforms.py, split.py
        │  (đọc → nội suy -9999 → tách wd thành sin/cos → StandardScaler chỉ fit trên train
        │   → cửa sổ 144→144 → chia theo năm 2009–2014/2015/2016)
        v
data/processed + data/splits ──► DataLoader: (B,144,15) → (B,144), B = 64
        │  src/models/architecture.py, loss.py
        │  (a) LSTMS2S cơ sở | (b) LSTMS2SAttention Bahdanau | (c) MiniTransformer
        │  hàm mất mát MSE
        v
src/training/trainer.py, scheduler.py
(Adam lr 1e-3, teacher forcing 0,5 giảm dần ×0,98/epoch, clip 1,0,
 early stopping patience 5 theo loss kiểm định)
        │  ──► models/checkpoints/*.pt + models/trained/jena_lstm_v1.pt
        v
src/evaluation/metrics.py, visualize.py, visualize_html.py, visualize_eda_html.py
(inverse-transform về °C → MAE/RMSE toàn tầm và tại 1/24/72/144
 → persistence → results/runs/<ts>/{metrics.json, plots/, report.html})
```

Tầng dữ liệu biến CSV thô thành các cửa sổ ngày-tới-ngày đã chuẩn hóa; tầng mô hình/huấn luyện học ánh xạ chuỗi-tới-chuỗi cho 1 ngày tiếp theo rồi sinh checkpoint; tầng đánh giá biến checkpoint thành `metrics.json` kèm `report.html` offline. Báo cáo EDA (`results/eda/eda_report.html`) là sản phẩm trực quan đầu tiên, render từ `eda_summary.json` bằng `src/evaluation/visualize_eda_html.py`.

## 2. Các mô-đun và trách nhiệm

| Mô-đun | Thư mục | Trách nhiệm | Trạng thái |
|--------|---------|-------------|------------|
| Dữ liệu | `src/data/` (`dataset.py`, `transforms.py`, `split.py`) | Đọc CSV Jena, nội suy mã thiếu `-9999` ở 2 cột gió, tách `wd` thành `wd_sin/wd_cos` (F = 15), StandardScaler chỉ fit trên train, cửa sổ 144→144 (`stride_train = 12`, `stride_eval = 144`), chia theo năm | Chưa code (kế tiếp) |
| Mô hình | `src/models/` (`architecture.py`, `loss.py`) | 3 kiến trúc: LSTM S2S cơ sở (`hidden = 128`, 2 lớp), kèm Attention Bahdanau, Transformer mini (`d_model = 64`, 4 đầu, 2 lớp); loss MSE | Chưa code |
| Huấn luyện | `src/training/` (`trainer.py`, `scheduler.py`) | Vòng train/val, teacher forcing giảm dần, early stopping, checkpoint tốt nhất theo loss kiểm định | Chưa code |
| Đánh giá | `src/evaluation/` (`metrics.py`, `visualize.py`, `visualize_html.py`, `visualize_eda_html.py`) | Inverse-transform 144 giá trị về °C, MAE/RMSE toàn tầm và tại 1/24/72/144, persistence, dựng `report.html` offline | Đã có `visualize_eda_html.py` ✅ |
| EDA | `notebooks/2026-11-03_eda.py` + `results/eda/` | Script EDA tái lập được (thay notebook để dễ chạy lại), 6 hình PNG + `eda_summary.json` + `eda_report.html` | ✅ Xong |
| Demo | `src/demo.py` | Nạp checkpoint tốt nhất, dự báo 1 ngày mẫu 144 bước | Chưa code (mức Could) |

## 3. Quyết định kiến trúc chính

| Quyết định | Lựa chọn | Vì sao | Lựa chọn đã loại |
|------------|----------|--------|------------------|
| Kiến trúc chính | Encoder–Decoder LSTM | Đúng yêu cầu đề; xử lý phụ thuộc dài hạn của chuỗi 10 phút tốt hơn RNN thường | GRU, TCN |
| Attention | Bahdanau (dạng cộng) | Cho decoder dài 144 bước "nhìn lại" giờ quan trọng, cải thiện nửa sau ngày dự báo; cài đặt nhẹ | Luong (dự phòng) |
| Đối sánh | Transformer mini chỉ encoder (2 lớp, 4 đầu) | Đúng phần mở rộng 2, tự chú ý xử lý song song 144 bước | Informer/Autoformer (quá nặng) |
| Hàm mất mát | MSE khi train, báo cáo thêm MAE/RMSE | MSE phạt mạnh sai số lớn; MAE diễn giải trực tiếp theo °C | Huber (dự phòng) |
| Khung công nghệ | PyTorch + scikit-learn + pandas/numpy + matplotlib + PyYAML | Quen thuộc, tài liệu Seq2Seq nhiều, dễ xuất mô hình và vẽ `report.html` offline | TensorFlow/Keras |
| Báo cáo | HTML đơn file, CSS nội tuyến, ảnh base64, không CDN | Mở bằng double-click trên máy chấm không mạng; số render từ JSON, không gõ tay | Jupyter notebook, dashboard online |

## 4. Rủi ro kiến trúc và giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Rò rỉ do xáo trộn/chuẩn hóa sai | Trung bình | Cao | Chia theo năm trước khi chuẩn hóa; scaler chỉ fit trên train; unit test rò rỉ | Test `scaler.fit` không gọi trên val/test |
| Sai số tích lũy nửa sau ngày (MAE@144 vọt) | Cao | Trung bình | Teacher forcing giảm dần + Attention; báo cáo tại 1/24/72/144 | Đồ thị MAE theo tầm |
| Train chậm trên CPU (chuỗi dài 144) | Trung bình | Trung bình | Batch 64, hidden 128, stride eval 144; dùng GPU nếu có | Đo epoch đầu |
| Quá khớp | Trung bình | Trung bình | Early stopping patience 5, dropout 0,1, weight decay 1e-5 | Đường loss train/val |
| Hướng gió dạng góc gây nhiễu | Thấp | Thấp | Đã tách sin/cos từ EDA (F = 15) | EDA Hình 4 + ablation |

Chi tiết luồng dữ liệu xem `docs/data-flow.md`, giao diện từng module xem `docs/api.md`, quy ước dữ liệu xem `docs/data.md`.
