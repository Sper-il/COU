# Dự báo thời tiết Seq2Seq Jena — Tình trạng tiến độ

> Nguồn đúng duy nhất cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-09 (kiểm tra lại Bước 3, đồng bộ F = 15)

## Đối chiếu với đề (khớp từng chữ)

| Dòng trong đề | Thể hiện trong dự án |
|---------------|----------------------|
| Task: Regression | Hồi quy, hàm mất mát MSE, chỉ số MAE/RMSE theo °C |
| Datasets: Jena Climate | Nguồn duy nhất `data/raw/jena_climate_2009_2016.csv`, 14 biến, chu kỳ 10 phút |
| Objective: Weather Forecasting | Dự báo nhiệt độ theo ngày phục vụ nông nghiệp, năng lượng, giao thông |
| Exercises: Sequence-to-Sequence Weather Forecasting | Bộ mã hóa nén 1 ngày quá khứ, bộ giải mã sinh 1 ngày tương lai |
| Goal: Predict temperature sequences for upcoming days | Đầu ra `(144,)` = 1 ngày tiếp theo, mức Could thử 2–3 ngày |
| Model: Encoder–Decoder LSTM | Mô hình cơ sở mức Must đúng tên trong đề |
| Task Type: Regression (multi-step) | Đánh giá toàn tầm 144 và tại tầm 1/24/72/144 |
| Extension: Add attention mechanism | Mức Should: LSTM kèm Attention Bahdanau |
| Extension: compare against Transformer | Mức Should: Transformer mini cùng bảng |

## Bối cảnh dự án (Giai đoạn 1 — Định nghĩa yêu cầu)

| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Dự báo thời tiết chuỗi-tới-chuỗi trên bộ dữ liệu khí hậu Jena (tên gốc tiếng Anh: Sequence-to-Sequence Weather Forecasting on Jena Climate) |
| Tên ngắn | `jena-seq2seq` |
| Bài toán | Hồi quy chuỗi-tới-chuỗi: từ 1 ngày quan trắc (144 bước × 10 phút, 15 đặc trưng sau tách gió) dự báo chuỗi nhiệt độ `T (degC)` của 1 ngày tiếp theo (144 bước). Mở rộng đa ngày {144, 288, 432} ở mức Could |
| Người dùng mục tiêu | Sinh viên làm bài Seq2Seq; giảng viên chấm phần hồi quy và mở rộng; người tham khảo đường cơ sở |
| Đầu vào | Một mẫu `X (144, 15)` với `F = 15` sau khi tách `wd` thành sin/cos, đã chuẩn hóa bằng StandardScaler chỉ khớp trên tập huấn luyện |
| Đầu ra | Một véc-tơ `y (144,)` là nhiệt độ 1 ngày tiếp theo (°C), chuyển ngược về đơn vị gốc khi đánh giá |
| Nguồn dữ liệu | `data/raw/jena_climate_2009_2016.csv` (khoảng 43 MB, khoảng 420.551 dòng, 2009–2016). Chia theo thời gian: huấn luyện 2009–2014 (khoảng 70%), kiểm định 2015 (khoảng 15%), kiểm thử 2016 (khoảng 15%). Cửa sổ trượt 144→144, bước trượt huấn luyện 12, đánh giá 144 |
| Tính năng cốt lõi | Must: quy trình dữ liệu tái lập + LSTM Encoder–Decoder cơ sở + đánh giá nhiều bước (MAE/RMSE toàn tầm 144 và tại 1/24/72/144) + `metrics.json`/`report.html`. Should: Attention Bahdanau + Transformer mini + teacher forcing giảm dần/dừng sớm. Could: đa ngày, demo một lệnh |
| Xây dựng công nghệ | Máy cá nhân (CPU chạy được, có GPU càng tốt). PyTorch kèm scikit-learn, pandas/numpy, matplotlib, PyYAML. Lô 64, Adam lr 1e-3, tối đa 30 epoch, dừng sớm kiên nhẫn 5, teacher forcing 0,5 giảm dần |
| Tiêu chí đánh giá | Tập kiểm thử tầm 144: MAE ≤ 1,5 °C, RMSE ≤ 2,0 °C, MAE@24 ≤ 1,2 °C, MAE@144 ≤ 2,2 °C, vượt persistence. Suy luận một mẫu một ngày trên CPU ≤ 100 ms. Chi tiết xem `docs/description.md`, `docs/define.md` |

## Ảnh chụp kiến trúc (Giai đoạn 2 — Thiết kế)

- Kiến trúc hệ thống: quy trình theo lô gồm 4 tầng `Dữ liệu → Mô hình → Huấn luyện → Đánh giá`, không máy chủ, chi tiết xem `docs/design.md`
- Các mô-đun: `src/data` (đọc/làm sạch/chuẩn hóa/cửa sổ 144→144/chia theo năm) | `src/models` (LSTM S2S đúng đề, kèm Attention, Transformer mini) | `src/training` (teacher forcing giảm dần, dừng sớm, checkpoint) | `src/evaluation` (chỉ số tại 1/24/72/144, persistence, `report.html`) | `src/demo.py` (demo một ngày mẫu)
- Cấu trúc thư mục: theo chuẩn (`data/{raw,processed,splits}`, `src/{data,models,training,evaluation}`, `models/{checkpoints,trained,configs}`, `configs/`, `results/runs/<thời gian>/`), chi tiết xem `docs/design.md` mục 3
- Luồng dữ liệu: `CSV Jena thô → đã xử lý (chuẩn hóa + tách gió F = 15) → chia theo năm → cửa sổ trượt (144→144) → DataLoader (B,144,15) → huấn luyện → checkpoint → chuyển ngược tỉ lệ → chỉ số theo tầm 1/24/72/144 → report.html`
- Giao diện: `build_dataloaders(config) → {train,val,test}`; `LSTMS2S.forward(x, y_in) → (B,144)` với `x: (B,144,15)`; `train(config) → best_ckpt`; `evaluate(ckpt, loader) → {mae, rmse, mae_h1/h24/h72/h144, persistence_mae}`
- Cột mốc: tuần 1 dựng khung và phân tích khám phá | tuần 2 dữ liệu | tuần 3 mô hình và huấn luyện (cơ sở rồi 2 mở rộng) | tuần 4 đánh giá, trực quan và tài liệu
- Rủi ro kỹ thuật: rò rỉ do chia/chuẩn hóa sai; sai số tích lũy ở nửa sau ngày dự báo; huấn luyện chậm trên CPU vì chuỗi dài 144; quá khớp; nhiễu hướng gió dạng góc (bảng chi tiết trong `docs/design.md` mục 7)

## Mô-đun đang làm: Mô hình `src/models` (kế tiếp)

- **Mô tả** — Cài đặt 3 kiến trúc đúng đề trong `src/models/architecture.py`: LSTM Encoder–Decoder cơ sở, kèm Attention Bahdanau, Transformer mini; hàm mất mát MSE trong `loss.py`
- **Mục tiêu** — Cả 3 mô hình chạy `forward` `(B,144,15) → (B,144)` không lỗi trên CPU
- **Trạng thái** — Mô tả ⏳ / Sinh mã ⏳ / Duyệt ⏳ / Chạy ⏳ / Kiểm thử ⏳ / Gọn mã ⏳ / Chốt ⏳

## Mô-đun đã xong: Dữ liệu `src/data` (2026-11-08)

- **Mô tả** — Đọc CSV Jena, thay mã thiếu `-9999` ở 2 cột gió bằng NaN rồi nội suy tuyến tính, tách `wd` thành `wd_sin/wd_cos` (`F = 15`), `StandardScaler` chỉ fit trên huấn luyện 2009–2014, cửa sổ 144→144, chia theo năm
- **Mục tiêu** — `DataLoader` chạy được với shape `(B,144,15) → (B,144)`, không rò rỉ scaler, không chồng lấn thời gian giữa các tập
- **Quy trình** — `CSV thô → transforms → split theo năm + loại cửa sổ qua gap → Dataset → DataLoader → scaler.pkl + splits.npz`
- **Kế hoạch thử nghiệm (việc nhỏ)**
  - [x] Viết `src/data/transforms.py` (mã thiếu, nội suy, gió sin/cos)
  - [x] Viết `src/data/split.py` (chia theo năm, loại cửa sổ cắt qua gap > 15 phút)
  - [x] Viết `src/data/dataset.py` (`JenaWindowDataset`, `build_dataloaders`, `get_scaler`, lưu `scaler.pkl` + `splits.npz`)
  - [x] Viết `src/data/__main__.py` kiểm tra 1 lô
  - [x] Viết `tests/unit/test_dataset.py` (4 kiểm thử)
- **Trạng thái** — Mô tả ✅ / Sinh mã ✅ / Duyệt ✅ / Chạy ✅ / Kiểm thử ✅ / Gọn mã ✅ / Chốt ⏳
- **Lần chạy gần nhất** — 2026-11-09: `python -m src.data --config configs/data_config.yaml` → train 26.192 mẫu (410 lô), val 364 mẫu (6 lô), test 359 mẫu (6 lô), `F = 15`; `splits.npz` ghi nhận `{'train': 26192, 'val': 364, 'test': 359}`; `pytest tests/unit/test_dataset.py` → 4 passed
- **Danh sách hoàn thành** — xem danh sách trong quy trình `ai-fusion-workflow`

## Mô-đun đã xong: Dựng khung và phân tích khám phá

- **Mô tả** — Dựng thư mục chuẩn, viết `configs/*.yaml` với `input_len=144, pred_len=144`, sổ tay phân tích khám phá để chốt `F = 15`
- **Mục tiêu** — Có nền tái lập được trước khi viết mã dữ liệu
- **Quy trình** — `CSV thô → sổ tay phân tích → configs chốt`
- **Kế hoạch thử nghiệm (việc nhỏ)**
  - [x] Dựng thư mục kèm `.gitignore` và `requirements.txt`
  - [x] Viết `configs/data_config.yaml`, `model_config.yaml`, `train_config.yaml` (tầm 144)
  - [x] Sổ tay phân tích: phân phối, NaN, tách gió sin/cos, chốt tầm 144
- **Trạng thái** — Mô tả ✅ / Sinh mã ✅ / Duyệt ✅ / Chạy ✅ / Kiểm thử ✅ / Gọn mã ⏳ / Chốt ⏳
- **Lần chạy gần nhất** — 2026-11-08: chuẩn hoá tiếng Việt có dấu + lược bỏ câu nhận xét chủ quan trong `notebooks/2026-11-03_eda.py` và `src/evaluation/visualize_eda_html.py` (tiêu đề đồ thị và chú thích hình chỉ mô tả nội dung); chạy lại EDA → `results/eda/eda_report.html` (~615 KB, offline 100%, 6 ảnh base64, số khớp JSON)
- **Báo cáo visualize** — `results/eda/eda_report.html` (mở bằng double-click): mục 1 bài toán và mục tiêu EDA, mục 2 tổng quan dữ liệu, mục 3 biến mục tiêu, mục 4 tương quan, mục 5 bảng persistence kèm mục tiêu Seq2Seq, mục 6 cả 6 hình EDA kèm chú thích, mục 7 hạn chế và bước tiếp theo. Script render: `src/evaluation/visualize_eda_html.py`
- **Docs đã bổ sung** — `README.md` (10 mục), `docs/architecture.md`, `docs/data-flow.md`, `docs/api.md`, `docs/data.md`, `docs/AI_Usage.md` (7 mục), `docs/CHANGELOG.md`. Bộ docs cũ giữ nguyên: `description.md`, `define.md`, `design.md`
- **Kết quả EDA chốt**:
  - Dữ liệu: 420.551 dòng, 01-01-2009 → 01-01-2017, chu kỳ 10 phút, 0 NaN; mã thiếu `-9999` chỉ ở `wv` (18) và `max. wv` (20) → nội suy tuyến tính
  - Nhiệt độ `T`: trung bình 9,45 °C, độ lệch chuẩn 8,42 °C, biên −23,01…37,28 °C; chu kỳ ngày rõ (thấp nhất ~5 giờ, cao nhất ~15 giờ), mùa vụ rõ (tháng 1 ~ −0,1 °C, tháng 7 ~ 19,3 °C)
  - Tương quan với `T`: `Tpot` 0,997, `rho` −0,963, `VPmax` 0,951; gió (`wv`, `wd`) gần như độc lập → giữ gió vì mang tín hiệu thời tiết dù tương quan tuyến tính thấp
  - Khoảng trống lớn nhất 3 ngày 02:20 (25–28/10/2016, 362 ngày đánh giá 2016 còn nguyên); chia train 2009–2014 (75,1%), val 2015 (12,5%), test 2016 (12,4%)
  - Chốt `F = 15`: tách `wd (deg)` thành `wd_sin/wd_cos` (tránh đứt đoạn 359° so với 1°)
  - Cửa sổ 144→144: `stride_train = 12` cho ~35.022 mẫu, `stride_eval = 144` cho ~2.919 mẫu
  - Ngưỡng persistence 2016: MAE toàn tầm 2,806 °C (MAE@1 = 0,12, MAE@24 = 0,94, MAE@72 = 1,77, MAE@144 = 2,81) — Seq2Seq phải vượt con số này
- **Danh sách hoàn thành** — xem danh sách trong quy trình `ai-fusion-workflow`

## Nhật ký mô-đun

| # | Mô-đun | Ngày | Trạng thái | Commit | Ghi chú |
|---|--------|------|------------|--------|---------|
| 0 | Định nghĩa và thiết kế | 2026-11-03 | ✅ | — | Xong `docs/description.md`, `docs/define.md`, `docs/design.md` kèm tệp này, khớp đủ 9 dòng đề |
| 1 | Dựng khung và EDA | 2026-11-03 | ✅ | — | `notebooks/2026-11-03_eda.py` → 6 PNG + `eda_summary.json`; chốt F=15, persistence 2,806 °C |
| 2 | Visualize EDA HTML + bổ sung docs | 2026-11-03 | ✅ | — | `src/evaluation/visualize_eda_html.py` → `results/eda/eda_report.html` (~638 KB, offline); thêm README 10 mục, architecture, data-flow, api, data, AI_Usage, CHANGELOG |
| 3 | Dữ liệu `src/data` | 2026-11-09 | ✅ | — | `transforms/split/dataset/__main__` + `tests/unit/test_dataset.py` (4 passed); train 26.192 mẫu, val 364, test 359; `scaler.pkl` + `splits.npz` (kiểm tra lại) |

## Nhật ký quyết định

| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-11-03 | Đầu vào/ra chốt 144→144 (ngày-tới-ngày) | 144→24 (4 giờ) | Khớp Goal upcoming days trong đề, đúng bản chất dự báo theo ngày |
| 2026-11-03 | Kiến trúc chính là Encoder–Decoder LSTM | GRU, TCN | Đúng yêu cầu đề, xử lý phụ thuộc dài hạn tốt hơn RNN thường |
| 2026-11-03 | Attention dùng Bahdanau | Luong dạng tích vô hướng | Nhẹ, hợp chuỗi khí tượng, Luong để dự phòng |
| 2026-11-03 | Đối sánh bằng Transformer mini (2 lớp, 4 đầu) | Informer/Autoformer | Đúng phần mở rộng 2, vừa sức bài tập |
| 2026-11-03 | Hàm mất mát MSE khi huấn luyện, báo cáo thêm MAE/RMSE | Huber | MSE phạt mạnh sai số lớn, MAE dễ diễn giải theo °C |
| 2026-11-03 | Chia tập theo năm kèm bộ chuẩn hóa chỉ khớp trên huấn luyện | Xáo trộn ngẫu nhiên | Chống rò rỉ thời gian, đúng bản chất dự báo |
| 2026-11-03 | Đánh giá tại tầm 1/24/72/144 kèm persistence | Chỉ một số trung bình | Khớp Task Type multi-step, lộ sai số tích lũy cuối ngày |
| 2026-11-03 | Chốt F = 15 (tách wd thành wd_sin/wd_cos) | Giữ 14 biến gốc | EDA chứng minh wd là biến vòng tròn đứt đoạn 0/360, tách sin/cos giữ tính liên tục |
| 2026-11-03 | Mã thiếu -9999 ở wv/max.wv (38 dòng) nội suy tuyến tính | Xóa dòng | Chỉ 0,005% dữ liệu, xóa sẽ phá vỡ cửa sổ 144→144 liên tục |

## Rủi ro và giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Rò rỉ do xáo trộn/chuẩn hóa sai | Trung bình | Cao | Chia theo năm trước khi chuẩn hóa, bộ chuẩn hóa chỉ khớp trên huấn luyện | Kiểm thử đơn vị về rò rỉ |
| Sai số tích lũy nửa sau ngày (MAE@144 vọt) | Cao | Trung bình | Teacher forcing giảm dần kèm Attention, báo cáo tại 1/24/72/144 | Đồ thị MAE theo tầm |
| Huấn luyện chậm trên CPU vì chuỗi dài 144 | Trung bình | Trung bình | Lô 64, hidden 128, bước trượt đánh giá 144 | Đo epoch đầu |
| Quá khớp | Trung bình | Trung bình | Dừng sớm kiên nhẫn 5, dropout 0,1 | Đường hàm mất mát huấn luyện/kiểm định |

## Câu hỏi còn mở

- [x] Có tách `wd (deg)` thành sin/cos (`F = 15`) hay giữ 14 đặc trưng gốc? → đã chốt `F = 15` từ EDA (`results/eda/04_gio_sincos.png`)
- [ ] Trình bày chính 1 ngày (144) hay thêm 2–3 ngày (288/432)? → mặc định 144, thử đa ngày ở mức Could
- [ ] Cấu hình CPU/GPU máy huấn luyện cụ thể? → ghi khi chạy thử epoch đầu

## Định nghĩa hoàn thành (áp dụng mỗi mô-đun)

- [ ] Đúng yêu cầu
- [ ] Mã chạy đầu-cuối
- [ ] Kiểm thử đạt
- [ ] Không có khóa bí mật / không có tệp lớn
- [ ] Mã đã tự duyệt
- [ ] Có thể giải thích được
- [ ] Tài liệu đã cập nhật
- [ ] Đã chốt commit (Conventional Commit)
