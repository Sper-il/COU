# Mô tả dự án — Dự báo nhiệt độ đa bước Jena bằng Seq2Seq (Encoder–Decoder LSTM)

> Câu giới thiệu: Từ 1 ngày quan trắc khí hậu (144 bước × 10 phút), dự báo chuỗi nhiệt độ của 1 ngày tiếp theo (144 bước) bằng Encoder–Decoder LSTM, có mở rộng Attention và đối sánh với Transformer đúng theo đề.

## 0. Đối chiếu với đề (đảm bảo khớp từng chữ)

| Dòng trong đề | Thể hiện trong tài liệu này |
|---------------|----------------------------|
| Task: Regression | Bài toán hồi quy, hàm mất mát MSE, chỉ số MAE/RMSE theo °C |
| Datasets: Jena Climate | Nguồn duy nhất `data/raw/jena_climate_2009_2016.csv`, 14 biến, chu kỳ 10 phút |
| Objective: Weather Forecasting | Mục tiêu dự báo nhiệt độ phục vụ nông nghiệp, năng lượng, giao thông |
| Exercises: Sequence-to-Sequence Weather Forecasting | Kiến trúc chuỗi-tới-chuỗi: bộ mã hóa nén quá khứ, bộ giải mã sinh từng bước tương lai |
| Goal: Predict temperature sequences for upcoming days | Đầu ra là chuỗi 144 bước = 1 ngày tiếp theo (mức Could thử 2–3 ngày); không làm dự báo 1 bước |
| Model: Encoder–Decoder LSTM | Mô hình cơ sở mức Must là LSTM Encoder–Decoder, đúng tên trong đề |
| Task Type: Regression (multi-step) | Đánh giá nhiều bước: toàn tầm và tại tầm 1/24/72/144, kèm đường cơ sở persistence |
| Extension: Add attention mechanism | Mức Should: LSTM Encoder–Decoder kèm Attention Bahdanau |
| Extension: compare against Transformer | Mức Should: Transformer mini đối sánh cùng đặc trưng, cùng tầm, cùng bảng |

## 1. Tên dự án

| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Dự báo thời tiết chuỗi-tới-chuỗi trên bộ dữ liệu khí hậu Jena (tên gốc tiếng Anh: Sequence-to-Sequence Weather Forecasting on Jena Climate) |
| Tên ngắn | `jena-seq2seq` |
| Câu giới thiệu | Dự báo chuỗi nhiệt độ nhiều bước cho ngày tiếp theo từ dữ liệu trạm khí tượng Jena |

## 2. Bài toán — Bài toán là gì, vì sao quan trọng?

- **Bài toán:** Hồi quy chuỗi-tới-chuỗi (sequence-to-sequence). Cho một cửa sổ quá khứ 1 ngày gồm 14 biến khí tượng đo mỗi 10 phút (144 bước), dự báo chuỗi nhiệt độ `T (degC)` của 1 ngày tiếp theo (144 bước). Đây là bài toán nhiều bước (multi-step) đúng như đề, không phải dự báo một điểm.
- **Vì sao quan trọng:** dự báo nhiệt độ theo ngày phục vụ nông nghiệp (gieo trồng, nhà kính), năng lượng (điều hòa, điện mặt trời), giao thông và cảnh báo thời tiết cực đoan. Bài toán nhiều bước tới 144 bước phơi bày hiện tượng tích lũy sai số theo tầm dự báo — đúng trọng tâm của kiến trúc Encoder–Decoder.
- **Ví dụ minh họa:** cho 144 bản ghi của ngày hôm qua (nhiệt độ, áp suất, độ ẩm, tốc độ gió…), mô hình sinh ra 144 giá trị nhiệt độ cho ngày mai `[2,1, 2,3, 2,5, …, 3,0]` (°C).

## 3. Người dùng mục tiêu — Ai dùng, trong bối cảnh nào?

| Nhóm | Bối cảnh | Kỳ vọng ở đầu ra |
|------|----------|------------------|
| Sinh viên / học viên môn trí tuệ nhân tạo | Làm bài tập chuỗi-tới-chuỗi, nộp báo cáo kèm `report.html` | Chạy được bằng một lệnh, có đồ thị hàm mất mát (loss), ví dụ dự báo tốt/xấu, bảng đối sánh 3 mô hình |
| Giảng viên / người chấm | Chấm điểm phần hồi quy và phần mở rộng Attention/Transformer | Chỉ số (metric) rõ ràng theo từng tầm dự báo, mã nguồn tái lập được qua `configs/*.yaml` |
| Người nghiên cứu thời tiết (mức Could) | Tham khảo đường cơ sở (baseline) trên Jena | Điểm kiểm tra (checkpoint) và kịch bản `predict` tái sử dụng được |

## 4. Dữ liệu vào / ra — Cụ thể ra sao

### 4.1. Nguồn dữ liệu

- Tệp: `data/raw/jena_climate_2009_2016.csv` (khoảng 43 MB, khoảng 420.551 dòng, chu kỳ 10 phút, giai đoạn 01-01-2009 → 31-12-2016). Đúng tập Jena trong đề.
- Gồm 14 cột đặc trưng số và 1 cột `Date Time`:

```python
['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
```

- Biến mục tiêu duy nhất: `T (degC)` — chuỗi nhiệt độ các ngày tiếp theo, đúng Goal trong đề.

### 4.2. Định dạng sau tiền xử lý (chốt cho bản chạy tối thiểu, khớp đề)

| Giai đoạn | Kích thước (shape) | Diễn giải |
|-----------|--------------------|-----------|
| Một mẫu đầu vào `X` | `(144, 14)` | 144 bước quá khứ = 1 ngày × 144 mẫu/ngày, 14 đặc trưng đã chuẩn hóa bằng StandardScaler (chỉ khớp trên tập huấn luyện) |
| Một mẫu đầu ra `y` | `(144,)` | 144 bước tương lai của `T (degC)` = 1 ngày tiếp theo (mặc định đúng Goal); mức Could thử `{144, 288, 432}` tương ứng 1–3 ngày |
| Lô huấn luyện | `(B, 144, 14) → (B, 144)` | `B = 64` mặc định |
| Đầu ra suy luận | `(144,)` kèm đồ thị | Chuỗi nhiệt độ 1 ngày (°C), chuyển ngược về đơn vị gốc để tính chỉ số |

- **Chia tập theo thời gian (không xáo trộn, chống rò rỉ):** huấn luyện 2009–2014 (khoảng 70%), kiểm định 2015 (khoảng 15%), kiểm thử 2016 (khoảng 15%). Chuẩn hóa chỉ khớp (fit) trên tập huấn luyện.
- **Cửa sổ trượt (sliding window):** bước trượt `stride = 12` (2 giờ) cho huấn luyện, `stride = 144` (1 ngày) cho kiểm định/kiểm thử để giảm tương quan giữa các cửa sổ ngày.

## 5. Tính năng cốt lõi — Mức Must / Should / Could (ánh xạ đề)

| Mức | Tính năng | Ánh xạ đề | Vì sao |
|-----|-----------|-----------|--------|
| Must (bắt buộc) | Quy trình dữ liệu tái lập được: đọc CSV Jena → làm sạch → chuẩn hóa → cửa sổ 144→144 → `DataLoader` | Datasets: Jena Climate | Không có thì không huấn luyện được mô hình nào |
| Must (bắt buộc) | Mô hình cơ sở Encoder–Decoder LSTM (không attention), hàm mất mát MSE, kỹ thuật teacher forcing | Model: Encoder–Decoder LSTM | Đúng tên mô hình trong đề, làm mốc đối sánh |
| Must (bắt buộc) | Đánh giá hồi quy nhiều bước: MAE/RMSE toàn tầm 144 và theo từng tầm (1, 24, 72, 144) | Task Type: Regression (multi-step) | Đề là nhiều bước cho nhiều ngày, một con số trung bình là chưa đủ |
| Must (bắt buộc) | `report.html` chạy ngoại tuyến kèm `metrics.json`, đồ thị hàm mất mát và ví dụ dự báo theo ngày | Exercises: Sequence-to-Sequence Weather Forecasting | Chuẩn nghiệm thu của quy trình |
| Should (nên có) | Encoder–Decoder LSTM kèm Attention (Bahdanau, dự phòng Luong) | Extension: Add attention mechanism | Phần mở rộng thứ nhất trong đề |
| Should (nên có) | Transformer mini đối sánh cùng đặc trưng, cùng tầm 144, cùng bảng | Extension: compare against Transformer | Phần mở rộng thứ hai trong đề |
| Should (nên có) | Tỉ lệ teacher forcing kèm dừng sớm (early stopping) và checkpoint tốt nhất theo hàm mất mát kiểm định | Ổn định Seq2Seq tầm dài 144 | Giảm quá khớp, ổn định giải mã dài ngày |
| Could (có thì tốt) | Dự báo đa ngày `{144, 288, 432}` và lấy mẫu lại theo giờ | Goal: upcoming days (số nhiều) | Chứng minh tổng quát nhiều ngày |
| Could (có thì tốt) | Demo bằng một lệnh `python -m src.demo` với đầu vào mẫu một ngày | Trình bày | Phục vụ trình bày trực tiếp |

### Ngoài phạm vi (bản chạy tối thiểu không làm)

- Không dự báo đa biến (chỉ dự báo `T (degC)` cho các ngày tiếp theo, không dự báo đồng thời áp suất/độ ẩm).
- Không dùng dữ liệu ngoài Jena (không thêm dữ liệu vệ tinh, không thu thập thời tiết bên ngoài).
- Không triển khai API/đám mây, không làm giao diện web — chỉ kịch bản lệnh, sổ tay (notebook) và `report.html`.
- Không tối ưu siêu tham số quy mô lớn (chỉ lưới nhỏ thủ công).

## 6. Tiêu chí thành công — Đo bằng gì, ngưỡng bao nhiêu?

| Nhóm | Chỉ số | Ngưỡng bản chạy tối thiểu (tầm 144 = 1 ngày) |
|------|--------|----------------------------------------------|
| Mô hình (tập kiểm thử 2016, tầm 144) | MAE toàn tầm | ≤ 1,5 °C |
| Mô hình | RMSE toàn tầm | ≤ 2,0 °C |
| Theo tầm dự báo | MAE@1 / MAE@24 / MAE@72 / MAE@144 | Tăng đơn điệu hợp lý, MAE@24 ≤ 1,2 °C, MAE@144 ≤ 2,2 °C |
| Đường cơ sở đối sánh | Persistence (lặp lại giá trị cuối ngày hôm qua) | Seq2Seq phải vượt persistence trên tập kiểm thử |
| Mở rộng | Attention tốt hơn bản không attention; Transformer được báo cáo cùng bảng tầm 144 | Chênh lệch có ý nghĩa, không cần thắng mọi tầm |
| Quy trình | Thời gian suy luận một mẫu một ngày trên CPU | ≤ 100 ms; huấn luyện một epoch trên CPU ≤ 20 phút với cấu hình tối thiểu |
| Tái lập | Chạy lại từ `configs/*.yaml` cho cùng `metrics.json` (sai số hạt giống cho phép) | Đạt |

## 7. Danh sách kiểm tra lại bản mô tả

- [x] Khớp đủ 9 dòng đề: Regression, Jena, Weather Forecasting, Seq2Seq, upcoming days (144 bước), Encoder–Decoder LSTM, multi-step, attention, Transformer.
- [x] Đọc xong có hình dung được đầu vào `(144, 14)` → đầu ra `(144,)` cụ thể không? Có.
- [x] Có phân biệt Must / Should / Could không? Có, kèm cột ánh xạ đề.
- [x] Có mục ngoài phạm vi không? Có.
- [x] Có số liệu (kích thước, tầm dự báo, chỉ số, ngưỡng) hay toàn chữ chung chung? Có số liệu.
- [x] Đã đồng bộ vào `docs/progress_status.md` mục Bối cảnh dự án chưa? Đã đồng bộ.
