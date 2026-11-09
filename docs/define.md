# Định nghĩa yêu cầu — Dự án Seq2Seq Jena

> Trả lời đủ bộ câu hỏi định nghĩa 6 mục trước khi viết bất kỳ dòng mã nào. Tệp này là bản chi tiết, bản rút gọn được đồng bộ trong `docs/progress_status.md` mục Bối cảnh dự án. Mọi chốt đều khớp 9 dòng trong đề.

## 0. Tên dự án — Tên dự án là gì?

- **Tên đầy đủ:** Dự báo thời tiết chuỗi-tới-chuỗi trên bộ dữ liệu khí hậu Jena (tên gốc tiếng Anh: Sequence-to-Sequence Weather Forecasting on Jena Climate)
- **Tên ngắn:** `jena-seq2seq`
- **Câu giới thiệu:** Từ 1 ngày quan trắc khí tượng (144 bước × 10 phút), dự báo chuỗi nhiệt độ của 1 ngày tiếp theo (144 bước) bằng Encoder–Decoder LSTM, có mở rộng Attention và đối sánh với Transformer.
- **Khớp đề:** Task Regression + Datasets Jena Climate + Objective Weather Forecasting + Exercises Sequence-to-Sequence + Goal upcoming days + Model Encoder–Decoder LSTM + Task Type multi-step + Extension attention/Transformer.

## 1. Bài toán — Bài toán, người dùng, dữ liệu vào/ra là gì?

### Bài toán cụ thể

- **Loại bài toán:** Hồi quy chuỗi-tới-chuỗi (sequence-to-sequence), nhiều bước (multi-step) — đúng Task Type trong đề.
- **Phát biểu chính thức:** cho ma trận quá khứ `X ∈ R^{144×15}` (144 bước × 10 phút = 1 ngày, 15 đặc trưng sau khi tách gió `wd` thành `sin/cos`), dự báo véc-tơ tương lai `y ∈ R^{144}` là nhiệt độ `T (degC)` của 144 bước kế tiếp (1 ngày tiếp theo). Mức Could thử `{144, 288, 432}` tương ứng 1–3 ngày để bao phủ chữ days số nhiều trong Goal.
- **Vì sao quan trọng:** dự báo theo ngày phục vụ nông nghiệp, điều độ năng lượng, giao thông; bài toán 144 bước phơi bày hiện tượng tích lũy sai số theo tầm dự báo — đúng điểm cần kiến trúc Encoder–Decoder và Attention.
- **Đường cơ sở ngây thơ cần vượt qua:** Persistence (lấy giá trị `T` cuối cùng của ngày hôm qua rồi lặp lại cho cả 144 bước ngày mai).

### Người dùng mục tiêu

| Vai trò | Bối cảnh | Kỳ vọng ở đầu ra |
|---------|----------|------------------|
| Sinh viên (người làm bài) | Làm bài tập Seq2Seq, chạy trên máy tính xách tay có CPU/GPU | Quy trình chạy được bằng một lệnh, `report.html` có bảng chỉ số và đồ thị theo ngày |
| Giảng viên (người chấm) | Chấm phần hồi quy và phần mở rộng Attention/Transformer | Chỉ số theo tầm 1/24/72/144, đối sánh 3 mô hình, mã nguồn tái lập qua `configs/*.yaml` |
| Người tham khảo (mức Could) | Tái dùng đường cơ sở Jena | Điểm kiểm tra (checkpoint) và kịch bản suy luận mẫu một ngày |

### Dữ liệu vào / ra

- **Nguồn dữ liệu:** `data/raw/jena_climate_2009_2016.csv` (khoảng 43 MB, khoảng 420.551 dòng, chu kỳ 10 phút, từ 01-01-2009 đến 31-12-2016). Đúng tập Jena trong đề. Gồm 14 đặc trưng số và cột `Date Time`. Biến mục tiêu duy nhất: `T (degC)`.
- **Tiền xử lý chốt cho bản chạy tối thiểu:**
  - Bỏ cột `Date Time` khỏi đặc trưng, chỉ dùng để chia tập theo thời gian và vẽ đồ thị theo ngày.
  - Xử lý hướng gió `wd (deg)` dạng vòng tròn: tách thành `wd_sin`, `wd_cos` (chốt `F = 15` từ EDA 2026-11-03: `wd` gốc đứt đoạn tại 0/360, xem `results/eda/04_gio_sincos.png`; bỏ cột `wd` gốc).
  - `StandardScaler` chỉ khớp (fit) trên tập huấn luyện, rồi áp dụng cho kiểm định/kiểm thử; biến mục tiêu cũng chuẩn hóa khi huấn luyện rồi chuyển ngược (inverse-transform) khi đánh giá.
- **Cửa sổ trượt:** `input_len = 144` (1 ngày quá khứ), `pred_len = 144` (1 ngày tương lai), `stride_train = 12` (2 giờ), `stride_eval = 144` (1 ngày, không chồng lấn khi đánh giá).
- **Chia tập theo thời gian (chống rò rỉ):** huấn luyện 2009–2014 (khoảng 70%), kiểm định 2015 (khoảng 15%), kiểm thử 2016 (khoảng 15%). Không xáo trộn giữa các tập.
- **Kích thước chuẩn:**

| Đối tượng | Kích thước | Ghi chú |
|-----------|------------|---------|
| Một mẫu `X` | `(144, 15)` với `F = 15` sau tách gió | 1 ngày quá khứ, đã chuẩn hóa |
| Một mẫu `y` | `(144,)` | Nhiệt độ 1 ngày tiếp theo, đã chuẩn hóa lúc huấn luyện |
| Lô dữ liệu | `(B, 144, 15) → (B, 144)`, `B = 64` | Khi suy luận cho ra `(144,)` theo đơn vị °C gốc |

## 2. Tính năng cốt lõi — Phân loại Must / Should / Could (kèm ánh xạ đề)

| Mức | Tính năng | Ánh xạ đề | Lý do một dòng |
|-----|-----------|-----------|----------------|
| Must (bắt buộc) | Quy trình dữ liệu tái lập: đọc CSV Jena → làm sạch → chuẩn hóa → cửa sổ 144→144 → DataLoader | Datasets: Jena Climate | Không có thì không huấn luyện được mô hình nào |
| Must (bắt buộc) | Encoder–Decoder LSTM cơ sở (không attention), hàm mất mát MSE, kỹ thuật teacher forcing | Model: Encoder–Decoder LSTM | Đúng tên mô hình trong đề, làm mốc đối sánh |
| Must (bắt buộc) | Đánh giá hồi quy nhiều bước: MAE/RMSE toàn tầm 144 và tại tầm 1/24/72/144, kèm persistence | Task Type: Regression (multi-step) | Đề là nhiều bước cho ngày tiếp theo, một số trung bình là chưa đủ |
| Must (bắt buộc) | `metrics.json` kèm `report.html` ngoại tuyến, đồ thị hàm mất mát, mẫu dự báo tốt/xấu theo ngày | Exercises: Sequence-to-Sequence Weather Forecasting | Chuẩn nghiệm thu |
| Should (nên có) | Encoder–Decoder LSTM kèm Attention Bahdanau (dự phòng Luong) | Extension: Add attention mechanism | Phần mở rộng thứ nhất trong đề |
| Should (nên có) | Transformer mini đối sánh cùng đặc trưng, cùng tầm 144 | Extension: compare against Transformer | Phần mở rộng thứ hai trong đề |
| Should (nên có) | Teacher forcing có lịch giảm dần kèm dừng sớm và checkpoint theo hàm mất mát kiểm định | Ổn định Seq2Seq tầm dài 144 | Chống quá khớp khi giải mã dài một ngày |
| Could (có thì tốt) | Thử đa ngày `{144, 288, 432}` và lấy mẫu lại theo giờ | Goal: upcoming days (số nhiều) | Chứng minh tổng quát nhiều ngày |
| Could (có thì tốt) | Demo bằng một lệnh `python -m src.demo` với đầu vào một ngày mẫu | Trình bày | Phục vụ trình bày |

**Ngoài phạm vi (bản chạy tối thiểu không làm):** chỉ dự báo `T (degC)` cho ngày tiếp theo (không đa biến); không thêm dữ liệu ngoài Jena; không triển khai API/đám mây hay giao diện web; không tìm kiếm siêu tham số quy mô lớn.

## 3. Vì sao — Tại sao chọn cách này?

| Quyết định | Lựa chọn | Vì sao | Lựa chọn đã loại |
|------------|----------|--------|------------------|
| Kiến trúc chính | Encoder–Decoder LSTM | Đúng yêu cầu đề; LSTM xử lý phụ thuộc dài hạn của chuỗi 10 phút tốt hơn RNN thường; bộ mã hóa nén 1 ngày thành véc-tơ ngữ cảnh, bộ giải mã sinh từng bước của ngày mai | GRU (ít sức biểu đạt hơn cho chuỗi dài 144), TCN (không đúng tinh thần Seq2Seq của đề) |
| Attention | Bahdanau (dạng cộng) trên trạng thái ẩn của bộ mã hóa | Cho phép bộ giải mã dài 144 bước "nhìn lại" giờ quan trọng (ví dụ biến đổi đột ngột trước bão), cải thiện nửa sau của ngày dự báo; cài đặt nhẹ | Luong dạng tích vô hướng (để dự phòng nếu Bahdanau không ổn định) |
| Đối sánh | Transformer mini dạng chỉ encoder (2 lớp, 4 đầu) | Đúng phần mở rộng 2; cơ chế tự chú ý xử lý song song 144 bước, đối trọng với LSTM tuần tự | Informer/Autoformer (quá nặng cho bài tập) |
| Hàm mất mát | MSE khi huấn luyện, báo cáo thêm MAE/RMSE | MSE phạt mạnh sai số lớn, phù hợp hồi quy nhiệt độ; MAE dễ diễn giải (°C) | Huber (để dự phòng nếu điểm ngoại lai nhiều) |
| Khung công nghệ | PyTorch kèm scikit-learn, pandas/numpy, matplotlib | Quen thuộc, tài liệu Seq2Seq nhiều, dễ xuất mô hình và vẽ `report.html` | TensorFlow/Keras (nhóm ít quen hơn) |

## 4. Ở đâu / Khi nào / Bằng gì — Kèm tiêu chí công nghệ

- **Ở đâu (chạy ở đâu):** máy tính cá nhân (local), ưu tiên chạy được trên CPU; có GPU thì dùng để tăng tốc giải mã 144 bước (không bắt buộc). RAM từ 8 GB trở lên.
- **Khi nào (khi nào chạy):** huấn luyện theo lô (ngoại tuyến), suy luận theo lô ngoại tuyến kèm demo một ngày mẫu trực tiếp. Không yêu cầu phát trực tuyến thời gian thực.
- **Bằng gì (dùng khung / thư viện / công cụ nào):**
  - Học sâu: PyTorch (`torch`, `DataLoader`), cố định hạt giống (seed) để tái lập.
  - Dữ liệu: pandas, numpy, scikit-learn (`StandardScaler`), PyYAML (đọc `configs/*.yaml`).
  - Theo dõi và vẽ: matplotlib (bắt buộc cho `plots/`), TensorBoard hoặc nhật ký CSV (tùy chọn).
  - Báo cáo: HTML thuần ngoại tuyến (không dùng CDN), kịch bản `src/evaluation/visualize_html.py`.
- **Tiêu chí công nghệ:**
  - Phần cứng: phải huấn luyện được trên CPU trong khoảng 20 phút/mỗi epoch với cấu hình tối thiểu tầm 144 (`hidden=128`, `layers=2`, `batch=64`).
  - Hiệu năng: suy luận một mẫu một ngày trên CPU ≤ 100 ms; MAE kiểm thử toàn tầm 144 ≤ 1,5 °C.
  - Giấy phép/chi phí: toàn bộ mã nguồn mở, không phí đám mây.
  - Kỹ năng nhóm: dùng ngăn xếp đã quen (PyTorch kèm sklearn), không học khung mới trong bản tối thiểu.
  - Triển khai/bảo trì: điểm kiểm tra `.pt` kèm cấu hình YAML định phiên bản `jena_lstm_v1`, chưa cần ONNX/Docker ở bản tối thiểu.

## 5. Làm thế nào? — Luồng thực hiện ra sao?

- **Luồng quy trình:** `CSV thô Jena → đã xử lý (làm sạch + chuẩn hóa + tách gió) → chia theo năm → cửa sổ trượt 144→144 → DataLoader → huấn luyện (teacher forcing) → checkpoint → dự báo 144 bước → chuyển ngược tỉ lệ → chỉ số theo tầm 1/24/72/144 → report.html`.
- **Huấn luyện:** Adam (`lr = 1e-3`, `weight_decay = 1e-5`), lô 64, tối đa 30 epoch, dừng sớm với kiên nhẫn (patience) 5 theo hàm mất mát kiểm định, giảm tốc độ học khi bão hòa, cắt chuẩn (clipping) 1,0. Tỉ lệ teacher forcing khởi đầu 0,5 rồi giảm dần vì bộ giải mã dài 144 bước dễ lệch khi tự hồi tiếp.
- **Siêu tham số chính:** `input_len=144`, `pred_len=144`, `hidden=128`, `num_layers=2`, `dropout=0,1`, Transformer mini `d_model=64, nhead=4, layers=2`.
- **Đánh giá:** MSE/hàm mất mát kiểm định để chọn checkpoint; chỉ số chính MAE/RMSE trên °C gốc, chi tiết tại tầm 1/24/72/144; đối sánh với persistence; phần mở rộng đối sánh 3 mô hình cùng bảng tầm 144.
- **Tiền xử lý đầu vào:** StandardScaler (chỉ khớp trên huấn luyện), tách `wd` thành sin/cos, không tăng cường chuỗi ở bản tối thiểu (mức Could: thêm nhiễu Gauss nhỏ).
- **Hậu xử lý đầu ra:** chuyển ngược 144 giá trị về °C, chặn biên vật lý hợp lý (ví dụ −40…50 °C để phát hiện lỗi), lưu CSV dự báo theo ngày kèm vẽ đồ thị.

## 6. Đánh giá — Đo lường thế nào?

### Mức mô hình (tập kiểm thử năm 2016, tầm 144 = 1 ngày)

- MAE toàn tầm ≤ 1,5 °C; RMSE toàn tầm ≤ 2,0 °C; MAE@1 < MAE@24 < MAE@72 < MAE@144 và MAE@24 ≤ 1,2 °C, MAE@144 ≤ 2,2 °C.
- Vượt đường cơ sở persistence trên cùng tập kiểm thử.
- Suy luận một mẫu một ngày trên CPU ≤ 100 ms; kích thước checkpoint ≤ 50 MB.

### Mức toàn quy trình

- Độ trễ đầu-cuối (đọc + tiền xử lý + suy luận 144 bước + vẽ một ngày mẫu) ≤ 3 giây trên CPU.
- Tài nguyên: RAM ≤ 4 GB lúc huấn luyện với lô 64; nhật ký đầy đủ `metrics.json` kèm `logs.txt`.
- Xử lý lỗi: thiếu cột, NaN, sai kích thước phải báo lỗi rõ ràng, không sập ngầm.
- Tái lập: chạy lại từ `configs/*.yaml` cho chỉ số tương đương (sai số hạt giống cho phép).

### Tiêu chí đạt / không đạt (định nghĩa hoàn thành của Define)

- [x] Đủ 6 mục định nghĩa, khớp đủ 9 dòng đề, không còn mục nào ghi TBD chặn bản tối thiểu.
- [x] Kích thước vào/ra 144→144 và cách chia tập đã chốt bằng số.
- [x] Chỉ số và ngưỡng cho tầm 144 đã chốt bằng số.
- [x] Đã đồng bộ bối cảnh dự án vào `docs/progress_status.md`.

## 7. Câu hỏi còn mở (không chặn bản tối thiểu)

- [x] Có tách `wd (deg)` thành sin/cos (F=15) hay giữ 14 đặc trưng gốc? → đã chốt `F = 15` từ EDA 2026-11-03 (`results/eda/04_gio_sincos.png`, `docs/progress_status.md`).
- [ ] Trình bày chính 1 ngày (144) hay thêm 2–3 ngày (288/432)? → mặc định 144, thử đa ngày ở mức Could.
- [ ] Cấu hình CPU/GPU cụ thể của máy huấn luyện là gì? → ghi vào tiến độ khi chạy thử epoch đầu.
