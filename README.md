# Dự báo thời tiết chuỗi-tới-chuỗi trên bộ dữ liệu khí hậu Jena (`jena-seq2seq`)

> Từ 1 ngày quan trắc khí tượng (144 bước × 10 phút), dự báo chuỗi nhiệt độ của 1 ngày tiếp theo (144 bước) bằng Encoder–Decoder LSTM, có mở rộng Attention và đối sánh với Transformer.

## 1. Project Overview

Dự án `jena-seq2seq` giải bài toán hồi quy chuỗi-tới-chuỗi (Sequence-to-Sequence) trên bộ dữ liệu khí hậu Jena (2009–2016, chu kỳ 10 phút). Mô hình bộ mã hóa (Encoder) nén 1 ngày quá khứ `X (144, F)` thành véc-tơ ngữ cảnh, bộ giải mã (Decoder) sinh từng bước nhiệt độ `T (degC)` của 1 ngày tiếp theo `y (144,)`.

Đối tượng phục vụ: sinh viên làm bài tập Seq2Seq (chạy được bằng một lệnh, có `report.html`), giảng viên chấm phần hồi quy và mở rộng Attention/Transformer (chỉ số theo tầm 1/24/72/144), người nghiên cứu thời tiết cần đường cơ sở Jena tái lập được. Dự báo nhiệt độ theo ngày phục vụ nông nghiệp, điều độ năng lượng và cảnh báo thời tiết cực đoan.

Báo cáo EDA mới nhất: mở `results/eda/eda_report.html` bằng double-click (offline 100%, 6 hình, số liệu khớp `results/eda/eda_summary.json`).

## 2. Problem Statement

Cho ma trận quá khứ `X ∈ R^{144×15}` (1 ngày, 15 đặc trưng đã chuẩn hóa sau khi tách gió), dự báo véc-tơ tương lai `y ∈ R^{144}` là nhiệt độ `T (degC)` của 144 bước kế tiếp. Hàm mất mát huấn luyện là MSE; chỉ số báo cáo là MAE/RMSE trên °C gốc, chi tiết tại tầm 1/24/72/144, kèm đường cơ sở persistence (lặp giá trị cuối ngày hôm qua). Ngưỡng bản chạy tối thiểu trên tập kiểm thử 2016: MAE toàn tầm ≤ 1,5 °C, RMSE ≤ 2,0 °C, vượt persistence (MAE 2,806 °C đo từ EDA).

## 3. Main Features

| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Quy trình dữ liệu tái lập: đọc CSV Jena → nội suy mã thiếu `-9999` → tách gió sin/cos → StandardScaler (chỉ fit trên train) → cửa sổ 144→144 → DataLoader | Không có thì không huấn luyện được mô hình nào |
| Must | Encoder–Decoder LSTM cơ sở (không attention), MSE, teacher forcing | Đúng tên mô hình trong đề, làm mốc đối sánh |
| Must | Đánh giá nhiều bước: MAE/RMSE toàn tầm 144 và tại 1/24/72/144 kèm persistence | Đề là multi-step, một số trung bình là chưa đủ |
| Must | `metrics.json` kèm `report.html` offline, đồ thị loss, mẫu tốt/xấu theo ngày | Chuẩn nghiệm thu |
| Should | LSTM Encoder–Decoder kèm Attention Bahdanau | Phần mở rộng thứ nhất trong đề |
| Should | Transformer mini đối sánh cùng đặc trưng, cùng tầm 144 | Phần mở rộng thứ hai trong đề |
| Should | Teacher forcing giảm dần kèm early stopping và checkpoint theo loss kiểm định | Ổn định giải mã dài 144 bước |
| Could | Dự báo đa ngày `{144, 288, 432}` và demo một lệnh `python -m src.demo` | Bao phủ chữ "days" số nhiều, phục vụ trình bày |

Ngoài phạm vi: không dự báo đa biến, không thêm dữ liệu ngoài Jena, không triển khai API/đám mây hay giao diện web, không tìm kiếm siêu tham số quy mô lớn.

## 4. System Architecture

```text
data/raw/jena_*.csv (10 phút, 14 biến + Date Time)
        │  src/data (đọc, nội suy -9999, tách wd → sin/cos, chuẩn hóa, cửa sổ 144→144, chia theo năm)
        v
data/processed + data/splits ──► DataLoader (B,144,15) → (B,144)
        │  src/models (LSTM S2S / kèm Attention Bahdanau / Transformer mini)
        v
src/training (teacher forcing giảm dần, early stopping) ──► models/checkpoints
        │  src/evaluation (inverse-transform, MAE/RMSE tại 1/24/72/144, persistence, report.html)
        v
results/runs/<thời gian>/{metrics.json, plots/, report.html}
```

Chi tiết xem `docs/architecture.md`, luồng dữ liệu xem `docs/data-flow.md`, giao diện module xem `docs/api.md`.

## 5. Folder Structure

```text
e:/COU/
├── data/{raw,processed,splits}      # dữ liệu (gitignore trừ raw gốc)
├── src/{data,models,training,evaluation}  # mã nguồn (evaluation chứa visualize_eda_html.py)
├── models/{checkpoints,trained,configs}
├── configs/{data_config.yaml,model_config.yaml,train_config.yaml}
├── notebooks/2026-11-03_eda.py      # script EDA (chạy thay notebook để tái lập)
├── tests/{unit,integration}
├── docs/{description.md,define.md,design.md,architecture.md,data-flow.md,api.md,
│         data.md,progress_status.md,AI_Usage.md,CHANGELOG.md}
├── results/eda/{eda_report.html,eda_summary.json,*.png}  # báo cáo EDA offline
├── results/runs/<thời gian>/{metrics.json,plots/,report.html}
├── requirements.txt / .gitignore / README.md
```

## 6. Installation / Run / Test

```bash
# 1. Tạo môi trường
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Cài dependency
pip install -r requirements.txt

# 3. Dữ liệu: file CSV nằm sẵn tại
# data/raw/jena_climate_2009_2016.csv/jena_climate_2009_2016.csv (~43 MB)

# 4. Chạy EDA (sinh 6 hình + eda_summary.json)
python notebooks/2026-11-03_eda.py

# 5. Sinh báo cáo EDA HTML offline (638 KB, double-click là xem)
python -m src.evaluation.visualize_eda_html
# → results/eda/eda_report.html

# 6. Test (khi có tests ở giai đoạn Data)
pytest -q
```

## 7. Example Input → Output

**Input** — một mẫu một ngày quá khứ (đã chuẩn hóa, `F = 15` sau tách gió):

```json
{
  "X_shape": [144, 15],
  "X_sample": "1 ngày quá khứ, 144 bước × 10 phút, 15 đặc trưng (p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max_wv, wd_sin, wd_cos)",
  "date": "2016-06-15"
}
```

**Output** — chuỗi nhiệt độ 1 ngày tiếp theo:

```json
{
  "y_shape": [144],
  "y_unit": "degC (đã inverse-transform về đơn vị gốc)",
  "y_preview": [18.2, 18.0, 17.8, "...", 21.5],
  "inference_time_ms": "<= 100 (mục tiêu CPU)"
}
```

## 8. Results

Báo cáo EDA: `results/eda/eda_report.html` + `results/eda/eda_summary.json` (nguồn số đúng duy nhất của giai đoạn EDA).

| Metric (tập 2016, stride 144) | Target (cho Seq2Seq) | Persistence (đường cơ sở) | Seq2Seq |
|-------------------------------|----------------------|---------------------------|---------|
| MAE toàn tầm 144 | ≤ 1,50 °C | 2,806 °C | TBD (giai đoạn Training) |
| RMSE toàn tầm 144 | ≤ 2,00 °C | TBD | TBD |
| MAE@1 / @24 / @72 / @144 | — / ≤ 1,20 / — / ≤ 2,20 °C | 0,12 / 0,94 / 1,77 / 2,81 °C | TBD |
| Suy luận 1 mẫu/ngày (CPU) | ≤ 100 ms | — | TBD |

Kết quả huấn luyện sẽ ghi vào `results/runs/<timestamp>/metrics.json` kèm `report.html` ở giai đoạn Eval.

## 9. Limitations

- Có 1 khoảng trống 3 ngày trong tập kiểm thử 10/2016 — các cửa sổ trượt cắt qua đoạn này phải loại bỏ khi tạo `src/data`.
- Biến `Tpot` tương quan 0,997 với `T` — nguy cơ mô hình ỷ lại một biến; cần kiểm tra ablation khi đánh giá.
- EDA mới dừng ở thống kê mô tả, chưa đo MAE theo tầm cho từng mùa — để dành cho `report.html` sau huấn luyện.
- Chưa có mã `src/data`, `src/models`, `src/training` — mới xong giai đoạn dựng khung và EDA.

## 10. Future Improvements

- Cài đặt `src/data` với kiểm thử đơn vị về shape `(144,15)/(144,)`, chống rò rỉ scaler và chia thời gian.
- Huấn luyện LSTM Encoder–Decoder cơ sở tầm 144, đối sánh với mốc persistence 2,806 °C.
- Mở rộng Attention Bahdanau và Transformer mini, báo cáo cùng bảng MAE tại 1/24/72/144.
- Thử đa ngày `{288, 432}` và demo một lệnh ở mức Could.

---
Được duy trì với skill **ai-fusion-workflow**. Xem `docs/AI_Usage.md` để biết cách AI được sử dụng trong dự án.
