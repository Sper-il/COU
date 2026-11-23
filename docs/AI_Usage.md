# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong dự án `jena-seq2seq`.

## 1. Tools Used

| Vai trò | Tool / Model | Phiên bản | Khi nào |
|---------|--------------|-----------|---------|
| Code generation | Cursor + Claude | 2026-11-23 | Viết `src/models/architecture.py`, `loss.py`, `smoke_test.py`, `tests/unit/test_models.py` |
| Debugging | Cursor + Claude | 2026-11-23 | Tìm và sửa lỗi teacher forcing off-by-one; tối ưu Attention Bahdanau cho ngưỡng 100 ms; thống nhất dạng đầu ra 3 kiến trúc |
| Code generation | Cursor + Claude | 2026-11-08 | Viết `src/data/transforms.py`, `split.py`, `dataset.py`, `__main__.py`, `tests/unit/test_dataset.py` |
| Documentation | Cursor + Claude | 2026-11-03 | Nháp `README.md`, `docs/architecture.md`, `data-flow.md`, `api.md`, `data.md`, báo cáo EDA HTML |
| Debugging | Cursor + Claude | 2026-11-03 | Sửa lỗi encoding `cp932` khi in tiếng Việt trên Windows (ép `stdout` UTF-8 + `PYTHONIOENCODING`) |
| Review | Cursor + Claude | 2026-11-03 | Tự review script EDA và HTML theo checklist offline (base64, không CDN, số khớp JSON) |

## 2. Plain Purposes

- **Code generation** — scaffold script EDA và script render HTML từ thiết kế đã duyệt (`docs/design.md`, chuẩn `modules/visualization.md`).
- **Code generation** — viết 3 kiến trúc mô hình (`LSTMS2S`, `LSTMS2SAttention`, `MiniTransformer`) theo `configs/model_config.yaml`, kèm hàm mất mát MSE và script kiểm thử nhanh trên dữ liệu thật.
- **Debugging** — phát hiện lỗi teacher forcing lệch 1 bước (off-by-one) bằng cách đọc kỹ vòng giải mã và đối chiếu hàm mất mát kiểm định trước/sau khi sửa (0,26253 → 0,16659 sau 1 epoch).
- **Debugging** — đo độ trễ suy luận từng thành phần, phát hiện Attention Bahdanau vượt ngưỡng 100 ms, tối ưu bằng cách chiếu `W_e·enc_out` một lần (108,9 ms → 90,5 ms).
- **Documentation** — nháp README 10 mục và bộ docs kỹ thuật từ `docs/define.md` + kết quả EDA thật.
- **Debugging** — chẩn đoán `UnicodeEncodeError: 'cp932'` khi chạy script có tiếng Việt trên PowerShell, đề xuất ép UTF-8.
- **Review** — kiểm tra `eda_report.html` offline thật sự (tách base64 rồi quét link ngoài), kiểm tra số trong HTML khớp `eda_summary.json`.

## 3. Student Contributions (con người làm gì)

- Định nghĩa bài toán 144→144, chia tập theo năm và ngưỡng MAE/RMSE (trong `docs/define.md`, `docs/description.md`).
- Quyết định kiến trúc cuối cùng: LSTM Encoder–Decoder + Attention Bahdanau + Transformer mini (kèm lý do và lựa chọn đã loại).
- Quyết định chốt `F = 15` (tách gió sin/cos) và xử lý `-9999` bằng nội suy — dựa trên bằng chứng EDA, không phó mặc cho AI.
- Toàn bộ review cuối cùng cho code AI generate; toàn bộ commit do người quyết định.
- Xác minh số EDA bằng truy vấn độc lập trên CSV trước khi chấp nhận kết quả script.

## 4. Important AI-Generated Artifacts

| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| `notebooks/2026-11-03_eda.py` | Script EDA: đọc CSV Jena, thống kê, 6 hình PNG, `eda_summary.json` | "Chạy EDA sâu trên CSV Jena: mã thiếu -9999, tương quan với T, baseline persistence 2016, 6 hình matplotlib" |
| `src/data/transforms.py`, `split.py`, `dataset.py`, `__main__.py` | Pipeline dữ liệu Jena 144→144, `F = 15`, scaler chỉ fit trên train, splits tái lập | "Làm bước 3 module Dữ liệu: transforms, split theo năm, dataset, DataLoader, scaler.pkl, splits.npz" |
| `tests/unit/test_dataset.py` | 4 kiểm thử: shape lô, chống rò rỉ scaler, không chồng lấn thời gian, gió đơn vị | "Viết kiểm thử đơn vị cho module Data theo checklist chống leakage" |
| `src/evaluation/visualize_eda_html.py` | Render `eda_report.html` offline từ JSON + PNG (base64, CSS nội tuyến) | "Viết script render report.html EDA offline theo chuẩn visualization.md, số render từ JSON không gõ tay" |
| `results/eda/eda_report.html` | Báo cáo EDA 1 file (~638 KB), 6 hình, bảng metric persistence | Sinh tự động từ script trên, không viết tay |
| `configs/*.yaml` | Cấu hình data/model/train chốt tầm 144 | "Chép cấu hình chốt từ docs/design.md mục 8 vào 3 file YAML" |
| `src/models/architecture.py` | 3 kiến trúc đúng đề (LSTM S2S / Attention Bahdanau / Transformer mini), dạng đầu ra neo cùng giờ + phần dư, cổng trộn 2 neo | "Làm Bước 4 module Mô hình: 3 kiến trúc đúng configs/model_config.yaml, forward (B,144,15)→(B,144), chữ ký giống docs/api.md mục 2" |
| `src/models/loss.py` | `mse_loss`, `mae_loss`, `masked_mse_loss` | "Viết hàm mất mát MSE cho hồi quy Seq2Seq theo docs/define.md mục 3" |
| `src/models/smoke_test.py` | Kiểm thử nhanh 3 kiến trúc trên dữ liệu Jena thật, 1 lệnh | "Viết script kiểm thử model trên dữ liệu thật: shape, MAE °C gốc so với persistence, loss, biên vật lý, độ trễ suy luận" |
| `tests/unit/test_models.py` | 13 kiểm thử: shape, tự hồi tiếp, teacher forcing, `mse_loss`, kích thước checkpoint, validate shape, `build_model`, cổng trộn, độ trễ | "Viết kiểm thử đơn vị cho module Mô hình theo checklist nghiệm thu Bước 4, kèm test chống tái phát lỗi đã sửa" |
| `README.md`, `docs/architecture.md`, `data-flow.md`, `api.md`, `data.md` | Bộ docs bổ sung theo khung skill | "Bổ sung docs còn thiếu theo templates, số liệu từ eda_summary.json" |

## 5. Prompt Purpose (cho mỗi artifact)

- **`notebooks/2026-11-03_eda.py`** — yêu cầu AI viết script EDA tái lập được (thay notebook) với đầy đủ thống kê NaN/`-9999`/tương quan/persistence/cửa sổ 144→144.
- **`src/evaluation/visualize_eda_html.py`** — yêu cầu AI render HTML 1 file offline, nhúng ảnh base64, bảng persistence có cột mục tiêu, mục hạn chế và bước tiếp theo.
- **Bộ docs** — yêu cầu AI viết docs kỹ thuật bám số thật trong `eda_summary.json`, giữ thuật ngữ tiếng Anh gốc (DataLoader, StandardScaler, teacher forcing, checkpoint).

## 6. Student Modifications

- **`notebooks/2026-11-03_eda.py`** — thêm ép `stdout` UTF-8 sau khi AI đề xuất bản đầu bị lỗi `cp932` trên Windows; giữ logic EDA của AI vì số đã đối chiếu đúng.
- **`src/evaluation/visualize_eda_html.py`** — giữ nguyên logic render của AI sau khi kiểm tra offline đạt (không link ngoài ngoài từ "CDN" trong câu chữ).
- **Docs** — người bổ sung số persistence (2,806 °C) và quyết định `F = 15` từ EDA vào `progress_status.md`, AI chỉ nháp văn bản.

## 7. Validation Method

- [x] Chạy code end-to-end trên môi trường sạch (`python notebooks/2026-11-03_eda.py`, `python -m src.evaluation.visualize_eda_html`).
- [x] So sánh output với truy vấn độc lập (shape 420.551, NaN = 0, MAE persistence ≈ 2,806 °C).
- [x] Kiểm tra HTML offline: tách base64 rồi quét link ngoài → không có link ngoài; 6 ảnh base64 đầy đủ; số khớp `eda_summary.json`.
- [x] Sanity-check metric theo Definition of Done (MAE persistence > mục tiêu 1,5 °C — đúng kỳ vọng đường cơ sở ngây thơ).
- [x] Module Data: `python -m src.data --config configs/data_config.yaml` chạy đạt shape `(64,144,15) → (64,144)` — kiểm tra lại ngày 2026-11-09: train 26.192 mẫu (410 lô), val 364, test 359, `splits.npz` khớp; `pytest tests/unit/test_dataset.py` 4 passed (shape, chống rò rỉ scaler, không chồng lấn thời gian, gió đơn vị).
- [x] Module Mô hình: `python -m src.models.smoke_test` chạy đạt trên **dữ liệu Jena thật** — cả 3 kiến trúc cho shape `(8,144,15) → (8,144)`, loss MSE giảm, 0 giá trị ngoài biên vật lý, độ trễ suy luận LSTM 73 ms / Attention 90 ms / Transformer 3 ms (đều ≤ 100 ms); `pytest tests/unit/test_models.py -q` 13 passed; `pytest -q` 25 passed.
- [x] Xác minh lỗi teacher forcing đã sửa bằng số đo: hàm mất mát kiểm định sau 1 epoch giảm từ 0,26253 (mã cũ) xuống 0,16659 (mã mới).
- [x] Chạy đầu-cuối `train → checkpoint → evaluate → report.html`: MAE 2,4923 °C trên tập 2016 (đã vượt persistence 2,8150 °C), `report.html` 366 KB, 5 ảnh base64, 0 link ngoài (quét bằng regex).
- [ ] Reproduce từ fresh clone (để dành khi có git remote).
