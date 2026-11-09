# Kế hoạch chia việc Bước 4–7 — Dự án `jena-seq2seq`

> Nguồn đối chiếu: `đề.txt` (9 dòng), `docs/define.md`, `docs/design.md`, `docs/description.md`,
> `docs/api.md`, `docs/progress_status.md`, `configs/data_config.yaml`,
> `configs/model_config.yaml`, `configs/train_config.yaml`.
> Mọi số liệu dưới đây sao nguyên từ các tài liệu trên, không tự đặt số mới.

## 1. Bối cảnh chung

- **Bài toán:** hồi quy chuỗi-tới-chuỗi, nhiều bước. Vào `X ∈ R^{144×15}`
  (1 ngày quá khứ, 144 bước × 10 phút, `F = 15` sau khi tách `wd (deg)` thành
  `wd_sin`/`wd_cos`), ra `y ∈ R^{144}` là nhiệt độ `T (degC)` 1 ngày tiếp theo.
- **Dữ liệu:** `data/raw/jena_climate_2009_2016.csv` (~420.551 dòng).
  Chia theo năm: huấn luyện 2009–2014, kiểm định 2015, kiểm thử 2016.
  `stride_train = 12`, `stride_eval = 144`. `StandardScaler` chỉ khớp trên huấn luyện.
- **Hiện trạng:** Bước 3 `src/data` đã xong (train 26.192 mẫu / 410 lô, val 364 / 6 lô,
  test 359 / 6 lô, shape `(64,144,15) → (64,144)`, `pytest` 4 passed,
  có `data/processed/scaler.pkl` + `data/splits/splits.npz`).
- **Đường cơ sở đã có:** `persistence` 2016 (lặp giá trị `T` cuối ngày hôm qua cho cả
  144 bước ngày mai): `MAE_full = 2,806 °C`, `MAE@1 = 0,119`, `MAE@24 = 0,942`,
  `MAE@72 = 1,770`, `MAE@144 = 2,806` (`results/eda/eda_summary.json`).
  Mọi mô hình Seq2Seq phải vượt mốc này.
- **Ngưỡng nghiệm thu (tập kiểm thử 2016, tầm 144):**
  `MAE ≤ 1,5 °C`, `RMSE ≤ 2,0 °C`, `MAE@24 ≤ 1,2 °C`, `MAE@144 ≤ 2,2 °C`,
  suy luận 1 mẫu/ngày trên CPU `≤ 100 ms`, đầu-cuối demo `≤ 3 s`,
  checkpoint `≤ 50 MB`, RAM huấn luyện `≤ 4 GB` với lô 64.

## 2. Sơ đồ phân công 4 người

| Người | Bước | Phạm vi tệp | Ánh xạ đề | Nhận từ ai | Giao cho ai |
|-------|------|-------------|-----------|------------|-------------|
| **A** | Bước 4 `Model` | `src/models/architecture.py`, `src/models/loss.py`, `tests/unit/test_models.py` | Model Encoder–Decoder LSTM (Must), Attention Bahdanau (Should), Transformer mini (Should) | `src/data` (DataLoader `(B,144,15)`) | B (checkpoint logic qua `forward`) |
| **B** | Bước 5 `Training` | `src/training/trainer.py`, `src/training/scheduler.py`, `models/checkpoints/`, `results/runs/<ts>/logs.txt` | Teacher forcing giảm dần + dừng sớm + checkpoint (Should) | A (3 lớp mô hình chạy `forward` không lỗi) | C (đường dẫn `best_ckpt`) |
| **C** | Bước 6 `Eval + Visualize HTML` | `src/evaluation/metrics.py`, `src/evaluation/visualize_html.py`, `results/runs/<ts>/{metrics.json, plots/, report.html}` | Task Type multi-step tại 1/24/72/144 + persistence (Must) | B (`best_ckpt`) + `src/data.get_scaler()` | D + cả nhóm (bảng 3 mô hình) |
| **D** | Bước 7 `Demo + chốt Document` | `src/demo.py`, `data/processed/demo/`, `results/demo/<ts>/`, `README.md`, `docs/progress_status.md`, `docs/AI_Usage.md`, `docs/CHANGELOG.md` | Exercises Seq2Seq + Goal upcoming days, mức Could demo 1 lệnh | C (`best_ckpt` + `metrics.json`) | Giảng viên / người chấm |

Nguyên tắc chống dẫm chân: A chốt chữ ký `forward` trong `docs/api.md` trước thì B mới viết
vòng huấn luyện; B chốt định dạng checkpoint (`.pt` gồm `state_dict` + `config` + `epoch`)
thì C mới viết `evaluate()`; C chốt `metrics.json` thì D mới viết demo và tài liệu cuối.

## 3. Người A — Bước 4 Module Mô hình (`src/models`)

### 3.1. Mục tiêu

Cả 3 kiến trúc chạy `forward` `(B,144,15) → (B,144)` không lỗi trên CPU, đúng siêu tham số
trong `configs/model_config.yaml` và `docs/design.md` mục 8.

### 3.2. Việc chi tiết

1. `src/models/loss.py`: hàm `mse_loss(pred: (B,144), target: (B,144)) → scalar`
   (MSE khi huấn luyện, khớp `docs/define.md` mục 3).
2. `src/models/architecture.py`:
   - `LSTMS2S`: Encoder–Decoder LSTM cơ sở, `hidden_dim = 128`, `num_layers = 2`,
     `dropout = 0,1`, `pred_len = 144`. `forward(x, y_in)` với `y_in: (B,144) | None`.
     Lúc huấn luyện `y_in` là mục tiêu dịch phải 1 bước (teacher forcing);
     lúc suy luận `y_in = None`, bộ giải mã tự hồi tiếp đủ 144 bước.
   - `LSTMS2SAttention`: cùng cấu hình trên + Attention Bahdanau (dạng cộng) trên
     trạng thái ẩn bộ mã hóa. Dự phòng Luong nếu Bahdanau mất ổn định
     (đã ghi trong `docs/define.md` mục 3).
   - `MiniTransformer`: chỉ encoder, `d_model = 64`, `nhead = 4`, `num_layers = 2`,
     `dropout = 0,1`, cộng đầu chiếu `15 → 64` và đầu hồi quy `64 → 1` cho từng bước
     trong 144 bước.
3. `tests/unit/test_models.py`: 4 kiểm thử — shape `(2,144,15) → (2,144)` cho cả 3 lớp,
   chế độ `y_in = None` sinh đủ 144 bước, `mse_loss` đúng công thức, số tham số
   checkpoint ước tính `≤ 50 MB`.
4. Chạy `pytest tests/unit/test_models.py -q` và đo thời gian `forward` 1 mẫu trên CPU
   (mục tiêu `≤ 100 ms`, khớp `docs/define.md` mục 4).

### 3.3. Nghiệm thu

- [ ] 3 lớp `forward` đúng chữ ký `docs/api.md` mục 2.
- [ ] `pytest` đạt, log dán vào `docs/progress_status.md`.
- [ ] Không đổi `F = 15`, `input_len/pred_len = 144`.

## 4. Người B — Bước 5 Module Huấn luyện (`src/training`)

### 4.1. Mục tiêu

Huấn luyện đầu-cuối LSTM cơ sở tầm 144 trước, sau đó tái dùng cho 2 mở rộng.
Xuất checkpoint tốt nhất theo MSE kiểm định (`docs/design.md` mục 5).

### 4.2. Việc chi tiết

1. `src/training/scheduler.py`: `build_scheduler(optimizer, config)` giảm tốc độ học
   khi hàm mất mát kiểm định bão hòa (khớp `docs/api.md` mục 3).
2. `src/training/trainer.py`: hàm `train(config) → best_ckpt_path: str` với đúng số
   trong `configs/train_config.yaml`:
   `batch_size = 64`, `epochs = 30`, `lr = 1e-3`, `optimizer = adam`,
   `weight_decay = 1e-5`, `teacher_forcing = 0,5 × 0,98 mỗi epoch`,
   `clip = 1,0`, `early_stop = 5`, `seed = 42`.
3. Luồng 1 epoch: đọc `build_dataloaders(config)` từ `src/data`, bộ giải mã nhận `y`
   dịch phải 1 bước với tỉ lệ teacher forcing hiện hành; cắt chuẩn 1,0; ghi
   `results/runs/<thời gian>/logs.txt`; lưu `models/checkpoints/jena_lstm_v1.pt`
   (đặt tên theo quy tắc `{domain}_{architecture}_{version}` trong skill
   `ai-fusion-workflow`). Checkpoint lưu `state_dict`, `model_config`, `epoch`,
   `val_loss`, `teacher_forcing` hiện hành để C tái nạp.
4. Chạy thử epoch đầu trên CPU, ghi thời gian/epoch (ngưỡng `≤ 20 phút/epoch`
   với `hidden = 128`, `layers = 2`, `batch = 64` theo `docs/define.md` mục 4).
   Nếu quá chậm thì báo cả nhóm, không tự đổi `hidden/batch` khi chưa ghi quyết định.
5. Xuất 3 checkpoint: `jena_lstm_v1.pt` (Must, bắt buộc), `jena_attn_v1.pt`,
   `jena_trans_v1.pt` (Should, nếu kịp).

### 4.3. Nghiệm thu

- [ ] `train(config)` trả về đường dẫn checkpoint tốt nhất.
- [ ] Dừng sớm patience 5 hoạt động, có `logs.txt` chứng minh.
- [ ] Checkpoint nạp lại được trên máy khác chỉ từ `configs/*.yaml`.

## 5. Người C — Bước 6 Đánh giá + Visualize HTML (`src/evaluation`)

### 5.1. Mục tiêu

Biến checkpoint thành `metrics.json` và `report.html` ngoại tuyến, đúng Task Type
multi-step trong đề (toàn tầm + tại tầm, kèm persistence).

### 5.2. Việc chi tiết

1. `src/evaluation/metrics.py`: hàm `evaluate(ckpt, loader)` (chữ ký theo `docs/api.md`
   mục 4):
   - Chuyển ngược 144 giá trị về °C bằng `get_scaler()` (scaler chỉ khớp trên huấn luyện,
     khớp `docs/define.md` mục 1).
   - Tính `MAE/RMSE` toàn tầm 144 và `MAE@1/@24/@72/@144` trên °C gốc.
   - Tính `persistence_mae` cùng tập kiểm thử (lặp giá trị cuối) để đối sánh;
     mốc tham chiếu EDA là 2,806 °C.
   - Chặn biên vật lý `−40…50 °C` để phát hiện lỗi (theo `docs/define.md` mục 5).
   - Ghi `results/runs/<thời gian>/metrics.json` gồm cả 3 mô hình trên cùng bảng tầm 144
     (yêu cầu Should trong `docs/description.md` mục 5).
2. `src/evaluation/visualize_html.py`: hàm `build_report(run_dir) → report.html`:
   - Đọc `metrics.json + config.yaml + plots/*.png` trong `results/runs/<ts>/`.
   - Vẽ: đường hàm mất mát huấn luyện/kiểm định, đường MAE theo tầm 1/24/72/144,
     1 mẫu dự báo tốt + 1 mẫu xấu theo ngày, 1 trường hợp tệ nhất.
   - Render `report.html` ngoại tuyến 100% (CSS nội tuyến, ảnh `base64`, không CDN),
     tái dùng khung `src/evaluation/visualize_eda_html.py` đã làm ở EDA.
   - Kiểm tra nháy đúp mở được, số trên HTML khớp `metrics.json`
     (chuẩn nghiệm thu `docs/design.md` mục 6).
3. Đối chiếu ngưỡng `docs/description.md` mục 6 và đánh dấu đạt/không đạt từng chỉ số
   trong `metrics.json` (`MAE ≤ 1,5`, `RMSE ≤ 2,0`, `MAE@24 ≤ 1,2`, `MAE@144 ≤ 2,2`,
   vượt persistence).

### 5.3. Nghiệm thu

- [ ] `metrics.json` có đủ `mae, rmse, mae_h1/h24/h72/h144, persistence_mae`.
- [ ] `report.html` mở ngoại tuyến, có bảng đạt/không đạt + đồ thị hàm mất mát +
      mẫu tốt/xấu + trường hợp tệ nhất + độ trễ suy luận.
- [ ] Bảng đối sánh 3 mô hình cùng đặc trưng, cùng tầm 144.

## 6. Người D — Bước 7 Demo + chốt Document (`src/demo.py` + `docs/`)

### 6.1. Mục tiêu

Biến lần chạy tốt nhất thành demo 1 lệnh và chốt bộ tài liệu nghiệm thu
(theo skill `ai-fusion-workflow` Giai đoạn 5–6 và `docs/design.md` mốc tuần 4).

### 6.2. Việc chi tiết

1. `src/demo.py`: lệnh `python -m src.demo --date 2016-06-15` (cú pháp theo `docs/api.md`
   mục 5):
   - Nạp `best_ckpt` của C, lấy 1 mẫu `(144,15)` đúng ngày trong tập kiểm thử 2016.
   - Dự báo `(144,)` theo °C gốc, vẽ nhanh so sánh thực tế/dự báo, in độ trễ đầu-cuối
     (mục tiêu `≤ 3 s` trên CPU).
   - Lưu vào `results/demo/<thời gian>/` kèm `index.html` rút gọn.
2. Chuẩn bị đầu vào mẫu nhẹ trong `data/processed/demo/` (không chép CSV 43 MB).
3. Chốt Document (song song, D chịu trách nhiệm tổng hợp):
   - `README.md` 10 mục: bổ sung khối Example Input → Output từ demo.
   - `docs/progress_status.md`: ghi module log Bước 4–7, decisions log, risks,
     cập nhật `Cập nhật lần cuối` và bảng `Nhật ký mô-đun`.
   - `docs/AI_Usage.md` 7 mục: ghi công cụ AI đã dùng cho từng bước.
   - `docs/CHANGELOG.md`: ghi phiên bản `jena_lstm_v1` (+ 2 mở rộng nếu có).
4. Chạy kiểm tra cuối: `pytest -q`, mở lại `results/eda/eda_report.html` và
   `results/runs/<ts>/report.html` để xác nhận số khớp nhau, kiểm tra `.gitignore`
   không để lọt tệp lớn/checkpoint lên git.

### 6.3. Nghiệm thu

- [ ] 1 lệnh demo chạy được trên máy khác, có hình + độ trễ live.
- [ ] `README`, `progress_status`, `AI_Usage`, `CHANGELOG` đã cập nhật.
- [ ] Tick đủ Definition of Done mỗi mô-đun (đúng yêu cầu, chạy đầu-cuối, kiểm thử đạt,
      không bí mật/tệp lớn, đã tự duyệt, giải thích được, tài liệu đủ, commit
      Conventional Commit).

## 7. Lịch và điểm đồng bộ

| Tuần | Người A | Người B | Người C | Người D |
|------|---------|---------|---------|---------|
| Tuần 3a | Chốt `forward` + `pytest` mô hình | Viết `trainer/scheduler`, chạy epoch đầu LSTM cơ sở | Viết khung `metrics.py` với persistence (dùng checkpoint giả) | Chuẩn bị khung `demo.py` + đầu vào mẫu |
| Tuần 3b | Hỗ trợ B gỡ lỗi shape `(B,144)` | Xuất `jena_lstm_v1.pt` rồi 2 mở rộng | Chạy `evaluate` thật khi có checkpoint | Chạy demo thử với checkpoint tạm |
| Tuần 4 | Review chéo + đo `≤ 100 ms` | Chốt `logs.txt` + 3 checkpoint | Chốt `metrics.json` + `report.html` | Chốt demo + README/docs + `pytest -q` |

Họp đồng bộ 3 mốc: (1) A xong chữ ký `forward`, (2) B xong checkpoint đầu,
(3) C xong `metrics.json` đầu. Mỗi mốc ghi 1 dòng vào `docs/progress_status.md`
mục nhật ký quyết định theo mẫu `[YYYY-MM-DD] Decision | Reasoning | Alternatives`.

## 8. Rủi ro liên mô-đun (từ `docs/design.md` mục 7)

| Rủi ro | Ai chịu chính | Giảm thiểu |
|--------|---------------|------------|
| Rò rỉ do chuẩn hóa/split sai | A + B | Không gọi `scaler.fit` ngoài huấn luyện; C kiểm tra lại bằng kiểm thử rò rỉ Bước 3 |
| Sai số tích lũy nửa sau ngày (`MAE@144` vọt) | A + B + C | Teacher forcing giảm dần + Attention; C báo cáo tại 1/24/72/144, không giấu |
| Huấn luyện chậm trên CPU (> 20 phút/epoch) | B | Giữ `hidden = 128, batch = 64`; đo epoch đầu; dùng GPU nếu có |
| Quá khớp | B + C | Dừng sớm patience 5, `dropout = 0,1`; C vẽ đường hàm mất mát huấn luyện/kiểm định |
| Hướng gió dạng góc gây nhiễu | A | Giữ `F = 15` (`wd_sin/wd_cos`); không tự đưa cột `wd` gốc trở lại |

## 9. Quy ước làm việc

- Ngôn ngữ: tiếng Việt có dấu; thuật ngữ giữ nguyên tiếng Anh
  (`DataLoader`, `checkpoint`, `teacher forcing`, `early stopping`).
- Commit: Conventional Commit, một mô-đun một commit
  (ví dụ `feat(models): thêm LSTMS2S và Attention Bahdanau`).
- Chạy/kiểm thử: `python -m src.data --config configs/data_config.yaml`,
  `pytest tests/unit/test_models.py -q`, `pytest -q` trước khi giao.
- Không commit CSV 43 MB, checkpoint `.pt`, `results/runs/`, `results/demo/`
  nếu `.gitignore` đã chặn; chỉ commit mã + `configs/*.yaml` + `docs/`.
