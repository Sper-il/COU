# CHANGELOG — `jena-seq2seq`

Mọi thay đổi đáng chú ý của dự án được ghi tại đây (mới nhất ở trên).

## [Chưa phát hành] — giai đoạn Model (2026-11-23)

### Thêm

- Module `src/models`:
  - `architecture.py` — 3 kiến trúc đúng đề: `LSTMS2S` (Encoder–Decoder LSTM cơ sở, **Must**), `LSTMS2SAttention` (+ Attention Bahdanau, **Should**), `MiniTransformer` (Transformer mini encoder-only, **Should**), kèm `build_model` / `count_parameters` / `estimated_checkpoint_mb`.
  - `loss.py` — `mse_loss`, `mae_loss`, `masked_mse_loss`.
  - `smoke_test.py` — kiểm thử nhanh **trên dữ liệu Jena thật**, 1 lệnh: shape, MAE (°C gốc) so với persistence/seasonal-naive, gradient + hàm mất mát, biên vật lý, độ trễ suy luận.
- Kiểm thử `tests/unit/test_models.py`: 13 kiểm thử đạt (shape 3 kiến trúc, tự hồi tiếp đủ 144 bước, đường teacher forcing, `mse_loss` đúng công thức, checkpoint ≤ 50 MB, kiểm tra shape đầu vào, `build_model` lọc kwargs, cổng trộn hai neo, độ trễ ≤ 100 ms).
- `src/models/__init__.py` xuất toàn bộ kiến trúc và hàm mất mát.
- Bổ sung đường cơ sở **seasonal-naive** (`y[t] = x[t]`) vào `src/evaluation/metrics.py` (`compute_seasonal_naive_horizons`), `plots.py` và `visualize_html.py`.

### Sửa

- **Lỗi teacher forcing lệch 1 bước (off-by-one)** trong `architecture.py`: chuỗi teacher input cũ dài `T` và được tra đúng chỗ `t`, khiến bộ giải mã ăn `y[:, t-2]` thay vì `y[:, t-1]`. Sửa thành chuỗi dài `T+1` (`teacher_input[:, k] = y[:, k-1]`) và tra chỉ số `t+1`. Ảnh hưởng đo được: hàm mất mát kiểm định sau 1 epoch giảm từ **0,26253 → 0,16659**.
- **Tối ưu Attention Bahdanau**: chiếu `W_e · enc_out` **một lần** thay vì tính lại ở mỗi bước giải mã (`enc_out` là hằng trong suốt 144 bước). Độ trễ suy luận giảm từ **108,9 ms → 90,5 ms**, đạt ngưỡng nghiệm thu ≤ 100 ms.
- **Thống nhất dạng đầu ra cho cả 3 kiến trúc**: `MiniTransformer` nay cũng dùng dạng *neo cùng giờ + phần dư* như hai mô hình LSTM, để phép so sánh "compare against Transformer" trong đề là công bằng.
- Kiểm tra shape đầu vào rõ ràng: sai số đặc trưng, sai số bước mục tiêu, lệch batch đều ném `ValueError` kèm thông điệp rõ ràng thay vì lỗi lặp vụng.
- `build_model` lọc tham số theo từng kiến trúc (`ARCH_PARAMS`) thay vì dùng một tập chung.
- `src/evaluation/run_eval.py`: không còn gọi `next(iter(loader))` ba lần (mỗi lần dựng lại iterator) khi đo độ trễ suy luận.
- `tests/unit/test_metrics.py`: khẳng định tiêu chí `status` theo tập con thay vì so sánh bằng phép bằng (thêm `beats_seasonal_naive`).

### Kết quả chạy (2026-11-23)

- `python -m src.models.smoke_test` → cả 3 kiến trúc OK trên dữ liệu Jena thật: shape `(8,144,15) → (8,144)`, loss MSE giảm, 0 giá trị ngoài biên vật lý.
- Tham số / checkpoint / độ trễ suy luận (CPU): LSTM 471.825 tham số, ~1,80 MB, ~73 ms/mẫu | Attention 504.849, ~1,93 MB, ~90 ms/mẫu | Transformer 105.489, ~0,90 MB, ~3 ms/mẫu.
- `pytest -q` → **25 passed**.
- Chạy thử 1 epoch LSTM (`smoke_lstm` → `model_check`): val loss 0,16659; đánh giá tập 2016 MAE 2,4923 °C (đã vượt persistence 2,8150 °C), RMSE 3,3755 °C, độ trễ 72,3 ms. Báo cáo `report.html` sinh offline đầy đủ.

## [Chưa phát hành] — giai đoạn Data (2026-11-08, kiểm tra lại 2026-11-09)

### Thêm

- Module `src/data`: `transforms.py` (mã thiếu, nội suy, gió sin/cos, `F = 15`), `split.py` (chia theo năm, loại cửa sổ qua gap), `dataset.py` (`JenaWindowDataset`, `build_dataloaders`, `get_scaler`, lưu `scaler.pkl` + `splits.npz`), `__main__.py` kiểm tra 1 lô.
- Kiểm thử `tests/unit/test_dataset.py`: 4 kiểm thử đạt (shape lô, scaler chỉ fit trên train, không chồng lấn thời gian, gió nằm trên vòng tròn đơn vị).
- Kết quả chạy: train 26.192 mẫu (410 lô), val 364 mẫu (6 lô), test 359 mẫu (6 lô), shape `(64,144,15) → (64,144)`; kiểm tra lại ngày 2026-11-09 cho cùng kết quả, `splits.npz` ghi nhận `{'train': 26192, 'val': 364, 'test': 359}`.

## [Chưa phát hành] — giai đoạn EDA

### Thêm

- Script EDA `notebooks/2026-11-03_eda.py`: thống kê 420.551 dòng Jena, 6 hình PNG, `results/eda/eda_summary.json`.
- Báo cáo EDA offline `results/eda/eda_report.html` (~638 KB, CSS nội tuyến, ảnh base64, không CDN), render bằng `src/evaluation/visualize_eda_html.py`.
- Khung dự án: `requirements.txt`, `.gitignore`, `configs/{data_config.yaml,model_config.yaml,train_config.yaml}`, thư mục `src/{data,models,training,evaluation}`, `data/{processed,splits}`, `models/{checkpoints,trained}`, `results/{eda,runs}`.
- Bộ docs: `README.md`, `docs/architecture.md`, `docs/data-flow.md`, `docs/api.md`, `docs/data.md`, `docs/AI_Usage.md` (tệp này khởi tạo CHANGELOG).

### Quyết định

- Chốt `F = 15`: tách `wd (deg)` thành `wd_sin/wd_cos` (biến vòng tròn đứt đoạn 0/360).
- Mã thiếu `-9999` ở `wv`/`max. wv` (38 dòng): nội suy tuyến tính, không xóa dòng.
- Ngưỡng persistence 2016: MAE toàn tầm 144 = 2,806 °C (MAE@1/@24/@72/@144 = 0,12/0,94/1,77/2,81) — Seq2Seq phải vượt.

## [2026-11-03] — Định nghĩa và thiết kế

### Thêm

- `docs/description.md`, `docs/define.md`, `docs/design.md`, `docs/progress_status.md` — khớp đủ 9 dòng đề (Regression, Jena, Weather Forecasting, Seq2Seq, upcoming days 144 bước, Encoder–Decoder LSTM, multi-step, attention, Transformer).
- Chốt cấu hình bản chạy tối thiểu: `input_len = 144`, `pred_len = 144`, `stride_train = 12`, `stride_eval = 144`, chia theo năm 2009–2014/2015/2016, batch 64, Adam lr 1e-3, teacher forcing 0,5 giảm dần.
