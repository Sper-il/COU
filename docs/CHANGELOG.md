# CHANGELOG — `jena-seq2seq`

Mọi thay đổi đáng chú ý của dự án được ghi tại đây (mới nhất ở trên).

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
