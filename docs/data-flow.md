# Luồng dữ liệu — `jena-seq2seq`

> Dữ liệu đi từ CSV thô Jena đến `report.html`. Tần suất: theo lô ngoại tuyến (offline batch); không thời gian thực. Mọi shape dưới đây là chốt cho bản chạy tối thiểu (tầm 144 = 1 ngày).

## 1. Sơ đồ luồng kèm kích thước

```text
[1] Thô: data/raw/jena_climate_2009_2016.csv
    ~420.551 dòng × (14 cột số + Date Time), chu kỳ 10 phút, 2009-01-01 → 2017-01-01
        │ sắp xếp theo thời gian; thay -9999 (wv: 18, max_wv: 20 dòng) bằng NaN → nội suy tuyến tính
        v
[2] Đặc trưng: 14 → 15 (tách wd (deg) [0,360] thành wd_sin + wd_cos, bỏ wd gốc)
    F = 15: [p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max_wv, wd_sin, wd_cos]
        │ StandardScaler: CHỈ fit trên 2009–2014, transform cho mọi tập; target cũng chuẩn hóa lúc train
        v
[3] Chia theo năm (không xáo trộn liên tập, chống rò rỉ):
    train 2009–2014: 315.731 dòng (75,1%) | val 2015: 52.560 (12,5%) | test 2016: 52.259 (12,4%)
    Lưu ý: khoảng trống 3 ngày 25–28/10/2016 nằm trong test → loại cửa sổ cắt qua đoạn này
        │ cửa sổ trượt ngày-tới-ngày: X = 144 bước quá khứ, y = 144 bước tương lai của T (degC)
        v
[4] Cửa sổ 144→144:
    train stride = 12 (2 giờ) → ~35.022 mẫu | val/test stride = 144 (1 ngày, không chồng lấn) → ~2.919 mẫu toàn tập
    Một mẫu: X (144, 15) đã chuẩn hóa | y (144,) đã chuẩn hóa lúc train
        │ DataLoader batch B = 64, num_workers = 0 (CPU), seed = 42
        v
[5] Lô huấn luyện: (64, 144, 15) → (64, 144)
    Decoder lúc train nhận y dịch phải 1 bước với teacher forcing 0,5 giảm dần ×0,98/epoch
        │ checkpoint tốt nhất theo loss val → dự báo 144 bước (tự hồi tiếp lúc inference)
        v
[6] Đầu ra: véc-tơ (144,) đã chuẩn hóa → inverse-transform về °C → chặn biên vật lý (−40…50 °C để phát hiện lỗi)
        │ MAE/RMSE toàn tầm + tại 1/24/72/144, persistence (lặp giá trị cuối ngày hôm qua)
        v
[7] results/runs/<ts>/{metrics.json, plots/, report.html} + results/eda/ (sản phẩm EDA độc lập)
```

## 2. Quy ước thư mục dữ liệu

| Thư mục | Nội dung | Có commit không |
|---------|----------|-----------------|
| `data/raw/` | CSV gốc Jena, bất biến, không bao giờ sửa trực tiếp | Không (file 43 MB; giữ local theo `.gitignore`) |
| `data/processed/` | Dữ liệu đã làm sạch + scaler đã fit (`scaler.pkl`), đặc trưng F = 15 | Không (tái tạo từ raw + config) |
| `data/splits/` | Chỉ mục cửa sổ train/val/test (file `.npz` hoặc `.csv`), seed cố định | Không (tái tạo từ `split.py`) |
| `results/eda/` | `eda_summary.json` + 6 PNG + `eda_report.html` offline | Có (nhẹ, dưới 1 MB không tính ảnh thì commit JSON; ảnh/HTML giữ local hoặc LFS) |

## 3. Điểm kiểm soát chất lượng trên luồng

1. Sau bước [1]: số dòng, khoảng thời gian, tần suất mode 10 phút, mã thiếu `-9999` (xem `eda_summary.json`).
2. Sau bước [2]: `wd_sin² + wd_cos² ≈ 1`, không còn cột `wd` gốc, F = 15.
3. Sau bước [3]: không chồng lấn thời gian giữa train/val/test; scaler chỉ fit trên train (unit test bắt buộc).
4. Sau bước [4]: shape mẫu `(144,15)/(144,)`; loại cửa sổ chứa điểm nội suy dài (gap 3 ngày 10/2016).
5. Sau bước [6]: MAE persistence 2016 = 2,806 °C làm mốc sàn; Seq2Seq phải vượt mốc này.
