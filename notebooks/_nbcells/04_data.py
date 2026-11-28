# === 3. Tải dữ liệu Jena ===

import subprocess
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/raw/jena_climate_2009_2016.csv")
CSV_PATH = DATA_DIR / "jena_climate_2009_2016.csv"

if CSV_PATH.exists():
    print("Đã có CSV trong repo:", CSV_PATH, CSV_PATH.stat().st_size // (1024 * 1024), "MB")
else:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Bộ Jena Climate gốc của đề bài (kaggle: punnenga/jena-climate)
    url = "https://archive.org/download/jena-climate/jena_climate_2009_2016.csv"
    print("Đang tải từ", url, "…")
    urllib.request.urlretrieve(url, CSV_PATH)
    print("Tải xong")

assert CSV_PATH.exists(), f"Không tìm thấy {CSV_PATH}"
print("Dung lượng:", CSV_PATH.stat().st_size // (1024 * 1024), "MB")
