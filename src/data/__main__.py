"""Chạy kiểm tra nhanh module dữ liệu: in shape 1 lô."""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.split import load_config
from src.data.dataset import build_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    loaders = build_dataloaders(config)
    for name in ("train", "val", "test"):
        batch = next(iter(loaders[name]))
        print(f"{name}: input {tuple(batch['input'].shape)} target {tuple(batch['target'].shape)} "
              f"số lô: {len(loaders[name])} số mẫu: {len(loaders[name].dataset)}")
    print("Đặc trưng F =", len(loaders["_meta"]["feature_scaler"].mean_))
    print("Scaler đã lưu: data/processed/scaler.pkl | splits: data/splits/splits.npz")


if __name__ == "__main__":
    main()
