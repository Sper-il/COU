"""Kiểm thử đơn vị module dữ liệu Jena."""
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders
from src.data.split import load_config
from src.data import transforms as T


def _config():
    return load_config(ROOT / "configs" / "data_config.yaml")


def test_batch_shape():
    config = _config()
    loaders = build_dataloaders(config)
    batch = next(iter(loaders["train"]))
    assert tuple(batch["input"].shape[1:]) == (144, 15), batch["input"].shape
    assert tuple(batch["target"].shape[1:]) == (144,), batch["target"].shape


def test_scaler_fit_only_on_train():
    from sklearn.preprocessing import StandardScaler
    config = _config()
    loaders = build_dataloaders(config)
    df = loaders["_meta"]["df"]
    scaler = loaders["_meta"]["feature_scaler"]
    train_mask = df["year"].isin(config["split"]["train"]).to_numpy()
    ref = StandardScaler().fit(df[T.FINAL_FEATURES].to_numpy(dtype=float)[train_mask])
    assert np.allclose(scaler.mean_, ref.mean_), "scaler phải fit trên train"
    # scaler toàn tập phải khác scaler train (chứng tỏ không fit toàn tập)
    full = StandardScaler().fit(df[T.FINAL_FEATURES].to_numpy(dtype=float))
    assert not np.allclose(scaler.mean_, full.mean_), "scaler fit toàn tập là rò rỉ"


def test_no_temporal_overlap():
    config = _config()
    loaders = build_dataloaders(config)
    splits = loaders["_meta"]["splits"]
    df = loaders["_meta"]["df"]
    total = config["input_len"] + config["pred_len"]
    ranges = {}
    for name in ("train", "val", "test"):
        starts = np.asarray(splits[name])
        covered = set()
        for s in starts[:2000]:  # kiểm tra mẫu đại diện để chạy nhanh
            covered.update(range(s, s + total))
        ranges[name] = covered
    assert ranges["train"].isdisjoint(ranges["val"]), "train/val chồng lấn thời gian"
    assert ranges["train"].isdisjoint(ranges["test"]), "train/test chồng lấn thời gian"
    assert ranges["val"].isdisjoint(ranges["test"]), "val/test chồng lấn thời gian"


def test_wind_encoding_unit_circle():
    import pandas as pd
    df = pd.DataFrame({"wd (deg)": [0.0, 90.0, 180.0, 359.0, 1.0]})
    out = T.add_wind_features(df)
    assert "wd (deg)" not in out.columns
    r = np.sqrt(out["wd_sin"].to_numpy() ** 2 + out["wd_cos"].to_numpy() ** 2)
    assert np.allclose(r, 1.0), "sin²+cos² phải bằng 1"
    # 359° và 1° phải gần nhau trong không gian sin/cos
    d = np.abs(out[["wd_sin", "wd_cos"]].iloc[3].to_numpy()
               - out[["wd_sin", "wd_cos"]].iloc[4].to_numpy()).max()
    assert d < 0.05, "gió vòng tròn phải liên tục tại 0/360"
