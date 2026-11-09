"""Chia tập theo năm cho Jena, chống rò rỉ thời gian."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GAP_THRESHOLD_MIN = 15  # khoảng cách lớn hơn ngưỡng coi là mất sóng


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(config: dict) -> pd.DataFrame:
    from src.data.transforms import add_wind_features, interpolate_numeric, replace_sentinel

    root = Path(config.get("_root", "."))
    csv = root / config["data_path"]
    df = pd.read_csv(csv)
    df["dt"] = pd.to_datetime(df[config["datetime_col"]], format=config["datetime_format"])
    df = df.sort_values("dt").reset_index(drop=True)
    sentinel_cols = ["wv (m/s)", "max. wv (m/s)"]
    df = replace_sentinel(df, sentinel_cols, config.get("sentinel_value", -9999))
    num_cols = config["feature_cols_original"]
    df = interpolate_numeric(df, num_cols)
    df = add_wind_features(df, config.get("wind_col", "wd (deg)"))
    df["year"] = df["dt"].dt.year
    return df


def bad_gap_mask(dt: pd.Series, threshold_min: float = GAP_THRESHOLD_MIN) -> np.ndarray:
    """Mảng bool độ dài n: True tại vị trí i nếu khoảng dt[i]-dt[i-1] vượt ngưỡng."""
    diff_min = dt.diff().dt.total_seconds().div(60).to_numpy()
    mask = np.zeros(len(dt), dtype=bool)
    mask[1:] = diff_min[1:] > threshold_min
    return mask


def window_has_gap(start: int, length: int, bad: np.ndarray) -> bool:
    """Cửa sổ [start, start+length) có chứa điểm mất sóng không."""
    return bool(bad[start + 1:start + length].any())


def make_splits(config: dict, df: pd.DataFrame | None = None) -> dict:
    """Sinh chỉ mục bắt đầu cửa sổ cho từng tập, loại cửa sổ cắt qua gap dài."""
    from src.data.transforms import FINAL_FEATURES

    if df is None:
        df = load_raw(config)
    input_len = config["input_len"]
    pred_len = config["pred_len"]
    total = input_len + pred_len
    bad = bad_gap_mask(df["dt"])
    years = df["year"].to_numpy()
    out: dict = {}
    for split_name, key in (("train", "train"), ("val", "val"), ("test", "test")):
        split_years = set(config["split"][key])
        rows = np.where(np.isin(years, list(split_years)))[0]
        if len(rows) == 0:
            out[split_name] = np.array([], dtype=np.int64)
            continue
        r0, r1 = int(rows[0]), int(rows[-1])
        stride = config["stride_train"] if split_name == "train" else config["stride_eval"]
        starts = []
        s = r0
        while s + total <= r1 + 1:
            # cửa sổ phải nằm gọn trong tập (không lấn năm khác)
            if years[s + total - 1] not in split_years:
                s += stride
                continue
            if not window_has_gap(s, total, bad):
                starts.append(s)
            s += stride
        out[split_name] = np.array(starts, dtype=np.int64)
    out["n_rows"] = len(df)
    out["feature_names"] = FINAL_FEATURES
    return out
