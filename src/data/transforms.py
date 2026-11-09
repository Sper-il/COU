"""Các phép biến đổi dữ liệu Jena: mã thiếu, gió sin/cos, thứ tự đặc trưng."""
import numpy as np
import pandas as pd

FINAL_FEATURES = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd_sin",
    "wd_cos",
]

TARGET_COL = "T (degC)"


def replace_sentinel(df: pd.DataFrame, cols: list, sentinel: float = -9999) -> pd.DataFrame:
    """Thay mã thiếu quy ước bằng NaN để nội suy ở bước sau."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(sentinel, np.nan)
    return df


def interpolate_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Nội suy tuyến tính 2 chiều cho các cột số."""
    df = df.copy()
    df[cols] = df[cols].interpolate(limit_direction="both")
    return df


def add_wind_features(df: pd.DataFrame, wind_col: str = "wd (deg)") -> pd.DataFrame:
    """Tách hướng gió dạng vòng tròn thành wd_sin/wd_cos, bỏ cột gốc."""
    df = df.copy()
    theta = np.deg2rad(df[wind_col].to_numpy(dtype=float))
    df["wd_sin"] = np.sin(theta)
    df["wd_cos"] = np.cos(theta)
    df = df.drop(columns=[wind_col])
    return df


def build_train_transforms():
    """Mô tả các bước áp dụng cho tập huấn luyện (dùng khi dựng tài liệu)."""
    return ["replace_sentinel", "interpolate", "add_wind_features", "standard_scale_fit"]


def build_val_transforms():
    """Mô tả các bước áp dụng cho kiểm định/kiểm thử (không fit lại scaler)."""
    return ["replace_sentinel", "interpolate", "add_wind_features", "standard_scale_transform"]
