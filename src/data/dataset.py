"""Dataset và DataLoader Jena: cửa sổ 144→144, scaler chỉ fit trên train."""
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import joblib

from src.data.split import load_raw, make_splits
from src.data.transforms import FINAL_FEATURES, TARGET_COL

ROOT = Path(__file__).resolve().parents[2]


class JenaWindowDataset(Dataset):
    """Mỗi mẫu trả về dict {'input': (144,15), 'target': (144,)} đã chuẩn hóa."""

    def __init__(self, feats_scaled: np.ndarray, target_scaled: np.ndarray,
                 starts: np.ndarray, input_len: int, pred_len: int):
        self.feats = feats_scaled.astype(np.float32)
        self.tgt = target_scaled.astype(np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        s = int(self.starts[idx])
        x = self.feats[s:s + self.input_len]
        y = self.tgt[s + self.input_len:s + self.input_len + self.pred_len]
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(y)}


def _fit_scalers(df, config: dict):
    train_years = set(config["split"]["train"])
    train_mask = df["year"].isin(train_years).to_numpy()
    feat_mat = df[FINAL_FEATURES].to_numpy(dtype=float)
    tgt_vec = df[[TARGET_COL]].to_numpy(dtype=float)
    feature_scaler = StandardScaler().fit(feat_mat[train_mask])
    target_scaler = StandardScaler().fit(tgt_vec[train_mask])
    return feature_scaler, target_scaler


def build_dataloaders(config: dict) -> dict:
    """Dựng 3 DataLoader train/val/test. Lưu scaler vào data/processed/scaler.pkl."""
    config = dict(config)
    config["_root"] = ROOT
    df = load_raw(config)
    splits = make_splits(config, df)
    feature_scaler, target_scaler = _fit_scalers(df, config)

    feat_mat = df[FINAL_FEATURES].to_numpy(dtype=float)
    tgt_vec = df[[TARGET_COL]].to_numpy(dtype=float)
    feats_scaled = feature_scaler.transform(feat_mat)
    target_scaled = target_scaler.transform(tgt_vec)[:, 0]

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"feature_scaler": feature_scaler, "target_scaler": target_scaler,
         "feature_names": FINAL_FEATURES, "target_col": TARGET_COL},
        proc / "scaler.pkl",
    )
    # lưu chỉ mục splits để tái lập
    sp_dir = ROOT / "data" / "splits"
    sp_dir.mkdir(parents=True, exist_ok=True)
    np.savez(sp_dir / "splits.npz",
             train=splits["train"], val=splits["val"], test=splits["test"])

    out = {}
    batch = int(config.get("batch_size", 64))
    for name in ("train", "val", "test"):
        ds = JenaWindowDataset(feats_scaled, target_scaled, splits[name],
                               int(config["input_len"]), int(config["pred_len"]))
        shuffle = (name == "train")
        g = torch.Generator()
        g.manual_seed(int(config.get("seed", 42)))
        out[name] = DataLoader(ds, batch_size=batch, shuffle=shuffle,
                               num_workers=int(config.get("num_workers", 0)),
                               generator=g if shuffle else None)
    # đính kèm siêu dữ liệu để kiểm thử dùng lại
    out["_meta"] = {"df": df, "splits": splits,
                    "feature_scaler": feature_scaler, "target_scaler": target_scaler}
    return out


def get_scaler(path: str | Path | None = None):
    """Nạp scaler đã fit trên train để inverse-transform khi đánh giá."""
    p = Path(path) if path else (ROOT / "data" / "processed" / "scaler.pkl")
    return joblib.load(p)
