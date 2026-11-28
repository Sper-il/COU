# === 6. Đường cơ sở: persistence và seasonal-naive (mốc sàn để mô hình vượt) ===

from src.evaluation.metrics import compute_persistence_baseline, compute_seasonal_naive, inverse_transform_target


def collect(loader):
    """ Gom (input, target, pred) của một DataLoader về numpy. """
    xs, ys = [], []
    for batch in loader:
        x, y = batch["input"], batch["target"]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def baseline_report(loader, name):
    """Tính MAE của persistence và seasonal-naive trên một tập (đơn vị °C gốc)."""
    inputs, targets = collect(loader)
    _, targets_c = inverse_transform_target(targets, targets, SCALER)
    pers = compute_persistence_baseline(inputs, targets_c, SCALER)
    seas = compute_seasonal_naive(inputs, targets_c, SCALER)
    print(f"[{name}] persistence = {pers:.4f} °C | seasonal-naive = {seas:.4f} °C")
    return {"persistence_mae": pers, "seasonal_naive_mae": seas}


BASE_TEST = baseline_report(LOADERS["test"], "test 2016")
BASE_VAL = baseline_report(LOADERS["val"], "val 2015")
print("Mục tiêu: mô hình phải vượt cả hai đường cơ sở trên tập kiểm thử 2016.")
