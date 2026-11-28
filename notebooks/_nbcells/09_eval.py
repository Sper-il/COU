# === 8. Đánh giá một mô hình đã huấn luyện trên tập kiểm thử 2016 ===

from src.evaluation.metrics import compute_persistence_baseline, compute_persistence_horizons, compute_seasonal_naive, compute_seasonal_naive_horizons, inverse_transform_target


def evaluate_model(model, loader, scaler=None):
    """Dự báo toàn bộ một tập và trả về chỉ số đầy đủ (đơn vị °C gốc)."""
    model.eval()
    xs, ys, ps = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["input"].to(DEVICE), batch["target"].to(DEVICE)
            out = model(x, y_in=None)
            xs.append(x.cpu().numpy())
            ys.append(y.cpu().numpy())
            ps.append(out.cpu().numpy())

    inputs = np.concatenate(xs, 0)
    targets = np.concatenate(ys, 0)
    preds = np.concatenate(ps, 0)
    preds_c, targets_c = inverse_transform_target(preds, targets, scaler)

    m = {
        "mae": float(np.mean(np.abs(preds_c - targets_c))),
        "rmse": float(np.sqrt(np.mean((preds_c - targets_c) ** 2))),
    }
    for h in HORIZONS:
        err = preds_c[:, h - 1] - targets_c[:, h - 1]
        m[f"mae_h{h}"] = float(np.mean(np.abs(err)))
        m[f"rmse_h{h}"] = float(np.sqrt(np.mean(err ** 2)))

    m["persistence_mae"] = compute_persistence_baseline(inputs, targets_c, scaler)
    m.update(compute_persistence_horizons(inputs, targets_c, scaler))
    m["seasonal_naive_mae"] = compute_seasonal_naive(inputs, targets_c, scaler)
    m.update(compute_seasonal_naive_horizons(inputs, targets_c, scaler))
    m["out_of_bounds_count"] = int(np.sum((preds_c < -40) | (preds_c > 50)))
    m["n_samples"] = int(preds.shape[0])
    return m, preds_c, targets_c


def print_scoreboard(m, title=""):
    """In bảng điểm kèm đánh dấu đạt/không đạt theo ngưỡng nghiệm thu."""
    print("=" * 66)
    print(title)
    print("=" * 66)
    for key, label, thr in (
        ("mae", "MAE toan tang", THRESHOLDS["mae"]),
        ("rmse", "RMSE toan tang", THRESHOLDS["rmse"]),
        ("mae_h24", "MAE@24", THRESHOLDS["mae_h24"]),
        ("mae_h144", "MAE@144", THRESHOLDS["mae_h144"]),
    ):
        val = m[key]
        print(f"  {label:<16}{val:8.4f} °C  (<= {thr})  -> "
              f"{'DAT' if val <= thr else 'KHONG DAT'}")
    for key, label in (("persistence_mae", "Persistence"),
                       ("seasonal_naive_mae", "Seasonal-naive")):
        beat = m["mae"] < m[key]
        print(f"  {label:<16}{m[key]:8.4f} °C  -> "
              f"{'DAT (da vuot)' if beat else 'KHONG DAT'}")
    print(f"  MAE theo tam: " + " | ".join(
        f"@{h} = {m[f'mae_h{h}']:.3f}" for h in HORIZONS))
    print("=" * 66)
