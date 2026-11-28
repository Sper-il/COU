# === 20. Tổng kết và kiểm tra nghiệm thu ===

print("=" * 74)
print("TONG KET KET QUA")
print("=" * 74)

print("\n[1] Cau hinh tot nhat (sweep tren tap kiem dinh 2015):")
print(f"    {BEST_NAME} | val MAE = {BEST_ROW['val_mae']:.4f} °C | "
      f"best epoch {BEST_ROW['best_epoch']}")

print("\n[2] Ket qua kiem thu 2016 cua luan chay chinh:")
for k, label, thr in (("mae", "MAE toan tang", THRESHOLDS["mae"]),
                      ("rmse", "RMSE toan tang", THRESHOLDS["rmse"]),
                      ("mae_h24", "MAE@24", THRESHOLDS["mae_h24"]),
                      ("mae_h144", "MAE@144", THRESHOLDS["mae_h144"])):
    v = final_metrics[k]
    print(f"    {label:<16}{v:8.4f} °C  (≤ {thr})  -> {'ĐẠT' if v <= thr else 'KHÔNG ĐẠT'}")
for k, label in (("persistence_mae", "Persistence"),
                 ("seasonal_naive_mae", "Seasonal-naive")):
    beat = final_metrics["mae"] < final_metrics[k]
    print(f"    {label:<16}{final_metrics[k]:8.4f} °C  -> "
          f"{'ĐẠT (đã vượt)' if beat else 'KHÔNG ĐẠT'}")
print(f"    MAE theo tầm: " + " | ".join(
    f"@{h} = {final_metrics[f'mae_h{h}']:.3f} °C" for h in HORIZONS))
print(f"    Giá trị ngoài biên vật lý: {final_metrics['out_of_bounds_count']}")

print("\n[3] So sanh 3 kien truc (tap kiem thu 2016):")
display(COMPARE_DF.round(4))

print("\n[4] Do tre suy luan:")
display(latency_df)

print("\n[5] File ket qua da luu:")
for p in ("results/sweep_results.csv", "results/compare_architectures.csv",
          "results/latency.csv", "results/final_metrics.json", "results/final_model.pt"):
    print(f"    {p}")

print("\n[6] So sanh voi moc EDA:")
print(f"    persistence EDA 2016   : {PERSISTENCE_2016:.4f} °C")
print(f"    seasonal-naive EDA 2016: {SEASONAL_2016:.4f} °C")
print(f"    mo hinh cua ban        : {final_metrics['mae']:.4f} °C")
