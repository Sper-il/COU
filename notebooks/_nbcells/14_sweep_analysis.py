# === 14. Phân tích kết quả sweep: cấu hình nào tốt nhất? ===

plot_config_comparison(SWEEP_RESULTS)
plot_learning_curves_grid(SWEEP_HISTORIES)

BEST_NAME = SWEEP_DF.iloc[0]["name"]
BEST_ROW = SWEEP_DF.iloc[0]
print(f"Cau hinh tot nhat theo val MAE: {BEST_NAME}")
print(SWEEP_DF.iloc[0].to_string())

# Phân tích thêm: yếu tố nào ảnh hưởng mạnh nhất
print("\n" + "=" * 70)
print("PHAN TICH YEU TO")
print("=" * 70)

for arch in SWEEP_DF["arch"].unique():
    sub = SWEEP_DF[SWEEP_DF["arch"] == arch]
    if len(sub) > 1:
        spread = sub["val_mae"].max() - sub["val_mae"].min()
        print(f"{arch:<18} best {sub['val_mae'].min():.4f} | worst {sub['val_mae'].max():.4f} "
              f"| chenh lech {spread:.4f} °C")

lr_sub = SWEEP_DF[SWEEP_DF["arch"] == "lstm_s2s"]
if len(lr_sub) > 1:
    print(f"\nTac dong cua learning rate (LSTM S2S):")
    for _, r in lr_sub.iterrows():
        print(f"  lr = {r['lr']:.0e} -> val MAE {r['val_mae']:.4f} °C")

# So sanh 3 kien truc o cung cau hinh goc
print("\nSo sanh 3 kien truc (cau hinh goc):")
base = SWEEP_DF[SWEEP_DF["name"].isin(
    ["lstm_s2s_goc", "attn_bahdanau_goc", "transformer_goc"])]
if len(base) == 3:
    print(base[["name", "val_mae", "val_rmse", "params", "train_time_s"]].to_string(index=False))
