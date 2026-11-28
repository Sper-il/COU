# === 18. So sánh 3 kiến trúc trên cùng tập kiểm thử (bảng bắt buộc của đề) ===

# Để so sánh công bằng, huấn luyện mỗi kiến trúc với cùng số epoch và cùng
# cấu hình siêu tham số tốt nhất tìm được ở bước sweep.
COMPARE_ARCHS = ["lstm_s2s", "lstm_attention", "mini_transformer"]
COMPARE_EPOCHS = 15
compare_results = []
compare_models = {}

for arch in COMPARE_ARCHS:
    # Cùng bộ siêu tham số với cấu hình thắng, nhưng tham số riêng của kiến
    # trúc (hidden_dim / d_model) lấy theo mặc định của kiến trúc đó.
    cfg = make_cfg(best_spec, epochs=COMPARE_EPOCHS, early_stop=COMPARE_EPOCHS,
                   decay_epochs=10)
    kwargs = make_kwargs(arch, cfg)

    print("=" * 70)
    print(f"SO SANH: {arch} | {COMPARE_EPOCHS} epoch")
    print("=" * 70)
    set_seed(cfg["seed"])
    m = build_model(arch, **kwargs)
    res = train_loop(m, LOADERS, cfg, run_name=f"cmp_{arch}", verbose=True)
    compare_models[arch] = res["model"]

    metrics, _, _ = evaluate_model(res["model"], LOADERS["test"], SCALER)
    compare_results.append({
        "arch": arch,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mae_h1": metrics["mae_h1"],
        "mae_h24": metrics["mae_h24"],
        "mae_h72": metrics["mae_h72"],
        "mae_h144": metrics["mae_h144"],
        "params": count_parameters(res["model"]),
    })

COMPARE_DF = pd.DataFrame(compare_results)
print("\n" + "=" * 70)
print("BANG SO SANH 3 KIEN TRUC (tap kiem thu 2016, tam 144)")
print("=" * 70)
display(COMPARE_DF.round(4))
COMPARE_DF.to_csv("results/compare_architectures.csv", index=False, encoding="utf-8")
print("Đã lưu results/compare_architectures.csv")

# Biểu đồ so sánh
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
x = np.arange(len(COMPARE_DF))
ax.bar(x - 0.2, COMPARE_DF["mae"], 0.4, label="MAE", color="steelblue")
ax.bar(x + 0.2, COMPARE_DF["rmse"], 0.4, label="RMSE", color="indianred")
ax.axhline(THRESHOLDS["mae"], ls="--", color="navy", label="nguong MAE 1.5")
ax.set_xticks(x)
ax.set_xticklabels(COMPARE_DF["arch"], rotation=20, ha="right")
ax.set_ylabel("°C")
ax.set_title("MAE / RMSE theo kien truc")
ax.legend()
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
for _, r in COMPARE_DF.iterrows():
    ax.plot(HORIZONS, [r[f"mae_h{h}"] for h in HORIZONS], "o-", label=r["arch"], ms=5)
ax.set_xlabel("buoc du bao (10 phut)")
ax.set_ylabel("MAE (°C)")
ax.set_title("Sai so theo tam du bao")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
