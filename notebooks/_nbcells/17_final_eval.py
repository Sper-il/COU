# === 17. Đánh giá cuối trên tập kiểm thử 2016 ===

final_metrics, preds_c, targets_c = evaluate_model(final_model, LOADERS["test"], SCALER)
print_scoreboard(final_metrics, f"KET QUA KIEM THU 2016 - {BEST_NAME}")

# Vẽ mẫu dự báo tốt nhất và xấu nhất theo MAE toàn tầm
per_sample_mae = np.mean(np.abs(preds_c - targets_c), axis=1)
best_i = int(np.argmin(per_sample_mae))
worst_i = int(np.argmax(per_sample_mae))

df_meta = LOADERS["_meta"]["df"]
starts = np.asarray(LOADERS["_meta"]["splits"]["test"])
date_best = df_meta["dt"].iloc[int(starts[best_i]) + 144].strftime("%Y-%m-%d %H:%M")
date_worst = df_meta["dt"].iloc[int(starts[worst_i]) + 144].strftime("%Y-%m-%d %H:%M")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, idx, title, d in (
    (axes[0], best_i, "Du bao tot nhat (MAE thap nhat)", date_best),
    (axes[1], worst_i, "Du bao xau nhat (MAE cao nhat)", date_worst),
):
    ax.plot(targets_c[idx], label="Thuc te", lw=1.6)
    ax.plot(preds_c[idx], label="Du bao", lw=1.4, ls="--")
    ax.fill_between(range(144), targets_c[idx], preds_c[idx], alpha=0.2, color="orange")
    ax.set_title(f"{title} | {d} | MAE = {per_sample_mae[idx]:.3f} °C")
    ax.set_xlabel("buoc (10 phut)")
    ax.set_ylabel("nhiet do (°C)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Lưu kết quả
Path("results").mkdir(exist_ok=True)
with open("results/final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2, default=float)
torch.save({"state_dict": final_model.state_dict(),
            "model_name": BEST_NAME, "arch": BEST_ARCH,
            "model_config": {"arch": BEST_ARCH, "arch_kwargs": FINAL_KWARGS}},
           "results/final_model.pt")
print("Đã lưu results/final_metrics.json và results/final_model.pt")
