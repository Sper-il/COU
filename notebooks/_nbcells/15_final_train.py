# === 15. Huấn luyện chính 30 epoch với cấu hình tốt nhất ===

# Lấy cấu hình thắng trong sweep, nhưng nâng `epochs` lên 30 và giữ nguyên
# mọi siêu tham số khác. `early_stop` bật lại để dừng sớm khi bão hoà.
best_spec = next(s for s in SWEEP if s["name"] == BEST_NAME)
BEST_ARCH = best_spec["arch"]

FINAL_CFG = make_cfg(best_spec, epochs=30, early_stop=5, decay_epochs=10)
FINAL_KWARGS = make_kwargs(BEST_ARCH, FINAL_CFG)

print("=" * 70)
print(f"CHAY CHINH: {BEST_NAME} | arch={BEST_ARCH} | 30 epoch | {DEVICE}")
print("=" * 70)
for k, v in FINAL_CFG.items():
    if k != "scheduler":
        print(f"  {k} = {v}")
print(f"  kwargs mo hinh = {FINAL_KWARGS}")

set_seed(FINAL_CFG["seed"])
final_model = build_model(BEST_ARCH, **FINAL_KWARGS)
print(f"Số tham số: {count_parameters(final_model):,}")

final_res = train_loop(final_model, LOADERS, FINAL_CFG, run_name=BEST_NAME, verbose=True)
final_model = final_res["model"]
print(f"Epoch tốt nhất: {final_res['best_epoch']} | val loss {final_res['best_val_loss']:.5f}")
