# === 5. Dựng DataLoader và kiểm tra chia tập ===

t0 = time.time()
LOADERS = load_data()
SCALER = get_scaler()
print(f"Dựng DataLoader xong sau {time.time() - t0:.1f}s")

for name in ("train", "val", "test"):
    ds = LOADERS[name].dataset
    print(f"{name:>5}: {len(ds):>6} mẫu | {len(LOADERS[name]):>4} lô")

sample = next(iter(LOADERS["train"]))
print("shape một lô:", sample["input"].shape, "->", sample["target"].shape)
print("F = 15 (14 biến gốc + wd_sin + wd_cos)")
