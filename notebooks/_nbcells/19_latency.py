# === 19. Đo độ trễ suy luận (ngưỡng ≤ 100 ms/mẫu) ===

def measure_latency(model, sample_x, repeats=30):
    """Đo trung bình thời gian suy luận 1 mẫu/ngày (ms)."""
    model.eval()
    x = sample_x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(x, y_in=None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x, y_in=None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / repeats * 1000


sample_x = next(iter(LOADERS["test"]))["input"][0]
latency_rows = []
for arch, mdl in compare_models.items():
    ms = measure_latency(mdl, sample_x)
    latency_rows.append({"arch": arch, "latency_ms": round(ms, 2),
                         "dat": ms <= 100})
    print(f"{arch:<18} {ms:7.2f} ms/mẫu  -> {'ĐẠT' if ms <= 100 else 'KHÔNG ĐẠT'} (≤ 100)")

latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv("results/latency.csv", index=False, encoding="utf-8")
print("\nĐã lưu results/latency.csv")
