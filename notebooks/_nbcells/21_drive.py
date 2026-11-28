# === 21. Lưu kết quả lên Google Drive (không mất khi phiên Colab kết thúc) ===

# Mount Drive để lưu kết quả lâu dài. Nếu không muốn dùng Drive, bỏ qua ô này.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/jena-seq2seq")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("sweep_results.csv", "compare_architectures.csv",
                 "latency.csv", "final_metrics.json", "final_model.pt"):
        src = Path("results") / name
        if src.exists():
            import shutil
            shutil.copy2(src, DRIVE_DIR / name)
    print("Đã lưu kết quả lên Drive:", DRIVE_DIR)
    for p in sorted(DRIVE_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
except Exception as exc:  # noqa: BLE001
    print("Không mount được Drive (bỏ qua ô này):", exc)

print("\nKết quả vẫn nằm trong thư mục results/ của phiên Colab hiện tại.")
