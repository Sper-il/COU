# === 16. Quan sát chi tiết quá trình train của lần chạy chính ===

plot_training_curves(final_res["history"], title=f"Quan sat luan chay chinh - {BEST_NAME}")

# Bảng số liệu từng epoch
final_df = pd.DataFrame(final_res["history"])
print("Bang chi tiet tung epoch:")
display(final_df[["epoch", "train_loss", "val_loss", "train_mae", "val_mae",
                  "teacher_forcing", "lr", "train_time_s", "val_time_s"]].round(5))

# Nhận xét tự động dựa trên đường cong
best_h = min(final_res["history"], key=lambda h: h["val_loss"])
last_h = final_res["history"][-1]
train_gap_last = last_h["train_loss"] - last_h["val_loss"]
val_mae_best = best_h["val_mae"]

print("\n" + "=" * 70)
print("DIEN GIAI QUA TRINH TRAIN")
print("=" * 70)
print(f"Epoch tot nhat         : {best_h['epoch']} (val loss {best_h['val_loss']:.5f})")
print(f"Val loss cuoi cung     : {last_h['val_loss']:.5f} sau {len(final_res['history'])} epoch")
print(f"Val MAE tot nhat       : {val_mae_best:.5f} (don vi chuan hoa)")
if train_gap_last < -0.005:
    print("-> Dau hieu QUA KHOP: val loss thap hon train loss, mo hinh dang chua hoa.")
elif train_gap_last > 0.02:
    print("-> Dau hieu QUA KHOP: train loss cao hon val loss, can chinh regularization.")
else:
    print("-> Train va val khop nhau, chua thay hien qua khop ro net.")
if last_h["teacher_forcing"] <= 0.01:
    print("-> Teacher forcing da ve 0: mo hinh da duoc huấn luyện o dung che do tu hoi tiep.")
else:
    print(f"-> Teacher forcing con {last_h['teacher_forcing']:.3f}, chua ve 0 hoan toan.")
if len(final_res["history"]) < FINAL_CFG["epochs"]:
    print(f"-> Da dung som sau {len(final_res['history'])} epoch (val loss khong cai thien).")
else:
    print("-> Da chay het so epoch cho phep.")
