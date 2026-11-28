# === 10. Bộ công cụ vẽ để quan sát quá trình huấn luyện ===


def plot_training_curves(history, title="Quan sat qua trinh huan luyen", save=None):
    """Vẽ 6 biểu đồ theo dõi: loss, MAE, teacher forcing, lr, thời gian, gap."""
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. hàm mất mát train vs val
    ax = axes[0, 0]
    ax.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss", ms=3)
    ax.plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (tu hoi tiep)", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (chu dan hoa)")
    ax.set_title("Ham mat tat train vs val")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. MAE train vs val
    ax = axes[0, 1]
    ax.plot(ep, [h["train_mae"] for h in history], "o-", label="train MAE", ms=3)
    ax.plot(ep, [h["val_mae"] for h in history], "s-", label="val MAE", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (chu dan hoa)")
    ax.set_title("MAE train vs val")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. tỉ lệ teacher forcing
    ax = axes[0, 2]
    ax.plot(ep, [h["teacher_forcing"] for h in history], "o-", color="darkorange", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ty le teacher forcing")
    ax.set_title("Lich giam dan teacher forcing ve 0")
    ax.grid(alpha=0.3)

    # 4. tốc độ học
    ax = axes[1, 0]
    ax.semilogy(ep, [h["lr"] for h in history], "o-", color="purple", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("ReduceLROnPlateau")
    ax.grid(alpha=0.3)

    # 5. thời gian mỗi epoch
    ax = axes[1, 1]
    ax.bar(ep, [h["train_time_s"] for h in history], label="train", color="steelblue")
    ax.bar(ep, [h["val_time_s"] for h in history],
           bottom=[h["train_time_s"] for h in history], label="val", color="orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("giay")
    ax.set_title("Thoi gian moi epoch")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 6. khoảng cách train - val (dau hieu qua khớp)
    ax = axes[1, 2]
    gap = [h["train_loss"] - h["val_loss"] for h in history]
    ax.plot(ep, gap, "o-", color="crimson", ms=3)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss - val loss")
    ax.set_title("Khoang cach train - val (qua khop?)")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        plt.savefig(save, dpi=110, bbox_inches="tight")
    plt.show()
