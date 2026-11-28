# === 11. So sánh nhiều cấu hình: biểu đồ + bảng xếp hạng ===


def _col(df, *names):
    """Trả về TÊN cột đầu tiên có trong df trong số các tên đưa ra."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Không tìm thấy cột nào trong {names} (có: {list(df.columns)})")


def plot_config_comparison(results):
    """Vẽ biểu đồ so sánh các cấu hình đã thử.

    `results` là list dict. Chấp nhận cả hai kiểu tên cột:
    - dùng trong sweep: `val_mae`, `val_rmse`, `val_mae_h24`, `val_mae_h144`
    - dùng trong đánh giá: `mae`, `rmse`, `mae_h24`, `mae_h144`
    """
    df = pd.DataFrame(results).copy()
    c_mae = _col(df, "mae", "val_mae")
    c_rmse = _col(df, "rmse", "val_rmse")
    df = df.sort_values(c_mae).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("So sanh cau hinh sieu tham so (sap xep theo MAE tot nhat)", fontsize=14)

    # 1. MAE / RMSE
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df[c_mae], 0.4, label="MAE toan tang", color="steelblue")
    ax.bar(x + 0.2, df[c_rmse], 0.4, label="RMSE toan tang", color="indianred")
    ax.axhline(THRESHOLDS["mae"], ls="--", color="navy", label="nguong MAE 1.5")
    ax.axhline(THRESHOLDS["rmse"], ls=":", color="darkred", label="nguong RMSE 2.0")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("°C")
    ax.set_title("MAE / RMSE theo cau hinh")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 2. MAE theo tầm dự báo
    ax = axes[1]
    for _, row in df.iterrows():
        vals = []
        for h in HORIZONS:
            v = row.get(f"mae_h{h}", row.get(f"val_mae_h{h}", np.nan))
            vals.append(v)
        ax.plot(HORIZONS, vals, "o-", label=row["name"], ms=4)
    ax.set_xlabel("buoc du bao (10 phut)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Sai so tich luy theo tam du bao")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 3. thời gian train (hoặc số tham số)
    ax = axes[2]
    if "train_time_s" in df.columns:
        ax.barh(df["name"], df["train_time_s"], color="teal")
        ax.set_xlabel("giay")
        ax.set_title("Thoi gian huấn luyện (giay)")
    else:
        ax.barh(df["name"], df.get("params", pd.Series(range(len(df)))), color="teal")
        ax.set_xlabel("so tham so")
        ax.set_title("So tham so")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return df


def plot_learning_curves_grid(all_histories):
    """Vẽ đường val loss của mọi cấu hình trên cùng một trục để so sánh tốc độ hội tụ."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in all_histories.items():
        h = res["history"]
        ax.plot([x["epoch"] for x in h], [x["val_loss"] for x in h], "o-", label=name, ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss (MSE, chuan hoa)")
    ax.set_title("Duong val loss cua cac cau hinh")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
