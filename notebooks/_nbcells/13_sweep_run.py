# === 13. Chạy sweep: huấn luyện 15 epoch cho từng cấu hình ===

SWEEP_RESULTS = []
SWEEP_HISTORIES = {}

for spec in SWEEP:
    name = spec["name"]
    arch = spec["arch"]
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in spec.items() if k not in ("name", "arch")})
    kwargs = make_kwargs(arch, cfg)

    print("=" * 70)
    print(f"CAU HINH: {name}  (arch={arch})  |  lr={cfg['lr']}  "
          f"tf={cfg['teacher_forcing']}->{cfg['teacher_forcing_min']}")
    print("=" * 70)

    set_seed(cfg["seed"])
    model = build_model(arch, **kwargs)
    n_params = count_parameters(model)
    print(f"Số tham số: {n_params:,}")

    res = train_loop(model, LOADERS, cfg, run_name=name, verbose=True)
    SWEEP_HISTORIES[name] = res

    # Đánh giá trên tập kiểm định để chọn cấu hình (không dùng tập kiểm thử)
    m_val, _, _ = evaluate_model(res["model"], LOADERS["val"], SCALER)
    total_time = sum(h["train_time_s"] + h["val_time_s"] for h in res["history"])

    row = {
        "name": name,
        "arch": arch,
        "val_mae": m_val["mae"],
        "val_rmse": m_val["rmse"],
        "val_mae_h24": m_val["mae_h24"],
        "val_mae_h144": m_val["mae_h144"],
        "best_epoch": res["best_epoch"],
        "epochs_run": len(res["history"]),
        "params": n_params,
        "train_time_s": round(total_time, 1),
        "lr": cfg["lr"],
        "teacher_forcing": cfg["teacher_forcing"],
    }
    SWEEP_RESULTS.append(row)
    print(f"  >> val MAE = {m_val['mae']:.4f} °C | val RMSE = {m_val['rmse']:.4f} °C "
          f"| tốt nhất ở epoch {res['best_epoch']}")

SWEEP_DF = pd.DataFrame(SWEEP_RESULTS).sort_values("val_mae").reset_index(drop=True)
print("\n" + "=" * 70)
print("BANG XEP HANG THEO VAL MAE (thu nho)")
print("=" * 70)
display(SWEEP_DF[["name", "arch", "val_mae", "val_rmse", "val_mae_h24",
                  "val_mae_h144", "best_epoch", "params", "train_time_s"]])
SWEEP_DF.to_csv("results/sweep_results.csv", index=False, encoding="utf-8")
print("Đã lưu results/sweep_results.csv")
