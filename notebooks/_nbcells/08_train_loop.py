# === 7. Hàm huấn luyện có đầy đủ công cụ quan sát ===

import torch.nn as nn


def train_loop(model, loaders, cfg, run_name="run", verbose=True, log_every_batch=False):
    """Huấn luyện 1 kiến trúc theo cấu hình `cfg` và trả về lịch sử chi tiết.

    Mọi thứ cần để quan sát quá trình train đều được ghi vào `history`:
    loss/MAE huấn luyện và kiểm định theo từng epoch, tỉ lệ teacher forcing,
    tốc độ học, thời gian mỗi epoch, và độ trễ suy luận.

    Args:
        model: mô hình PyTorch đã dựng.
        loaders: dict {train, val, test} DataLoader.
        cfg: dict cấu hình (xem `DEFAULT_CFG`).
        run_name: tên hiển thị.
        verbose: in tiến độ mỗi epoch.
        log_every_batch: in loss từng lô (để thấy độ trễ trong epoch).

    Returns:
        dict gồm `history` (list theo epoch) và `best` (epoch tốt nhất).
    """
    set_seed(cfg.get("seed", SEED))
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-5)
    )
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}))
    criterion = mse_loss

    epochs = cfg["epochs"]
    tf_start = cfg.get("teacher_forcing", 0.5)
    tf_decay_epochs = cfg.get("teacher_forcing_decay_epochs", 10)
    tf_warmup = cfg.get("teacher_forcing_warmup_epochs", 1)
    clip = cfg.get("clip", 1.0)
    patience = cfg.get("early_stop", 5)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        # Lịch teacher forcing: giữ ở mức khởi đầu ở epoch đầu, sau đó giảm
        # TUYẾN TÍNH về 0. Khi về 0, mô hình học đúng chế độ tự hồi tiếp như
        # lúc suy luận nên không còn "sốc" (exposure bias).
        if epoch <= tf_warmup:
            tf_ratio = tf_start
        else:
            span = max(tf_decay_epochs, 1)
            progress = min((epoch - tf_warmup) / span, 1.0)
            tf_ratio = max(tf_start * (1.0 - progress), cfg.get("teacher_forcing_min", 0.0))

        # ---- vòng huấn luyện ----
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        tot_mae = 0.0
        n_batch = 0
        for batch in loaders["train"]:
            x, y = batch["input"].to(DEVICE), batch["target"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, y_in=y, teacher_forcing_ratio=tf_ratio)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            tot_loss += float(loss.item())
            tot_mae += float(mae_loss(pred, y).item())
            n_batch += 1
            if log_every_batch and n_batch % 50 == 0:
                print(f"    lô {n_batch:>4} | loss {tot_loss / n_batch:.5f} "
                      f"| {time.time() - t0:.0f}s")
        train_time = time.time() - t0

        # ---- kiểm định ở chế độ tự hồi tiếp (giống hệt lúc suy luận) ----
        model.eval()
        t1 = time.time()
        v_loss = 0.0
        v_mae = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in loaders["val"]:
                x, y = batch["input"].to(DEVICE), batch["target"].to(DEVICE)
                out = model(x, y_in=None)
                v_loss += float(criterion(out, y).item())
                v_mae += float(mae_loss(out, y).item())
                n_val += 1
        val_time = time.time() - t1
        v_loss /= max(n_val, 1)
        v_mae /= max(n_val, 1)

        scheduler.step(v_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": tot_loss / max(n_batch, 1),
            "train_mae": tot_mae / max(n_batch, 1),
            "val_loss": v_loss,
            "val_mae": v_mae,
            "teacher_forcing": tf_ratio,
            "lr": lr_now,
            "train_time_s": train_time,
            "val_time_s": val_time,
        })

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            mark = "  <-- tot nhat" if improved else ""
            print(f"  Epoch {epoch:>3}/{epochs} | train loss {tot_loss / max(n_batch, 1):.5f} "
                  f"mae {tot_mae / max(n_batch, 1):.5f} | val loss {v_loss:.5f} "
                  f"mae {v_mae:.5f} | tf {tf_ratio:.3f} | lr {lr_now:.2e} "
                  f"| {train_time:.0f}s + {val_time:.0f}s{mark}")

        if bad_epochs >= patience:
            print(f"  DUNG SOM sau {epoch} epoch (val loss khong cai thien {patience} lan)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val, "model": model}
