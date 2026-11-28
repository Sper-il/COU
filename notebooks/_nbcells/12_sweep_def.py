# === 12. Sweep siêu tham số (15 epoch mỗi cấu hình) ===

# Danh sách cấu hình thử. Mỗi cấu hình là một dict ghi đè lên `DEFAULT_CFG`.
# Bắt đầu từ cấu hình gốc trong `configs/train_config.yaml` rồi thử biến thiên
# có kiểm soát để biết yếu tố nào thực sự ảnh hưởng tới chất lượng.
DEFAULT_CFG = {
    "epochs": 15,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "teacher_forcing": 0.5,
    "teacher_forcing_decay_epochs": 5,
    "teacher_forcing_warmup_epochs": 1,
    "teacher_forcing_min": 0.0,
    "clip": 1.0,
    "early_stop": 15,
    "seed": SEED,
    "scheduler": {"lr_factor": 0.5, "lr_patience": 3, "min_lr": 1e-6},
}

# Tham số khởi tạo mô hình theo từng kiến trúc (khớp configs/model_config.yaml)
ARCH_KWARGS = {
    "lstm_s2s": {"input_dim": 15, "hidden_dim": 128, "num_layers": 2, "dropout": 0.1, "pred_len": 144},
    "lstm_attention": {"input_dim": 15, "hidden_dim": 128, "num_layers": 2, "dropout": 0.1, "pred_len": 144},
    "mini_transformer": {"input_dim": 15, "d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1, "pred_len": 144},
}

ARCH_KEYS = ("hidden_dim", "num_layers", "dropout", "d_model", "nhead")


def make_cfg(spec, epochs, early_stop=None, decay_epochs=10):
    """Dựng cấu hình huấn luyện từ một spec của sweep."""
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in spec.items() if k not in ("name", "arch")})
    cfg["epochs"] = epochs
    if early_stop is not None:
        cfg["early_stop"] = early_stop
    cfg["teacher_forcing_decay_epochs"] = decay_epochs
    return cfg


def make_kwargs(arch, cfg):
    """Tham số khởi tạo mô hình theo kiến trúc và các override trong cfg."""
    kwargs = dict(ARCH_KWARGS[arch])
    for key in ARCH_KEYS:
        if key in cfg:
            kwargs[key] = cfg[key]
    return kwargs


# Nhóm 1: quanh cấu hình gốc (baseline LSTM + Attention + Transformer)
# Nhóm 2: biến thiên tốc độ học
# Nhóm 3: biến thiên kiến trúc / kích thước ẩn
SWEEP = [
    {"name": "lstm_s2s_goc", "arch": "lstm_s2s"},
    {"name": "attn_bahdanau_goc", "arch": "lstm_attention"},
    {"name": "transformer_goc", "arch": "mini_transformer"},
    {"name": "lstm_s2s_lr3e-4", "arch": "lstm_s2s", "lr": 3e-4},
    {"name": "lstm_s2s_lr3e-3", "arch": "lstm_s2s", "lr": 3e-3},
    {"name": "attn_lr3e-4", "arch": "lstm_attention", "lr": 3e-4},
    {"name": "attn_hidden64", "arch": "lstm_attention", "hidden_dim": 64},
    {"name": "attn_tf_tu0", "arch": "lstm_attention", "teacher_forcing": 0.0},
    {"name": "attn_tf_decay10", "arch": "lstm_attention", "teacher_forcing_decay_epochs": 10},
]

print(f"Sweep {len(SWEEP)} cau hinh x {DEFAULT_CFG['epochs']} epoch tren {DEVICE}")
print("Lưu ý: chạy sweep trên tập kiểm định để chọn cấu hình, tập kiểm thử 2016 giữ lại để đánh giá một lần cuối.")
