"""Kiem tra day du chuoi o cua notebook Colab (dung tap con de chay nhanh).

Chay: python notebooks/_verify_nb.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NB = ROOT / "notebooks" / "colab_train.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"So o code: {len(code)}")
    ns: dict = {}

    # 3=common 4=data_check 5=baseline 6=train_loop 7=eval
    # 8=plot_curves 9=plot_compare 10=sweep_def
    for i in range(3, 11):
        exec(compile(code[i], f"cell{i}", "exec"), ns)
    print("[OK] o 4-11")

    # Thay bang tap con de verify nhanh (van giu _meta de o danh gia lay ngay)
    full = ns["LOADERS"]
    small = dict(full)
    for name in ("train", "val", "test"):
        small[name] = list(full[name])[:2]
    ns["LOADERS"] = small
    print("[OK] da thay bang tap con (2 lo/bo) de verify")

    # Rut gon sweep: 1 cau hinh, 1 epoch
    ns["SWEEP"] = [{"name": "lstm_s2s_goc", "arch": "lstm_s2s"}]
    ns["DEFAULT_CFG"]["epochs"] = 1
    ns["DEFAULT_CFG"]["early_stop"] = 1

    # 11=sweep_run 12=sweep_analysis
    exec(compile(code[11], "cell11", "exec"), ns)
    exec(compile(code[12], "cell12", "exec"), ns)
    print("[OK] o 12-13 (sweep + phan tich)")

    # 13=final_train: chay 2 epoch thay vi 30
    src = code[13].replace("epochs=30", "epochs=2")
    exec(compile(src, "cell13", "exec"), ns)
    print("[OK] o 14 (train chinh)")

    # 14=observe_final
    exec(compile(code[14], "cell14", "exec"), ns)
    print("[OK] o 15 (quan sat)")

    # 15=final_eval
    exec(compile(code[15], "cell15", "exec"), ns)
    print("[OK] o 16 (danh gia cuoi)")

    # 16=compare_archs: rut gon con 1 kien truc + 1 epoch
    src = code[16].replace(
        'COMPARE_ARCHS = ["lstm_s2s", "lstm_attention", "mini_transformer"]',
        'COMPARE_ARCHS = ["lstm_s2s"]').replace("COMPARE_EPOCHS = 15", "COMPARE_EPOCHS = 1")
    exec(compile(src, "cell16", "exec"), ns)
    print("[OK] o 17 (so sanh kien truc)")

    # 17=latency
    exec(compile(code[17], "cell17", "exec"), ns)
    print("[OK] o 18 (latency)")

    # 18=summary
    exec(compile(code[18], "cell18", "exec"), ns)
    print("[OK] o 19 (tong ket)")

    print("\n=== TAT CA O CHAY DUOC ===")


if __name__ == "__main__":
    main()
