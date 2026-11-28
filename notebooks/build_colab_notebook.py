"""Sinh file notebook Colab `notebooks/colab_train.ipynb` cho du an `jena-seq2seq`.

Notebook duoc sinh tu ma nguon trong `notebooks/_nbcells/` de tranh loi escape
JSON khi viet truc tiep. Chay:

    python notebooks/build_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_train.ipynb"
CELLS_DIR = Path(__file__).resolve().parent / "_nbcells"


def load_cell(name: str) -> dict:
    """Doc mot o tu tep trong `notebooks/_nbcells/`.

    Ten tep `.md` = o markdown, `.py` = o code.
    """
    path = CELLS_DIR / name
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if name.endswith(".md"):
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build() -> dict:
    """Ghep cac o theo thu tu ten trong `ORDER` thanh mot notebook."""
    order = [ln.strip() for ln in (CELLS_DIR / "ORDER.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    cells = [load_cell(name) for name in order]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True, "name": "colab_train.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Da tao {OUT} ({len(nb['cells'])} o)")
