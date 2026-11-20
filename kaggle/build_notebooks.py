"""把 kaggle/*.py（`# %%` 分格）轉成 .ipynb，方便上傳到 Kaggle。`# @include 路徑` 會內嵌該檔。

    python kaggle/build_notebooks.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def to_cells(src: str):
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        head, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if head.strip() == "[markdown]":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.splitlines(True)})
    return cells


def inline_includes(src: str) -> str:
    """`# @include 路徑`（相對 kaggle/）整行換成該檔內容，讓 notebook 與本機共用同一份程式碼（合成器、轉正裁角）。"""
    return re.sub(r"^# @include (\S+)\s*$",
                  lambda m: f"# ---- 內嵌自 {m.group(1)} ----\n" + (HERE / m.group(1)).read_text(encoding="utf-8").rstrip()
                  + f"\n# ---- 內嵌結束 {m.group(1)} ----",
                  src, flags=re.M)


def main():
    for py in sorted(HERE.glob("train_*.py")):
        nb = {"cells": to_cells(inline_includes(py.read_text(encoding="utf-8"))),
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        out = py.with_suffix(".ipynb")
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("產生", out.name, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
