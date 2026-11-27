# %% [markdown]
# # 方案 B-2 牌值分類 — 轉正牌角裁圖＋YOLO26s-cls（52 類＋none；v8 版改 MODEL 即可）
#
# 本機流程：整張牌分割 → 旋轉框轉正 → 裁左上與轉 180° 的右下牌角 → 分類（兩角平均）。
# 零訓練版借用舊牌角分類器只有約 89%（人工判讀 24/27），主因是訓練資料的裁法與牌面不同。
# 這裡用整張牌合成器重新產生資料：每張可見的牌依「本機完全相同的轉正裁角程式」（內嵌 upright.py）裁圖，
# 裁到真正牌角的標牌值，裁到空白角／別張牌／手指的標 `none`，讓推論時能忽略沒有牌角的那一角。
#
# **Kaggle 設定**：GPU、Internet On、Add Input：
# `gpiosenka/cards-image-datasetclassification`、`jmexpert/describable-textures-dataset-dtd`

# %%
N_TRAIN, N_VAL = 9000, 900     # 合成場景數（每場景約 2~6 張可見牌、每張 2 個裁角）
IMGSZ = 128
MODEL = "yolo26s-cls.pt"       # 可改 yolov8s-cls.pt
OUT_NAME = "value_cls26"
EPOCHS = 30
BATCH = 256
NONE_RATIO = 1.5               # none 類別上限＝每個牌值類別平均張數 × 此倍數
SEED = 0
NB_TAG = "value26-1"
print("NB_TAG=" + NB_TAG)

# %%
import json, os, random, shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np


# @include synth_cards.py


# @include ../blackjack_vision/upright.py


# @include value_cls_data.py


# %%
if os.path.exists("/kaggle/input"):
    cards, bgs = load_sources()
    print("牌照片", len(cards), "背景", len(bgs))
    assert len(cards) > 1000, "找不到 gpiosenka 牌照片"
    DS = Path("/tmp/value_cls")
    if not (DS / "train").exists():
        STATS = build(cards, bgs, DS, N_TRAIN, N_VAL)

# %% [markdown]
# ## 訓練
# 裁角已轉正，YOLO 端關閉左右翻轉與大角度旋轉（6／9、牌面鏡像都會混淆）。

# %%
if os.path.exists("/kaggle/input"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"], check=True)
    from ultralytics import YOLO
    model = YOLO(MODEL)
    model.train(data=str(DS), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, workers=4,
                fliplr=0.0, degrees=8, patience=10, seed=SEED,
                project="/kaggle/working/runs", name="value_cls", exist_ok=True, plots=True)
    run = Path("/kaggle/working/runs/value_cls")
    best = YOLO(str(run / "weights" / "best.pt"))
    r = best.val(data=str(DS), imgsz=IMGSZ, batch=BATCH, device=0)
    metrics = {"model": MODEL, "top1": float(r.top1), "top5": float(r.top5), "imgsz": IMGSZ,
               "crop": {"INDEX_DX": INDEX_DX, "INDEX_DY": INDEX_DY, "INDEX_SIZE": INDEX_SIZE, "CROP_PAD": CROP_PAD},
               "nb_tag": NB_TAG}
    print(metrics)
    OUT = Path("/kaggle/working/out") / OUT_NAME; OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "weights" / "best.pt", OUT / "best.pt")
    for f in ("results.csv", "results.png", "confusion_matrix_normalized.png"):
        if (run / f).exists():
            shutil.copy(run / f, OUT / f)
    # 每類附一張裁圖樣本，供檢查標註
    sheet = [cv2.imread(str(sorted((DS / "val" / k).glob("*.jpg"))[0])) for k in sorted(os.listdir(DS / "val"))[:53]]
    sheet += [np.zeros_like(sheet[0])] * (-len(sheet) % 9)
    cv2.imwrite(str(OUT / "samples.jpg"), np.vstack([np.hstack(sheet[i:i + 9]) for i in range(0, len(sheet), 9)]))
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    shutil.make_archive("/kaggle/working/cards_value_cls", "zip", "/kaggle/working/out")
    print("下載 cards_value_cls.zip，解壓到本機專案的 models/ 底下")
