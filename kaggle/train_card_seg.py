# %% [markdown]
# # 整張牌偵測（露出區域分割）— 合成資料＋YOLOv8n-seg
#
# 兩階段方案 B 的第一階段：先把畫面中每張牌**露出的區域**分割出來（手拿扇形時多半只露出一條邊＋牌角），
# 之後再依框做切片辨識。單一類別 `card`。
#
# 沒有現成的整張牌標註，所以自己合成：把 `gpiosenka` 的單張牌照片（多種牌面設計）以扇形／散放／橫排重疊
# 貼到 DTD 紋理背景上，依圖層順序算出每張牌的可見遮罩 → 多邊形標註。
#
# **Kaggle 設定**：GPU、Internet On、Add Input：
# `gpiosenka/cards-image-datasetclassification`、`jmexpert/describable-textures-dataset-dtd`

# %%
N_TRAIN, N_VAL = 10000, 1000
MODEL = "yolov8n-seg.pt"
EPOCHS = 60
BATCH = 32
SEED = 0
NB_TAG = "seg-3"
print("NB_TAG=" + NB_TAG)

# %%
import glob, json, math, os, random, shutil
from pathlib import Path
import cv2
import numpy as np


# @include synth_cards.py


def build(cards, bgs, out_dir: Path, n_train, n_val, seed=SEED):
    for split, n in (("train", n_train), ("val", n_val)):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + (0 if split == "train" else 999))
        for i in range(n):
            img, polys, _ = render(cards, bgs, rng)
            cv2.imwrite(str(out_dir / "images" / split / f"{i:05d}.jpg"), img)
            (out_dir / "labels" / split / f"{i:05d}.txt").write_text(
                "\n".join("0 " + " ".join(f"{v:.5f}" for v in p.reshape(-1)) for p in polys) + ("\n" if polys else ""))
    (out_dir / "data.yaml").write_text(f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: card\n")


# %%
if os.path.exists("/kaggle/input"):
    cards, bgs = load_sources()
    print("牌照片", len(cards), "背景", len(bgs))
    assert len(cards) > 1000, "找不到 gpiosenka 牌照片"
    DS = Path("/tmp/card_seg")
    if not (DS / "data.yaml").exists():
        build(cards, bgs, DS, N_TRAIN, N_VAL)
    print("合成完成")

# %% [markdown]
# ## 訓練
# 合成器已含旋轉／透視，YOLO 端關閉左右翻轉（牌面鏡像不自然），其餘用預設增強。

# %%
if os.path.exists("/kaggle/input"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"], check=True)
    from ultralytics import YOLO
    model = YOLO(MODEL)
    model.train(data=str(DS / "data.yaml"), epochs=EPOCHS, imgsz=S, batch=BATCH, device=0, workers=4,
                fliplr=0.0, degrees=10, mosaic=0.5, close_mosaic=10, patience=15, seed=SEED,
                project="/kaggle/working/runs", name="card_seg", exist_ok=True, plots=True)
    run = Path("/kaggle/working/runs/card_seg")
    best = YOLO(str(run / "weights" / "best.pt"))
    r = best.val(data=str(DS / "data.yaml"), imgsz=S, batch=BATCH, device=0)
    metrics = {"model": MODEL, "box_mAP50": float(r.box.map50), "box_mAP50-95": float(r.box.map),
               "mask_mAP50": float(r.seg.map50), "mask_mAP50-95": float(r.seg.map)}
    print(metrics)
    OUT = Path("/kaggle/working/out/card_seg"); OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "weights" / "best.pt", OUT / "best.pt")
    for f in ("results.csv", "results.png", "val_batch0_pred.jpg", "val_batch1_pred.jpg"):
        if (run / f).exists():
            shutil.copy(run / f, OUT / f)
    for i in range(6):   # 附幾張合成樣本供檢查
        shutil.copy(DS / "images" / "val" / f"{i:05d}.jpg", OUT / f"sample_{i}.jpg")
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    shutil.make_archive("/kaggle/working/cards_card_seg", "zip", "/kaggle/working/out")
    print("下載 cards_card_seg.zip，解壓到本機專案的 models/ 底下")
