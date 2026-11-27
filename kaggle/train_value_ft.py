# %% [markdown]
# # value_cls 真牌微調 — 從現有權重出發，合成資料＋實拍錄影牌角混合訓練
#
# 現有 value_cls 只用合成資料訓練；實拍錄影（牌離鏡頭遠、牌角約 20px）常把 5♣ 認成 5♠、4♥ 認成 A♥，
# 也會把海報、牌中央的大花色讀成牌值。這裡把錄影裁出、人工複查過的牌角（含海報／書櫃／大花色等 none）
# 加進合成資料一起微調。為避免過擬合：真牌只佔一部分（過採樣倍數可調）、學習率調低、合成驗證集同時監看。
#
# **Kaggle 設定**：GPU、Internet On、Add Input：
# `gpiosenka/cards-image-datasetclassification`、`jmexpert/describable-textures-dataset-dtd`、
# `yun193/cards-real-crops`（real/<類別>/*.jpg 與 base/ 或 base26/best.pt，依 BASE_DIR）

# %%
N_TRAIN, N_VAL = 4000, 900     # 合成場景數；N_VAL 與 SEED 同原訓練 → 合成驗證集相同，可直接比 top-1
IMGSZ = 128
EPOCHS = 12
BATCH = 256
LR0 = 0.0005
REAL_POS_TARGET = 250          # 真牌牌值類別過採樣到約此張數（每類倍數 1~20，避免張數多的 6♦ 壓過其他類）
REAL_NONE_REPEAT = 2           # 真牌 none 過採樣倍數
NONE_RATIO = 1.5
SEED = 0
NB_TAG = "value26-ft-1"
BASE_DIR = "base26"             # 資料集裡的基底權重資料夾：base = v8（value-1）、base26 = YOLO26（value26-1）
OUT_NAME = "value26_ft"
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
def find_real():
    for p in Path("/kaggle/input").rglob("real"):
        if p.is_dir() and any(p.iterdir()):
            return p, next(Path("/kaggle/input").rglob(f"{BASE_DIR}/best.pt"))
    raise FileNotFoundError("找不到 cards-real-crops（real/、base/ 或 base26/best.pt，依 BASE_DIR）")


def add_real(real: Path, ds: Path):
    """真牌裁圖縮到 IMGSZ 後複製進 train/<類別>/：牌值類別過採樣到約 REAL_POS_TARGET 張、none ×REAL_NONE_REPEAT。"""
    cnt = Counter()
    for d in sorted(real.iterdir()):
        files = sorted(d.glob("*.jpg"))
        rep = REAL_NONE_REPEAT if d.name == "none" else min(20, max(1, round(REAL_POS_TARGET / len(files))))
        out = ds / "train" / d.name
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            img = cv2.resize(cv2.imread(str(f)), (IMGSZ, IMGSZ))
            for k in range(rep):
                cv2.imwrite(str(out / f"real_{f.stem}_{k}.jpg"), img)
                cnt[d.name] += 1
    print("真牌加入", dict(cnt))
    return dict(cnt)


if os.path.exists("/kaggle/input"):
    cards, bgs = load_sources()
    print("牌照片", len(cards), "背景", len(bgs))
    REAL, BASE = find_real()
    DS = Path("/tmp/value_ft")
    if not (DS / "train").exists():
        STATS = build(cards, bgs, DS, N_TRAIN, N_VAL)
        REAL_STATS = add_real(REAL, DS)

# %% [markdown]
# ## 微調
# 從現有權重出發、學習率調低。訓練前後都在同一個合成驗證集上量 top-1，確認合成能力沒有明顯退步。

# %%
if os.path.exists("/kaggle/input"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"], check=True)
    from ultralytics import YOLO
    before = YOLO(str(BASE)).val(data=str(DS), imgsz=IMGSZ, batch=BATCH, device=0)
    model = YOLO(str(BASE))
    model.train(data=str(DS), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, workers=4,
                optimizer="AdamW", lr0=LR0, warmup_epochs=1, fliplr=0.0, degrees=8, patience=6, seed=SEED,
                project="/kaggle/working/runs", name="value_ft", exist_ok=True, plots=True)
    run = Path("/kaggle/working/runs/value_ft")
    best = YOLO(str(run / "weights" / "best.pt"))
    r = best.val(data=str(DS), imgsz=IMGSZ, batch=BATCH, device=0)
    metrics = {"base": BASE_DIR, "top1": float(r.top1), "top5": float(r.top5),
               "synth_val_top1_before": float(before.top1), "imgsz": IMGSZ, "epochs": EPOCHS, "lr0": LR0,
               "real_pos_target": REAL_POS_TARGET, "real_none_repeat": REAL_NONE_REPEAT, "real": REAL_STATS,
               "nb_tag": NB_TAG}
    print(metrics)
    OUT = Path("/kaggle/working/out") / OUT_NAME; OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "weights" / "best.pt", OUT / "best.pt")
    for f in ("results.csv", "results.png", "confusion_matrix_normalized.png"):
        if (run / f).exists():
            shutil.copy(run / f, OUT / f)
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    shutil.make_archive("/kaggle/working/cards_value_ft", "zip", "/kaggle/working/out")
    print("下載 cards_value_ft.zip，解壓到本機專案的 models/ 底下")
