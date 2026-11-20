# %% [markdown]
# # 方案 B-2 牌值分類 — 轉正牌角裁圖＋YOLOv8s-cls（52 類＋none）
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
MODEL = "yolov8s-cls.pt"
EPOCHS = 30
BATCH = 256
NONE_RATIO = 1.5               # none 類別上限＝每個牌值類別平均張數 × 此倍數
SEED = 0
NB_TAG = "value-1"
print("NB_TAG=" + NB_TAG)

# %%
import json, os, random, shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np


# @include synth_cards.py


# @include ../blackjack_vision/upright.py


# %% [markdown]
# ## 標註裁角
# 牌照片貼上去的四角 `quad` 依序是牌的左上、右上、右下、左下，所以真正牌角（左上、右下）的位置可由同一個透視算出。
# 裁切中心離真牌角 ≤ 0.12×牌高、且該處沒被蓋住 → 標牌值；否則 → `none`。

# %%
RANKS = {"ace": "A", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
         "nine": "9", "ten": "10", "jack": "J", "queen": "Q", "king": "K"}
SUITS = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


def to_label(name: str):
    parts = name.lower().split(" of ")
    if len(parts) != 2 or parts[0] not in RANKS or parts[1] not in SUITS:
        return None
    return RANKS[parts[0]] + SUITS[parts[1]]


def true_index_points(q):
    """牌照片座標 → 原圖：左上、右下兩個牌角索引中心。"""
    cw, ch = 200.0, 280.0   # 與 render 貼圖尺寸一致
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), q.astype(np.float32))
    pts = np.float32([[[INDEX_DX * ch, INDEX_DY * ch]], [[cw - INDEX_DX * ch, ch - INDEX_DY * ch]]])
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2), float(np.linalg.norm(q[3] - q[0]))


def visible_at(mask, pt, r):
    x, y, r = int(pt[0]), int(pt[1]), max(2, int(r))
    win = mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return win.size > 0 and (win > 0).mean() >= 0.6


def crops_from_scene(img, infos):
    out = []
    for inf in infos:
        lab = to_label(inf["label"])
        m = inf["mask"]
        if lab is None or cv2.countNonZero(m) < 400:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        rbox = cv2.boxPoints(cv2.minAreaRect(c))                     # 與本機 CardSegmenter 相同的旋轉框
        idx_pts, card_h = true_index_points(inf["quad"])
        centers = dict(upright_corner_points(rbox))
        for name, crop in upright_corner_crops(img, rbox):
            cpt = centers[name]
            hit = any(np.linalg.norm(cpt - t) <= 0.12 * card_h and visible_at(m, t, 0.04 * card_h) for t in idx_pts)
            out.append((lab if hit else "none", cv2.resize(crop, (IMGSZ, IMGSZ))))
    return out


def build(cards, bgs, out_dir: Path, n_train, n_val, seed=SEED):
    stats = {}
    for split, n in (("train", n_train), ("val", n_val)):
        rng = random.Random(seed + (0 if split == "train" else 999))
        buf, cnt = [], Counter()
        for _ in range(n):
            img, _, infos = render(cards, bgs, rng)
            for lab, crop in crops_from_scene(img, infos):
                buf.append((lab, crop)); cnt[lab] += 1
        avg = np.mean([v for k, v in cnt.items() if k != "none"])
        keep_none = int(avg * NONE_RATIO)
        rng.shuffle(buf)
        written = Counter()
        for lab, crop in buf:
            if lab == "none" and written["none"] >= keep_none:
                continue
            d = out_dir / split / lab
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{written[lab]:05d}.jpg"), crop)
            written[lab] += 1
        stats[split] = dict(written)
        print(split, "類別數", len(written), "總數", sum(written.values()), "none", written["none"],
              "最少", min(v for k, v in written.items() if k != "none"))
    return stats


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
    OUT = Path("/kaggle/working/out/value_cls"); OUT.mkdir(parents=True, exist_ok=True)
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
