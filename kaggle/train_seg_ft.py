# %% [markdown]
# # card_seg 真牌微調 — 從部署中的 seg-2 權重出發，用實拍錄影的人工多邊形標註微調
#
# 合成資料連兩輪改都沒把真實扇形的「多張牌被併成一個框」壓下來（seg-3 還變差），這次直接用真實畫面：
# 錄影影格以現有模型預標、人工在 labelme 把併在一起的扇形拆成每張牌，加上沒有牌的影格當背景負樣本。
# 真實資料很少（數十張），所以：凍結骨幹前段、學習率調低、保留時間上最後幾張當驗證；
# 另外混入合成資料（同一套合成器）防止遺忘，並用合成驗證集確認沒有退步。
# 扇形標註：人工 labelme ＋幾何擬合補拆（7 張同尺寸牌繞握點旋轉，擬合整把外框後算每張露出區域）。
#
# **Kaggle 設定**：GPU、Internet On、Add Input：`yun193/cards-seg-real`（images/、labels/、base/ 或 base26/best.pt，依 BASE_DIR）、
# `gpiosenka/cards-image-datasetclassification`、`jmexpert/describable-textures-dataset-dtd`

# %%
IMGSZ = 640
EPOCHS = 40
BATCH = 16
LR0 = 0.001
FREEZE = 10                    # 凍結骨幹前 10 層，只調頸部與分割頭，降低對少量資料過擬合
N_VAL = 6                      # 時間上最後幾張有標註的影格當驗證
REPEAT = 10                    # 真牌訓練清單重複次數（約 370 張次，與合成約 1:5）
N_SYN_TRAIN, N_SYN_VAL = 2000, 300
SEED = 0
NB_TAG = "seg26-ft-1"
BASE_DIR = "base26"             # 資料集裡的基底權重資料夾：base = v8（seg-2）、base26 = YOLO26（seg26-1）
OUT_NAME = "card_seg26_ft"
S = IMGSZ
print("NB_TAG=" + NB_TAG)

# %%
import glob, json, math, os, random, shutil
from pathlib import Path
import cv2
import numpy as np


# @include synth_cards.py


def build_synth(cards, bgs, out_dir: Path):
    for split, n in (("train", N_SYN_TRAIN), ("val", N_SYN_VAL)):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(SEED + (0 if split == "train" else 999))
        for i in range(n):
            img, polys, _ = render(cards, bgs, rng)
            cv2.imwrite(str(out_dir / "images" / split / f"{i:05d}.jpg"), img)
            (out_dir / "labels" / split / f"{i:05d}.txt").write_text(
                "\n".join("0 " + " ".join(f"{v:.5f}" for v in p.reshape(-1)) for p in polys) + ("\n" if polys else ""))


def find_input():
    for p in Path("/kaggle/input").rglob("images"):
        if p.is_dir() and (p.parent / "labels").is_dir():
            return p.parent, next(Path("/kaggle/input").rglob(f"{BASE_DIR}/best.pt"))
    raise FileNotFoundError("找不到 cards-seg-real（images/、labels/、base/ 或 base26/best.pt，依 BASE_DIR）")


def make_lists(root: Path, out: Path, syn: Path):
    """依檔名（時間戳）排序；有牌的影格最後 N_VAL 張當驗證，其餘（含背景）重複 REPEAT 次＋合成訓練集當訓練。"""
    imgs = sorted((root / "images").glob("*.jpg"), key=lambda p: p.name.replace("empty_", ""))
    labeled = [p for p in imgs if (root / "labels" / f"{p.stem}.txt").read_text().strip()]
    val = labeled[-N_VAL:]
    train = [p for p in imgs if p not in val]
    out.mkdir(parents=True, exist_ok=True)
    syn_train = sorted((syn / "images" / "train").glob("*.jpg"))
    (out / "train.txt").write_text("\n".join(str(p) for p in train * REPEAT + syn_train) + "\n")
    (out / "val.txt").write_text("\n".join(str(p) for p in val) + "\n")
    (out / "data.yaml").write_text(f"train: {out / 'train.txt'}\nval: {out / 'val.txt'}\nnames:\n  0: card\n")
    (out / "syn.yaml").write_text(f"path: {syn}\ntrain: images/train\nval: images/val\nnames:\n  0: card\n")
    print("真牌訓練", len(train), "張（×", REPEAT, "）＋合成", len(syn_train), "張；真牌驗證", len(val), "張")
    return out / "data.yaml", out / "syn.yaml"


# %%
if os.path.exists("/kaggle/input"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"], check=True)
    from ultralytics import YOLO
    ROOT, BASE = find_input()
    SYN = Path("/tmp/seg_syn")
    if not (SYN / "images" / "val").exists():
        cards, bgs = load_sources()
        assert len(cards) > 1000, "找不到 gpiosenka 牌照片"
        build_synth(cards, bgs, SYN)
    DATA, SYNY = make_lists(ROOT, Path("/tmp/seg_ft"), SYN)
    before = YOLO(str(BASE)).val(data=str(DATA), imgsz=IMGSZ, batch=BATCH, device=0)
    before_syn = YOLO(str(BASE)).val(data=str(SYNY), imgsz=IMGSZ, batch=BATCH, device=0)
    model = YOLO(str(BASE))
    model.train(data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, workers=4, freeze=FREEZE,
                optimizer="AdamW", lr0=LR0, warmup_epochs=2, fliplr=0.0, degrees=10, mosaic=0.5, close_mosaic=5,
                patience=15, seed=SEED, project="/kaggle/working/runs", name="seg_ft", exist_ok=True, plots=True)
    run = Path("/kaggle/working/runs/seg_ft")
    best = YOLO(str(run / "weights" / "best.pt"))
    r = best.val(data=str(DATA), imgsz=IMGSZ, batch=BATCH, device=0)
    rs = best.val(data=str(SYNY), imgsz=IMGSZ, batch=BATCH, device=0)
    metrics = {"base": BASE_DIR, "mask_mAP50": float(r.seg.map50), "mask_mAP50-95": float(r.seg.map),
               "box_mAP50": float(r.box.map50), "real_val_mask_mAP50-95_before": float(before.seg.map),
               "real_val_mask_mAP50_before": float(before.seg.map50),
               "syn_val_mask_mAP50-95": float(rs.seg.map), "syn_val_mask_mAP50-95_before": float(before_syn.seg.map),
               "epochs": EPOCHS, "lr0": LR0, "freeze": FREEZE,
               "nb_tag": NB_TAG}
    print(metrics)
    OUT = Path("/kaggle/working/out") / OUT_NAME; OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "weights" / "best.pt", OUT / "best.pt")
    for f in ("results.csv", "results.png", "val_batch0_pred.jpg", "val_batch0_labels.jpg"):
        if (run / f).exists():
            shutil.copy(run / f, OUT / f)
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    shutil.make_archive("/kaggle/working/cards_seg_ft", "zip", "/kaggle/working/out")
