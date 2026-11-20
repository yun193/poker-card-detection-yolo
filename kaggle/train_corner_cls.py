# %% [markdown]
# # 牌角二次分類 — Ultralytics 分類模型（52 類）
#
# 兩階段辨識的第二階段：偵測器先找出牌角，本機再從**原始全解析度畫面**裁出牌角放大，
# 交給這個分類模型判斷點數＋花色，用來修正偵測器的誤判（例：K♣→A♠）與遠距小牌。
#
# 訓練資料：直接用 andy8744 資料集的牌角框裁圖（約 6 萬個），不需新標註。
# **Kaggle 設定**：GPU、Internet On、Add Input `andy8744/playing-cards-object-detection-dataset`。

# %%
MODEL = "yolov8s-cls.pt"
IMGSZ = 128          # 牌角裁圖統一縮放到 128×128
EPOCHS = 30
BATCH = 256
PAD = 0.25           # 裁切時框四周外擴比例（保留一點背景，貼近實際裁切）
NB_TAG = "cls-1"
print("NB_TAG=" + NB_TAG)

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"], check=True)
import glob, json, os, shutil, random
from pathlib import Path
import cv2, yaml
from concurrent.futures import ThreadPoolExecutor

cands = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True)
         if os.path.isdir(os.path.join(os.path.dirname(p), "train", "images"))]
assert cands, "找不到資料集"
root = Path(os.path.dirname(cands[0]))
names = yaml.safe_load(open(cands[0]))["names"]
OUT_DS = Path("/tmp/corner_cls")   # 裁圖放 /tmp，不進 Output

# %% [markdown]
# ## 裁出牌角
# 依 YOLO 標註框外擴 `PAD` 後裁成正方形。split 對應：train→train、valid→val、test→test。

# %%
def crop_split(split, dst_split):
    imgs = sorted((root / split / "images").glob("*.jpg"))
    for n in names:
        (OUT_DS / dst_split / n).mkdir(parents=True, exist_ok=True)

    def work(p):
        lab = root / split / "labels" / (p.stem + ".txt")
        if not lab.exists():
            return 0
        im = cv2.imread(str(p)); H, W = im.shape[:2]; k = 0
        for i, line in enumerate(lab.read_text().splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            s = max(w * W, h * H) * (1 + 2 * PAD)
            x1, y1 = int(cx * W - s / 2), int(cy * H - s / 2)
            x2, y2 = int(x1 + s), int(y1 + s)
            crop = im[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if crop.size == 0 or min(crop.shape[:2]) < 6:
                continue
            cv2.imwrite(str(OUT_DS / dst_split / names[c] / f"{p.stem}_{i}.jpg"), crop)
            k += 1
        return k
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(work, imgs))
    print(split, "→", dst_split, "裁出", n, "個牌角")

if not OUT_DS.exists():
    for a, b in (("train", "train"), ("valid", "val"), ("test", "test")):
        crop_split(a, b)

# %% [markdown]
# ## 訓練
# 關閉左右翻轉（牌角文字鏡像）；資料集本身已含上下兩個對角（下方那個是 180° 旋轉），模型會學到兩種方向。

# %%
from ultralytics import YOLO
model = YOLO(MODEL)
model.train(data=str(OUT_DS), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, workers=4,
            fliplr=0.0, flipud=0.0, scale=0.25,   # RandomResizedCrop 面積保留 75~100%，避免切掉牌角文字
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, erasing=0.2,
            project="/kaggle/working/runs", name="corner_cls", exist_ok=True, patience=8, plots=True)

# %%
run = Path("/kaggle/working/runs/corner_cls")
best = YOLO(str(run / "weights" / "best.pt"))
r = best.val(data=str(OUT_DS), split="test", imgsz=IMGSZ, batch=BATCH, device=0)
metrics = {"model": MODEL, "top1": float(r.top1), "top5": float(r.top5), "imgsz": IMGSZ, "pad": PAD}
print(metrics)
OUT = Path("/kaggle/working/out/corner_cls"); OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(run / "weights" / "best.pt", OUT / "best.pt")
for f in ("results.csv", "results.png", "confusion_matrix_normalized.png"):
    if (run / f).exists():
        shutil.copy(run / f, OUT / f)
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
shutil.make_archive("/kaggle/working/cards_corner_cls", "zip", "/kaggle/working/out")
print("下載 cards_corner_cls.zip，解壓到本機專案的 models/ 底下")
