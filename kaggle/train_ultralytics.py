# %% [markdown]
# # 撲克牌偵測微調 — Ultralytics（YOLOv8 / YOLO12 / YOLO26）
#
# **Kaggle 設定**：Accelerator = GPU（T4 或 P100）、Internet = On、
# Add Input → `andy8744/playing-cards-object-detection-dataset`。
# 建議用 **Save Version → Save & Run All (Commit)** 背景跑，跑完到 Output 下載 `cards_<模型>.zip`（例：`cards_yolov8n.zip`）。
#
# 一次 commit 只訓練 `MODELS` 裡列的模型；三個一起跑約 4~6 小時（Kaggle 單次上限 12 小時）。
# 若擔心超時，可以分三次 commit，每次只留一個模型。

# %%
MODELS = ["yolov8n.pt"]   # 逐一訓練；可選 yolov8n.pt / yolo12n.pt / yolo26n.pt（或 s 版）
EPOCHS = 60
NB_TAG = "v8-only-1"       # 版本標記：本機監看用來辨識 log 是否來自這一版
print("NB_TAG=" + NB_TAG)
IMGSZ = 416          # 資料集原圖就是 416x416
BATCH = 32
PATIENCE = 15
SEED = 0

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"], check=True)
import ultralytics
print("ultralytics", ultralytics.__version__)
ultralytics.checks()

# %%
import glob, json, os, shutil
from pathlib import Path
import yaml

cands = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True)
         if os.path.isdir(os.path.join(os.path.dirname(p), "train", "images"))]
assert cands, "找不到資料集，請確認已 Add Input: andy8744/playing-cards-object-detection-dataset"
root = os.path.dirname(cands[0])
names = yaml.safe_load(open(cands[0]))["names"]
data_cfg = {"path": root, "train": "train/images", "val": "valid/images", "test": "test/images",
            "nc": len(names), "names": names}
DATA = "/kaggle/working/cards.yaml"
yaml.safe_dump(data_cfg, open(DATA, "w"), allow_unicode=True)
for s in ("train", "valid", "test"):
    print(s, len(os.listdir(f"{root}/{s}/images")))

# %% [markdown]
# **增強策略**：資料集是合成圖，要彌補「合成 → 真實相機」落差：
# - `fliplr=0`：左右翻轉會讓牌角的字鏡像，關掉。
# - 加大色彩／亮度、旋轉、透視、縮放的擾動，模擬燈光與手持角度。
# - `close_mosaic` 最後 10 epoch 關 mosaic，讓模型適應完整畫面。

# %%
from ultralytics import YOLO

OUT = Path("/kaggle/working/out")
OUT.mkdir(exist_ok=True)
AUG = dict(fliplr=0.0, flipud=0.0, degrees=15, perspective=0.0005, scale=0.5, translate=0.1,
           hsv_h=0.015, hsv_s=0.7, hsv_v=0.5, mosaic=1.0, close_mosaic=10, mixup=0.0)

for w in MODELS:
    tag = w.replace(".pt", "")
    short = tag[:-1]                      # yolov8n → yolov8（對應本機 models/<short>/）
    run_dir = Path("/kaggle/working/runs") / tag
    YOLO(w).train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE, seed=SEED,
                  project="/kaggle/working/runs", name=tag, exist_ok=True, device=0, workers=4,
                  cos_lr=True, plots=True, **AUG)
    best = run_dir / "weights" / "best.pt"
    m = YOLO(str(best))
    r = m.val(data=DATA, split="test", imgsz=IMGSZ, batch=BATCH, device=0, plots=False)
    metrics = {"model": tag, "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
               "precision": float(r.box.mp), "recall": float(r.box.mr),
               "speed_ms": r.speed, "params": sum(p.numel() for p in m.model.parameters())}
    print(metrics)
    dst = OUT / short
    dst.mkdir(exist_ok=True)
    shutil.copy(best, dst / "best.pt")
    for f in ("results.csv", "results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, dst / f)
    json.dump(metrics, open(dst / "metrics.json", "w"), indent=2)

# %%
ZIP = "/kaggle/working/cards_" + "_".join(m.replace(".pt", "") for m in MODELS)
shutil.make_archive(ZIP, "zip", OUT)
for p in sorted(OUT.rglob("metrics.json")):
    print(p.parent.name, json.load(open(p)))
print(f"下載 {ZIP}.zip，解壓到本機專案的 models/ 底下")
