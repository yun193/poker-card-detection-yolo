"""三模型比較：真牌照片的牌級辨識準確度＋本機推論速度，合併 Kaggle 端的 mAP。

真牌測試集：data/real_test/*.jpg，檔名前綴寫正解，用 '-' 分隔、'_' 之後忽略，例：
    As-10h-Kd_001.jpg      （同一張牌出現兩次就寫兩次：As-As_002.jpg）
    none_003.jpg           （畫面中沒有牌）
App 按 S 截圖會自動用辨識結果命名，人工確認／修正檔名後移到 data/real_test/ 即可。

用法：python evaluate.py [--models yolov8 yolo26] [--conf 0.45]
輸出：reports/compare_<日期>.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.cards import group_corners, normalize_label
from blackjack_vision.detector import ROOT, build_detector, load_registry


def truth_from_name(p: Path) -> Counter:
    tag = p.stem.split("_")[0]
    if tag.lower() == "none":
        return Counter()
    return Counter(normalize_label(t) for t in tag.split("-") if t)


def imread(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def evaluate_model(det, images: list[Path], warmup: int = 5) -> dict:
    tp = fp = fn = exact = 0
    times = []
    errors = []
    frames = [imread(p) for p in images]
    for f in frames[:warmup]:
        det(f)
    for p, f in zip(images, frames):
        t0 = time.perf_counter()
        dets = det(f)
        times.append((time.perf_counter() - t0) * 1000)
        pred = Counter(c.label for c in group_corners(dets))
        truth = truth_from_name(p)
        hit = sum((pred & truth).values())
        tp += hit
        fp += sum(pred.values()) - hit
        fn += sum(truth.values()) - hit
        if pred == truth:
            exact += 1
        else:
            errors.append(f"{p.name}: 預測 {'-'.join(sorted(pred.elements())) or 'none'}")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ms = statistics.median(times) if times else float("nan")
    return dict(precision=prec, recall=rec, f1=f1, exact=exact / max(1, len(images)),
                ms=ms, fps=1000 / ms if ms else 0, errors=errors)


def speed_only(det, n: int = 50) -> dict:
    f = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    for _ in range(5):
        det(f)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        det(f)
        ts.append((time.perf_counter() - t0) * 1000)
    ms = statistics.median(ts)
    return dict(ms=ms, fps=1000 / ms)


def kaggle_metrics(name: str) -> dict:
    p = ROOT / "models" / name / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--data", default=str(ROOT / "data" / "real_test"))
    args = ap.parse_args()

    reg = load_registry()
    names = [n for n in (args.models or reg) if n in reg]
    if not names:
        raise SystemExit("沒有可用模型（檢查 models.yaml 與 models/ 目錄）")
    images = sorted(p for p in Path(args.data).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"模型：{names}；真牌照片 {len(images)} 張")

    rows, err_sections = [], []
    for n in names:
        det = build_detector(reg[n], conf=args.conf)
        r = evaluate_model(det, images) if images else speed_only(det)
        km = kaggle_metrics(n)
        rows.append((n, r, km))
        if r.get("errors"):
            err_sections.append(f"### {n}\n" + "\n".join(f"- {e}" for e in r["errors"][:30]))
        print(n, {k: v for k, v in r.items() if k != "errors"}, km)

    def f(x, pct=False):
        return "-" if x is None else (f"{x:.1%}" if pct else f"{x:.3f}")

    lines = [f"# 三模型比較（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"- 真牌照片：{len(images)} 張　conf={args.conf}　本機推論含前處理，取中位數", "",
             "| 模型 | 合成 test mAP50 | 合成 test mAP50-95 | 真牌 Precision | 真牌 Recall | 真牌 F1 | 整張全對率 | 推論 ms | FPS |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r, km in rows:
        lines.append(f"| {n} | {f(km.get('mAP50'))} | {f(km.get('mAP50-95'))} | {f(r.get('precision'), 1)} | "
                     f"{f(r.get('recall'), 1)} | {f(r.get('f1'), 1)} | {f(r.get('exact'), 1)} | {r['ms']:.1f} | {r['fps']:.1f} |")
    if err_sections:
        lines += ["", "## 錯誤樣本（每模型最多 30 筆）", *err_sections]
    out = ROOT / "reports" / f"compare_{datetime.now():%Y%m%d_%H%M}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("報告：", out)


if __name__ == "__main__":
    main()
