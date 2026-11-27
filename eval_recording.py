"""錄影驗收：依錄影資料夾的 labels.json（時間段正解）重播，算手牌精確率／召回率、有錯牌的幀、每幀分割框數。

只計 train_until 之後的區段（驗收段，沒拿來訓練或標註）；--all 改算全部有正解的區段。
比較分割權重：
  .venv/Scripts/python.exe eval_recording.py data/recordings/20261121_182108 --seg models/card_seg/best.pt
  .venv/Scripts/python.exe eval_recording.py data/recordings/20261121_182108 --seg models/card_seg_ft/best.pt
比較牌值分類權重加 --cls models/value_cls26_ft/best.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import web.server as srv
from blackjack_vision.cards import normalize_label
from blackjack_vision.detector import CardSegmenter, ValueClassifier


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--seg", default=None, help="分割權重（預設 models.yaml）")
    ap.add_argument("--cls", default=None, help="牌值分類權重（預設 models.yaml）")
    ap.add_argument("--all", action="store_true", help="連訓練段一起算")
    a = ap.parse_args()

    folder = Path(a.folder)
    spec = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    if a.seg:
        seg = CardSegmenter(a.seg, imgsz=640)
        srv.get_card_seg = lambda: seg
    if a.cls:
        clf = ValueClassifier(a.cls, imgsz=128)
        srv.get_value_cls = lambda: clf
    frames = sorted(folder.glob("*.jpg"))
    t0 = srv.frame_time(frames[0])
    for f in frames[:10]:  # 熱機
        srv.Session().process(srv.imread(f))

    tp = fp = fn = wrong = n = 0
    nseg, per_seg = [], Counter()

    def on_frame(f, img, r):
        nonlocal tp, fp, fn, wrong, n
        t = srv.frame_time(f) - t0
        if not a.all and t <= spec.get("train_until", 0):
            return
        gt = next((g for s, e, g in spec["segments"] if s <= t <= e), None)
        if gt is None:
            return
        truth = Counter(normalize_label(x) for x in gt)
        pred = Counter(normalize_label(x) for x in r["hand"])
        hit = sum((pred & truth).values())
        tp, fp, fn = tp + hit, fp + sum(pred.values()) - hit, fn + sum(truth.values()) - hit
        wrong += sum(pred.values()) > hit
        n += 1
        if len(gt) == 7:
            nseg.append(len(r["segs"]))
        per_seg[len(gt)] += 1

    rep = srv.replay_folder(folder, {}, on_frame)
    out = {"seg": a.seg or "models.yaml", "cls": a.cls or "models.yaml", "frames": n, "precision": round(tp / max(tp + fp, 1), 3),
           "recall": round(tp / max(tp + fn, 1), 3), "frames_with_wrong": wrong,
           "fan7_segs_mean": round(float(np.mean(nseg)), 2) if nseg else None,
           "hand_changes": rep["hand_changes"], "ms_median": rep["ms_median"]}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
