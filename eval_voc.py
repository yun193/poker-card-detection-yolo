"""外部資料集測試：VOC xml 標註（物件名如 "king of diamonds"）的照片逐張單獨辨識（不追蹤），比對手牌。

  .venv/Scripts/python.exe eval_voc.py data/external/gunhcolab --out reports/eval_gunhcolab
標註只有 21 點點數（物件名 "1"~"10"，如 Roboflow jallchin/blackjack-cards）時自動改比點數（J/Q/K 算 10、A 算 1）。
Roboflow 匯出夾帶的授權字樣物件（License／Roboflow／https）會忽略。
輸出整體精確率／召回率／整張全對率、依張數分組結果，--out 另存每張的疊圖與 results.json 供人工抽查。
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import web.server as srv

RANKS = {"ace": "A", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
         "nine": "9", "ten": "10", "jack": "J", "queen": "Q", "king": "K"}
SUITS = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


IGNORE = ("license", "roboflow", "https")


def to_label(name: str):
    n = name.strip().lower()
    if n.isdigit() and 1 <= int(n) <= 10:
        return int(n)  # 只有點數
    p = n.split(" of ")
    return RANKS[p[0]] + SUITS[p[1]] if len(p) == 2 and p[0] in RANKS and p[1] in SUITS else None


def points(label: str) -> int:
    r = label[:-1].upper()
    return 1 if r == "A" else 10 if r in ("10", "J", "Q", "K") else int(r)


def load_items(root: Path):
    for x in sorted(root.rglob("*.xml")):
        names = [o.findtext("name") or "" for o in ET.parse(x).getroot().iter("object")]
        names = [n for n in names if not any(k in n.lower() for k in IGNORE)]
        labels = [to_label(n) for n in names]
        if not labels or None in labels:  # 有非牌物件（狗、人…）或無法解析的就跳過
            continue
        img = next((x.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".JPG", ".PNG") if x.with_suffix(e).exists()), None)
        if img is not None:
            yield img, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seg", default=None, help="分割權重（預設 models.yaml）")
    ap.add_argument("--cls", default=None, help="牌值分類權重（預設 models.yaml）")
    ap.add_argument("--min-cards", type=int, default=1, help="只算正解至少幾張牌的影像（jallchin 用 4，排除標註不全的單張圖）")
    a = ap.parse_args()
    out = Path(a.out) if a.out else None
    if a.seg:
        from blackjack_vision.detector import CardSegmenter
        seg = CardSegmenter(a.seg, imgsz=640)
        srv.get_card_seg = lambda: seg
    if a.cls:
        from blackjack_vision.detector import ValueClassifier
        clf = ValueClassifier(a.cls, imgsz=128)
        srv.get_value_cls = lambda: clf
    if out:
        (out / "vis").mkdir(parents=True, exist_ok=True)

    tp = fp = fn = exact = 0
    groups = defaultdict(lambda: [0, 0, 0, 0, 0])  # 張數 → [張圖數, tp, fp, fn, 全對]
    rows, ms, skipped, tot_ok = [], [], 0, 0
    for f, labels in load_items(Path(a.root)):
        if len(labels) < a.min_cards:
            continue
        img = srv.imread(f)
        if img is None or min(img.shape[:2]) < 64:
            skipped += 1
            continue
        s = srv.Session(); s.apply({"smooth": False})
        r = s.process(img)
        hand = [points(h) for h in r["hand"]] if isinstance(labels[0], int) else r["hand"]
        truth, pred = Counter(labels), Counter(hand)
        tot_ok += sum(points(h) for h in r["hand"]) == sum(l if isinstance(l, int) else points(l) for l in labels)
        hit = sum((pred & truth).values())
        e = (pred == truth)
        tp, fp, fn, exact = tp + hit, fp + sum(pred.values()) - hit, fn + sum(truth.values()) - hit, exact + e
        g = groups[min(len(labels), 5)]
        g[0] += 1; g[1] += hit; g[2] += sum(pred.values()) - hit; g[3] += sum(truth.values()) - hit; g[4] += e
        ms.append(r["ms"]["total"])
        rows.append({"file": str(f), "truth": sorted(labels), "pred": sorted(hand), "hand": sorted(r["hand"]), "segs": len(r["segs"]), "exact": e})
        if out:
            vis = img.copy()
            for sg in r["segs"]:
                cv2.polylines(vis, [np.array(sg["poly"], np.int32)], True, (0, 255, 0), 2)
            k = 900 / max(vis.shape[:2])
            vis = cv2.resize(vis, None, fx=k, fy=k) if k < 1 else vis
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(vis, f"GT {' '.join(map(str, sorted(labels)))}", (6, 20), 0, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"PR {' '.join(sorted(r['hand'])) or '-'}  ({sum(points(h) for h in r['hand'])})", (6, 44), 0, 0.6,
                        (0, 255, 0) if e else (0, 0, 255), 2)
            cv2.imencode(".jpg", vis)[1].tofile(str(out / "vis" / f"{'ok' if e else 'ng'}_{f.stem[:40]}.jpg"))

    res = {"images": len(rows), "skipped": skipped, "precision": round(tp / max(tp + fp, 1), 3),
           "recall": round(tp / max(tp + fn, 1), 3), "exact": round(exact / max(len(rows), 1), 3), "total_exact": round(tot_ok / max(len(rows), 1), 3),
           "ms_median": round(float(np.median(ms)), 1) if ms else None,
           "by_cards": {("5+" if k == 5 else str(k)): {"images": g[0], "precision": round(g[1] / max(g[1] + g[2], 1), 3),
                                                     "recall": round(g[1] / max(g[1] + g[3], 1), 3),
                                                     "exact": round(g[4] / max(g[0], 1), 3)}
                        for k, g in sorted(groups.items())}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if out:
        (out / "results.json").write_text(json.dumps({**res, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
