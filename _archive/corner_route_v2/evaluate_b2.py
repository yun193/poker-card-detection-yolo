"""方案 B-2 驗收：用測試台 DEBUG 存下的實測影格（data/debug_frames*/）。

主指標（驗收用）：
  0. 人工答案正確率：data/b2_gt.json 的人工判讀牌值（目標 ≥ 95%）
參考指標（不需標註，但參考答案本身會錯、扇形窄條常沒結果，只看趨勢）：
  1. 一致率：兩條路都有結果的牌，B-2 牌值與牌角辨識相同的比例
  2. 補漏：牌角辨識沒結果、B-2 信心 ≥ --recover-conf 的框數（另存總覽圖人工抽看）
  3. 延遲：B-2 每幀額外耗時中位數（目標 < 30ms）
分類器：有 models/value_cls/ 用它，否則用牌角二次分類器（零訓練替代），與伺服器相同。

用法：.venv/Scripts/python evaluate_b2.py [--out 總覽圖目錄] [--recover-conf 0.8]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.cards import Detection, _inside, verify_segs
from blackjack_vision.detector import ROOT, CornerClassifier, load_refiner, load_value_cls


def reference_labels(segs: list[dict], dets: list[dict]) -> list[str | None]:
    """每個框內信心最高的牌角標籤（牌角只歸含它的最小框）。"""
    best: list[tuple[float, str] | None] = [None] * len(segs)
    polys = [[tuple(p) for p in s["poly"]] for s in segs]
    areas = [cv2.contourArea(np.array(p, np.float32)) for p in polys]
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        hit = [i for i, p in enumerate(polys) if _inside(((x1 + x2) / 2, (y1 + y2) / 2), p)]
        if hit:
            i = min(hit, key=lambda k: areas[k])
            if best[i] is None or d["conf"] > best[i][0]:
                best[i] = (d["conf"], d["label"])
    return [b[1] if b else None for b in best]


def kept_segs(frame_json: Path) -> tuple[list[dict], dict]:
    d = json.loads(frame_json.read_text(encoding="utf-8"))
    r = d["result"]
    dets = [Detection(x["label"], x["conf"], tuple(x["box"])) for x in r["detections"]]
    segs, _ = verify_segs([dict(s) for s in r.get("segs", []) + r.get("segs_rejected", [])], dets)
    return segs, d


def gt_accuracy(clf) -> None:
    gt = ROOT / "data" / "b2_gt.json"
    if not gt.exists():
        return
    items = json.loads(gt.read_text(encoding="utf-8"))["items"]
    ok, wrong = 0, []
    for it in items:
        f = ROOT / "data" / it["frame"]
        segs, _ = kept_segs(f)
        img = cv2.imdecode(np.fromfile(str(f.with_suffix(".jpg")), np.uint8), cv2.IMREAD_COLOR)
        p = clf.classify_segs(img, [segs[it["seg"]]])[0]
        if p.get("label") == it["label"]:
            ok += 1
        else:
            wrong.append(f"{it['label']}→{p.get('label')}")
    print(f"人工答案正確率 {ok}/{len(items)} = {ok / len(items):.1%}　錯：{' '.join(wrong) or '無'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="補漏與不一致的總覽圖輸出目錄")
    ap.add_argument("--recover-conf", type=float, default=0.8)
    a = ap.parse_args()

    spec = load_value_cls() or load_refiner()
    assert spec, "需要 models/value_cls/ 或 models/corner_cls/"
    print("分類器：", spec["weights"])
    clf = CornerClassifier(str(ROOT / spec["weights"]), imgsz=spec.get("imgsz", 128), pad=spec.get("pad", 0.25))
    gt_accuracy(clf)

    stats, ms, samples = Counter(), [], {"mismatch": [], "recover": []}
    for f in sorted(glob.glob(str(ROOT / "data" / "debug_frames*" / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d["settings"].get("view") != "both":
            continue
        r = d["result"]
        segs = r.get("segs", []) + r.get("segs_rejected", [])
        dets = [Detection(x["label"], x["conf"], tuple(x["box"])) for x in r["detections"]]
        segs, _ = verify_segs([dict(s) for s in segs], dets)    # 只評通過驗證的框（背景不算）
        if not segs:
            continue
        img = cv2.imdecode(np.fromfile(f[:-5] + ".jpg", np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        pred = clf.classify_segs(img, segs)
        ms.append((time.perf_counter() - t0) * 1000)
        for s, ref, p in zip(segs, reference_labels(segs, r["detections"]), pred):
            if ref and p:
                ok = p["label"] == ref
                stats["agree" if ok else "disagree"] += 1
                if not ok:
                    samples["mismatch"].append((f, s, f"{ref}->{p['label']} {p['conf']:.2f}"))
            elif ref:
                stats["b2_none"] += 1
            elif p and p["conf"] >= a.recover_conf:
                stats["recover"] += 1
                samples["recover"].append((f, s, f"{p['label']} {p['conf']:.2f} "))
            else:
                stats["no_ref_low_conf"] += 1

    both = stats["agree"] + stats["disagree"]
    print(f"一致率 {stats['agree']}/{both} = {stats['agree'] / max(both, 1):.1%}")
    print(f"補漏（無牌角結果、B-2 ≥ {a.recover_conf}）：{stats['recover']}　無牌角且低信心：{stats['no_ref_low_conf']}"
          f"　B-2 裁不出：{stats['b2_none']}")
    print(f"B-2 延遲中位數 {np.median(ms):.1f} ms（{len(ms)} 幀）")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for kind, items in samples.items():
            tiles = []
            for f, s, text in items[:24]:
                img = cv2.imdecode(np.fromfile(f[:-5] + ".jpg", np.uint8), cv2.IMREAD_COLOR)
                cv2.polylines(img, [np.array(s["rbox"], np.int32)], True, (0, 0, 255), 3)
                x, y = map(int, s["center"])
                cv2.putText(img, text, (max(0, x - 120), y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
                tiles.append(cv2.resize(img, (480, 270)))
            if tiles:
                while len(tiles) % 4:
                    tiles.append(np.zeros_like(tiles[0]))
                sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
                cv2.imencode(".jpg", sheet)[1].tofile(str(out / f"b2_{kind}.jpg"))
        print("總覽圖：", out)


if __name__ == "__main__":
    main()
