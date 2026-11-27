"""從錄影裁出真牌牌角當 value_cls 微調資料（依人工標的時間段正解自動標註，再出總覽圖人工複查）。

正解檔（JSON，放在錄影資料夾裡 labels.json）：
    {"train_until": 32.0,                       # 只用前 32 秒（之後留給驗收）
     "segments": [[0, 8.3, []],                 # [起, 迄（秒，相對第一幀）, 這段畫面裡的牌]；[] = 沒有牌
                  [9.0, 21.8, ["3c", "4d", "3d", "5c", "4h", "5d", "6d"]]]}
不在任何區段內的影格（換牌、模糊過渡）不使用。

標註規則（每個分割框的左上／右下轉正裁角，與推論完全同一套裁法）：
  - 沒有牌的區段、或外觀不像牌的框（looks_like_card 不通過）：裁角標 none（海報、書櫃 → 困難負樣本）
  - 有牌的區段：模型讀值在正解內且信心 ≥ 0.6 → 用讀值；讀值不在正解內，但正解裡「同點數同顏色」只有一張
    （5♠ → 5♣）→ 改成那張；none 機率 ≥ 0.9 → none；其他不確定的丟掉
自動標註可能把同點數同色的牌互換（4♥／4♦），一定要看 review_*.jpg 人工複查。

用法：
  .venv/Scripts/python.exe make_real_crops.py data/recordings/20261121_182108 --out data/real_crops/rec182108
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import web.server as srv
from blackjack_vision.cards import looks_like_card
from blackjack_vision.upright import upright_corner_crops

RED = set("hd")


def fix_label(pred: str, gt: set[str]) -> str | None:
    if pred in gt:
        return pred
    same = [g for g in gt if g[:-1] == pred[:-1] and (g[-1] in RED) == (pred[-1] in RED)]
    return same[0] if len(same) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None, help="正解檔（預設 <folder>/labels.json）")
    ap.add_argument("--every", type=int, default=1, help="每幾幀取一幀（相鄰幀幾乎一樣時可調大）")
    a = ap.parse_args()

    folder, out = Path(a.folder), Path(a.out)
    spec = json.loads(Path(a.labels or folder / "labels.json").read_text(encoding="utf-8"))
    seg, clf = srv.get_card_seg(), srv.get_value_cls()
    frames = sorted(folder.glob("*.jpg"))
    t0 = srv.frame_time(frames[0])
    cnt, dropped, items = Counter(), Counter(), []
    for k, f in enumerate(frames[::a.every]):
        t = srv.frame_time(f) - t0
        if t > spec.get("train_until", 1e9):
            break
        gt = next((set(g) for s, e, g in spec["segments"] if s <= t <= e), None)
        if gt is None:
            continue
        img = srv.imread(f)
        segs = seg(img, conf=0.4)
        faces = clf.classify_segs(img, segs)   # 只為了取外觀（邊框白底、彩色比例）
        crops, bg = [], []
        for s, fc in zip(segs, faces):
            for _, c in upright_corner_crops(img, s["rbox"]):
                crops.append(c); bg.append(not looks_like_card(fc))
        for c, p, is_bg in zip(crops, clf.predict(crops), bg):
            if not gt or is_bg:   # 沒有牌的區段、或外觀不像牌的框（海報、書櫃）→ none
                lab = "none"
            elif p.get("none", 0.0) >= 0.9:
                lab = "none"
            else:
                real = {x: q for x, q in p.items() if x != "none"}
                pred, q = max(real.items(), key=lambda kv: kv[1]) if real else (None, 0.0)
                lab = fix_label(pred, gt) if pred and q >= 0.6 else None
                if lab is None:   # 不確定的另存 _unsure/<讀值>/，人工看過可整批改標後移入正式類別
                    dropped[pred or "?"] += 1
                    d = out / "_unsure" / (pred or "unknown")
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imencode(".jpg", c)[1].tofile(str(d / f"{f.stem}_{dropped[pred or '?']:04d}.jpg"))
                    continue
            d = out / lab
            d.mkdir(parents=True, exist_ok=True)
            name = f"{f.stem}_{cnt[lab]:04d}.jpg"
            cv2.imencode(".jpg", c)[1].tofile(str(d / name))
            items.append({"label": lab, "file": f"{lab}/{name}", "frame": f.name, "t": round(t, 2)})
            cnt[lab] += 1

    (out / "manifest.json").write_text(json.dumps({"source": str(folder), "counts": cnt, "dropped": dropped,
                                                   "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    for lab in sorted(cnt):   # 每類一張總覽（最多 60 張）供人工複查
        files = sorted((out / lab).glob("*.jpg"))
        step = max(1, len(files) // 60)
        tiles = [cv2.resize(cv2.imdecode(np.fromfile(str(p), np.uint8), 1), (80, 80)) for p in files[::step][:60]]
        tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 12)
        sheet = np.vstack([np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)])
        cv2.imencode(".jpg", sheet)[1].tofile(str(out / f"review_{lab}.jpg"))
    print("各類張數", dict(cnt), "\n丟棄（讀值不在正解內）", dict(dropped))


if __name__ == "__main__":
    main()
