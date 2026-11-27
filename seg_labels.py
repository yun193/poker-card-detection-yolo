"""整張牌分割（card_seg）的真實畫面標註工具：挑影格＋產生 labelme 預標、把修好的 labelme 標註轉成 YOLO 分割格式。

1) 產生預標（用現有 card_seg 的分割結果當起點，人工只要把併在一起的扇形拆成每張牌）：
   .venv/Scripts/python.exe seg_labels.py prelabel data/recordings/20261121_182108 --out data/seg_label/rec182108 \
       --cards 9.0-21.8 --empty 0-8.3,25.3-32 --n-cards 40 --n-empty 8
   → <out>/ 裡每張影格一個 .jpg＋同名 .json（labelme 格式，類別名 card）；沒有牌的影格只放 .jpg（背景負樣本）

2) 用 labelme 修（每張「露出的牌」一個多邊形，類別 card；被完全蓋住的牌不標）：
   labelme data/seg_label/rec182108 --labels card
   （新版 labelme 預設就自動存檔、不把影像內嵌進 JSON）

3) 轉成 YOLO 分割資料（images/、labels/，沒有 .json 的影格輸出空標註＝背景）：
   .venv/Scripts/python.exe seg_labels.py export data/seg_label/rec182108 --out data/seg_yolo/rec182108
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import web.server as srv


def parse_ranges(s: str) -> list[tuple[float, float]]:
    return [tuple(map(float, r.split("-"))) for r in s.split(",") if r]


def pick(frames: list[Path], t0: float, ranges, n: int) -> list[Path]:
    """在時間區段內平均挑 n 幀。"""
    cand = [f for f in frames if any(a <= srv.frame_time(f) - t0 <= b for a, b in ranges)]
    if len(cand) <= n:
        return cand
    idx = np.linspace(0, len(cand) - 1, n).round().astype(int)
    return [cand[i] for i in sorted(set(idx))]


def prelabel(a):
    frames = sorted(Path(a.folder).glob("*.jpg"))
    t0 = srv.frame_time(frames[0])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seg = srv.get_card_seg()
    for f in pick(frames, t0, parse_ranges(a.cards), a.n_cards):
        img = srv.imread(f)
        h, w = img.shape[:2]
        shapes = [{"label": "card", "points": [[round(x, 1), round(y, 1)] for x, y in s["poly"][::max(1, len(s["poly"]) // 40)]],
                   "group_id": None, "description": f"預標 conf={s['conf']}", "shape_type": "polygon", "flags": {}}
                  for s in seg(img, conf=0.3)]
        shutil.copy(f, out / f.name)
        (out / f"{f.stem}.json").write_text(json.dumps(
            {"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": f.name, "imageData": None,
             "imageHeight": h, "imageWidth": w}, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in pick(frames, t0, parse_ranges(a.empty), a.n_empty):
        shutil.copy(f, out / f"empty_{f.name}")
    print("輸出", out, "：有牌", len(list(out.glob("*.json"))), "張、背景", len(list(out.glob("empty_*.jpg"))), "張")


def export(a):
    src, out = Path(a.folder), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    n_poly = 0
    for img in sorted(src.glob("*.jpg")):
        lines = []
        js = img.with_suffix(".json")
        if js.exists():
            d = json.loads(js.read_text(encoding="utf-8"))
            w, h = d["imageWidth"], d["imageHeight"]
            for s in d["shapes"]:
                if s["label"] != "card" or s.get("shape_type", "polygon") != "polygon" or len(s["points"]) < 3:
                    continue
                lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in s["points"]))
        shutil.copy(img, out / "images" / img.name)
        (out / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_poly += len(lines)
    print("輸出", out, "：影格", len(list((out / "images").glob("*.jpg"))), "張、多邊形", n_poly, "個")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prelabel")
    p.add_argument("folder"); p.add_argument("--out", required=True)
    p.add_argument("--cards", required=True, help="有牌的時間區段（秒，相對第一幀），例 9-21.8,33-40")
    p.add_argument("--empty", default="", help="沒有牌的時間區段（背景負樣本）")
    p.add_argument("--n-cards", type=int, default=40); p.add_argument("--n-empty", type=int, default=8)
    e = sub.add_parser("export")
    e.add_argument("folder"); e.add_argument("--out", required=True)
    a = ap.parse_args()
    prelabel(a) if a.cmd == "prelabel" else export(a)


if __name__ == "__main__":
    main()
