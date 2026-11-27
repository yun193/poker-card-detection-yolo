"""實拍影格回測：用目前部署的完整流程（models.yaml 的整張牌分割＋牌角交叉驗證＋B-2 牌值＋一框多牌值），
重播測試台 DEBUG 模式存下的實拍影格（data/debug_frames_round*/），產出每輪統計與轉正牌面總覽，供人工判讀。

每輪各自一個 Session，跨幀記憶依影格原始時間戳；所有影格一律以「兩者並列」處理（不論當初拍攝時的模式）。
換分割權重比較時用 --seg（不改 models.yaml）。

輸出（--out，預設 reports/backtest_<時間>/）：
  summary.md     每輪統計（框數、剔除數、單／多牌值、每幀耗時中位數）
  items.json     每個保留框：輪次、影格、牌值、驗證方式
  sheet_N.jpg    轉正牌面總覽（每張 #編號＋牌值，紅字有牌值、灰字「?」），人工判讀用

用法：
  .venv/Scripts/python.exe backtest_real.py
  .venv/Scripts/python.exe backtest_real.py --rounds "data/debug_frames_round4" --seg models/card_seg_seg1/best.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import web.server as srv
from blackjack_vision.detector import ROOT, CardSegmenter
from blackjack_vision.upright import upright_card

TILE_W, TILE_H, PER_ROW, PER_SHEET = 150, 230, 12, 60


def tile_of(img: np.ndarray, seg: dict, idx: int) -> np.ndarray | None:
    u = upright_card(img, seg["rbox"])
    if u is None:
        return None
    c = u[0]
    k = (TILE_H - 30) / c.shape[0]
    c = cv2.resize(c, (max(8, min(TILE_W, int(c.shape[1] * k))), TILE_H - 30))
    tile = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    tile[30:, :c.shape[1]] = c
    vals = seg.get("values", [])
    lab = "+".join(v["label"] for v in vals) if vals else "?"
    cv2.putText(tile, f"#{idx} {lab}", (2, 20), 0, 0.5, (0, 0, 255) if vals else (150, 150, 150), 2)
    return tile


def replay_round(round_dir: str, tiles: list, items: list) -> dict:
    s = srv.Session()
    s.apply(dict(view="both", seg_verify=True, seg_value=True, smooth=False, conf=0.5))
    st, ms = Counter(), []
    for f in sorted(glob.glob(round_dir + "/*.json")):
        t = datetime.strptime(Path(f).stem, "%Y%m%d_%H%M%S_%f").timestamp()
        img = cv2.imdecode(np.fromfile(f[:-5] + ".jpg", np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t0 = time.perf_counter()
        r = s.process(img, t=t)
        ms.append((time.perf_counter() - t0) * 1000)
        st["frames"] += 1
        st["frames_with_cards"] += bool(r["segs"])
        st["kept"] += len(r["segs"])
        st["rejected"] += len(r["segs_rejected"])
        for g in r["segs"]:
            n = len(g.get("values", []))
            st["multi_val" if n > 1 else ("one_val" if n else "no_val")] += 1
            tile = tile_of(img, g, len(items))
            if tile is None:
                continue
            tiles.append(tile)
            items.append({"round": Path(round_dir).name, "frame": Path(f).name,
                          "values": [v["label"] for v in g.get("values", [])], "verify": g.get("verify")})
    st = dict(st)
    st["ms_median"] = round(float(np.median(ms)), 1) if ms else 0.0
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", default=str(ROOT / "data" / "debug_frames_round*"), help="影格資料夾 glob")
    ap.add_argument("--seg", default=None, help="改用指定的整張牌分割權重（預設用 models.yaml）")
    ap.add_argument("--out", default=None, help="輸出目錄（預設 reports/backtest_<時間>/）")
    a = ap.parse_args()

    rounds = sorted(glob.glob(a.rounds))
    assert rounds, f"找不到影格資料夾：{a.rounds}"
    if a.seg:
        seg = CardSegmenter(a.seg, imgsz=640)
        srv.get_card_seg = lambda: seg
    assert srv.get_card_seg() is not None, "沒有整張牌分割模型（models/card_seg/）"
    out = Path(a.out) if a.out else ROOT / "reports" / f"backtest_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    first = sorted(glob.glob(rounds[0] + "/*.jpg"))[:10]   # 熱機 10 幀（首次載入、CUDA 初始化、各尺寸首次推論），不計入耗時
    for f in first:
        w = srv.Session(); w.apply(dict(view="both", smooth=False))
        w.process(cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR))

    tiles, items, report = [], [], {}
    for rd in rounds:
        report[Path(rd).name] = replay_round(rd, tiles, items)
        print(Path(rd).name, report[Path(rd).name], flush=True)

    json.dump({"seg": a.seg or "models.yaml", "report": report, "items": items},
              open(out / "items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for p in range(0, len(tiles), PER_SHEET):
        page = tiles[p:p + PER_SHEET]
        page += [np.full_like(tiles[0], 255)] * (-len(page) % PER_ROW)
        sheet = np.vstack([np.hstack(page[i:i + PER_ROW]) for i in range(0, len(page), PER_ROW)])
        cv2.imencode(".jpg", sheet)[1].tofile(str(out / f"sheet_{p // PER_SHEET}.jpg"))   # imwrite 不吃中文路徑

    cols = ["frames", "frames_with_cards", "kept", "rejected", "one_val", "multi_val", "no_val", "ms_median"]
    lines = [f"# 實拍影格回測 {datetime.now():%Y-%m-%d %H:%M}", "", f"分割權重：{a.seg or 'models.yaml'}", "",
             "| 輪次 | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {k} | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |" for k, v in report.items()]
    lines += ["", f"共 {len(items)} 個保留框，總覽圖 {(len(tiles) + PER_SHEET - 1) // PER_SHEET} 張（sheet_N.jpg），"
              "人工判讀：主牌值是否正確、附加牌值是否多出錯牌、「?」是否為模糊／扇形。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("輸出：", out)


if __name__ == "__main__":
    main()
