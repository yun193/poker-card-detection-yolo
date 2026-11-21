"""實拍影格回測：用測試台的完整流程（整張牌分割＋背景框過濾＋轉正讀牌角＋牌內切片＋追蹤投票），
依原始時間戳重播 DEBUG 監看存檔（data/debug_frames*/）或錄製（data/recordings/*/），產出統計與轉正牌面總覽，供人工判讀。

注意：DEBUG 監看每 2 秒才存一幀，追蹤投票（記憶 1 秒）在這種資料上等於逐幀獨立；
要驗證追蹤投票，請用測試台的「錄製」功能存連續影格。

輸出（--out，預設 reports/backtest_<時間>/）：
  summary.md     每個資料夾的統計（框數、剔除數、單／多／無牌值、手牌變動次數、每幀耗時中位數）
  items.json     每個保留框：資料夾、影格、牌值、驗證方式、切片讀值
  sheet_N.jpg    轉正牌面總覽（每張 #編號＋牌值，紅字有牌值、灰字「?」）

用法：
  .venv/Scripts/python.exe backtest_real.py
  .venv/Scripts/python.exe backtest_real.py --folders "data/recordings/*" --seg models/card_seg_seg1/best.pt
  .venv/Scripts/python.exe backtest_real.py --no-scan --no-smooth
"""
from __future__ import annotations

import argparse
import glob
import json
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
    lab = "+".join(seg["labels"]) or "?"
    cv2.putText(tile, f"#{idx} {lab}", (2, 20), 0, 0.5, (0, 0, 255) if seg["labels"] else (150, 150, 150), 2)
    return tile


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folders", default=str(ROOT / "data" / "debug_frames_round*"), help="影格資料夾 glob")
    ap.add_argument("--seg", default=None, help="改用指定的整張牌分割權重（預設用 models.yaml）")
    ap.add_argument("--no-scan", action="store_true", help="關閉牌內切片")
    ap.add_argument("--no-smooth", action="store_true", help="關閉追蹤投票（逐幀獨立）")
    ap.add_argument("--out", default=None, help="輸出目錄（預設 reports/backtest_<時間>/）")
    a = ap.parse_args()

    folders = sorted(p for p in glob.glob(a.folders) if Path(p).is_dir())
    assert folders, f"找不到影格資料夾：{a.folders}"
    if a.seg:
        seg = CardSegmenter(a.seg, imgsz=640)
        srv.get_card_seg = lambda: seg
    assert srv.get_card_seg() is not None and srv.get_value_cls() is not None, "缺少 models/card_seg 或 models/value_cls"
    out = Path(a.out) if a.out else ROOT / "reports" / f"backtest_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    settings = {"scan": not a.no_scan, "smooth": not a.no_smooth}

    for f in sorted(glob.glob(folders[0] + "/*.jpg"))[:10]:   # 熱機 10 幀（首次載入、CUDA 初始化），不計入耗時
        srv.Session().process(srv.imread(Path(f)))

    tiles, items, report = [], [], {}

    def collect(f, img, r):
        for g in r["segs"]:
            tile = tile_of(img, g, len(items))
            if tile is None:
                continue
            tiles.append(tile)
            items.append({"folder": f.parent.name, "frame": f.name, "labels": g["labels"], "verify": g.get("verify"),
                          "scan": g.get("scan", [])})

    for fd in folders:
        report[Path(fd).name] = srv.replay_folder(Path(fd), settings, collect)
        print(Path(fd).name, report[Path(fd).name], flush=True)

    json.dump({"seg": a.seg or "models.yaml", "settings": settings, "report": report, "items": items},
              open(out / "items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for p in range(0, len(tiles), PER_SHEET):
        page = tiles[p:p + PER_SHEET]
        page += [np.full_like(tiles[0], 255)] * (-len(page) % PER_ROW)
        sheet = np.vstack([np.hstack(page[i:i + PER_ROW]) for i in range(0, len(page), PER_ROW)])
        cv2.imencode(".jpg", sheet)[1].tofile(str(out / f"sheet_{p // PER_SHEET}.jpg"))   # imwrite 不吃中文路徑

    cols = ["frames", "kept", "rejected", "single", "multi", "unknown", "hand_changes", "ms_median"]
    lines = [f"# 實拍影格回測 {datetime.now():%Y-%m-%d %H:%M}", "",
             f"分割權重：{a.seg or 'models.yaml'}　牌內切片：{settings['scan']}　追蹤投票：{settings['smooth']}", "",
             "| 資料夾 | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {k} | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |" for k, v in report.items()]
    lines += ["", f"共 {len(items)} 個保留框，總覽圖 {(len(tiles) + PER_SHEET - 1) // PER_SHEET} 張（sheet_N.jpg），"
              "人工判讀：主牌值是否正確、附加牌值是否多出錯牌、「?」是否為模糊／扇形。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("輸出：", out)


if __name__ == "__main__":
    main()
