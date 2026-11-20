"""即時攝影機：撲克牌辨識＋21 點勝率提示。

用法：
    python app.py                 # 預設第一個可用模型、攝影機 0
    python app.py --model yolo26 --cam 1 --decks 6

按鍵：
    1-9  切換模型（依 models.yaml 順序）    S  存截圖到 data/captures（檔名為目前辨識結果）
    N    本局結束：桌上牌計入牌靴（算牌）    R  重置牌靴
    F    上下對調莊家／玩家區               [ ]  移動分區線
    Q / Esc 離開
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from blackjack_vision.advisor import Advisor
from blackjack_vision.blackjack import Rules
from blackjack_vision.cards import StableTable, group_corners, split_zones
from blackjack_vision.detector import ROOT, build_detector, load_registry
from blackjack_vision.overlay import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--decks", type=int, default=6)
    ap.add_argument("--h17", action="store_true", help="莊家軟 17 要牌")
    ap.add_argument("--no-peek", action="store_true", help="莊家不偷看底牌（歐式）")
    args = ap.parse_args()

    registry = load_registry()
    if not registry:
        raise SystemExit("models.yaml 裡沒有找到任何已下載的權重，請先把 Kaggle 輸出放到 models/ 底下")
    names = list(registry)
    cur = args.model if args.model in registry else names[0]
    det = build_detector(registry[cur], conf=args.conf)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟攝影機 {args.cam}")

    advisor = Advisor(Rules(decks=args.decks, dealer_hits_soft17=args.h17, dealer_peeks=not args.no_peek))
    table = StableTable()
    split, dealer_top = 0.45, True
    fps, t_prev = 0.0, time.perf_counter()
    cap_dir = ROOT / "data" / "captures"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw = frame.copy()
        t0 = time.perf_counter()
        dets = det(frame)
        infer_ms = (time.perf_counter() - t0) * 1000
        cards = group_corners(dets)
        h = frame.shape[0]
        d_cards, p_cards = split_zones(cards, h, split, dealer_top)
        dealer, player = table.update(d_cards, p_cards)
        adv = advisor.advise(dealer, player)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        status = (f"[{cur}] 推論 {infer_ms:.0f}ms  FPS {fps:.1f}  已記牌 {len(advisor.shoe.seen)} 張  "
                  f"真數 {advisor.shoe.hi_lo_true_count():+.1f}  | 1-{len(names)}換模型 S截圖 N下一局 R重置 F對調 Q離開")
        out = draw(frame, dets, cards, int(h * split), dealer, player, adv, status)
        cv2.imshow("Blackjack Vision", out)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if ord("1") <= k <= ord("9") and k - ord("1") < len(names):
            cur = names[k - ord("1")]
            det = build_detector(registry[cur], conf=args.conf)
            table.reset()
        elif k == ord("s"):
            cap_dir.mkdir(parents=True, exist_ok=True)
            tag = "-".join(sorted(dealer + player)) or "none"
            path = cap_dir / f"{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imencode(".jpg", raw)[1].tofile(str(path))
            print("已存：", path)
        elif k == ord("n"):
            advisor.shoe.end_round(dealer + player)
            table.reset()
        elif k == ord("r"):
            advisor.shoe.reset()
        elif k == ord("f"):
            dealer_top = not dealer_top
        elif k == ord("["):
            split = max(0.1, split - 0.05)
        elif k == ord("]"):
            split = min(0.9, split + 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
