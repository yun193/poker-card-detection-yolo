"""畫面疊加：偵測框、分區線、21 點資訊面板（PIL 繪製中文）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .advisor import Advice
from .cards import Card, Detection, pretty

_FONT_CANDIDATES = ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjhbd.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
_SYMBOL_CANDIDATES = ["C:/Windows/Fonts/seguisym.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_SUIT_CHARS = set("♣♦♥♠")
_fonts: dict[tuple, ImageFont.ImageFont] = {}


def font(size: int, symbol: bool = False):
    key = (size, symbol)
    if key not in _fonts:
        cands = _SYMBOL_CANDIDATES if symbol else _FONT_CANDIDATES
        path = next((p for p in cands if Path(p).exists()), None)
        _fonts[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _fonts[key]


def text(g: ImageDraw.ImageDraw, xy, s: str, size: int, fill, **kw):
    """逐段繪製：花色字元改用符號字型（中文字型多半缺 ♠♥♦♣）。"""
    x, y = xy
    for ch in s:
        f = font(size, ch in _SUIT_CHARS)
        g.text((x, y), ch, font=f, fill=fill, **kw)
        x += g.textlength(ch, font=f)


ACTION_LABEL = {"stand": "停牌", "hit": "要牌", "double": "加倍"}
GREEN, YELLOW, RED, WHITE = (60, 220, 60), (0, 210, 255), (60, 60, 230), (255, 255, 255)


def draw(frame: np.ndarray, dets: list[Detection], cards: list[Card], split_y: int,
         dealer: list[str], player: list[str], adv: Advice, status: str) -> np.ndarray:
    h, w = frame.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    cv2.line(frame, (0, split_y), (w, split_y), YELLOW, 1)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    g = ImageDraw.Draw(img, "RGBA")
    for c in cards:
        x, y = map(int, c.center)
        text(g, (x - 20, y - 14), pretty(c.label), 24, (255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    g.text((8, split_y - 26), "莊家區 ↑", font=font(18), fill=(255, 210, 0))
    g.text((8, split_y + 4), "玩家區 ↓", font=font(18), fill=(255, 210, 0))

    lines = [
        f"莊家：{' '.join(pretty(l) for l in dealer) or '-'}",
        f"玩家：{' '.join(pretty(l) for l in player) or '-'}"
        + (f"  = {'軟' if adv.player_soft else ''}{adv.player_total}" if adv.player_total else ""),
    ]
    if adv.strategy:
        lines.append(f"基本策略：{adv.strategy_name}")
    for k, o in adv.odds.items():
        mark = " ★" if k == adv.best_ev_action else ""
        mult = "（注×2）" if k == "double" else ""
        q = o.scaled(0.5) if k == "double" else o
        lines.append(f"{ACTION_LABEL[k]}{mult} 勝{q.win:.0%} 和{q.push:.0%} 負{q.lose:.0%} EV{o.ev:+.2f}{mark}")
    if adv.note:
        lines.append(adv.note)
    pw, lh = 470, 28
    ph = lh * len(lines) + 16
    x0 = w - pw - 8
    g.rectangle((x0, 8, x0 + pw, 8 + ph), fill=(0, 0, 0, 160))
    for i, t in enumerate(lines):
        text(g, (x0 + 10, 14 + i * lh), t, 20, (255, 255, 255))
    g.rectangle((0, h - 30, w, h), fill=(0, 0, 0, 160))
    g.text((8, h - 27), status, font=font(16), fill=(200, 200, 200))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
