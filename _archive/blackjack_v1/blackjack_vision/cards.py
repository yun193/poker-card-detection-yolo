"""牌面整合：標籤解析、牌角去重、莊家／玩家分區、時間平滑。

資料集標註的是「牌角索引」（點數＋花色），一張牌通常露出兩個對角，
所以同類別、距離夠近的偵測框要合併成一張實體牌。
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

from .blackjack import RANK_VALUES

SUITS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


def normalize_label(label: str) -> str:
    """'as' / 'AS' / 'As' / '10H' → 資料集格式 'As' / '10h'。"""
    s = label.strip()
    rank, suit = s[:-1].upper(), s[-1].lower()
    if rank == "T":
        rank = "10"
    if rank not in RANK_VALUES or suit not in SUITS:
        raise ValueError(f"無法解析牌名: {label!r}")
    return rank + suit


def card_value(label: str) -> int:
    return RANK_VALUES[normalize_label(label)[:-1]]


def pretty(label: str) -> str:
    n = normalize_label(label)
    return n[:-1] + SUITS[n[-1]]


@dataclass(frozen=True)
class Card:
    label: str
    conf: float
    center: tuple[float, float]
    corners: int  # 合併了幾個牌角


def group_corners(dets: list[Detection], dist_factor: float = 6.0) -> list[Card]:
    """同類別、中心距離 < dist_factor × 牌角框邊長 的偵測合併為一張牌。"""
    cards: list[Card] = []
    by_label: dict[str, list[Detection]] = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)
    for label, ds in by_label.items():
        parent = list(range(len(ds)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                lim = dist_factor * max(ds[i].size, ds[j].size)
                if math.dist(ds[i].center, ds[j].center) < lim:
                    parent[find(i)] = find(j)
        groups: dict[int, list[Detection]] = {}
        for i, d in enumerate(ds):
            groups.setdefault(find(i), []).append(d)
        for g in groups.values():
            cx = sum(d.center[0] for d in g) / len(g)
            cy = sum(d.center[1] for d in g) / len(g)
            cards.append(Card(label, max(d.conf for d in g), (cx, cy), len(g)))
    return cards


def split_zones(cards: list[Card], frame_h: int, split_ratio: float = 0.5,
                dealer_on_top: bool = True) -> tuple[list[Card], list[Card]]:
    """依水平分隔線分成 (莊家, 玩家)。"""
    line = frame_h * split_ratio
    top = [c for c in cards if c.center[1] < line]
    bottom = [c for c in cards if c.center[1] >= line]
    return (top, bottom) if dealer_on_top else (bottom, top)


class StableTable:
    """最近 N 幀投票：某張牌（含張數）需在 ≥ ratio 比例的幀出現才算穩定。"""

    def __init__(self, window: int = 10, ratio: float = 0.6):
        self.window = window
        self.ratio = ratio
        self.hist: deque[tuple[Counter, Counter]] = deque(maxlen=window)

    def update(self, dealer: list[Card], player: list[Card]) -> tuple[list[str], list[str]]:
        self.hist.append((Counter(c.label for c in dealer), Counter(c.label for c in player)))
        return self._stable(0), self._stable(1)

    def _stable(self, idx: int) -> list[str]:
        need = max(1, math.ceil(self.ratio * len(self.hist)))
        labels = set().union(*(h[idx].keys() for h in self.hist)) if self.hist else set()
        out: list[str] = []
        for lab in sorted(labels):
            counts = sorted((h[idx][lab] for h in self.hist), reverse=True)
            out += [lab] * counts[need - 1]
        return out

    def reset(self):
        self.hist.clear()
