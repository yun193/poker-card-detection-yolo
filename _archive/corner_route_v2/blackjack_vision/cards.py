"""牌面整合：標籤解析、牌角去重、時間平滑。

資料集標註的是「牌角索引」（點數＋花色），一張牌通常露出兩個對角，
所以同類別、距離夠近的偵測框要合併成一張實體牌。
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}
SUITS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    det_label: str = ""                     # 二次分類修正前，偵測器原本的標籤（空字串＝未經分類）
    cls_conf: float = 0.0                   # 二次分類器對最終標籤的機率

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


def normalize_label(label: str) -> str:
    """'as' / 'AS' / 'As' / '10H' / 'td' → 資料集格式 'As' / '10h' / '10d'。"""
    s = label.strip()
    rank, suit = s[:-1].upper(), s[-1].lower()
    if rank == "T":
        rank = "10"
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"無法解析牌名: {label!r}")
    return rank + suit


def pretty(label: str) -> str:
    n = normalize_label(label)
    return n[:-1] + SUITS[n[-1]]


@dataclass(frozen=True)
class Card:
    label: str
    conf: float
    center: tuple[float, float]
    corners: int        # 合併了幾個牌角
    size: float = 0.0   # 牌角框邊長（追蹤時用來決定配對距離）


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
            cards.append(Card(label, max(d.conf for d in g), (cx, cy), len(g), max(d.size for d in g)))
    return cards


# ---------------------------------------------------------------- 二次分類
def refine_with_classifier(dets: list[Detection], probs: list[dict[str, float]], w_cls: float = 0.6) -> list[Detection]:
    """偵測結果與牌角分類器的機率加權合併。

    分數[l] = (1-w)·偵測信心·[l==偵測標籤] + w·分類機率[l]，取最高者。分類器看的是全解析度放大的牌角，
    權重較高；分類器很有把握地不同意時就改判（例：偵測說 A♠、分類說 K♣ 0.9 → K♣）。
    """
    out = []
    for d, p in zip(dets, probs):
        scores = {k: w_cls * v for k, v in p.items()}
        scores[d.label] = scores.get(d.label, 0.0) + (1 - w_cls) * d.conf
        label = max(scores, key=scores.get)
        out.append(Detection(label, scores[label], d.box, det_label=d.label, cls_conf=p.get(label, 0.0)))
    return out


# ---------------------------------------------------------------- 整張牌框 × 牌角交叉驗證
def _inside(pt, poly) -> bool:
    """射線法判斷點是否在多邊形內。"""
    x, y = pt
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _bbox(poly) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a, b, min_frac: float = 0.2) -> bool:
    """兩個外接框的交集至少佔較小者面積的 min_frac（只擦邊不算，避免牌碰到畫框就把畫框串進來）。"""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return False
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return small > 0 and ix * iy >= min_frac * small


def _area(poly) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]))) / 2


def corner_owners(segs: list[dict], dets: list[Detection]) -> list[int | None]:
    """每個牌角偵測歸屬哪個整張牌框（索引）：中心在多邊形內，或框四角一半以上在內；
    同時落在多個框時歸給面積最小者（牌拿在書櫃前，牌角歸牌不歸書櫃）。不在任何框內為 None。"""
    polys = [[tuple(p) for p in s["poly"]] for s in segs]
    areas = [_area(p) for p in polys]
    out: list[int | None] = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        hits = [i for i, poly in enumerate(polys)
                if _inside(((x1 + x2) / 2, (y1 + y2) / 2), poly)
                or sum(_inside(c, poly) for c in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))) >= 2]
        out.append(min(hits, key=lambda i: areas[i]) if hits else None)
    return out


def seg_card_values(corner_preds: list[tuple[str, float]], corner_dets: list[Detection],
                    min_conf: float = 0.6, det_min_conf: float | None = None) -> list[dict]:
    """一框多牌值：分割常把疊在一起的牌框成一塊，但每張牌的牌角通常各自看得到。
    合併 B-2 兩個裁角各自的結果（corner_preds，≥ min_conf）與框內的牌角偵測（corner_dets，≥ det_min_conf，
    預設同 min_conf；已歸屬到通過驗證的框，可放寬）。同牌值取最高信心；同點數不同花色只留信心高者
    （實測多是同一張 A♦ 被另一路讀成 A♥，真的兩張同點數疊在同一框很少見）。依信心由高到低回傳 [{"label", "conf", "src"}]。"""
    det_min_conf = min_conf if det_min_conf is None else det_min_conf
    cands = [(lab, p, "b2") for lab, p in corner_preds if p >= min_conf] + \
            [(d.label, d.conf, "corner") for d in corner_dets if d.conf >= det_min_conf]
    best: dict[str, dict] = {}   # 以點數為鍵
    for lab, p, src in sorted(cands, key=lambda c: -c[1]):
        rank = lab[:-1]
        if rank not in best:
            best[rank] = {"label": lab, "conf": round(p, 3), "src": src}
    return sorted(best.values(), key=lambda v: -v["conf"])


def verify_segs(segs: list[dict], dets: list[Detection], memory=(), max_area_ratio: float = 1.5
                ) -> tuple[list[dict], list[dict]]:
    """用牌角偵測驗證整張牌框，濾掉門、畫框等長方形背景。三種通過方式（寫入 s["verify"]）：
    corner：框內有牌角偵測（中心在多邊形內，或框四角一半以上在內）。一個牌角只算給含它的最小框，
            牌拿在書櫃前時，牌角歸那張牌，書櫃不會跟著通過。
    chain ：與已通過的框明顯重疊，且面積不超過對方的 max_area_ratio 倍（扇形後排的窄條；大片的門不會被串進來）。
    memory：與最近通過的框明顯重疊且面積條件相同（動態模糊、牌角暫時沒偵測到的牌）。
    memory 為 [(bbox, area), ...]。每個 seg 另寫入 corners（命中數）。回傳 (保留, 剔除)。"""
    info = []
    for s in segs:
        poly = [tuple(p) for p in s["poly"]]
        s["corners"] = 0
        info.append((s, _bbox(poly), _area(poly)))
    for owner in corner_owners(segs, dets):
        if owner is not None:
            segs[owner]["corners"] += 1
    for s, _, _ in info:
        s["verify"] = "corner" if s["corners"] else None

    def near(bb, area, refs):
        return any(_overlap(bb, rb) and area <= max_area_ratio * ra for rb, ra in refs)

    changed = True
    while changed:   # 串接可能一層層傳下去（扇形）
        changed = False
        ok = [(bb, a) for s, bb, a in info if s["verify"]]
        for s, bb, a in info:
            if not s["verify"] and near(bb, a, ok):
                s["verify"], changed = "chain", True
    for s, bb, a in info:
        if not s["verify"] and near(bb, a, memory):
            s["verify"] = "memory"
    kept = [s for s, _, _ in info if s["verify"]]
    return kept, [s for s, _, _ in info if not s["verify"]]


def seg_memory_entry(segs: list[dict]) -> list[tuple]:
    """把本幀通過驗證的框轉成 verify_segs 的 memory 條目（只收 corner／chain，避免 memory 自我延續）。"""
    out = []
    for s in segs:
        if s.get("verify") in ("corner", "chain"):
            poly = [tuple(p) for p in s["poly"]]
            out.append((_bbox(poly), _area(poly)))
    return out


# ---------------------------------------------------------------- 多尺度集成
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fuse_scales(per_scale: list[list[Detection]], conf: float = 0.45, iou_thr: float = 0.45) -> list[Detection]:
    """合併多個推論尺度的偵測：同位置（IoU 夠高、不分類別）的框聚成一群，
    以各類別信心度總和投票決定標籤，分數＝該標籤在各尺度最高信心度的平均。

    只在單一尺度出現、或各尺度標籤不一致的框，分數會被拉低，用來壓掉偶發誤判。
    """
    n = len(per_scale)
    if n == 1:
        return [d for d in per_scale[0] if d.conf >= conf]
    items = sorted(((d, si) for si, ds in enumerate(per_scale) for d in ds), key=lambda x: -x[0].conf)
    clusters: list[list[tuple[Detection, int]]] = []
    for d, si in items:
        for cl in clusters:
            if iou(cl[0][0].box, d.box) >= iou_thr:
                cl.append((d, si))
                break
        else:
            clusters.append([(d, si)])
    out = []
    for cl in clusters:
        votes: dict[str, float] = {}
        for d, _ in cl:
            votes[d.label] = votes.get(d.label, 0.0) + d.conf
        label = max(votes, key=votes.get)
        best = [0.0] * n
        for d, si in cl:
            if d.label == label:
                best[si] = max(best[si], d.conf)
        score = sum(best) / n
        if score < conf:
            continue
        ws = [(d, d.conf) for d, _ in cl if d.label == label]
        tw = sum(w for _, w in ws)
        box = tuple(sum(d.box[k] * w for d, w in ws) / tw for k in range(4))
        out.append(Detection(label, score, box))
    return out


# ---------------------------------------------------------------- 位置追蹤
@dataclass
class Track:
    id: int
    center: tuple[float, float]
    size: float
    votes: dict[str, float]
    label: str = ""     # 目前採用的標籤（有遲滯，不是每幀都取票數最高者）
    hits: int = 1
    missed: int = 0

    def vote(self, label: str, conf: float, hysteresis: float):
        self.votes[label] = self.votes.get(label, 0.0) + conf
        best = max(self.votes, key=self.votes.get)
        if not self.label or (best != self.label and self.votes[best] >= hysteresis * self.votes.get(self.label, 0.0)):
            self.label = best

    @property
    def score(self) -> float:
        """目前標籤佔總票數的比例（1＝歷史完全一致）。"""
        return self.votes[self.label] / sum(self.votes.values())


class CardTracker:
    """依位置追蹤每張實體牌，標籤用「信心度加權、隨時間衰減」的投票決定。

    同一位置先前一直是 K♣，偶爾幾幀被認成 A♠ 時，A♠ 的票數不足以翻盤；新標籤的票數
    要超過目前標籤的 hysteresis 倍才換。若真的換了牌，約 5~7 幀（24 FPS 下 0.2~0.3 秒）後勝出。
    """

    def __init__(self, memory: int = 8, decay: float = 0.9, min_hits: int = 3, match_factor: float = 5.0,
                 hysteresis: float = 1.3):
        self.memory, self.decay, self.min_hits, self.match_factor = memory, decay, min_hits, match_factor
        self.hysteresis = hysteresis
        self.tracks: list[Track] = []
        self._next = 0

    def update(self, cards: list[Card]) -> list[tuple[Card, Track]]:
        """回傳每張輸入牌對應的追蹤軌跡（軌跡的 label 為平滑後標籤）。"""
        pairs = []
        for ci, c in enumerate(cards):
            for ti, t in enumerate(self.tracks):
                d = math.dist(c.center, t.center)
                if d < self.match_factor * max(c.size, t.size, 1.0):
                    pairs.append((d * (1.0 if t.label == c.label else 2.0), ci, ti))
        pairs.sort()
        used_c, used_t, matched = set(), set(), {}
        for _, ci, ti in pairs:
            if ci not in used_c and ti not in used_t:
                used_c.add(ci); used_t.add(ti); matched[ci] = ti
        for t in self.tracks:
            for k in t.votes:
                t.votes[k] *= self.decay
        out = []
        for ci, c in enumerate(cards):
            if ci in matched:
                t = self.tracks[matched[ci]]
                t.vote(c.label, c.conf, self.hysteresis)
                t.center, t.size, t.hits, t.missed = c.center, c.size, t.hits + 1, 0
            else:
                t = Track(self._next, c.center, c.size, {c.label: c.conf}, label=c.label)
                self._next += 1
                self.tracks.append(t)
            out.append((c, t))
        alive = {id(t) for _, t in out}
        for t in self.tracks:
            if id(t) not in alive:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.memory]
        return out

    def confirmed(self) -> list[Track]:
        """已確認（出現 ≥ min_hits 幀、且最近 2 幀內還看得到）的牌。"""
        return [t for t in self.tracks if t.hits >= self.min_hits and t.missed <= 2]

    def reset(self):
        self.tracks.clear()


