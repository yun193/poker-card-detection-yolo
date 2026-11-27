"""整張牌路線的後處理：牌名解析、背景框過濾、一框多牌值、跨幀追蹤投票。

舊的牌角路線（牌角偵測、多尺度融合、牌角合併、牌角追蹤）已封存到 _archive/corner_route_v2/。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}
SUITS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


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


# ---------------------------------------------------------------- 幾何
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


def card_width(seg: dict) -> float:
    """旋轉框短邊（牌寬，像素）。"""
    p = seg["rbox"]
    return min(math.dist(p[0], p[1]), math.dist(p[1], p[2]))


# ---------------------------------------------------------------- 背景框過濾
MIN_BORDER, MAX_COLOR = 0.27, 0.2


def looks_like_card(face: dict) -> bool:
    """外觀像不像撲克牌：四周有白邊（邊框白底 ≥ MIN_BORDER）且幾乎沒有紅色以外的彩色（< MAX_COLOR）。
    實拍：海報邊框白底 ≤ 0.25、真牌（含窄條、人頭牌、牌背）≥ 0.27；亮色書櫃彩色 0.29～0.43、真牌 ≤ 0.07。
    value_cls 對海報這類沒見過的東西會很有把握地亂猜牌值（0.8 以上），所以外觀檢查必須獨立於牌值信心。"""
    return face.get("border", 0.0) >= MIN_BORDER and face.get("color", 1.0) < MAX_COLOR


def is_card_evidence(value: dict, min_white: float = 0.56, min_conf: float = 0.7) -> bool:
    """這個框本身算不算一張牌：外觀像牌（looks_like_card），至少一個角不是 none（value 有 label），
    且牌面夠白或牌值很有把握。"""
    return "label" in value and looks_like_card(value) and \
        (value.get("white", 0.0) >= min_white or value.get("top", 0.0) >= min_conf)


def verify_segs(segs: list[dict], evidence: list[bool], linkable: list[bool] | None = None, memory=(),
                max_area_ratio: float = 1.5) -> tuple[list[dict], list[dict]]:
    """濾掉門、海報、書櫃等長方形背景。三種通過方式（寫入 s["verify"]）：
    card  ：框本身有牌的證據（is_card_evidence）。
    chain ：與已通過的框明顯重疊，且面積不超過對方的 max_area_ratio 倍（扇形後排的窄條；大片的門不會被串進來）。
    memory：與最近通過的框明顯重疊且面積條件相同（動態模糊、這幀證據不足的牌）。
    chain／memory 另要求 linkable（外觀像牌，looks_like_card）：否則海報只要某一幀被當成牌，就會靠記憶一直留著。
    memory 為 [(bbox, area), ...]。回傳 (保留, 剔除)。"""
    linkable = linkable if linkable is not None else [True] * len(segs)
    info = []
    for s, ev, ok in zip(segs, evidence, linkable):
        poly = [tuple(p) for p in s["poly"]]
        s["verify"] = "card" if ev else None
        info.append((s, _bbox(poly), _area(poly), ok))

    def near(bb, area, refs):
        return any(_overlap(bb, rb) and area <= max_area_ratio * ra for rb, ra in refs)

    changed = True
    while changed:   # 串接可能一層層傳下去（扇形）
        changed = False
        ok = [(bb, a) for s, bb, a, _ in info if s["verify"]]
        for s, bb, a, linkable in info:
            if not s["verify"] and linkable and near(bb, a, ok):
                s["verify"], changed = "chain", True
    for s, bb, a, linkable in info:
        if not s["verify"] and linkable and near(bb, a, memory):
            s["verify"] = "memory"
    info = [(s, bb, a) for s, bb, a, _ in info]
    kept = [s for s, _, _ in info if s["verify"]]
    return kept, [s for s, _, _ in info if not s["verify"]]


def seg_memory_entry(segs: list[dict]) -> list[tuple]:
    """把本幀通過驗證的框轉成 verify_segs 的 memory 條目（只收 card／chain，避免 memory 自我延續）。"""
    out = []
    for s in segs:
        if s.get("verify") in ("card", "chain"):
            poly = [tuple(p) for p in s["poly"]]
            out.append((_bbox(poly), _area(poly)))
    return out


# ---------------------------------------------------------------- 一框多牌值
def seg_card_values(preds: list[tuple[str, float, str]], min_conf: float = 0.6) -> list[dict]:
    """合併一個框的所有牌角讀值（兩個角＋牌內切片），preds 為 [(牌名, 信心, 來源)]。
    同點數只留信心最高者（實測多是同一張 A♦ 被另一角讀成 A♥，真的兩張同點數疊在同一框很少見）。
    依信心由高到低回傳 [{"label", "conf", "src"}]。"""
    best: dict[str, dict] = {}   # 以點數為鍵
    for lab, p, src in sorted(preds, key=lambda c: -c[1]):
        if p >= min_conf and lab[:-1] not in best:
            best[lab[:-1]] = {"label": lab, "conf": round(p, 3), "src": src}
    return sorted(best.values(), key=lambda v: -v["conf"])


# ---------------------------------------------------------------- 跨幀追蹤投票
@dataclass
class SegTrack:
    id: int
    center: tuple[float, float]
    size: float
    t: float                                   # 最後一次看到的時間（秒）
    votes: dict[str, float] = field(default_factory=dict)
    seen: dict[str, int] = field(default_factory=dict)   # 每個牌值出現過的幀數（附加牌值要連續出現才列出）
    label: str = ""                            # 目前採用的主牌值（有遲滯）
    hits: int = 1

    def score(self) -> float:
        """主牌值佔總票數的比例（1＝歷史完全一致）。"""
        tot = sum(self.votes.values())
        return self.votes.get(self.label, 0.0) / tot if tot else 0.0


class SegTracker:
    """依位置追蹤每個整張牌框，牌值用「信心加權、隨時間衰減」的投票決定（規則式，無需訓練）。

    - 時間一律以秒計（不是幀數），離線重播傳入原始時間戳。
    - 糊掉的幀信心低、票也少，沿用前幾幀的結果；偶發的錯值票數不足，不會翻盤。
    - 新牌值要超過目前牌值的 hysteresis 倍才換主牌值。
    - 附加牌值（一框多張，主要來自牌內切片）要達主牌值票數的 extra_frac、且出現過 ≥ extra_seen 幀才列出：
      實拍單幀切片的附加牌值只有約一半正確（牌中央的大花色會被讀成牌角），錯的多半只出現一兩幀。
    """

    def __init__(self, half_life: float = 0.6, memory_s: float = 1.0, match_factor: float = 1.0,
                 hysteresis: float = 1.3, extra_frac: float = 0.3, extra_seen: int = 3, min_hits: int = 3,
                 min_vote: float = 0.3):
        self.half_life, self.memory_s, self.match_factor = half_life, memory_s, match_factor
        self.hysteresis, self.extra_frac, self.extra_seen = hysteresis, extra_frac, extra_seen
        self.min_hits, self.min_vote = min_hits, min_vote
        self.tracks: list[SegTrack] = []
        self._next = 0

    def reset(self):
        self.tracks.clear()

    def update(self, segs: list[dict], t: float) -> list[SegTrack]:
        """segs 每個需有 center、rbox、reads（[(牌名, 信心)]，這一幀所有讀值）。回傳每個 seg 對應的軌跡。"""
        pairs = []
        for si, s in enumerate(segs):
            w = card_width(s)
            for ti, tr in enumerate(self.tracks):
                d = math.dist(s["center"], tr.center)
                if d < self.match_factor * max(w, tr.size, 1.0):
                    pairs.append((d, si, ti))
        pairs.sort()
        used_s, used_t, matched = set(), set(), {}
        for _, si, ti in pairs:
            if si not in used_s and ti not in used_t:
                used_s.add(si); used_t.add(ti); matched[si] = ti
        out = []
        for si, s in enumerate(segs):
            if si in matched:
                tr = self.tracks[matched[si]]
                k = 0.5 ** (max(0.0, t - tr.t) / self.half_life)
                tr.votes = {lab: v * k for lab, v in tr.votes.items() if v * k > 1e-3}
                tr.seen = {lab: n for lab, n in tr.seen.items() if lab in tr.votes}
                tr.center, tr.size, tr.t, tr.hits = tuple(s["center"]), card_width(s), t, tr.hits + 1
            else:
                tr = SegTrack(self._next, tuple(s["center"]), card_width(s), t)
                self._next += 1
                self.tracks.append(tr)
            for lab, p in s.get("reads", []):
                tr.votes[lab] = tr.votes.get(lab, 0.0) + p
            for lab in {lab for lab, _ in s.get("reads", [])}:
                tr.seen[lab] = tr.seen.get(lab, 0) + 1
            if tr.votes:
                best = max(tr.votes, key=tr.votes.get)
                if not tr.label or tr.label not in tr.votes or \
                        (best != tr.label and tr.votes[best] >= self.hysteresis * tr.votes[tr.label]):
                    tr.label = best
            out.append(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.t <= self.memory_s]
        return out

    def labels(self, tr: SegTrack) -> list[str]:
        """軌跡的牌值清單：主牌值＋票數夠多、點數不同的附加牌值（一框多張）。票數不足 min_vote 時為空。"""
        if not tr.label or tr.votes.get(tr.label, 0.0) < self.min_vote:
            return []
        top = tr.votes[tr.label]
        out, ranks = [tr.label], {tr.label[:-1]}
        for lab, v in sorted(tr.votes.items(), key=lambda kv: -kv[1]):
            if lab[:-1] not in ranks and v >= self.extra_frac * top and v >= self.min_vote \
                    and tr.seen.get(lab, 0) >= self.extra_seen:
                out.append(lab); ranks.add(lab[:-1])
        return out

    def confirmed(self, t: float) -> list[SegTrack]:
        """已確認（出現 ≥ min_hits 幀、且 0.3 秒內還看得到）的軌跡。"""
        return [tr for tr in self.tracks if tr.hits >= self.min_hits and t - tr.t <= 0.3]

    def hand(self, t: float) -> list[str]:
        """已確認軌跡的牌值彙總成手牌（去重，依票數排序）。"""
        best: dict[str, float] = {}
        for tr in self.confirmed(t):
            for lab in self.labels(tr):
                best[lab] = max(best.get(lab, 0.0), tr.votes[lab])
        return sorted(best, key=lambda k: -best[k])


class HandSmoother:
    """手牌層級的時間平滑：牌值在最近 win 秒內出現的時間比例 ≥ on 才加入手牌，降到 off 以下才移除。
    追蹤軌跡會因分割框在扇形中分合而重建，手牌每幀跳動；錄影實測（9 FPS、7 張扇形）：
    手牌變動 75 → 36 次、有錯牌的幀 87 → 67，召回率 37% → 39%。"""

    def __init__(self, win: float = 1.0, on: float = 0.5, off: float = 0.2):
        self.win, self.on, self.off = win, on, off
        self.hist: list[tuple[float, frozenset]] = []
        self.cur: set[str] = set()

    def reset(self):
        self.hist.clear()
        self.cur.clear()

    def update(self, hand: list[str], t: float) -> list[str]:
        self.hist = [(tt, h) for tt, h in self.hist if t - tt <= self.win] + [(t, frozenset(hand))]
        ends = [tt for tt, _ in self.hist[1:]] + [t + 0.1]   # 每個手牌持續到下一幀；最新一幀算 0.1 秒
        frac: dict[str, float] = {}
        for (ta, h), tb in zip(self.hist, ends):
            for lab in h:
                frac[lab] = frac.get(lab, 0.0) + tb - ta
        tot = max(ends[-1] - self.hist[0][0], self.win / 2)   # 剛開始歷史很短時，不讓單一幀就佔滿比例
        frac = {k: v / tot for k, v in frac.items()}
        self.cur = {k for k, v in frac.items() if v >= self.on} | {k for k in self.cur if frac.get(k, 0.0) >= self.off}
        order = list(hand) + sorted(self.cur - set(hand))
        return [k for k in order if k in self.cur]
