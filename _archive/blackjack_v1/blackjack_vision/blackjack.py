"""21 點邏輯：手牌點數、基本策略、勝率／期望值計算。

點數表示：A=1、2..9、10（含 J/Q/K）。牌靴以長度 10 的 tuple 表示各點數剩餘張數
（index 0 = A，index 9 = 10 點牌）。

勝率計算採「莊家分佈固定」近似：莊家最終點數分佈只扣除桌上已知的牌，
不再隨玩家後續要的牌變動（多副牌下誤差極小，換來即時運算）。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

RANK_VALUES = {"A": 1, "J": 10, "Q": 10, "K": 10, "10": 10,
               **{str(i): i for i in range(2, 10)}}


@dataclass(frozen=True)
class Rules:
    decks: int = 6
    dealer_hits_soft17: bool = False   # S17
    dealer_peeks: bool = True          # 莊家明牌 A/10 時先偷看，沒 BJ 才繼續
    blackjack_payout: float = 1.5
    double_after_split: bool = True


def full_shoe(decks: int) -> tuple[int, ...]:
    return tuple([4 * decks] * 9 + [16 * decks])


def remove_cards(shoe: tuple[int, ...], values) -> tuple[int, ...]:
    counts = list(shoe)
    for v in values:
        if counts[v - 1] > 0:
            counts[v - 1] -= 1
    return tuple(counts)


def hand_value(values) -> tuple[int, bool]:
    """回傳 (最佳點數, 是否軟牌)。"""
    total = sum(values)
    if 1 in values and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_blackjack(values) -> bool:
    return len(values) == 2 and hand_value(values)[0] == 21


# ---------------------------------------------------------------- 莊家分佈
OUTCOMES = (17, 18, 19, 20, 21, "bust", "bj")


def _add(total: int, soft: bool, v: int) -> tuple[int, bool]:
    total += v
    if v == 1 and total + 10 <= 21:
        return total + 10, True
    if total > 21 and soft:
        return total - 10, False
    return total, soft


@lru_cache(maxsize=200_000)
def _dealer_draw(total: int, soft: bool, shoe: tuple[int, ...], h17: bool) -> tuple[float, ...]:
    """莊家從 (total, soft) 開始補牌，回傳 OUTCOMES[:6] 的機率。"""
    if total > 21:
        return (0, 0, 0, 0, 0, 1.0)
    if total >= 17 and not (h17 and soft and total == 17):
        out = [0.0] * 6
        out[total - 17] = 1.0
        return tuple(out)
    n = sum(shoe)
    acc = [0.0] * 6
    for i, c in enumerate(shoe):
        if not c:
            continue
        p = c / n
        nt, ns = _add(total, soft, i + 1)
        nxt = shoe[:i] + (c - 1,) + shoe[i + 1:]
        sub = _dealer_draw(nt, ns, nxt, h17)
        for k in range(6):
            acc[k] += p * sub[k]
    return tuple(acc)


def dealer_distribution(upcard: int, shoe: tuple[int, ...], rules: Rules) -> dict:
    """莊家明牌 upcard、底牌未知時的最終結果分佈（含 'bj'）。

    dealer_peeks=True 時回傳「已知莊家沒有 BJ」的條件分佈（'bj' 為 0）。
    """
    acc = dict.fromkeys(OUTCOMES, 0.0)
    t0, s0 = _add(0, False, upcard)
    weights = []
    for i, c in enumerate(shoe):
        if not c:
            continue
        hole = i + 1
        bj = {upcard, hole} == {1, 10}
        if bj and rules.dealer_peeks:
            continue
        weights.append((hole, c, bj))
    total_w = sum(c for _, c, _ in weights) or 1
    for hole, c, bj in weights:
        p = c / total_w
        if bj:
            acc["bj"] += p
            continue
        t, s = _add(t0, s0, hole)
        sub = _dealer_draw(t, s, remove_cards(shoe, [hole]), rules.dealer_hits_soft17)
        for k, o in enumerate(OUTCOMES[:6]):
            acc[o] += p * sub[k]
    return acc


# ---------------------------------------------------------------- 玩家勝率
@dataclass(frozen=True)
class Odds:
    win: float
    push: float
    lose: float

    @property
    def ev(self) -> float:
        return self.win - self.lose

    def scaled(self, k: float) -> "Odds":
        return Odds(self.win * k, self.push * k, self.lose * k)


def _stand_odds(total: int, dealer: dict) -> Odds:
    if total > 21:
        return Odds(0, 0, 1)
    win = dealer["bust"]
    push = 0.0
    lose = dealer["bj"]
    for t in (17, 18, 19, 20, 21):
        p = dealer[t]
        if total > t:
            win += p
        elif total == t:
            push += p
        else:
            lose += p
    # 玩家 <17 時莊家點數一律 >=17
    return Odds(win, push, lose)


def _combine(parts) -> Odds:
    w = p = l = 0.0
    for prob, o in parts:
        w += prob * o.win
        p += prob * o.push
        l += prob * o.lose
    return Odds(w, p, l)


_DEALER_REGISTRY: dict = {}


@lru_cache(maxsize=200_000)
def _hit_cached(total, soft, shoe, dealer_key, depth) -> Odds:
    """要一張牌後依 EV 最大策略繼續（要牌或停牌）。"""
    dealer = _DEALER_REGISTRY[dealer_key]
    n = sum(shoe)
    parts = []
    for i, c in enumerate(shoe):
        if not c:
            continue
        nt, ns = _add(total, soft, i + 1)
        if nt > 21:
            parts.append((c / n, Odds(0, 0, 1)))
            continue
        stand = _stand_odds(nt, dealer)
        if nt == 21 or depth >= 6:
            parts.append((c / n, stand))
            continue
        nxt = shoe[:i] + (c - 1,) + shoe[i + 1:]
        hit = _hit_cached(nt, ns, nxt, dealer_key, depth + 1)
        parts.append((c / n, hit if hit.ev > stand.ev else stand))
    return _combine(parts)


def action_odds(player: list[int], upcard: int, shoe: tuple[int, ...],
                rules: Rules = Rules()) -> dict[str, Odds]:
    """回傳各動作的勝/和/負機率。shoe 需已扣除桌上所有可見牌。

    double 的 Odds 已乘 2（代表下注倍數），ev 直接可比。
    """
    dealer = dealer_distribution(upcard, shoe, rules)
    key = (upcard, shoe, rules)
    if len(_DEALER_REGISTRY) > 512:
        _DEALER_REGISTRY.clear()
        _hit_cached.cache_clear()
    _DEALER_REGISTRY[key] = dealer
    total, soft = hand_value(player)
    res: dict[str, Odds] = {}
    if is_blackjack(player):
        # 莊家已偷看沒 BJ → 必勝；未偷看則與莊家 BJ 和局
        pb = dealer["bj"]
        res["stand"] = Odds(1 - pb, pb, 0)
        return res
    res["stand"] = _stand_odds(total, dealer)
    if total < 21:
        res["hit"] = _hit_cached(total, soft, shoe, key, 0)
        if len(player) == 2:
            n = sum(shoe)
            parts = []
            for i, c in enumerate(shoe):
                if c:
                    nt, _ = _add(total, soft, i + 1)
                    parts.append((c / n, _stand_odds(nt, dealer)))
            res["double"] = _combine(parts).scaled(2)
    return res


# ---------------------------------------------------------------- 基本策略
# 多副牌、S17、DAS、不可投降。dealer index: 2..10, 11(=A)
_D = list(range(2, 12))


def _row(spec: str) -> dict[int, str]:
    return dict(zip(_D, spec.split()))


HARD = {
    **{t: _row("H H H H H H H H H H") for t in range(4, 9)},
    9: _row("H D D D D H H H H H"),
    10: _row("D D D D D D D D H H"),
    11: _row("D D D D D D D D D H"),
    12: _row("H H S S S H H H H H"),
    **{t: _row("S S S S S H H H H H") for t in range(13, 17)},
    **{t: _row("S S S S S S S S S S") for t in range(17, 22)},
}
SOFT = {
    13: _row("H H H D D H H H H H"),
    14: _row("H H H D D H H H H H"),
    15: _row("H H D D D H H H H H"),
    16: _row("H H D D D H H H H H"),
    17: _row("H D D D D H H H H H"),
    18: _row("S Ds Ds Ds Ds S S H H H"),
    19: _row("S S S S S S S S S S"),
    20: _row("S S S S S S S S S S"),
    21: _row("S S S S S S S S S S"),
}
PAIRS = {
    1: _row("P P P P P P P P P P"),
    2: _row("P P P P P P H H H H"),
    3: _row("P P P P P P H H H H"),
    4: _row("H H H P P H H H H H"),
    6: _row("P P P P P H H H H H"),
    7: _row("P P P P P P H H H H"),
    8: _row("P P P P P P P P P P"),
    9: _row("P P P P P S P P S S"),
    10: _row("S S S S S S S S S S"),
}
ACTION_NAMES = {"H": "要牌", "S": "停牌", "D": "加倍", "P": "分牌"}


def basic_strategy(player: list[int], upcard: int, rules: Rules = Rules()) -> str:
    """回傳 'H' / 'S' / 'D' / 'P'。"""
    d = 11 if upcard == 1 else upcard
    total, soft = hand_value(player)
    if total >= 21:
        return "S"
    two = len(player) == 2
    if two and player[0] == player[1] and player[0] in PAIRS:
        a = PAIRS[player[0]][d]
        if a == "P":
            return "P"
    table = SOFT if soft else HARD
    a = table.get(total, HARD.get(total, _row("H " * 10)))[d]
    if a == "D":
        return "D" if two else "H"
    if a == "Ds":
        return "D" if two else "S"
    return a
