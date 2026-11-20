"""把穩定的桌面牌組轉成 21 點建議：點數、基本策略、各動作勝率，並追蹤牌靴（算牌）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .blackjack import (ACTION_NAMES, Odds, Rules, action_odds, basic_strategy,
                        full_shoe, hand_value, remove_cards)
from .cards import card_value


@dataclass
class Advice:
    player_total: int | None = None
    player_soft: bool = False
    dealer_up: str | None = None
    strategy: str | None = None          # 'H' / 'S' / 'D' / 'P'
    odds: dict[str, Odds] = field(default_factory=dict)
    note: str = ""

    @property
    def strategy_name(self) -> str:
        return ACTION_NAMES.get(self.strategy or "", "-")

    @property
    def best_ev_action(self) -> str | None:
        if not self.odds:
            return None
        return max(self.odds, key=lambda k: self.odds[k].ev)


class Shoe:
    """記錄前幾局已出過的牌（按 N 結束一局時累加），用於組合依存的勝率。"""

    def __init__(self, rules: Rules):
        self.rules = rules
        self.seen: list[int] = []

    def end_round(self, labels: list[str]):
        self.seen += [card_value(l) for l in labels]

    def reset(self):
        self.seen.clear()

    def remaining(self, table_labels: list[str]) -> tuple[int, ...]:
        shoe = remove_cards(full_shoe(self.rules.decks), self.seen)
        return remove_cards(shoe, [card_value(l) for l in table_labels])

    def hi_lo_true_count(self) -> float:
        rc = sum(1 if v in (2, 3, 4, 5, 6) else -1 if v in (1, 10) else 0 for v in self.seen)
        decks_left = max(0.5, (52 * self.rules.decks - len(self.seen)) / 52)
        return rc / decks_left


class Advisor:
    def __init__(self, rules: Rules = Rules()):
        self.rules = rules
        self.shoe = Shoe(rules)
        self._cache: tuple | None = None

    def advise(self, dealer: list[str], player: list[str]) -> Advice:
        key = (tuple(sorted(dealer)), tuple(sorted(player)), len(self.shoe.seen))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        adv = self._compute(dealer, player)
        self._cache = (key, adv)
        return adv

    def _compute(self, dealer: list[str], player: list[str]) -> Advice:
        adv = Advice()
        pv = [card_value(l) for l in player]
        if pv:
            adv.player_total, adv.player_soft = hand_value(pv)
        if len(dealer) != 1:
            adv.note = "請讓莊家區只露出 1 張明牌" if dealer else "等待莊家明牌"
            return adv
        adv.dealer_up = dealer[0]
        if len(pv) < 2:
            adv.note = "等待玩家 2 張牌"
            return adv
        up = card_value(dealer[0])
        shoe = self.shoe.remaining(dealer + player)
        adv.strategy = basic_strategy(pv, up, self.rules)
        if adv.player_total is not None and adv.player_total > 21:
            adv.note = "爆牌"
            adv.strategy = None
            return adv
        adv.odds = action_odds(pv, up, shoe, self.rules)
        return adv
