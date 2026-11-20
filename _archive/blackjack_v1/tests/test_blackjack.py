import pytest

from blackjack_vision.blackjack import (Rules, action_odds, basic_strategy, dealer_distribution,
                                        full_shoe, hand_value, remove_cards)

BIG = Rules(decks=200)  # 近似無限副牌，對照公開數值


def test_hand_value():
    assert hand_value([1, 6]) == (17, True)
    assert hand_value([1, 6, 10]) == (17, False)
    assert hand_value([1, 1, 9]) == (21, True)
    assert hand_value([10, 10, 5]) == (25, False)


# 期望值來自獨立的無限副牌遞迴（S17、2~9 明牌不受 peek 影響）
@pytest.mark.parametrize("up,bust", [(2, 0.3536), (4, 0.3945), (5, 0.4164), (6, 0.4232), (7, 0.2623), (9, 0.2284)])
def test_dealer_bust_matches_infinite_deck(up, bust):
    d = dealer_distribution(up, remove_cards(full_shoe(200), [up]), BIG)
    assert d["bust"] == pytest.approx(bust, abs=0.002)
    assert sum(d.values()) == pytest.approx(1.0)


def test_peek_removes_dealer_blackjack():
    shoe = remove_cards(full_shoe(6), [1])
    d = dealer_distribution(1, shoe, Rules(dealer_peeks=True))
    assert d["bj"] == 0
    d2 = dealer_distribution(1, shoe, Rules(dealer_peeks=False))
    assert d2["bj"] == pytest.approx(96 / 311, abs=1e-6)  # 6 副牌扣掉明牌 A


def test_odds_are_probabilities():
    shoe = remove_cards(full_shoe(6), [10, 6, 10])
    res = action_odds([10, 6], 10, shoe)
    for k, o in res.items():
        total = o.win + o.push + o.lose
        assert total == pytest.approx(2.0 if k == "double" else 1.0)


def test_obvious_decisions():
    shoe = remove_cards(full_shoe(6), [10, 10, 6])
    r = action_odds([10, 10], 6, shoe)
    assert r["stand"].ev > r["hit"].ev and r["stand"].ev > 0.5
    shoe = remove_cards(full_shoe(6), [6, 5, 6])
    r = action_odds([6, 5], 6, shoe)
    assert r["double"].ev > r["hit"].ev > r["stand"].ev


def test_blackjack_hand():
    shoe = remove_cards(full_shoe(6), [1, 10, 9])
    r = action_odds([1, 10], 9, shoe)
    assert r["stand"].win == pytest.approx(1.0)


@pytest.mark.parametrize("player,up,expect", [
    ([10, 6], 10, "H"), ([10, 6], 6, "S"), ([6, 5], 10, "D"), ([6, 5], 1, "H"),
    ([1, 7], 2, "S"), ([1, 7], 5, "D"), ([1, 7], 9, "H"), ([1, 7, 1], 5, "S"),
    ([8, 8], 10, "P"), ([1, 1], 1, "P"), ([10, 10], 6, "S"), ([9, 9], 7, "S"),
    ([5, 5], 9, "D"), ([4, 5, 2], 6, "H"), ([10, 2], 3, "H"), ([10, 2], 4, "S"),
])
def test_basic_strategy(player, up, expect):
    assert basic_strategy(player, up) == expect
