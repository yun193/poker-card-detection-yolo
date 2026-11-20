import pytest

from blackjack_vision.advisor import Advisor
from blackjack_vision.cards import Detection, StableTable, group_corners, normalize_label, split_zones


def det(label, cx, cy, s=20, conf=0.9):
    return Detection(label, conf, (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))


def test_normalize_label():
    assert normalize_label("AS") == "As"
    assert normalize_label("10H") == "10h"
    assert normalize_label("td") == "10d"
    with pytest.raises(ValueError):
        normalize_label("1x")


def test_two_corners_one_card_and_far_duplicate_is_two():
    dets = [det("As", 100, 100), det("As", 160, 190), det("As", 600, 100), det("Kh", 110, 105)]
    cards = group_corners(dets)
    assert sorted(c.label for c in cards) == ["As", "As", "Kh"]
    assert max(c.corners for c in cards) == 2


def test_split_zones():
    cards = group_corners([det("As", 100, 50), det("Kh", 100, 400)])
    dealer, player = split_zones(cards, frame_h=480)
    assert [c.label for c in dealer] == ["As"] and [c.label for c in player] == ["Kh"]


def test_stable_table_filters_flicker():
    t = StableTable(window=5, ratio=0.6)
    a = group_corners([det("As", 100, 100)])
    noisy = group_corners([det("As", 100, 100), det("2c", 300, 300)])
    for frame in [a, a, noisy, a, a]:
        dealer, _ = t.update(frame, [])
    assert dealer == ["As"]


def test_advisor_end_to_end():
    adv = Advisor().advise(["6h"], ["10s", "Kd"])
    assert adv.player_total == 20 and adv.strategy == "S"
    assert adv.best_ev_action == "stand"
    assert Advisor().advise([], ["10s"]).note


def test_truth_from_filename():
    from pathlib import Path
    from evaluate import truth_from_name
    assert truth_from_name(Path("As-10H-kd_001.jpg")) == {"As": 1, "10h": 1, "Kd": 1}
    assert truth_from_name(Path("As-As_2.jpg")) == {"As": 2}
    assert not truth_from_name(Path("none_3.jpg"))
