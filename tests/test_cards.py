import pytest

from blackjack_vision.cards import (SegTracker, is_card_evidence, looks_like_card, normalize_label, pretty, seg_card_values,
                                    seg_memory_entry, verify_segs)
from blackjack_vision.upright import scan_positions


def test_normalize_label():
    assert normalize_label("as") == "As"
    assert normalize_label("10H") == "10h"
    assert normalize_label("td") == "10d"
    assert pretty("Kc") == "K♣"
    with pytest.raises(ValueError):
        normalize_label("Zx")


# ---------------------------------------------------------------- 背景框過濾
def _seg(x1, y1, x2, y2):
    return {"poly": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


CARD_FACE = {"border": 0.6, "color": 0.03}


def test_card_evidence_needs_a_corner_and_white_face_or_confident_value():
    assert not is_card_evidence({"white": 0.9, **CARD_FACE})                         # 兩角都是 none
    assert is_card_evidence({"label": "Ks", "top": 0.3, "white": 0.7, **CARD_FACE})  # 白底牌面
    assert not is_card_evidence({"label": "Jd", "top": 0.67, "white": 0.54, **CARD_FACE})   # 不夠白、信心不足
    assert is_card_evidence({"label": "Qd", "top": 0.97, "white": 0.45, **CARD_FACE})       # 人頭牌：白底少但很有把握


def test_poster_and_bookshelf_fail_even_with_confident_value():
    # 錄影實例：海報被 value_cls 很有把握地讀成 6♦（0.82），但邊框是深色；亮色書櫃白底夠但大片藍色
    poster = {"label": "6d", "top": 0.82, "white": 0.35, "border": 0.18, "color": 0.12}
    shelf = {"label": "Jd", "top": 0.67, "white": 0.74, "border": 0.6, "color": 0.38}
    assert not looks_like_card(poster) and not is_card_evidence(poster)
    assert not looks_like_card(shelf) and not is_card_evidence(shelf)
    assert looks_like_card({"border": 0.28, "color": 0.07})                         # 窄條、人頭牌仍算


def test_verify_segs_keeps_evidence_and_drops_background():
    card, door = _seg(100, 100, 200, 240), _seg(0, 0, 80, 400)
    kept, rejected = verify_segs([card, door], [True, False])
    assert kept == [card] and card["verify"] == "card"
    assert rejected == [door]


def test_verify_segs_chains_fan_strip_but_not_big_door_or_colorful_poster():
    front = _seg(100, 100, 200, 240)
    strip = _seg(80, 100, 115, 240)          # 扇形後排窄條，壓在前牌下、這幀沒牌角
    door = _seg(0, 0, 110, 500)              # 門很大，雖與前牌重疊也不串入
    poster = _seg(150, 120, 230, 250)        # 與前牌重疊、大小相近，但幾乎沒有白底
    kept, rejected = verify_segs([front, strip, door, poster], [True, False, False, False], [True, True, True, False])
    assert kept == [front, strip] and strip["verify"] == "chain"
    assert rejected == [door, poster]


def test_verify_segs_memory_keeps_blurred_card_only():
    prev = _seg(100, 100, 200, 240)
    verify_segs([prev], [True])
    mem = seg_memory_entry([prev])
    blurred, picture = _seg(120, 110, 220, 250), _seg(500, 0, 600, 140)
    kept, rejected = verify_segs([blurred, picture], [False, False], [True, True], memory=mem)
    assert kept == [blurred] and blurred["verify"] == "memory"
    assert rejected == [picture]
    assert seg_memory_entry(kept) == []      # memory 通過的不再寫回，避免背景靠記憶自我延續
    poster = _seg(120, 110, 220, 250)        # 同位置但外觀不像牌（海報）：不能靠記憶留下
    assert verify_segs([poster], [False], [False], memory=mem)[0] == []


def test_verify_segs_grazing_contact_does_not_chain():
    card, frame = _seg(100, 100, 200, 240), _seg(195, 20, 260, 105)  # 畫框只擦到牌的角
    kept, rejected = verify_segs([card, frame], [True, False])
    assert kept == [card] and rejected == [frame]


# ---------------------------------------------------------------- 一框多牌值
def test_seg_card_values_merges_two_cards_in_one_box():
    vals = seg_card_values([("Ad", 0.95, "corner"), ("5h", 0.92, "scan"), ("9s", 0.3, "corner")])
    assert [(v["label"], v["src"]) for v in vals] == [("Ad", "corner"), ("5h", "scan")]   # 9s 信心不足不收


def test_seg_card_values_same_rank_keeps_higher():
    vals = seg_card_values([("Ad", 0.95, "corner"), ("Ah", 0.7, "corner"), ("Ks", 0.9, "corner"), ("Ks", 0.8, "scan")])
    assert [(v["label"], v["conf"]) for v in vals] == [("Ad", 0.95), ("Ks", 0.9)]


# ---------------------------------------------------------------- 牌內切片位置
def test_scan_positions_cover_edges_and_skip_standard_corners():
    w, h = 70, 100
    pos = scan_positions(w, h)
    assert pos and all(0 <= x <= w and 0 <= y <= h for (x, y), _ in pos)
    assert {flip for _, flip in pos} == {False, True}
    assert ((10.0, 17.0), False) not in pos                                 # 左上標準位置由 upright_corner_crops 負責
    assert len(scan_positions(20, 100)) < len(pos)                          # 窄條掃的位置較少


# ---------------------------------------------------------------- 追蹤投票
def _card(cx, cy, reads, w=70, h=100):
    return {"center": [cx, cy], "rbox": [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
                                         [cx - w / 2, cy + h / 2]], "reads": reads}


def test_tracker_resists_brief_misread_and_blur():
    tr = SegTracker()
    for i in range(6):
        out = tr.update([_card(200, 200, [("Kc", 0.9), ("Kc", 0.85)])], t=i * 0.05)
    tr.update([_card(203, 201, [("As", 0.7)])], t=0.30)                      # 一幀認錯
    out = tr.update([_card(205, 202, [])], t=0.35)                           # 一幀糊掉：沒有讀值
    assert out[0].label == "Kc" and tr.labels(out[0]) == ["Kc"]
    assert tr.hand(0.35) == ["Kc"]


def test_tracker_follows_real_card_change():
    tr = SegTracker()
    for i in range(6):
        tr.update([_card(200, 200, [("Kc", 0.9)])], t=i * 0.05)
    for i in range(10):
        out = tr.update([_card(200, 200, [("5h", 0.9), ("5h", 0.9)])], t=0.3 + i * 0.05)
    assert out[0].label == "5h"


def test_tracker_extra_value_needs_to_persist():
    tr = SegTracker()
    out = tr.update([_card(200, 200, [("Ad", 0.95), ("Ad", 0.9), ("9s", 0.95)])], t=0.0)
    assert tr.labels(out[0]) == ["Ad"]                                       # 附加牌值只出現一幀：不列
    for i in range(1, 4):
        out = tr.update([_card(200, 200, [("Ad", 0.95), ("Ad", 0.9), ("9s", 0.95)])], t=i * 0.05)
    assert tr.labels(out[0]) == ["Ad", "9s"]                                 # 連續出現：一框兩張


def test_tracker_separate_cards_and_drop_lost_ones():
    tr = SegTracker()
    for i in range(4):
        out = tr.update([_card(100, 200, [("Kc", 0.9)]), _card(400, 200, [("7d", 0.9)])], t=i * 0.05)
    assert out[0].id != out[1].id
    assert sorted(tr.hand(0.15)) == ["7d", "Kc"]
    tr.update([_card(100, 200, [("Kc", 0.9)])], t=1.5)                      # 7♦ 超過 1 秒沒出現
    assert [x.label for x in tr.tracks] == ["Kc"]


def test_hand_smoother_ignores_flicker_and_keeps_through_dropouts():
    from blackjack_vision.cards import HandSmoother
    hs = HandSmoother()
    out = [hs.update(h, i * 0.1) for i, h in enumerate(
        [["6d"]] * 6 + [["6d", "Ah"], ["6d"], ["6d"], [], ["6d"]] + [["6d", "5d"]] * 8)]
    assert all("Ah" not in h for h in out)              # 只閃一幀的錯牌不進手牌
    assert out[9] == ["6d"]                              # 掉一幀不會清空
    assert out[-1] == ["6d", "5d"]                       # 持續出現的新牌會加入
    for i in range(19, 35):
        last = hs.update([], i * 0.1)
    assert last == []                                    # 牌拿走後約 1 秒清空
