from pathlib import Path

import pytest

from blackjack_vision.cards import (CardTracker, Detection, fuse_scales, group_corners, normalize_label, pretty,
                                    refine_with_classifier, seg_card_values, verify_segs)


def det(label, cx, cy, s=20, conf=0.9):
    return Detection(label, conf, (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))


def test_normalize_label():
    assert normalize_label("AS") == "As"
    assert normalize_label("10H") == "10h"
    assert normalize_label("td") == "10d"
    assert pretty("kd") == "K♦"
    with pytest.raises(ValueError):
        normalize_label("1x")


def test_two_corners_one_card_and_far_duplicate_is_two():
    dets = [det("As", 100, 100), det("As", 160, 190), det("As", 600, 100), det("Kh", 110, 105)]
    cards = group_corners(dets)
    assert sorted(c.label for c in cards) == ["As", "As", "Kh"]
    assert max(c.corners for c in cards) == 2


def test_fuse_agreeing_scales_keeps_label():
    a = [det("Kc", 100, 100, conf=0.9)]
    b = [det("Kc", 101, 99, conf=0.8)]
    out = fuse_scales([a, b], conf=0.45)
    assert [d.label for d in out] == ["Kc"] and out[0].conf == pytest.approx(0.85)


def test_fuse_disagreement_votes_and_lowers_score():
    a = [det("Kc", 100, 100, conf=0.9)]
    b = [det("As", 100, 100, conf=0.6)]
    out = fuse_scales([a, b], conf=0.3)
    assert [d.label for d in out] == ["Kc"] and out[0].conf == pytest.approx(0.45)
    assert fuse_scales([a, b], conf=0.5) == []          # 兩尺度不一致 → 分數被壓到門檻以下


def test_fuse_single_scale_only_detection_is_penalized():
    out = fuse_scales([[det("7h", 50, 50, conf=0.8)], []], conf=0.45)
    assert out == []                                    # 只有一個尺度看到 → 0.4 < 0.45


def test_tracker_resists_brief_label_flip():
    t = CardTracker(memory=8)
    for _ in range(6):
        t.update(group_corners([det("Kc", 200, 200)]))
    for _ in range(3):                                  # 同位置連續 3 幀被認成 A♠
        pairs = t.update(group_corners([det("As", 202, 201, conf=0.9)]))
    assert pairs[0][1].label == "Kc" and pairs[0][0].label == "As"
    assert [x.label for x in t.confirmed()] == ["Kc"]


def test_tracker_follows_real_card_change():
    t = CardTracker(memory=8)
    for _ in range(6):
        t.update(group_corners([det("Kc", 200, 200)]))
    for _ in range(10):                                 # 真的換成 A♠ 且持續出現
        t.update(group_corners([det("As", 200, 200)]))
    assert [x.label for x in t.confirmed()] == ["As"]


def test_tracker_keeps_separate_cards_and_drops_lost_ones():
    t = CardTracker(memory=3)
    two = group_corners([det("As", 100, 100), det("As", 600, 100)])
    for _ in range(4):
        t.update(two)
    assert [x.label for x in t.confirmed()] == ["As", "As"]
    for _ in range(5):
        t.update([])
    assert t.confirmed() == [] and t.tracks == []


def test_refine_classifier_overrides_confident_disagreement():
    d = [det("As", 100, 100, conf=0.85)]
    out = refine_with_classifier(d, [{"Kc": 0.93, "As": 0.04}], w_cls=0.6)
    assert out[0].label == "Kc" and out[0].det_label == "As" and out[0].cls_conf == pytest.approx(0.93)


def test_refine_keeps_detector_when_classifier_unsure():
    d = [det("7h", 100, 100, conf=0.9)]
    out = refine_with_classifier(d, [{"8h": 0.5, "7h": 0.45}], w_cls=0.6)
    assert out[0].label == "7h"                         # 0.36+0.27 > 0.30


def test_refine_without_probs_keeps_label():
    out = refine_with_classifier([det("Qs", 0, 0, conf=0.8)], [{}])
    assert out[0].label == "Qs"


def test_truth_from_filename():
    from evaluate import truth_from_name
    assert truth_from_name(Path("As-10H-kd_001.jpg")) == {"As": 1, "10h": 1, "Kd": 1}
    assert truth_from_name(Path("As-As_2.jpg")) == {"As": 2}
    assert not truth_from_name(Path("none_3.jpg"))


def _seg(x1, y1, x2, y2, conf=0.6):
    return {"poly": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "conf": conf}


def test_verify_segs_keeps_card_with_corner_and_drops_background():
    card, door = _seg(100, 100, 200, 240), _seg(0, 0, 80, 400)
    kept, rejected = verify_segs([card, door], [det("Ks", 115, 120)])
    assert kept == [card] and kept[0]["corners"] == 1
    assert rejected == [door] and door["corners"] == 0


def test_verify_segs_counts_corner_straddling_mask_edge():
    # 牌角框中心剛好在遮罩外一點（遮罩略縮），但框有一半以上的角落在內 → 仍算
    card = _seg(100, 100, 200, 240)
    kept, _ = verify_segs([card], [det("Ah", 101, 160, s=20)])
    assert kept == [card]


def test_verify_segs_tilted_polygon():
    diamond = {"poly": [[150, 50], [250, 150], [150, 250], [50, 150]], "conf": 0.7}
    assert verify_segs([diamond], [det("9h", 150, 80)])[0] == [diamond]
    assert verify_segs([diamond], [det("9h", 60, 60)])[0] == []


def test_verify_segs_chains_fan_strip_but_not_big_door():
    front = _seg(100, 100, 200, 240)
    strip = _seg(80, 100, 115, 240)          # 扇形後排窄條，壓在前牌下、沒牌角偵測
    door = _seg(0, 0, 110, 500)              # 門很大，雖與前牌重疊也不串入
    kept, rejected = verify_segs([front, strip, door], [det("Ks", 150, 120)])
    assert kept == [front, strip] and strip["verify"] == "chain"
    assert rejected == [door]


def test_verify_segs_memory_keeps_blurred_card_only():
    from blackjack_vision.cards import seg_memory_entry
    prev = _seg(100, 100, 200, 240); verify_segs([prev], [det("Ks", 115, 120)])
    mem = seg_memory_entry([prev])
    blurred, picture = _seg(120, 110, 220, 250), _seg(500, 0, 600, 140)
    kept, rejected = verify_segs([blurred, picture], [], memory=mem)
    assert kept == [blurred] and blurred["verify"] == "memory"
    assert rejected == [picture]
    assert seg_memory_entry(kept) == []      # memory 通過的不再寫回，避免背景靠記憶自我延續


def test_verify_segs_corner_goes_to_smallest_box_so_shelf_behind_card_fails():
    shelf, card = _seg(0, 0, 400, 400), _seg(100, 100, 200, 240)   # 牌拿在書櫃前
    kept, rejected = verify_segs([shelf, card], [det("Ks", 115, 120)])
    assert kept == [card] and rejected == [shelf]                    # 書櫃比牌大太多，也不會被串接


def test_verify_segs_grazing_contact_does_not_chain():
    card, frame = _seg(100, 100, 200, 240), _seg(195, 20, 260, 105)  # 畫框只擦到牌的角
    kept, rejected = verify_segs([card, frame], [det("Ks", 150, 200)])
    assert kept == [card] and rejected == [frame]


def test_seg_card_values_merges_two_cards_in_one_box():
    # 分割把 A♦ 與後面露出牌角的 5♥ 框成一塊：B-2 左上認 A♦、右下認 5♥，牌角偵測也看到 5♥
    vals = seg_card_values([("Ad", 0.95), ("5h", 0.7)], [det("5h", 0, 0, conf=0.88), det("9s", 0, 0, conf=0.3)])
    assert [(v["label"], v["src"]) for v in vals] == [("Ad", "b2"), ("5h", "corner")]   # 9s 信心不足不收


def test_seg_card_values_single_card_dedupes():
    vals = seg_card_values([("Ks", 0.9), ("Ks", 0.8)], [det("Ks", 0, 0, conf=0.95)])
    assert [(v["label"], v["conf"]) for v in vals] == [("Ks", 0.95)]


def test_seg_card_values_same_rank_keeps_higher_and_det_threshold():
    vals = seg_card_values([("Ad", 0.95), ("Ah", 0.7)], [det("5h", 0, 0, conf=0.45)], 0.6, 0.4)
    assert [v["label"] for v in vals] == ["Ad", "5h"]   # Ah 同點數被去掉；5h 0.45 ≥ 牌角門檻 0.4 收下
