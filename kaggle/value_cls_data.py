"""value_cls 訓練資料：合成場景 → 轉正裁角 → 標牌值或 none。train_value_cls.py 與 train_value_ft.py 共用（# @include）。
需要先定義 IMGSZ、NONE_RATIO、SEED，並內嵌 synth_cards.py、upright.py。"""

# %% [markdown]
# ## 標註裁角
# 牌照片貼上去的四角 `quad` 依序是牌的左上、右上、右下、左下，所以真正牌角（左上、右下）的位置可由同一個透視算出。
# 裁切中心離真牌角 ≤ 0.12×牌高、且該處沒被蓋住 → 標牌值；否則 → `none`。

# %%
RANKS = {"ace": "A", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
         "nine": "9", "ten": "10", "jack": "J", "queen": "Q", "king": "K"}
SUITS = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


def to_label(name: str):
    parts = name.lower().split(" of ")
    if len(parts) != 2 or parts[0] not in RANKS or parts[1] not in SUITS:
        return None
    return RANKS[parts[0]] + SUITS[parts[1]]


def true_index_points(q):
    """牌照片座標 → 原圖：左上、右下兩個牌角索引中心。"""
    cw, ch = 200.0, 280.0   # 與 render 貼圖尺寸一致
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), q.astype(np.float32))
    pts = np.float32([[[INDEX_DX * ch, INDEX_DY * ch]], [[cw - INDEX_DX * ch, ch - INDEX_DY * ch]]])
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2), float(np.linalg.norm(q[3] - q[0]))


def visible_at(mask, pt, r):
    x, y, r = int(pt[0]), int(pt[1]), max(2, int(r))
    win = mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return win.size > 0 and (win > 0).mean() >= 0.6


def crops_from_scene(img, infos):
    out = []
    for inf in infos:
        lab = to_label(inf["label"])
        m = inf["mask"]
        if lab is None or cv2.countNonZero(m) < 400:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        rbox = cv2.boxPoints(cv2.minAreaRect(c))                     # 與本機 CardSegmenter 相同的旋轉框
        idx_pts, card_h = true_index_points(inf["quad"])
        centers = dict(upright_corner_points(rbox))
        for name, crop in upright_corner_crops(img, rbox):
            cpt = centers[name]
            hit = any(np.linalg.norm(cpt - t) <= 0.12 * card_h and visible_at(m, t, 0.04 * card_h) for t in idx_pts)
            out.append((lab if hit else "none", cv2.resize(crop, (IMGSZ, IMGSZ))))
    return out


def build(cards, bgs, out_dir: Path, n_train, n_val, seed=SEED):
    stats = {}
    for split, n in (("train", n_train), ("val", n_val)):
        rng = random.Random(seed + (0 if split == "train" else 999))
        buf, cnt = [], Counter()
        for _ in range(n):
            img, _, infos = render(cards, bgs, rng)
            for lab, crop in crops_from_scene(img, infos):
                buf.append((lab, crop)); cnt[lab] += 1
        avg = np.mean([v for k, v in cnt.items() if k != "none"])
        keep_none = int(avg * NONE_RATIO)
        rng.shuffle(buf)
        written = Counter()
        for lab, crop in buf:
            if lab == "none" and written["none"] >= keep_none:
                continue
            d = out_dir / split / lab
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{written[lab]:05d}.jpg"), crop)
            written[lab] += 1
        stats[split] = dict(written)
        print(split, "類別數", len(written), "總數", sum(written.values()), "none", written["none"],
              "最少", min(v for k, v in written.items() if k != "none"))
    return stats
