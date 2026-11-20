"""撲克牌合成器（整張牌分割與牌值分類兩個 Kaggle notebook 共用，以 `# @include synth_cards.py` 內嵌）。

把 gpiosenka 的單張牌照片以扇形／散放／橫排重疊貼到 DTD 紋理背景上：
- fan：手牌扇形，所有牌繞牌下方同一個樞紐點旋轉（最貼近手拿情境）
- scatter：散放、任意角度；row：橫排重疊（發牌）
之後加整體透視、手指遮擋（膚色橢圓）、模糊、亮度、雜訊、JPEG 壓縮。
render 回傳 (影像, 可見區域多邊形[正規化], 每張牌資訊)；可見面積 < 整張的 1% 的牌不出多邊形（實測常見只露出一條細邊＋牌角的後牌，要標成獨立的牌）。
"""
import glob
import math
import random
from pathlib import Path

import cv2
import numpy as np

S = 640                 # 合成圖邊長


def find_dir(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits


def load_sources(card_root=None, bg_root=None):
    cards = [p for p in find_dir(f"{card_root or '/kaggle/input'}/**/train/*/*.jpg") if "joker" not in p.lower()]
    bgs = find_dir(f"{bg_root or '/kaggle/input'}/**/dtd/images/*/*.jpg") or \
          [p for p in find_dir(f"{bg_root or '/kaggle/input'}/**/*.jpg") if "texture" in p.lower() or "dtd" in p.lower()]
    return cards, bgs



CARD_AR = 1.4   # 牌高 / 牌寬


def rand_bg(bgs, rng):
    if bgs and rng.random() < 0.85:
        im = cv2.imread(rng.choice(bgs))
        if im is not None:
            h, w = im.shape[:2]
            s = rng.uniform(0.3, 1.0) * min(h, w)
            x, y = rng.uniform(0, w - s), rng.uniform(0, h - s)
            return cv2.resize(im[int(y):int(y + s), int(x):int(x + s)], (S, S))
    c = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 12, (S, S, 3))
    return np.clip(c + noise, 0, 255).astype(np.uint8)


def card_quads(rng):
    n = rng.choice([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7])
    # stack：幾乎平行、往側邊錯開的疊牌（實測最常被分割合併成一塊的情況）
    mode = rng.choices(["fan", "scatter", "row", "stack"], [0.4, 0.15, 0.15, 0.3])[0]
    quads = []
    if mode == "stack":
        n = max(n, 2)
        h = rng.uniform(0.3, 0.65) * S
        w = h / CARD_AR
        ang = rng.uniform(-40, 40)
        # 每張往側邊露出的寬度：4 成是只露出牌角那一條的細邊（實測最常被併進前牌）
        shift = (rng.uniform(0.06, 0.15) if rng.random() < 0.4 else rng.uniform(0.15, 0.5)) * w
        drop = rng.uniform(-0.08, 0.08) * h          # 每張上下微錯
        cx0, cy0 = rng.uniform(0.25, 0.6) * S, rng.uniform(0.3, 0.7) * S
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for i in range(n):
            a = math.radians(ang + rng.uniform(-3, 3))
            local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            dx, dy = i * shift, i * drop
            quads.append(local @ R.T + [cx0 + dx * ca - dy * sa, cy0 + dx * sa + dy * ca])
    elif mode == "fan":
        h = rng.uniform(0.28, 0.6) * S
        w = h / CARD_AR
        px, py = rng.uniform(0.25, 0.75) * S, rng.uniform(0.55, 0.95) * S
        base = rng.uniform(-35, 35)
        # 實際手牌常是很密的小角度扇形：一半機率 3~8°，一半 7~18°
        step = (rng.uniform(3, 8) if rng.random() < 0.5 else rng.uniform(7, 18)) * rng.choice([1, -1])
        off = rng.uniform(0.05, 0.25) * h
        a0 = base - step * (n - 1) / 2
        for i in range(n):
            a = math.radians(a0 + step * i)
            local = np.array([[-w / 2, -h + off], [w / 2, -h + off], [w / 2, off], [-w / 2, off]])
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            quads.append(local @ R.T + [px + rng.uniform(-3, 3), py + rng.uniform(-3, 3)])
    elif mode == "row":
        h = rng.uniform(0.2, 0.45) * S
        w = h / CARD_AR
        gap = rng.uniform(0.25, 0.7) * w
        cx0 = rng.uniform(0.15, 0.5) * S
        cy = rng.uniform(0.25, 0.75) * S
        ang = rng.uniform(-20, 20)
        for i in range(n):
            a = math.radians(ang + rng.uniform(-5, 5))
            local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            quads.append(local @ R.T + [cx0 + i * gap * math.cos(math.radians(ang)), cy + i * gap * math.sin(math.radians(ang))])
    else:
        for _ in range(n):
            h = rng.uniform(0.15, 0.45) * S
            w = h / CARD_AR
            a = math.radians(rng.uniform(0, 360))
            local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            quads.append(local @ R.T + [rng.uniform(0.15, 0.85) * S, rng.uniform(0.15, 0.85) * S])
    # 整體透視（模擬鏡頭角度）
    src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    j = S * 0.08
    dst = np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)], [S + rng.uniform(-j, j), rng.uniform(-j, j)],
                      [S + rng.uniform(-j, j), S + rng.uniform(-j, j)], [rng.uniform(-j, j), S + rng.uniform(-j, j)]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    quads = [cv2.perspectiveTransform(q.reshape(-1, 1, 2).astype(np.float32), Hm).reshape(-1, 2) for q in quads]
    return quads, mode


def draw_distractors(img, rng):
    """背景負樣本：門、書櫃、畫框一類的長方形（不標註），教模型「長方形≠牌」。"""
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        w, h = rng.uniform(0.1, 0.5) * S, rng.uniform(0.15, 0.8) * S
        x, y = rng.uniform(-0.1, 0.9) * S, rng.uniform(-0.1, 0.9) * S
        base = np.array(rng.choice([(200, 225, 235), (90, 140, 190), (230, 230, 225), (60, 90, 140), (180, 200, 210)]),
                        np.float32) + np.random.default_rng(rng.randint(0, 1 << 30)).uniform(-25, 25, 3)
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(img, p1, p2, tuple(int(v) for v in np.clip(base, 0, 255)), -1)
        kind = rng.random()
        if kind < 0.35:      # 書櫃：橫隔板＋書背
            for k in range(1, rng.randint(2, 5)):
                yy = int(y + h * k / 5)
                cv2.line(img, (p1[0], yy), (p2[0], yy), tuple(int(v * 0.6) for v in base), 3)
                xx = p1[0]
                while xx < p2[0]:
                    bw = rng.randint(4, 14)
                    cv2.rectangle(img, (xx, yy - rng.randint(10, 40)), (xx + bw, yy),
                                  tuple(rng.randint(30, 220) for _ in range(3)), -1)
                    xx += bw + rng.randint(0, 3)
        elif kind < 0.65:    # 畫框：內框＋雜訊內容
            m = int(min(w, h) * rng.uniform(0.08, 0.2))
            inner = img[max(0, p1[1] + m):max(0, p2[1] - m), max(0, p1[0] + m):max(0, p2[0] - m)]
            if inner.size:
                inner[:] = np.random.default_rng(rng.randint(0, 1 << 30)).integers(0, 255, inner.shape, dtype=np.uint8)
                inner[:] = cv2.GaussianBlur(inner, (0, 0), 6)
        else:                # 門：門框線＋門把
            cv2.rectangle(img, p1, p2, tuple(int(v * 0.7) for v in base), 2)
            cv2.circle(img, (int(x + w * 0.85), int(y + h * 0.5)), max(2, int(w * 0.03)), (60, 160, 200), -1)


def render(cards, bgs, rng):
    img = rand_bg(bgs, rng)
    if rng.random() < 0.6:
        draw_distractors(img, rng)
    quads, mode = card_quads(rng)
    masks = []
    full_areas = []
    infos = []   # 每張牌：gpiosenka 類別名（例 "ace of spades"）、整張外框四角、可見遮罩（與 masks 同一物件）
    for q in quads:
        path = rng.choice(cards)
        cim = cv2.imread(path)
        if cim is None:
            continue
        ch, cw = 280, 200
        cim = cv2.resize(cim, (cw, ch))
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), q.astype(np.float32))
        warped = cv2.warpPerspective(cim, M, (S, S), flags=cv2.INTER_LINEAR)
        m = np.zeros((S, S), np.uint8)
        cv2.fillConvexPoly(m, q.astype(np.int32), 255)
        if masks and rng.random() < 0.6:   # 疊在別張牌上：牌緣外側一圈柔和陰影（真牌疊放時白牌之間唯一的邊界線索）
            sh = np.zeros((S, S), np.uint8)
            cv2.fillConvexPoly(sh, q.astype(np.int32), 255)
            sh = cv2.GaussianBlur(cv2.dilate(sh, np.ones((5, 5), np.uint8)), (0, 0), rng.uniform(1.5, 4))
            dark = 1 - rng.uniform(0.15, 0.4) * (sh.astype(np.float32) / 255)
            img[:] = (img * dark[..., None]).astype(np.uint8)
        img[m > 0] = warped[m > 0]
        if rng.random() < 0.4:   # 實際白牌疊白牌幾乎沒有輪廓線，只有一部分畫上細邊
            cv2.polylines(img, [q.astype(np.int32)], True, tuple(int(v) for v in rng.choice([(40, 40, 40), (200, 200, 200), (120, 120, 120)])), 1, cv2.LINE_AA)
        for pm in masks:
            pm[m > 0] = 0
        masks.append(m)
        full_areas.append(cv2.contourArea(q.astype(np.float32)))
        infos.append({"label": Path(path).parent.name, "quad": q, "mask": m})
    # 手指遮擋
    if mode == "fan" and rng.random() < 0.6 or rng.random() < 0.2:
        occ = np.zeros((S, S), np.uint8)
        for _ in range(rng.randint(1, 3)):
            cx, cy = rng.uniform(0.1, 0.9) * S, rng.uniform(0.5, 1.0) * S
            cv2.ellipse(occ, (int(cx), int(cy)), (int(rng.uniform(25, 70)), int(rng.uniform(50, 130))),
                        rng.uniform(0, 180), 0, 360, 255, -1)
        skin = np.array([rng.randint(60, 140), rng.randint(100, 180), rng.randint(150, 230)], np.uint8)
        img[occ > 0] = (img[occ > 0] * 0.15 + skin * 0.85).astype(np.uint8)
        for pm in masks:
            pm[occ > 0] = 0
    # 全域擾動
    alpha, beta = rng.uniform(0.6, 1.35), rng.uniform(-40, 40)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < 0.35:
        k = rng.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0) if rng.random() < 0.6 else cv2.blur(img, (k, 1))
    if rng.random() < 0.4:
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(3, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = rng.randint(35, 95)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    # 標註：可見區域多邊形
    polys = []
    for m, fa in zip(masks, full_areas):
        vis = cv2.countNonZero(m)
        if fa <= 0 or vis < 0.01 * fa:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < max(0.008 * fa, 60):
            continue
        c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append(np.clip(c / S, 0, 1))
    return img, polys, infos
