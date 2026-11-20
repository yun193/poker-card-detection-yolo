"""方案 B-2 的轉正裁角：整張牌旋轉框 → 透視轉正 → 裁左上與（轉 180° 的）右下牌角。

本檔只依賴 numpy／cv2、不用相對匯入：Kaggle 訓練 notebook 以 `# @include` 原樣內嵌，
確保訓練資料與本機推論的裁法完全一致。改這裡就要重新訓練 value_cls。
"""
import cv2
import numpy as np

# 牌角索引在轉正牌面中的位置（以牌高 H 為單位，實測：中心距最近角 (0.10H, 0.17H)，牌角框長邊約 0.22H）
INDEX_DX, INDEX_DY, INDEX_SIZE = 0.10, 0.17, 0.22
CROP_PAD = 0.25


def upright_card(frame, rbox):
    """把旋轉框透視轉正成直立牌面（短邊為寬、長邊為高，不鏡像）。回傳 (影像, 寬, 高) 或 None。
    方向只能確定到 180°（直立或倒立），所以牌角可能在左上或右下。"""
    p = np.array(rbox, np.float32)
    if len(p) != 4:
        return None
    # 確保來源點與目的點同為順時針（影像座標 y 向下），否則轉出來會是鏡像
    if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
        p = p[::-1].copy()
    if np.linalg.norm(p[1] - p[0]) > np.linalg.norm(p[2] - p[1]):   # 讓 p0→p1 為短邊（牌寬）
        p = np.roll(p, -1, axis=0)
    w, h = float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[1]))
    if w < 4 or h < 8:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    img = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(p, dst), (int(round(w)), int(round(h))),
                              borderMode=cv2.BORDER_REPLICATE)
    return img, w, h


def upright_corner_points(rbox):
    """兩個裁切中心在原圖上的座標（tl、br 順序同 upright_corner_crops），供合成資料判斷該處是否真有牌角。"""
    p = np.array(rbox, np.float32)
    if len(p) != 4:
        return []
    if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
        p = p[::-1].copy()
    if np.linalg.norm(p[1] - p[0]) > np.linalg.norm(p[2] - p[1]):
        p = np.roll(p, -1, axis=0)
    w, h = float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[1]))
    ex, ey = (p[1] - p[0]) / max(w, 1e-6), (p[3] - p[0]) / max(h, 1e-6)
    cx, cy = min(INDEX_DX * h, w / 2), INDEX_DY * h
    return [("tl", p[0] + ex * cx + ey * cy), ("br", p[0] + ex * (w - cx) + ey * (h - cy))]


def upright_corner_crops(frame, rbox, pad=CROP_PAD):
    """轉正後裁兩個牌角（正方形，邊長＝牌角框 ×(1+2·pad)）：
    tl＝左上原樣、br＝右下轉 180°（倒立的牌或扇形窄條，牌角會落在這裡）。"""
    u = upright_card(frame, rbox)
    if u is None:
        return []
    img, w, h = u
    side = INDEX_SIZE * h * (1 + 2 * pad)
    cx = min(INDEX_DX * h, w / 2)    # 扇形窄條比牌角還窄時，取窄條中線
    cy = INDEX_DY * h
    big = cv2.copyMakeBorder(img, int(side), int(side), int(side), int(side), cv2.BORDER_REPLICATE)
    out = []
    for name, (x, y) in (("tl", (cx, cy)), ("br", (w - cx, h - cy))):
        a, b = int(x - side / 2 + side), int(y - side / 2 + side)
        c = big[b:b + int(side), a:a + int(side)]
        if c.size and min(c.shape[:2]) >= 6:
            out.append((name, c if name == "tl" else cv2.rotate(c, cv2.ROTATE_180)))
    return out
