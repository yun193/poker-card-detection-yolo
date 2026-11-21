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


def upright_to_frame(rbox, fx, fy):
    """轉正牌面上的相對座標 (fx＝x/w, fy＝y/h) 換回原圖座標（前端畫牌內切片命中位置用）。"""
    p = np.array(rbox, np.float32)
    if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
        p = p[::-1].copy()
    if np.linalg.norm(p[1] - p[0]) > np.linalg.norm(p[2] - p[1]):
        p = np.roll(p, -1, axis=0)
    q = p[0] + (p[1] - p[0]) * fx + (p[3] - p[0]) * fy
    return [round(float(q[0]), 1), round(float(q[1]), 1)]


def face_white_ratio(img):
    """轉正牌面中「低彩度、高亮度」像素的比例（牌面白底）。背景框（書櫃、海報）通常 ≤ 0.55。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < 60) & (hsv[..., 2] > 140)).mean())


def face_border_white(img, inner=0.03, outer=0.10):
    """轉正牌面「邊框帶」（距邊緣 inner～outer 倍牌寬）的白底比例。撲克牌（含人頭牌、牌背）四周都有白邊，
    海報、畫框、書背的邊緣通常是深色或彩色。略過最外圈 inner，避開分割框邊緣混到的背景。"""
    h, w = img.shape[:2]
    a, b = max(1, int(inner * w)), max(2, int(outer * w))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 60) & (hsv[..., 2] > 140)
    band = np.zeros((h, w), bool)
    band[a:h - a, a:w - a] = True
    band[b:h - b, b:w - b] = False
    return float(white[band].mean()) if band.any() else 0.0


def face_color_ratio(img):
    """轉正牌面中「飽和、非紅色」像素的比例。撲克牌只有紅黑兩色加白底（人頭牌的黃藍綠面積很小，實拍 ≤ 0.07），
    書背、海報、抱枕通常大片彩色（實拍書櫃 0.29～0.43）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    colored = (hsv[..., 1] > 70) & (hsv[..., 2] > 50) & (hue > 12) & (hue < 160)
    return float(colored.mean())


def scan_positions(w, h, step=0.5):
    """牌內切片的裁切中心（轉正牌面座標）與方向：沿上緣橫向、沿左緣縱向各掃一排（正向），
    再沿下緣、右緣掃一排（轉 180°）。分割把扇形／錯開疊牌框成一塊時，後面那張牌的牌角索引就落在這些邊上。
    step 以牌角框邊長為單位。tl、br 兩個標準位置不在內（由 upright_corner_crops 負責）。"""
    cx, cy, s = min(INDEX_DX * h, w / 2), INDEX_DY * h, INDEX_SIZE * h * step
    xs = [cx + s * k for k in range(1, int((w - 2 * cx) / s) + 1)]
    ys = [cy + s * k for k in range(1, int((h - 2 * cy) / s) + 1)]
    out = [((x, cy), False) for x in xs] + [((cx, y), False) for y in ys]
    out += [((w - x, h - cy), True) for x in xs] + [((w - cx, h - y), True) for y in ys]
    return out


def upright_scan_crops(frame, rbox, pad=CROP_PAD, step=0.5):
    """牌內切片：轉正後沿四邊掃描裁切（大小同 upright_corner_crops），回傳 [((x, y), 是否轉180°, 影像)]。"""
    u = upright_card(frame, rbox)
    if u is None:
        return []
    img, w, h = u
    side = INDEX_SIZE * h * (1 + 2 * pad)
    big = cv2.copyMakeBorder(img, int(side), int(side), int(side), int(side), cv2.BORDER_REPLICATE)
    out = []
    for (x, y), flip in scan_positions(w, h, step):
        a, b = int(x - side / 2 + side), int(y - side / 2 + side)
        c = big[b:b + int(side), a:a + int(side)]
        if c.size and min(c.shape[:2]) >= 6:
            out.append(((x, y), flip, cv2.rotate(c, cv2.ROTATE_180) if flip else c))
    return out


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
