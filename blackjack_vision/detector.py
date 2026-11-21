"""整張牌路線的兩個模型：card_seg（整張牌露出區域分割）與 value_cls（轉正牌角牌值分類，52 類＋none）。

舊的牌角偵測（UltralyticsDetector）與牌角二次分類（refiner）已封存到 _archive/corner_route_v2/。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .upright import (face_border_white, face_color_ratio, face_white_ratio, upright_card, upright_corner_crops, upright_corner_points, upright_scan_crops,
                      upright_to_frame)

ROOT = Path(__file__).resolve().parent.parent
MULTI_CORNER_CONF = 0.6   # 兩角各自信心都 ≥ 此值卻認成不同牌 → 判定框住多張牌
NONE_CONF = 0.5           # 該角 none 機率 ≥ 此值 → 視為沒有牌角，不投票


def _device(device):
    import torch
    return device if device is not None else (0 if torch.cuda.is_available() else "cpu")


class ValueClassifier:
    """轉正牌角 → 52 類＋none 分類。裁法在 upright.py（與 Kaggle 訓練共用，改了要重訓）。"""

    def __init__(self, weights: str, imgsz: int = 128, device: str | int | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.device = imgsz, _device(device)
        self.names = self.model.names
        self.net = None   # 第一次推論時建立（直接批次前向，見 predict）

    def _forward(self, crops: list) -> np.ndarray:
        """整批一次前向。前處理與 Ultralytics 分類預測相同（BGR→RGB、PIL 雙線性縮放、/255），
        實測 944 張實拍裁圖 top-1 一致 943 張，速度約 4 倍（model.predict 逐張建 Results 物件）。"""
        import cv2
        import torch
        from PIL import Image

        if self.net is None:
            self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), imgsz=self.imgsz, device=self.device,
                               verbose=False)   # 讓 Ultralytics 完成裝置與精度設定
            self.net = self.model.predictor.model
        s = self.imgsz
        x = np.stack([np.asarray(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).resize((s, s), Image.BILINEAR))
                      for c in crops])
        n = len(x)
        pad = -n % 16   # 批次大小補到 16 的倍數：形狀固定，cuDNN 不會每換一個張數就重新挑演算法（實測會慢到 3 倍）
        if pad:
            x = np.concatenate([x, np.zeros((pad, s, s, 3), np.uint8)])
        t = torch.from_numpy(x).to(self.net.device).permute(0, 3, 1, 2)
        t = (t.half() if self.net.fp16 else t.float()) / 255
        with torch.no_grad():
            p = self.net(t)
        p = p[0] if isinstance(p, (list, tuple)) else p
        return p[:n].float().cpu().numpy()

    def predict(self, crops: list) -> list[dict[str, float]]:
        """每張裁圖回傳 {標籤: 機率}（只留前 5 名）；None 或空裁圖回傳 {}。"""
        idx = [i for i, c in enumerate(crops) if c is not None and c.size]
        out: list[dict[str, float]] = [{} for _ in crops]
        for k in range(0, len(idx), 128):   # 4GB 顯示卡：一批最多 128 張
            part = idx[k:k + 128]
            for i, p in zip(part, self._forward([crops[j] for j in part])):
                top = np.argsort(-p)[:5]
                out[i] = {self.names[int(c)]: float(p[c]) for c in top}
        return out

    def classify_segs(self, frame: np.ndarray, segs: list[dict]) -> list[dict]:
        """每個框轉正 → 裁左上與（轉 180° 的）右下牌角 → 分類，兩角機率平均後取最高。
        正立或倒立的牌，兩個角轉完都應是正立的同一牌值；只取單角最高信心會被倒過來的 6／9 誤導。
        回傳每個 seg 的 {"label", "conf", "top", "multi", "corners", "white", "border", "color"}；
        兩角都是 none 時沒有 label 等牌值欄位。top：單角最高信心；multi：兩角各自有把握地認成不同牌（多半框住兩張以上的牌）；
        white／border／color：牌面白底、邊框白底、非紅彩色比例（背景過濾用，見 cards.looks_like_card）。"""
        crops, owner, faces = [], [], []
        for i, s in enumerate(segs):
            u = upright_card(frame, s["rbox"]) if s.get("rbox") else None
            faces.append({"white": round(face_white_ratio(u[0]), 3), "border": round(face_border_white(u[0]), 3),
                          "color": round(face_color_ratio(u[0]), 3)} if u else {"white": 0.0, "border": 0.0, "color": 1.0})
            for _, c in (upright_corner_crops(frame, s["rbox"]) if u else []):
                crops.append(c); owner.append(i)
        votes: list[dict[str, float]] = [{} for _ in segs]
        tops: list[list[tuple[str, float]]] = [[] for _ in segs]
        n = [0] * len(segs)
        for i, probs in zip(owner, self.predict(crops)):
            if probs.get("none", 0.0) >= NONE_CONF:   # 這一角沒有牌角（空白角、別張牌、手指）→ 不投票
                continue
            real = {lab: p for lab, p in probs.items() if lab != "none"}
            if not real:
                continue
            n[i] += 1
            tops[i].append(max(real.items(), key=lambda kv: kv[1]))
            for lab, p in real.items():
                votes[i][lab] = votes[i].get(lab, 0.0) + p
        out: list[dict] = []
        for s, v, k, t, face in zip(segs, votes, n, tops, faces):
            pts = [[round(float(x), 1), round(float(y), 1)] for _, (x, y) in upright_corner_points(s["rbox"])] \
                if s.get("rbox") else []
            if not v:
                out.append({**face, "corner_pts": pts}); continue
            lab, p = max(v.items(), key=lambda kv: kv[1])
            multi = len(t) == 2 and t[0][0] != t[1][0] and min(t[0][1], t[1][1]) >= MULTI_CORNER_CONF
            out.append({"label": lab, "conf": round(p / k, 3), "top": round(max(q for _, q in t), 3), "multi": multi,
                        "corners": [[c, round(q, 3)] for c, q in t], **face, "corner_pts": pts})
        return out

    def scan_segs(self, frame: np.ndarray, segs: list[dict], min_conf: float = 0.9, step: float = 0.5) -> list[list]:
        """牌內切片：沿轉正牌面四邊掃描，找 tl／br 以外的牌角索引（一框多張時後面那張牌）。
        回傳每個 seg 的 [[牌名, 信心, 原圖 x, 原圖 y, 是否轉180°], ...]（只收非 none 且 ≥ min_conf）。"""
        crops, meta = [], []
        for i, s in enumerate(segs):
            u = upright_card(frame, s["rbox"]) if s.get("rbox") else None
            if u is None:
                continue
            _, w, h = u
            for (x, y), flip, c in upright_scan_crops(frame, s["rbox"], step=step):
                crops.append(c); meta.append((i, x / w, y / h, flip))
        out: list[list] = [[] for _ in segs]
        for (i, fx, fy, flip), probs in zip(meta, self.predict(crops)):
            if not probs or probs.get("none", 0.0) >= NONE_CONF:
                continue
            lab, p = max(((k, v) for k, v in probs.items() if k != "none"), key=lambda kv: kv[1], default=(None, 0))
            if lab and p >= min_conf:
                out[i].append([lab, round(p, 3), *upright_to_frame(segs[i]["rbox"], fx, fy), flip])
        return out


class CardSegmenter:
    """整張牌偵測：分割每張牌的露出區域，回傳多邊形、貼合的旋轉框（minAreaRect 四角）與信心度。"""

    def __init__(self, weights: str, imgsz: int = 640, device: str | int | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.device = imgsz, _device(device)

    def __call__(self, frame: np.ndarray, conf: float = 0.4) -> list[dict]:
        import cv2

        r = self.model.predict(frame, imgsz=self.imgsz, conf=conf, device=self.device, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            return []
        out = []
        for poly, cf in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
            if len(poly) < 3:
                continue
            pts = poly.astype(np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            out.append({"poly": pts.round(1).tolist(), "rbox": cv2.boxPoints(((cx, cy), (w, h), ang)).round(1).tolist(),
                        "center": [round(float(cx), 1), round(float(cy), 1)], "conf": round(float(cf), 3),
                        "area": round(float(cv2.contourArea(pts)), 1)})
        return out


def load_spec(key: str, path: str | Path = ROOT / "models.yaml") -> dict | None:
    """models.yaml 的指定區塊（card_seg／value_cls）；權重不存在回傳 None。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = cfg.get(key)
    return spec if spec and (ROOT / spec["weights"]).exists() else None
