"""偵測器後端：Ultralytics（YOLOv8 / YOLO12 / YOLO26）。

所有後端都是 callable：frame(BGR ndarray) -> list[Detection]。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .cards import Detection
from .upright import upright_card, upright_corner_crops  # noqa: F401（B-2 轉正裁角，供外部沿用）

ROOT = Path(__file__).resolve().parent.parent


class UltralyticsDetector:
    def __init__(self, weights: str, conf: float = 0.4, imgsz: int = 416,
                 device: str | int | None = None):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.names = self.model.names

    def __call__(self, frame: np.ndarray, imgsz: int | None = None, conf: float | None = None) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf if conf is None else conf, imgsz=imgsz or self.imgsz,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        return [Detection(self.names[c], float(p), tuple(map(float, b)))
                for b, c, p in zip(xyxy, cls, cf)]


class CornerClassifier:
    """第二階段：從原始全解析度畫面裁出牌角（外擴 pad 成正方形）→ 52 類分類。

    pad 需與訓練時一致（kaggle/train_corner_cls.py 的 PAD）。
    """

    def __init__(self, weights: str, imgsz: int = 128, pad: float = 0.25, device: str | int | None = None):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(weights)
        self.imgsz, self.pad = imgsz, pad
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.names = self.model.names

    def crop(self, frame: np.ndarray, box) -> np.ndarray | None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        s = max(x2 - x1, y2 - y1) * (1 + 2 * self.pad)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b = int(cx - s / 2), int(cy - s / 2)
        c = frame[max(0, b):min(H, b + int(s)), max(0, a):min(W, a + int(s))]
        return c if c.size and min(c.shape[:2]) >= 6 else None

    def _predict(self, crops: list) -> list[dict[str, float]]:
        idx = [i for i, c in enumerate(crops) if c is not None]
        out: list[dict[str, float]] = [{} for _ in crops]
        if not idx:
            return out
        res = self.model.predict([crops[i] for i in idx], imgsz=self.imgsz, device=self.device, verbose=False)
        for i, r in zip(idx, res):
            p = r.probs
            out[i] = {self.names[int(k)]: float(v) for k, v in zip(p.top5, p.top5conf)}
        return out

    def __call__(self, frame: np.ndarray, dets: list[Detection]) -> list[dict[str, float]]:
        """每個偵測框回傳 {標籤: 機率}（只留前 5 名）；裁不出來的回傳空 dict。"""
        return self._predict([self.crop(frame, d.box) for d in dets])

    def classify_segs(self, frame: np.ndarray, segs: list[dict]) -> list[dict]:
        """方案 B-2：整張牌框轉正 → 裁左上角與（轉 180° 的）右下角的牌角 → 分類，兩角機率平均後取最高。
        正立或倒立的牌，兩個角轉完都應是正立的同一牌值；只取單角最高信心會被倒過來的 6／9 誤導。
        回傳每個 seg 的 {"label", "conf", "multi"}；裁不出來的回傳 {}。
        multi：兩個角都有牌角、且各自有把握地認成不同牌 → 這個框多半框住了兩張以上的牌（分割合併），牌值不可信。"""
        crops, owner = [], []
        for i, s in enumerate(segs):
            for _, c in (upright_corner_crops(frame, s["rbox"]) if s.get("rbox") else []):
                crops.append(c); owner.append(i)
        votes: list[dict[str, float]] = [{} for _ in segs]
        tops: list[list[tuple[str, float]]] = [[] for _ in segs]
        n = [0] * len(segs)
        for i, probs in zip(owner, self._predict(crops)):
            if probs.get("none", 0.0) >= 0.5:   # value_cls 判定這一角沒有牌角（空白角、別張牌、手指）→ 不投票
                continue
            n[i] += 1
            real = {lab: p for lab, p in probs.items() if lab != "none"}
            if real:
                tops[i].append(max(real.items(), key=lambda kv: kv[1]))
            for lab, p in real.items():
                votes[i][lab] = votes[i].get(lab, 0.0) + p
        out: list[dict] = []
        for v, k, t in zip(votes, n, tops):
            if not v:
                out.append({}); continue
            lab, p = max(v.items(), key=lambda kv: kv[1])
            multi = len(t) == 2 and t[0][0] != t[1][0] and min(t[0][1], t[1][1]) >= MULTI_CORNER_CONF
            out.append({"label": lab, "conf": round(p / k, 3), "multi": multi})
        return out


MULTI_CORNER_CONF = 0.6   # 兩角各自信心都 ≥ 此值卻認成不同牌 → 判定框住多張牌


class CardSegmenter:
    """整張牌偵測：分割每張牌的露出區域，回傳多邊形、貼合的旋轉框（minAreaRect 四角）與信心度。"""

    def __init__(self, weights: str, imgsz: int = 640, device: str | int | None = None):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")

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


def load_card_seg(path: str | Path = ROOT / "models.yaml") -> dict | None:
    """models.yaml 的 card_seg 區塊（整張牌偵測）；權重不存在回傳 None。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = cfg.get("card_seg")
    return spec if spec and (ROOT / spec["weights"]).exists() else None


def load_value_cls(path: str | Path = ROOT / "models.yaml") -> dict | None:
    """models.yaml 的 value_cls 區塊（方案 B-2 轉正牌角分類，含 none 類）；權重不存在回傳 None。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = cfg.get("value_cls")
    return spec if spec and (ROOT / spec["weights"]).exists() else None


def load_refiner(path: str | Path = ROOT / "models.yaml") -> dict | None:
    """models.yaml 的 refiner 區塊（牌角二次分類）；權重不存在回傳 None。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = cfg.get("refiner")
    return spec if spec and (ROOT / spec["weights"]).exists() else None


def load_registry(path: str | Path = ROOT / "models.yaml") -> dict[str, dict]:
    """讀 models.yaml，只回傳權重檔已存在的模型。路徑相對於專案根目錄。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ready = {}
    for name, spec in cfg["models"].items():
        if (ROOT / spec["weights"]).exists():
            ready[name] = spec
    return ready


def build_detector(spec: dict, conf: float = 0.4):
    kind = spec["type"]
    if kind == "ultralytics":
        return UltralyticsDetector(str(ROOT / spec["weights"]), conf=conf, imgsz=spec.get("imgsz", 416))
    raise ValueError(f"未知模型類型: {kind}")
