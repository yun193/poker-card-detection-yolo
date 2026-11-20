"""偵測器後端：Ultralytics（YOLOv8 / YOLO12 / YOLO26）。

所有後端都是 callable：frame(BGR ndarray) -> list[Detection]。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .cards import Detection

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

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        return [Detection(self.names[c], float(p), tuple(map(float, b)))
                for b, c, p in zip(xyxy, cls, cf)]


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
