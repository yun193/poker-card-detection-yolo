"""真牌測試前端的後端：瀏覽器送影格 → YOLO 推論 → 牌角整合 → 21 點建議。

啟動：
    .venv/Scripts/python.exe -m uvicorn web.server:app --port 8765
然後瀏覽器開 http://localhost:8765

WebSocket /ws 協定：
    client → server  binary：一張 JPEG 影格
                     text  ：{"type": "settings", ...} 或 {"type": "cmd", "cmd": "next_round"|"reset_shoe"|"reset_table"}
    server → client  {"type": "result", ...}（回覆每張影格）、{"type": "state", ...}（回覆設定／指令）
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from blackjack_vision.advisor import Advice, Advisor
from blackjack_vision.blackjack import Rules
from blackjack_vision.cards import StableTable, group_corners, normalize_label, split_zones
from blackjack_vision.detector import ROOT, build_detector, load_registry

STATIC = Path(__file__).parent / "static"
REAL_TEST = ROOT / "data" / "real_test"

app = FastAPI(title="Blackjack Vision")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_detectors: dict = {}
_gpu_lock = threading.Lock()   # 多分頁同時連線時，推論排隊進 GPU


def get_detector(name: str):
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"模型不存在或權重未下載：{name}")
    if name not in _detectors:
        _detectors[name] = build_detector(reg[name])
    return _detectors[name]


def decode_jpeg(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("無法解碼影像")
    return img


def advice_json(adv: Advice) -> dict:
    odds = {}
    for k, o in adv.odds.items():
        q = o.scaled(0.5) if k == "double" else o   # 加倍顯示機率，EV 保留 2 倍注
        odds[k] = {"win": q.win, "push": q.push, "lose": q.lose, "ev": o.ev}
    return {"total": adv.player_total, "soft": adv.player_soft, "dealer_up": adv.dealer_up,
            "strategy": adv.strategy, "strategy_name": adv.strategy_name if adv.strategy else None,
            "best": adv.best_ev_action, "odds": odds, "note": adv.note}


class Session:
    RULE_KEYS = ("decks", "h17", "peek")

    def __init__(self):
        models = list(load_registry())
        self.s = dict(model=models[0] if models else None, conf=0.45, split=0.45, dealer_top=True,
                      decks=6, h17=False, peek=True, smooth=True, window=8)
        self.table = StableTable(window=self.s["window"])
        self.advisor = self._new_advisor()
        self._last_labels: list[str] = []

    def _new_advisor(self) -> Advisor:
        return Advisor(Rules(decks=int(self.s["decks"]), dealer_hits_soft17=bool(self.s["h17"]),
                             dealer_peeks=bool(self.s["peek"])))

    def apply(self, new: dict):
        old = dict(self.s)
        self.s.update({k: v for k, v in new.items() if k in self.s})
        if any(old[k] != self.s[k] for k in self.RULE_KEYS):
            self.advisor = self._new_advisor()
        if old["model"] != self.s["model"] or old["window"] != self.s["window"]:
            self.table = StableTable(window=int(self.s["window"]))

    def command(self, cmd: str):
        if cmd == "next_round":
            self.advisor.shoe.end_round(self._last_labels)
            self.table.reset()
        elif cmd == "reset_shoe":
            self.advisor.shoe.reset()
        elif cmd == "reset_table":
            self.table.reset()

    def state(self) -> dict:
        shoe = self.advisor.shoe
        return {"type": "state", "settings": self.s, "seen": len(shoe.seen),
                "true_count": round(shoe.hi_lo_true_count(), 2), "models": list(load_registry())}

    def process(self, img: np.ndarray) -> dict:
        if not self.s["model"]:
            return {"type": "result", "error": "沒有可用模型：請把權重放到 models/ 底下"}
        det = get_detector(self.s["model"])
        det.conf = float(self.s["conf"])
        t0 = time.perf_counter()
        with _gpu_lock:
            dets = det(img)
        infer_ms = (time.perf_counter() - t0) * 1000
        h, w = img.shape[:2]
        cards = group_corners(dets)
        dc, pc = split_zones(cards, h, float(self.s["split"]), bool(self.s["dealer_top"]))
        if self.s["smooth"]:
            dealer, player = self.table.update(dc, pc)
        else:
            dealer, player = sorted(c.label for c in dc), sorted(c.label for c in pc)
        self._last_labels = dealer + player
        adv = self.advisor.advise(dealer, player)
        dealer_ids = {id(c) for c in dc}
        return {
            "type": "result", "w": w, "h": h, "infer_ms": round(infer_ms, 1),
            "detections": [{"label": d.label, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box]}
                           for d in dets],
            "cards": [{"label": c.label, "conf": round(c.conf, 3), "center": [round(v, 1) for v in c.center],
                       "corners": c.corners, "zone": "dealer" if id(c) in dealer_ids else "player"} for c in cards],
            "dealer": dealer, "player": player, "advice": advice_json(adv),
            "seen": len(self.advisor.shoe.seen), "true_count": round(self.advisor.shoe.hi_lo_true_count(), 2),
        }


@app.on_event("startup")
def warmup():
    """背景預先載入預設模型並跑一次推論（首次載入＋CUDA 初始化約 10~15 秒）。"""
    def run():
        reg = load_registry()
        if reg:
            det = get_detector(next(iter(reg)))
            with _gpu_lock:
                det(np.zeros((416, 416, 3), np.uint8))
    threading.Thread(target=run, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def models():
    reg = load_registry()
    out = {}
    for name in reg:
        p = ROOT / "models" / name / "metrics.json"
        out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"models": out}


class CaptureReq(BaseModel):
    image: str           # dataURL 或純 base64 的 JPEG
    truth: str           # 例 "As-10h-Kd"，空牌寫 "none"


@app.post("/api/capture")
def capture(req: CaptureReq):
    tag = req.truth.strip()
    if tag.lower() in ("", "none"):
        tag = "none"
    else:
        try:
            tag = "-".join(sorted(normalize_label(t) for t in tag.replace(" ", "").split("-") if t))
        except ValueError as e:
            raise HTTPException(400, str(e))
    data = base64.b64decode(req.image.split(",", 1)[-1])
    decode_jpeg(data)
    REAL_TEST.mkdir(parents=True, exist_ok=True)
    path = REAL_TEST / f"{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    path.write_bytes(data)
    return {"saved": path.name, "count": len(list(REAL_TEST.glob("*.jpg")))}


@app.get("/api/realtest")
def realtest():
    files = sorted(REAL_TEST.glob("*.jpg")) if REAL_TEST.exists() else []
    return {"count": len(files), "recent": [f.name for f in files[-10:]]}


class EvalReq(BaseModel):
    model: str
    conf: float = 0.45


@app.post("/api/evaluate")
async def evaluate(req: EvalReq):
    from evaluate import evaluate_model
    files = sorted(REAL_TEST.glob("*.jpg")) if REAL_TEST.exists() else []
    if not files:
        raise HTTPException(400, "data/real_test/ 還沒有照片，先用「存成測試照」累積")

    def run():
        det = get_detector(req.model)
        det.conf = req.conf
        with _gpu_lock:
            return evaluate_model(det, files)
    r = await asyncio.to_thread(run)
    r["errors"] = r["errors"][:20]
    r["n"] = len(files)
    return r


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    sess = Session()
    await sock.send_json(sess.state())
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                try:
                    img = decode_jpeg(msg["bytes"])
                    res = await asyncio.to_thread(sess.process, img)
                except Exception as e:  # 單張壞圖不中斷連線
                    res = {"type": "result", "error": f"{type(e).__name__}: {e}"}
                await sock.send_json(res)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "settings":
                    sess.apply(data)
                elif data.get("type") == "cmd":
                    sess.command(data.get("cmd", ""))
                await sock.send_json(sess.state())
    except WebSocketDisconnect:
        pass
