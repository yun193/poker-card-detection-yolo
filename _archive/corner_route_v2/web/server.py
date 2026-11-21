"""真牌辨識測試台的後端：瀏覽器送影格 → YOLO 推論（可多尺度集成）→ 牌角合併 → 位置追蹤投票。

啟動：
    .venv/Scripts/python.exe -m uvicorn web.server:app --port 8765
然後瀏覽器開 http://localhost:8765
除錯監看（記錄最新影格與辨識歷史，供 /api/debug/*）預設關閉；要開時先設環境變數 CARD_DEBUG=1。

WebSocket /ws 協定：
    client → server  binary：一張 JPEG 影格
                     text  ：{"type": "settings", ...}
                             {"type": "cmd", "cmd": "reset"}
                             {"type": "cmd", "cmd": "report", "note": "Kc 被認成 As"}   ← 回報誤判，存下最後一幀
    server → client  {"type": "result", ...}（回覆每張影格）、{"type": "state", ...}（回覆設定／指令）
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from blackjack_vision.cards import (CardTracker, fuse_scales, group_corners, normalize_label, refine_with_classifier,
                                    corner_owners, seg_card_values, seg_memory_entry, verify_segs)
from blackjack_vision.detector import (ROOT, CardSegmenter, CornerClassifier, build_detector, load_card_seg,
                                      load_refiner, load_registry, load_value_cls)

STATIC = Path(__file__).parent / "static"
REAL_TEST = ROOT / "data" / "real_test"
HARD_CASES = ROOT / "data" / "hard_cases"
DEBUG_FRAMES = ROOT / "data" / "debug_frames"   # 僅 DEBUG 時每 2 秒存一幀
BASE_SCALE = 416                     # 訓練解析度：集成時固定包含，近距離牌最準
DEBUG = os.environ.get("CARD_DEBUG") == "1"

app = FastAPI(title="Card Vision")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_detectors: dict = {}
_gpu_lock = threading.Lock()   # 多分頁同時連線時，推論排隊進 GPU
_latest: dict = {"jpeg": None, "result": None, "t": 0.0}   # 僅 DEBUG 時記錄
_history: deque = deque(maxlen=600)


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


def scales_for(imgsz: int, ensemble: bool) -> list[int]:
    if not ensemble:
        return [imgsz]
    return [BASE_SCALE, imgsz if imgsz != BASE_SCALE else 640]


_refiner: dict = {}


def get_refiner():
    """牌角二次分類器（models.yaml 的 refiner）；權重不存在回傳 None。"""
    spec = load_refiner()
    if spec is None:
        return None
    if "model" not in _refiner:
        _refiner["model"] = CornerClassifier(str(ROOT / spec["weights"]), imgsz=spec.get("imgsz", 128),
                                             pad=spec.get("pad", 0.25))
        _refiner["weight"] = float(spec.get("weight", 0.6))
    return _refiner


VERIFY_CONF = 0.2   # 整張牌框交叉驗證用的牌角候選門檻：只問「這裡有沒有牌角」，不看標籤，可放很低
SEG_MEMORY_S = 1.0  # 整張牌框記憶的時間窗（秒）
SEG_VALUE_MIN = 0.6  # B-2 牌值顯示門檻：實測信心 ≥ 0.6 的 39 張全對，錯的 9 張都 < 0.6；多牌框也多半落在門檻下


def run_detect(det, img: np.ndarray, imgsz: int, ensemble: bool, conf: float, refine: bool = False,
               with_candidates: bool = False):
    """第一階段：單尺度直接推論；集成時各尺度先用較低門檻收集候選，再由 fuse_scales 投票並套用 conf。
    第二階段（refine）：從原始畫面裁出每個牌角放大，交給分類器，與偵測結果加權合併。
    with_candidates：另回傳各尺度 ≥ VERIFY_CONF 的原始牌角框（未投票），供整張牌框交叉驗證。"""
    scales = scales_for(imgsz, ensemble)
    per_conf = min(conf, 0.25) if len(scales) > 1 else conf
    if with_candidates:
        per_conf = min(per_conf, VERIFY_CONF)
    ref = get_refiner() if refine else None
    with _gpu_lock:
        per_scale = [det(img, imgsz=s, conf=per_conf) for s in scales]
        dets = fuse_scales(per_scale, conf=conf)
        if ref and dets:
            dets = refine_with_classifier(dets, ref["model"](img, dets), ref["weight"])
    if with_candidates:
        return dets, scales, [d for ds in per_scale for d in ds if d.conf >= VERIFY_CONF]
    return dets, scales


_card_seg: dict = {}


_value_cls: dict = {}


def get_value_cls():
    """方案 B-2 牌值分類器：優先用 value_cls（轉正牌角專用、含 none 類），沒有時退回牌角二次分類器（零訓練）。"""
    spec = load_value_cls()
    if spec is not None:
        if "model" not in _value_cls:
            _value_cls["model"] = CornerClassifier(str(ROOT / spec["weights"]), imgsz=spec.get("imgsz", 128))
        return _value_cls["model"]
    ref = get_refiner()
    return ref["model"] if ref else None


def get_card_seg():
    """整張牌偵測模型（models.yaml 的 card_seg）；權重不存在回傳 None。"""
    spec = load_card_seg()
    if spec is None:
        return None
    if "model" not in _card_seg:
        _card_seg["model"] = CardSegmenter(str(ROOT / spec["weights"]), imgsz=spec.get("imgsz", 640))
    return _card_seg["model"]


def seg_hand(segs: list[dict]) -> list[str]:
    """整張牌框（一框多牌值）彙總成手牌：所有框的牌值去重、依信心排序。"""
    best: dict[str, float] = {}
    for s in segs:
        for v in s.get("values", []):
            best[v["label"]] = max(best.get(v["label"], 0.0), v["conf"])
    return sorted(best, key=lambda k: -best[k])


class Session:
    def __init__(self):
        models = list(load_registry())
        self.s = dict(model=models[0] if models else None, conf=0.45, smooth=True, window=8,
                      dist_factor=6.0, imgsz=832, ensemble=True, refine=True,
                      view="corner",   # view：corner＝牌角辨識、card＝整張牌框、both＝兩者
                      seg_verify=True,  # both 模式：整張牌框內要有牌角偵測才保留（濾掉門、畫框等背景）
                      seg_value=True)   # 方案 B-2：整張牌框轉正裁牌角分類出牌值（需牌角分類器）
        self.tracker = CardTracker(memory=self.s["window"])
        self.seg_memory: deque = deque()   # [(時間, 通過牌角驗證的框)]，只留最近 SEG_MEMORY_S 秒（以時間計，與幀率無關）
        self.last_jpeg: bytes | None = None
        self.last_result: dict | None = None

    def apply(self, new: dict):
        old = dict(self.s)
        self.s.update({k: v for k, v in new.items() if k in self.s})
        if old["model"] != self.s["model"] or old["window"] != self.s["window"]:
            self.tracker = CardTracker(memory=int(self.s["window"]))

    def command(self, data: dict) -> dict | None:
        cmd = data.get("cmd", "")
        if cmd == "reset":
            self.tracker.reset()
        elif cmd == "report":
            return self.report(str(data.get("note", "")))
        return None

    def report(self, note: str) -> dict:
        """把最後一幀原圖＋偵測結果＋使用者備註存到 data/hard_cases/，供分析與第二輪微調。"""
        if self.last_jpeg is None:
            return {"type": "report", "ok": False, "msg": "還沒有影格"}
        HARD_CASES.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
        (HARD_CASES / f"{stem}.jpg").write_bytes(self.last_jpeg)
        meta = {"note": note, "settings": self.s, "result": self.last_result}
        (HARD_CASES / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"type": "report", "ok": True, "msg": f"已存 {stem}", "count": len(list(HARD_CASES.glob("*.jpg")))}

    def state(self) -> dict:
        return {"type": "state", "settings": self.s, "models": list(load_registry()), "refiner": load_refiner() is not None,
                "card_seg": load_card_seg() is not None}

    def add_seg_values(self, img: np.ndarray, segs: list[dict], dets=()) -> float:
        """方案 B-2：每個整張牌框轉正裁牌角分類，寫入 seg["value"] = {"label", "conf", "shown", ...}；
        再合併兩角各自結果與框內牌角偵測（dets，兩者並列時才有），寫入 seg["values"]（一框多牌值）。回傳耗時（ms）。"""
        clf = get_value_cls() if self.s["seg_value"] and segs else None
        if clf is None:
            return 0.0
        t0 = time.perf_counter()
        with _gpu_lock:
            preds = clf.classify_segs(img, segs)
        owners = corner_owners(segs, list(dets))
        for i, (s, v) in enumerate(zip(segs, preds)):
            if v:
                v["shown"] = v["conf"] >= SEG_VALUE_MIN   # 低於門檻前端顯示「?」（multi 時顯示「多張?」）
                s["value"] = v
            s["values"] = seg_card_values((v or {}).get("corners", []), [d for d, o in zip(dets, owners) if o == i],
                                          SEG_VALUE_MIN, 0.0)   # dets 已過顯示門檻（使用者設定），不再二次篩
        return round((time.perf_counter() - t0) * 1000, 1)


    def process(self, img: np.ndarray, t: float | None = None) -> dict:
        """t：影格時間（秒），離線重播時傳入原始時間戳，讓整張牌框記憶的時間窗正確；預設現在。"""
        view = self.s["view"]
        seg = get_card_seg() if view in ("card", "both") else None
        segs, seg_ms = [], 0.0
        if seg is not None:
            t0 = time.perf_counter()
            with _gpu_lock:
                segs = seg(img, conf=float(self.s["conf"]))
            seg_ms = (time.perf_counter() - t0) * 1000
        if view == "card":   # 只看整張牌框：不跑牌角偵測（牌值只來自 B-2）
            h, w = img.shape[:2]
            err = None if seg is not None else "尚未下載整張牌偵測模型（models/card_seg/）"
            seg_value_ms = self.add_seg_values(img, segs)
            res = {"type": "result", "w": w, "h": h, "infer_ms": round(seg_ms, 1), "scales": [],
                   "detections": [], "refined": False, "cards": [], "stable": [], "segs": segs, "seg_ms": round(seg_ms, 1),
                   "seg_value_ms": seg_value_ms, "seg_hand": seg_hand(segs)}
            if err:
                res["error"] = err
            return res
        if not self.s["model"]:
            return {"type": "result", "error": "沒有可用模型：請把權重放到 models/ 底下"}
        det = get_detector(self.s["model"])
        t0 = time.perf_counter()
        verify = bool(segs) and self.s["seg_verify"]
        dets, scales, *cands = run_detect(det, img, int(self.s["imgsz"]), bool(self.s["ensemble"]),
                                          float(self.s["conf"]), bool(self.s["refine"]), with_candidates=verify)
        infer_ms = (time.perf_counter() - t0) * 1000
        h, w = img.shape[:2]
        segs_rejected = []
        if verify:   # 用低門檻原始牌角候選驗證（顯示仍用使用者門檻的 dets）
            now = time.time() if t is None else t
            while self.seg_memory and now - self.seg_memory[0][0] > SEG_MEMORY_S:
                self.seg_memory.popleft()
            segs, segs_rejected = verify_segs(segs, cands[0], memory=[m for _, f in self.seg_memory for m in f])
            self.seg_memory.append((now, seg_memory_entry(segs)))
        # 一框多牌值只用顯示門檻、已投票＋二次分類的牌角：實測改用低門檻原始候選時，新增的牌值錯多於對
        seg_value_ms = self.add_seg_values(img, segs, dets)
        cards = group_corners(dets, float(self.s["dist_factor"]))
        card_json = []
        if self.s["smooth"]:
            confirmed = {id(t) for t in self.tracker.confirmed()}
            pairs = self.tracker.update(cards)
            confirmed |= {id(t) for t in self.tracker.confirmed()}
            for c, t in pairs:
                card_json.append({"label": t.label, "raw": c.label, "conf": round(c.conf, 3),
                                  "center": [round(v, 1) for v in c.center], "corners": c.corners,
                                  "track": t.id, "agree": round(t.score, 2), "confirmed": id(t) in confirmed})
            stable = sorted(t.label for t in self.tracker.confirmed())
        else:
            for c in cards:
                card_json.append({"label": c.label, "raw": c.label, "conf": round(c.conf, 3),
                                  "center": [round(v, 1) for v in c.center], "corners": c.corners,
                                  "track": None, "agree": 1.0, "confirmed": True})
            stable = sorted(c.label for c in cards)
        return {
            "type": "result", "w": w, "h": h, "infer_ms": round(infer_ms, 1), "scales": scales,
            "detections": [{"label": d.label, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box],
                            "det_label": d.det_label or d.label, "cls_conf": round(d.cls_conf, 3)} for d in dets],
            "refined": any(d.det_label for d in dets),
            "cards": card_json, "stable": stable, "segs": segs, "seg_ms": round(seg_ms, 1),
            "segs_rejected": segs_rejected, "seg_value_ms": seg_value_ms, "seg_hand": seg_hand(segs),
        }


@app.on_event("startup")
def warmup():
    """背景預先載入預設模型，並在常用尺度各跑一次（首次載入＋CUDA 初始化約 10~15 秒）。"""
    def run():
        reg = load_registry()
        if reg:
            det = get_detector(next(iter(reg)))
            with _gpu_lock:
                for s in (416, 832):
                    det(np.zeros((480, 640, 3), np.uint8), imgsz=s)
    threading.Thread(target=run, daemon=True).start()


if DEBUG:
    @app.get("/api/debug/latest.jpg")
    def debug_latest_jpg():
        """最新一幀，伺服器端畫上偵測框與標籤（供遠端除錯檢視）。"""
        if _latest["jpeg"] is None:
            raise HTTPException(404, "尚未收到任何影格")
        img = decode_jpeg(_latest["jpeg"])
        r = _latest["result"]
        for d in r.get("detections", []):
            x1, y1, x2, y2 = map(int, d["box"])
            col = (60, 207, 122) if d["conf"] >= 0.8 else (78, 193, 242) if d["conf"] >= 0.5 else (75, 83, 229)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        for c in r.get("cards", []):
            x, y = map(int, c["center"])
            cv2.putText(img, f'[{c["label"]}]x{c["corners"]}', (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")

    @app.get("/api/debug/history")
    def debug_history(n: int = 200):
        """最近 n 幀的辨識摘要。"""
        items = list(_history)[-n:]
        return {"age_s": round(time.time() - _latest["t"], 1) if _latest["t"] else None, "n": len(items), "items": items}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def models():
    out = {}
    for name in load_registry():
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
    hard = len(list(HARD_CASES.glob("*.jpg"))) if HARD_CASES.exists() else 0
    return {"count": len(files), "recent": [f.name for f in files[-10:]], "hard_cases": hard}


class EvalReq(BaseModel):
    model: str
    conf: float = 0.45
    imgsz: int = 640
    ensemble: bool = False
    refine: bool = True


@app.post("/api/evaluate")
async def evaluate(req: EvalReq):
    from evaluate import evaluate_model
    files = sorted(REAL_TEST.glob("*.jpg")) if REAL_TEST.exists() else []
    if not files:
        raise HTTPException(400, "data/real_test/ 還沒有照片，先用「存成測試照」累積")

    def run():
        det = get_detector(req.model)
        return evaluate_model(lambda f: run_detect(det, f, req.imgsz, req.ensemble, req.conf, req.refine)[0], files)
    r = await asyncio.to_thread(run)
    r["errors"] = r["errors"][:20]
    r["n"] = len(files)
    r["scales"] = scales_for(req.imgsz, req.ensemble)
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
                    sess.last_jpeg, sess.last_result = msg["bytes"], res
                    if DEBUG:
                        _latest.update(jpeg=msg["bytes"], result=res, t=time.time())
                        _history.append({"t": round(time.time(), 2), "stable": res["stable"],
                                         "raw": sorted(c["raw"] for c in res["cards"]),
                                         "corners": len(res["detections"]), "ms": res["infer_ms"],
                                         "segs": [round(g["conf"], 2) for g in res.get("segs", [])],
                                         "verify": [g.get("verify") for g in res.get("segs", [])],
                                         "values": [(g["value"]["label"], g["value"]["conf"]) if g.get("value") else None
                                                    for g in res.get("segs", [])],
                                         "rejected": len(res.get("segs_rejected", [])),
                                         "seg_ms": res.get("seg_ms", 0), "settings": dict(sess.s)})
                        if time.time() - _latest.get("saved", 0) >= 2:   # 每 2 秒存一幀原圖＋結果，供離線檢視
                            _latest["saved"] = time.time()
                            DEBUG_FRAMES.mkdir(parents=True, exist_ok=True)
                            stem = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
                            (DEBUG_FRAMES / f"{stem}.jpg").write_bytes(msg["bytes"])
                            (DEBUG_FRAMES / f"{stem}.json").write_text(
                                json.dumps({"settings": sess.s, "result": res}, ensure_ascii=False), encoding="utf-8")
                except Exception as e:  # 單張壞圖不中斷連線
                    res = {"type": "result", "error": f"{type(e).__name__}: {e}"}
                await sock.send_json(res)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                reply = None
                if data.get("type") == "settings":
                    sess.apply(data)
                elif data.get("type") == "cmd":
                    reply = sess.command(data)
                await sock.send_json(reply or sess.state())
    except WebSocketDisconnect:
        pass
