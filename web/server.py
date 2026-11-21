"""真牌辨識測試台的後端（整張牌路線）：
瀏覽器送影格 → card_seg 分割整張牌 → value_cls 轉正讀兩個牌角 → 背景框過濾 → 牌內切片找後面的牌 → 跨幀追蹤投票。

啟動：
    .venv/Scripts/python.exe -m uvicorn web.server:app --port 8765
然後瀏覽器開 http://localhost:8765
除錯監看（每 2 秒存一幀到 data/debug_frames/）預設關閉；要開時先設環境變數 CARD_DEBUG=1。
舊的牌角路線（yolov8 牌角偵測＋corner_cls）已封存到 _archive/corner_route_v2/。

WebSocket /ws 協定：
    client → server  binary：一張 JPEG 影格
                     text  ：{"type": "settings", ...}
                             {"type": "frame_t", "t": 1732170000.12}   ← 重播時，下一張影格的原始時間戳（秒）
                             {"type": "cmd", "cmd": "reset" | "record_start" | "record_stop"}
                             {"type": "cmd", "cmd": "report", "note": "Kc 被認成 As"}   ← 回報誤判，存下最後一幀
    server → client  {"type": "result", ...}（回覆每張影格）、{"type": "state", ...}（回覆設定／指令）
"""
from __future__ import annotations

import asyncio
import base64
import glob
import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from blackjack_vision.cards import HandSmoother, SegTracker, is_card_evidence, looks_like_card, normalize_label, seg_card_values, seg_memory_entry, verify_segs
from blackjack_vision.detector import ROOT, CardSegmenter, ValueClassifier, load_spec

STATIC = Path(__file__).parent / "static"
DATA = ROOT / "data"
REAL_TEST = DATA / "real_test"
HARD_CASES = DATA / "hard_cases"
RECORDINGS = DATA / "recordings"
DEBUG_FRAMES = DATA / "debug_frames"   # 僅 DEBUG 時每 2 秒存一幀
DEBUG = os.environ.get("CARD_DEBUG") == "1"
RECORD_MAX_S = 60                      # 單次錄製上限（秒）
SEG_MEMORY_S = 1.0                     # 背景過濾 memory 的時間窗（秒，與幀率無關）
SCAN_WEIGHT = 0.5                      # 牌內切片讀值的投票權重（相對兩個標準牌角）
STAMP = "%Y%m%d_%H%M%S_%f"             # 影格檔名＝時間戳，重播時依此還原時間

app = FastAPI(title="Card Vision")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_cache_pages(request, call_next):
    """頁面與前端檔案每次都向伺服器確認，改版後瀏覽器不會拿到舊的 app.js。"""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

_models: dict = {}
_gpu_lock = threading.Lock()   # 多分頁同時連線或回測時，推論排隊進 GPU


def get_card_seg() -> CardSegmenter | None:
    """整張牌分割模型；權重不存在回傳 None。"""
    if "seg" not in _models:
        spec = load_spec("card_seg")
        _models["seg"] = CardSegmenter(str(ROOT / spec["weights"]), imgsz=spec.get("imgsz", 640)) if spec else None
    return _models["seg"]


def get_value_cls() -> ValueClassifier | None:
    """轉正牌角牌值分類器；權重不存在回傳 None。"""
    if "cls" not in _models:
        spec = load_spec("value_cls")
        _models["cls"] = ValueClassifier(str(ROOT / spec["weights"]), imgsz=spec.get("imgsz", 128)) if spec else None
    return _models["cls"]


def decode_jpeg(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("無法解碼影像")
    return img


def imread(p: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)   # cv2.imread 不吃中文路徑


DEFAULTS = dict(conf=0.5,        # 分割信心門檻
                value_min=0.6,   # 牌值顯示門檻：實測 ≥ 0.6 的單幀牌值全對
                verify=True,     # 背景框過濾
                scan=True,       # 牌內切片（找一框多張時後面的牌）
                smooth=True)     # 跨幀追蹤投票（圖片模式關閉）


class Session:
    def __init__(self):
        self.s = dict(DEFAULTS)
        self.tracker = SegTracker()
        self.smoother = HandSmoother()
        self.seg_memory: deque = deque()   # [(時間, 通過驗證的框)]
        self.next_t: float | None = None   # 重播時由 frame_t 指定下一幀的時間
        self.last_jpeg: bytes | None = None
        self.last_result: dict | None = None
        self.rec_dir: Path | None = None
        self.rec_t0 = 0.0

    def apply(self, new: dict):
        self.s.update({k: type(DEFAULTS[k])(v) for k, v in new.items() if k in DEFAULTS})

    def reset(self):
        self.tracker.reset()
        self.smoother.reset()
        self.seg_memory.clear()

    def command(self, data: dict) -> dict | None:
        cmd = data.get("cmd", "")
        if cmd == "reset":
            self.reset()
        elif cmd == "report":
            return self.report(str(data.get("note", "")))
        elif cmd == "record_start":
            self.rec_dir = RECORDINGS / f"{datetime.now():%Y%m%d_%H%M%S}"
            self.rec_dir.mkdir(parents=True, exist_ok=True)
            self.rec_t0 = time.time()
        elif cmd == "record_stop":
            self.rec_dir = None
        return None

    def report(self, note: str) -> dict:
        """把最後一幀原圖＋辨識結果＋使用者備註存到 data/hard_cases/，供分析與微調。"""
        if self.last_jpeg is None:
            return {"type": "report", "ok": False, "msg": "還沒有影格"}
        HARD_CASES.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():{STAMP}}"
        (HARD_CASES / f"{stem}.jpg").write_bytes(self.last_jpeg)
        meta = {"note": note, "settings": self.s, "result": self.last_result}
        (HARD_CASES / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"type": "report", "ok": True, "msg": f"已存 {stem}", "count": len(list(HARD_CASES.glob("*.jpg")))}

    def state(self) -> dict:
        rec = None
        if self.rec_dir is not None:
            rec = {"dir": self.rec_dir.name, "s": round(time.time() - self.rec_t0, 1),
                   "frames": len(list(self.rec_dir.glob("*.jpg")))}
        return {"type": "state", "settings": self.s, "recording": rec,
                "models": {"card_seg": load_spec("card_seg") is not None, "value_cls": load_spec("value_cls") is not None}}

    def record(self, jpeg: bytes, res: dict):
        """錄製中：每一幀原圖＋結果都存（供離線重播、驗證追蹤投票）。超過上限自動停止。"""
        if self.rec_dir is None:
            return
        if time.time() - self.rec_t0 > RECORD_MAX_S:
            self.rec_dir = None
            return
        stem = f"{datetime.now():{STAMP}}"
        (self.rec_dir / f"{stem}.jpg").write_bytes(jpeg)
        (self.rec_dir / f"{stem}.json").write_text(json.dumps({"settings": self.s, "result": res}, ensure_ascii=False),
                                                   encoding="utf-8")

    def process(self, img: np.ndarray, t: float | None = None) -> dict:
        """t：影格時間（秒），重播時傳入原始時間戳，讓記憶與投票的時間窗正確；預設現在。"""
        if t is None:
            t, self.next_t = (self.next_t if self.next_t is not None else time.time()), None
        seg, clf = get_card_seg(), get_value_cls()
        h, w = img.shape[:2]
        if seg is None or clf is None:
            miss = "models/card_seg/" if seg is None else "models/value_cls/"
            return {"type": "result", "error": f"尚未下載模型（{miss}），見 README 的權重下載說明"}
        ms = {}
        t0 = time.perf_counter()
        with _gpu_lock:
            segs = seg(img, conf=self.s["conf"])
            ms["seg"] = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            vals = clf.classify_segs(img, segs) if segs else []
            ms["value"] = (time.perf_counter() - t1) * 1000
        for s, v in zip(segs, vals):
            if "label" in v:
                v["shown"] = v["conf"] >= self.s["value_min"]   # 低於門檻前端顯示「?」（multi 時「多張?」）
            s["value"] = v
        rejected = []
        if self.s["verify"] and segs:
            while self.seg_memory and t - self.seg_memory[0][0] > SEG_MEMORY_S:
                self.seg_memory.popleft()
            segs, rejected = verify_segs(segs, [is_card_evidence(v) for v in vals], [looks_like_card(v) for v in vals],
                                         memory=[m for _, f in self.seg_memory for m in f])
            self.seg_memory.append((t, seg_memory_entry(segs)))
        t2 = time.perf_counter()
        if self.s["scan"] and segs:
            with _gpu_lock:
                scans = clf.scan_segs(img, segs)
        else:
            scans = [[] for _ in segs]
        ms["scan"] = (time.perf_counter() - t2) * 1000
        for s, sc in zip(segs, scans):
            v = s["value"]
            corners = [(c, q) for c, q in v.get("corners", [])]
            s["scan"] = sc
            # 追蹤投票的讀值：主牌值用兩角平均信心（單幀時與顯示門檻 value_min 同尺度），
            # 另一角讀成別的點數（一框多張）與切片讀值各自投票；切片單幀只有約一半正確，票數減半
            main = [(v["label"], v["conf"])] if "label" in v else []
            s["reads"] = main + [(c, q) for c, q in corners if c[:-1] != v["label"][:-1]] + \
                [(x[0], x[1] * SCAN_WEIGHT) for x in sc]
            s["values"] = seg_card_values([(c, q, "corner") for c, q in corners] + [(x[0], x[1], "scan") for x in sc],
                                          self.s["value_min"])
        if self.s["smooth"]:
            self.tracker.min_vote = self.s["value_min"]   # 單幀要達顯示門檻；連續幾幀的弱讀值累積後也能通過（模糊的牌）
            confirmed = {tr.id for tr in self.tracker.confirmed(t)}
            tracks = self.tracker.update(segs, t)
            confirmed |= {tr.id for tr in self.tracker.confirmed(t)}
            for s, tr in zip(segs, tracks):
                s.update(track=tr.id, labels=self.tracker.labels(tr), agree=round(tr.score(), 2),
                         confirmed=tr.id in confirmed)
            hand = self.smoother.update(self.tracker.hand(t), t)   # 手牌再做一層時間平滑，避免每幀跳動
        else:   # 單張圖片：只用兩個牌角的讀值（單幀切片的附加牌值只有約一半正確，不列入手牌）
            best: dict[str, float] = {}
            for s in segs:
                s.update(track=None, labels=[v["label"] for v in s["values"] if v["src"] == "corner"], agree=1.0,
                         confirmed=True)
                for v in s["values"]:
                    if v["src"] == "corner":
                        best[v["label"]] = max(best.get(v["label"], 0.0), v["conf"])
            hand = sorted(best, key=lambda k: -best[k])
        for s in segs:
            s.pop("reads", None)
        ms["total"] = (time.perf_counter() - t0) * 1000
        return {"type": "result", "w": w, "h": h, "t": round(t, 3), "segs": segs, "segs_rejected": rejected,
                "hand": hand, "ms": {k: round(v, 1) for k, v in ms.items()}}


@app.on_event("startup")
def warmup():
    """背景預先載入兩個模型並各跑一次（首次載入＋CUDA 初始化約 10~15 秒）。"""
    def run():
        seg, clf = get_card_seg(), get_value_cls()
        with _gpu_lock:
            if seg:
                seg(np.zeros((720, 1280, 3), np.uint8))
            if clf:
                clf.predict([np.zeros((128, 128, 3), np.uint8)])
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- 重播／回測
def frame_folders() -> dict[str, Path]:
    """可重播的影格資料夾：DEBUG 監看存檔（data/debug_frames*）與錄製（data/recordings/*）。鍵為相對 data/ 的路徑。"""
    out = {}
    for p in sorted(glob.glob(str(DATA / "debug_frames*"))) + sorted(glob.glob(str(RECORDINGS / "*")), reverse=True):
        p = Path(p)
        if p.is_dir() and any(p.glob("*.jpg")):
            out[p.relative_to(DATA).as_posix()] = p
    return out


def frame_time(p: Path) -> float:
    try:
        return datetime.strptime(p.stem, STAMP).timestamp()
    except ValueError:
        return p.stat().st_mtime


def replay_folder(folder: Path, settings: dict | None = None, on_frame=None) -> dict:
    """用全新的 Session 依原始時間戳重播一個資料夾，回傳統計。on_frame(path, img, result) 供外部收集細節。"""
    s = Session()
    s.apply(settings or {})
    st, ms, hands = Counter(), [], []
    for f in sorted(folder.glob("*.jpg")):
        img = imread(f)
        if img is None:
            continue
        r = s.process(img, t=frame_time(f))
        if "error" in r:
            raise RuntimeError(r["error"])
        ms.append(r["ms"]["total"])
        st["frames"] += 1
        st["kept"] += len(r["segs"])
        st["rejected"] += len(r["segs_rejected"])
        for g in r["segs"]:
            n = len(g["labels"])
            st["multi" if n > 1 else ("single" if n else "unknown")] += 1
        hand = "-".join(sorted(r["hand"])) or "none"
        st["hand_changes"] += bool(hands) and hand != hands[-1]
        hands.append(hand)
        if on_frame:
            on_frame(f, img, r)
    out = dict(st)
    out["ms_median"] = round(float(np.median(ms)), 1) if ms else 0.0
    out["top_hands"] = Counter(hands).most_common(5)
    return out


class ReplayReq(BaseModel):
    folder: str
    settings: dict = {}


@app.get("/api/frames")
def list_frames():
    return {"folders": [{"id": k, "frames": len(list(p.glob("*.jpg")))} for k, p in frame_folders().items()]}


@app.get("/api/frames/{folder:path}/list")
def folder_frames(folder: str):
    p = frame_folders().get(folder)
    if p is None:
        raise HTTPException(404, "沒有這個資料夾")
    return {"frames": [{"name": f.name, "t": frame_time(f)} for f in sorted(p.glob("*.jpg"))]}


@app.get("/api/frames/{folder:path}/img/{name}")
def frame_image(folder: str, name: str):
    p = frame_folders().get(folder)
    if p is None or "/" in name or "\\" in name or not (p / name).is_file():
        raise HTTPException(404, "沒有這張影格")
    return FileResponse(p / name, media_type="image/jpeg")


@app.post("/api/backtest")
async def backtest(req: ReplayReq):
    p = frame_folders().get(req.folder)
    if p is None:
        raise HTTPException(404, "沒有這個資料夾")
    try:
        return await asyncio.to_thread(replay_folder, p, req.settings)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- 真牌測試照
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class CaptureReq(BaseModel):
    image: str           # dataURL 或純 base64 的 JPEG
    truth: str           # 例 "As-10h-Kd"，空牌寫 "none"


def truth_from_name(p: Path) -> Counter:
    """檔名前綴是正解：As-10h-Kd_xxx.jpg；none_xxx.jpg 表示沒有牌。"""
    tag = p.stem.split("_")[0]
    return Counter() if tag.lower() == "none" else Counter(normalize_label(x) for x in tag.split("-") if x)


@app.post("/api/capture")
def capture(req: CaptureReq):
    tag = req.truth.strip()
    if tag.lower() in ("", "none"):
        tag = "none"
    else:
        try:
            tag = "-".join(sorted(normalize_label(x) for x in tag.replace(" ", "").split("-") if x))
        except ValueError as e:
            raise HTTPException(400, str(e))
    data = base64.b64decode(req.image.split(",", 1)[-1])
    decode_jpeg(data)
    REAL_TEST.mkdir(parents=True, exist_ok=True)
    path = REAL_TEST / f"{tag}_{datetime.now():{STAMP}}.jpg"
    path.write_bytes(data)
    return {"saved": path.name, "count": len(list(REAL_TEST.glob("*.jpg")))}


@app.get("/api/realtest")
def realtest():
    files = sorted(REAL_TEST.glob("*.jpg")) if REAL_TEST.exists() else []
    hard = len(list(HARD_CASES.glob("*.jpg"))) if HARD_CASES.exists() else 0
    return {"count": len(files), "hard_cases": hard}


def evaluate_realtest(files: list[Path], settings: dict) -> dict:
    """每張測試照單獨辨識（不追蹤），手牌與檔名正解比對：精確率、召回率、整張全對率。"""
    tp = fp = fn = exact = 0
    errors, ms = [], []
    for f in files:
        s = Session(); s.apply({**settings, "smooth": False})
        img = imread(f)
        r = s.process(img) if img is not None else {"error": "讀不到圖"}
        if "error" in r:
            raise RuntimeError(r["error"])
        ms.append(r["ms"]["total"])
        pred, truth = Counter(r["hand"]), truth_from_name(f)
        hit = sum((pred & truth).values())
        tp, fp, fn = tp + hit, fp + sum(pred.values()) - hit, fn + sum(truth.values()) - hit
        if pred == truth:
            exact += 1
        else:
            errors.append(f"{f.name}：辨識成 {'-'.join(sorted(pred.elements())) or 'none'}")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(files), "precision": prec, "recall": rec, "exact": exact / max(1, len(files)),
            "ms": float(np.median(ms)) if ms else 0.0, "errors": errors[:20]}


class EvalReq(BaseModel):
    settings: dict = {}


@app.post("/api/evaluate")
async def evaluate(req: EvalReq):
    files = sorted(REAL_TEST.glob("*.jpg")) if REAL_TEST.exists() else []
    if not files:
        raise HTTPException(400, "data/real_test/ 還沒有照片，先用「存成測試照」累積")
    try:
        return await asyncio.to_thread(evaluate_realtest, files, req.settings)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- 除錯監看
_latest: dict = {"jpeg": None, "result": None, "t": 0.0, "saved": 0.0}

if DEBUG:
    @app.get("/api/debug/latest.jpg")
    def debug_latest_jpg():
        """最新一幀，伺服器端畫上整張牌框與牌值（供遠端除錯檢視）。"""
        if _latest["jpeg"] is None:
            raise HTTPException(404, "尚未收到任何影格")
        img = decode_jpeg(_latest["jpeg"])
        for g in _latest["result"].get("segs", []):
            cv2.polylines(img, [np.array(g["rbox"], np.int32)], True, (60, 207, 122), 2)
            x, y = map(int, g["center"])
            cv2.putText(img, "+".join(g.get("labels", [])) or "?", (x - 30, y), 0, 0.7, (255, 255, 255), 2)
        return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")


def debug_save(jpeg: bytes, res: dict, settings: dict):
    _latest.update(jpeg=jpeg, result=res, t=time.time())
    if time.time() - _latest["saved"] >= 2:   # 每 2 秒存一幀原圖＋結果，供離線檢視
        _latest["saved"] = time.time()
        DEBUG_FRAMES.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():{STAMP}}"
        (DEBUG_FRAMES / f"{stem}.jpg").write_bytes(jpeg)
        (DEBUG_FRAMES / f"{stem}.json").write_text(json.dumps({"settings": settings, "result": res}, ensure_ascii=False),
                                                   encoding="utf-8")


# ---------------------------------------------------------------- WebSocket
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
                    replaying = sess.next_t is not None   # 重播的影格不再錄製／監看存檔（避免重複）
                    img = decode_jpeg(msg["bytes"])
                    res = await asyncio.to_thread(sess.process, img)
                    sess.last_jpeg, sess.last_result = msg["bytes"], res
                    if not replaying:
                        sess.record(msg["bytes"], res)
                        if DEBUG:
                            debug_save(msg["bytes"], res, dict(sess.s))
                    if sess.rec_dir is not None:
                        res["recording"] = sess.state()["recording"]
                except Exception as e:  # 單張壞圖不中斷連線
                    sess.next_t = None
                    res = {"type": "result", "error": f"{type(e).__name__}: {e}"}
                await sock.send_json(res)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "frame_t":   # 不回覆，下一張影格才回
                    sess.next_t = float(data["t"])
                    continue
                reply = None
                if data.get("type") == "settings":
                    sess.apply(data)
                elif data.get("type") == "cmd":
                    reply = sess.command(data)
                await sock.send_json(reply or sess.state())
    except WebSocketDisconnect:
        pass
