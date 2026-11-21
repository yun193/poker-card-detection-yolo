"""測試台後端：用假模型測流程與協定；有真權重時另跑一次真模型的冒煙測試。"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
HAS_MODELS = (ROOT / "models/card_seg/best.pt").exists() and (ROOT / "models/value_cls/best.pt").exists()


def jpeg(img=None) -> bytes:
    img = np.full((360, 640, 3), 255, np.uint8) if img is None else img
    return cv2.imencode(".jpg", img)[1].tobytes()


def rect_seg(x1, y1, x2, y2):
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return {"poly": pts, "rbox": pts, "center": [(x1 + x2) / 2, (y1 + y2) / 2], "conf": 0.9,
            "area": float((x2 - x1) * (y2 - y1))}


class FakeSeg:
    def __init__(self, segs):
        self.segs = segs

    def __call__(self, img, conf=0.4):
        return json.loads(json.dumps(self.segs))


class FakeCls:
    """第一個框讀成 Ks（白底）、第二個框兩角都是 none（背景）；切片在第一個框多讀到 5h。"""

    def classify_segs(self, img, segs):
        out = [{"label": "Ks", "conf": 0.9, "top": 0.92, "multi": False, "corners": [["Ks", 0.92], ["Ks", 0.88]],
                "white": 0.8, "border": 0.7, "color": 0.02}]
        return (out + [{"white": 0.2, "border": 0.1, "color": 0.5}] * len(segs))[:len(segs)]

    def scan_segs(self, img, segs, **kw):
        return [[["5h", 0.95, 0.5, 0.17, False]]] + [[] for _ in segs[1:]]

    def predict(self, crops):
        return [{} for _ in crops]


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    import web.server as srv
    monkeypatch.setattr(srv, "get_card_seg", lambda: FakeSeg([rect_seg(100, 50, 170, 150), rect_seg(400, 0, 600, 300)]))
    monkeypatch.setattr(srv, "get_value_cls", lambda: FakeCls())
    monkeypatch.setattr(srv, "HARD_CASES", tmp_path / "hard")
    monkeypatch.setattr(srv, "REAL_TEST", tmp_path / "real")
    monkeypatch.setattr(srv, "RECORDINGS", tmp_path / "rec")
    monkeypatch.setattr(srv, "DATA", tmp_path)
    return srv


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from web.server import app
    return TestClient(app)


def test_index(client):
    assert "撲克牌辨識測試台" in client.get("/").text


def test_process_filters_background_and_reads_values(fake):
    s = fake.Session()
    s.apply({"smooth": False})
    r = s.process(cv2.imdecode(np.frombuffer(jpeg(), np.uint8), 1))
    assert len(r["segs"]) == 1 and len(r["segs_rejected"]) == 1
    g = r["segs"][0]
    assert g["verify"] == "card" and g["value"]["shown"]
    assert [(v["label"], v["src"]) for v in g["values"]] == [("5h", "scan"), ("Ks", "corner")]   # 依信心排序
    assert g["labels"] == ["Ks"] and r["hand"] == ["Ks"]             # 但單張圖片不把切片讀值算進手牌
    assert set(r["ms"]) >= {"seg", "value", "scan", "total"}


def test_process_tracking_confirms_extra_value_over_time(fake):
    s = fake.Session()
    img = cv2.imdecode(np.frombuffer(jpeg(), np.uint8), 1)
    for i in range(12):                                            # 手牌平滑：要持續約半秒才加入
        r = s.process(img, t=100 + i * 0.05)
    g = r["segs"][0]
    assert g["confirmed"] and g["labels"] == ["Ks", "5h"]
    assert sorted(r["hand"]) == ["5h", "Ks"]


def test_ws_protocol_settings_frame_t_record_and_report(fake, client):
    with client.websocket_connect("/ws") as ws:
        st = ws.receive_json()
        assert st["type"] == "state" and st["settings"]["scan"] is True
        ws.send_text(json.dumps({"type": "settings", "scan": False}))
        assert ws.receive_json()["settings"]["scan"] is False
        ws.send_text(json.dumps({"type": "cmd", "cmd": "record_start"}))
        assert ws.receive_json()["recording"] is not None
        ws.send_bytes(jpeg())
        r = ws.receive_json()
        assert r["segs"][0]["values"][-1]["label"] == "Ks" and r["recording"]["frames"] == 1
        ws.send_text(json.dumps({"type": "frame_t", "t": 1234.5}))     # 重播影格：不回覆、用原始時間、不錄製
        ws.send_bytes(jpeg())
        r = ws.receive_json()
        assert r["t"] == 1234.5 and r["recording"]["frames"] == 1
        ws.send_text(json.dumps({"type": "cmd", "cmd": "record_stop"}))
        assert ws.receive_json()["recording"] is None
        ws.send_text(json.dumps({"type": "cmd", "cmd": "report", "note": "測試"}))
        rep = ws.receive_json()
        assert rep["type"] == "report" and rep["ok"]
        ws.send_bytes(b"not a jpeg")
        assert "error" in ws.receive_json()
    assert len(list((fake.HARD_CASES).glob("*.json"))) == 1


def test_recording_can_be_listed_and_backtested(fake, client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "cmd", "cmd": "record_start"}))
        ws.receive_json()
        for _ in range(3):
            ws.send_bytes(jpeg()); ws.receive_json()
    folders = client.get("/api/frames").json()["folders"]
    assert len(folders) == 1 and folders[0]["frames"] == 3
    fid = folders[0]["id"]
    frames = client.get(f"/api/frames/{fid}/list").json()["frames"]
    assert client.get(f"/api/frames/{fid}/img/{frames[0]['name']}").status_code == 200
    assert client.get(f"/api/frames/{fid}/img/..%5Cx.jpg").status_code == 404
    r = client.post("/api/backtest", json={"folder": fid}).json()
    assert r["frames"] == 3 and r["kept"] == 3 and r["rejected"] == 3


def test_capture_validates_truth_and_evaluates(fake, client):
    img = "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()
    assert client.post("/api/capture", json={"image": img, "truth": "Zz"}).status_code == 400
    assert client.post("/api/capture", json={"image": img, "truth": "ks"}).json()["saved"].startswith("Ks_")
    r = client.post("/api/evaluate", json={}).json()
    assert r["n"] == 1 and r["exact"] == 1.0


def test_missing_model_reports_error(monkeypatch):
    import web.server as srv
    monkeypatch.setattr(srv, "get_card_seg", lambda: None)
    r = srv.Session().process(np.zeros((10, 10, 3), np.uint8))
    assert "card_seg" in r["error"]


@pytest.mark.skipif(not HAS_MODELS, reason="需要 models/card_seg 與 models/value_cls 權重")
def test_real_models_smoke():
    import web.server as srv
    frames = sorted((ROOT / "data").glob("debug_frames_round*/*.jpg"))
    img = srv.imread(frames[0]) if frames else np.full((720, 1280, 3), 255, np.uint8)
    r = srv.Session().process(img)
    assert "error" not in r and r["ms"]["total"] > 0
