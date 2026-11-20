"""前端後端的端對端測試：用資料集 test 圖（CC0）走 WebSocket，結果需與標註一致。"""
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "yolov8" / "best.pt").exists(), reason="需要 models/yolov8/best.pt")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from web.server import app
    return TestClient(app)


def truth(img: Path) -> set[str]:
    names = yaml.safe_load((FIX / "data.yaml").read_text())["names"]
    return {names[int(l.split()[0])] for l in img.with_suffix(".txt").read_text().splitlines() if l.strip()}


def test_index_and_models(client):
    assert "真牌辨識測試台" in client.get("/").text
    assert "yolov8" in client.get("/api/models").json()["models"]


@pytest.mark.parametrize("img", sorted(FIX.glob("*.jpg")), ids=lambda p: p.name[:9])
def test_ws_detects_dataset_image(client, img):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        ws.send_text(json.dumps({"type": "settings", "model": "yolov8", "smooth": False}))
        ws.receive_json()
        ws.send_bytes(img.read_bytes())
        r = ws.receive_json()
    assert "error" not in r, r.get("error")
    assert {c["label"] for c in r["cards"]} == truth(img)
    assert set(r["stable"]) == truth(img)
    assert r["infer_ms"] > 0


def test_ws_bad_frame_keeps_connection(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(b"not a jpeg")
        assert "error" in ws.receive_json()
        ws.send_text(json.dumps({"type": "cmd", "cmd": "reset"}))
        assert ws.receive_json()["type"] == "state"


@pytest.mark.parametrize("ensemble", [False, True])
def test_ws_resolution_and_ensemble(client, ensemble):
    img = sorted(FIX.glob("*.jpg"))[0]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "model": "yolov8", "smooth": False, "imgsz": 832, "ensemble": ensemble}))
        ws.receive_json()
        ws.send_bytes(img.read_bytes())
        r = ws.receive_json()
    assert r["scales"] == ([416, 832] if ensemble else [832])
    assert set(r["stable"]) == truth(img)


def test_ws_refine_path_with_fake_classifier(client, monkeypatch):
    """用假的分類器驗證第二階段串接：分類器一律說 Kc → 所有牌角被改判並標記 det_label。"""
    import web.server as srv

    class FakeCls:
        def __call__(self, frame, dets):
            return [{"Kc": 0.99} for _ in dets]
    monkeypatch.setattr(srv, "get_refiner", lambda: {"model": FakeCls(), "weight": 0.6})
    img = sorted(FIX.glob("*.jpg"))[0]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "model": "yolov8", "smooth": False, "refine": True}))
        ws.receive_json()
        ws.send_bytes(img.read_bytes())
        r = ws.receive_json()
    assert r["refined"] and r["detections"]
    assert all(d["label"] == "Kc" for d in r["detections"])
    assert {d["det_label"] for d in r["detections"]} == truth(img)


@pytest.mark.parametrize("view", ["card", "both"])
def test_ws_card_view_with_fake_segmenter(client, monkeypatch, view):
    import web.server as srv
    seg = {"poly": [[10, 10], [60, 10], [60, 90], [10, 90]], "rbox": [[10, 10], [60, 10], [60, 90], [10, 90]],
           "center": [35, 50], "conf": 0.9, "area": 4000}
    monkeypatch.setattr(srv, "get_card_seg", lambda: (lambda img, conf=0.4: [seg]))
    img = sorted(FIX.glob("*.jpg"))[0]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "model": "yolov8", "smooth": False, "view": view,
                                 "seg_verify": False}))
        ws.receive_json()
        ws.send_bytes(img.read_bytes())
        r = ws.receive_json()
    assert r["segs"] == [seg]
    assert (r["detections"] == []) == (view == "card")    # card 模式不跑牌角辨識


def test_ws_both_view_verifies_segs_with_corners(client, monkeypatch):
    import web.server as srv
    img = sorted(FIX.glob("*.jpg"))[0]
    full = {"poly": [[0, 0], [9999, 0], [9999, 9999], [0, 9999]], "conf": 0.9}   # 涵蓋整張圖 → 必含牌角
    tiny = {"poly": [[-50, -50], [-48, -50], [-48, -48], [-50, -48]], "conf": 0.6}  # 圖外、不與 full 重疊 → 不含牌角也串不到
    monkeypatch.setattr(srv, "get_card_seg", lambda: (lambda im, conf=0.4: [dict(full), dict(tiny)]))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "model": "yolov8", "smooth": False, "view": "both",
                                 "seg_verify": True}))
        ws.receive_json()
        ws.send_bytes(img.read_bytes())
        r = ws.receive_json()
    assert r["detections"]
    # 驗證用的是低門檻原始候選（各尺度、未投票），所以命中數 ≥ 顯示的牌角數
    assert [s["poly"] for s in r["segs"]] == [full["poly"]] and r["segs"][0]["corners"] >= len(r["detections"])
    assert [s["poly"] for s in r["segs_rejected"]] == [tiny["poly"]]


def test_ws_card_view_without_model_reports_error(client, monkeypatch):
    import web.server as srv
    monkeypatch.setattr(srv, "get_card_seg", lambda: None)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "view": "card"}))
        ws.receive_json()
        ws.send_bytes(sorted(FIX.glob("*.jpg"))[0].read_bytes())
        assert "card_seg" in ws.receive_json()["error"]


def test_report_saves_hard_case(client, tmp_path, monkeypatch):
    import web.server as srv
    monkeypatch.setattr(srv, "HARD_CASES", tmp_path)
    img = sorted(FIX.glob("*.jpg"))[0]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "cmd", "cmd": "report", "note": "x"}))
        assert ws.receive_json()["ok"] is False             # 還沒有影格
        ws.send_bytes(img.read_bytes())
        ws.receive_json()
        ws.send_text(json.dumps({"type": "cmd", "cmd": "report", "note": "Kc 被認成 As"}))
        r = ws.receive_json()
    assert r["ok"] and r["count"] == 1
    meta = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert meta["note"] == "Kc 被認成 As" and meta["result"]["detections"]


def test_capture_validates_truth(client, tmp_path, monkeypatch):
    import base64
    import web.server as srv
    monkeypatch.setattr(srv, "REAL_TEST", tmp_path)
    img = sorted(FIX.glob("*.jpg"))[0]
    b64 = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
    ok = client.post("/api/capture", json={"image": b64, "truth": "10H-as"})
    assert ok.status_code == 200 and ok.json()["saved"].startswith("10h-As_")
    assert client.post("/api/capture", json={"image": b64, "truth": "ZZ"}).status_code == 400


@pytest.mark.parametrize("on", [True, False])
def test_ws_card_view_adds_b2_value(client, monkeypatch, on):
    import web.server as srv
    seg = {"poly": [[10, 10], [60, 10], [60, 90], [10, 90]], "rbox": [[10, 10], [60, 10], [60, 90], [10, 90]],
           "center": [35, 50], "conf": 0.9, "area": 4000}

    class FakeCls:
        def classify_segs(self, img, segs):
            return [{"label": "9h", "conf": 0.88} for _ in segs]
    monkeypatch.setattr(srv, "get_card_seg", lambda: (lambda im, conf=0.4: [dict(seg)]))
    monkeypatch.setattr(srv, "get_value_cls", lambda: FakeCls())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "view": "card", "seg_value": on}))
        ws.receive_json()
        ws.send_bytes(sorted(FIX.glob("*.jpg"))[0].read_bytes())
        r = ws.receive_json()
    assert (r["segs"][0].get("value") == {"label": "9h", "conf": 0.88, "shown": True}) == on


def test_ws_b2_value_below_threshold_is_hidden(client, monkeypatch):
    import web.server as srv
    seg = {"poly": [[10, 10], [60, 10], [60, 90], [10, 90]], "rbox": [[10, 10], [60, 10], [60, 90], [10, 90]],
           "center": [35, 50], "conf": 0.9, "area": 4000}

    class FakeCls:
        def classify_segs(self, img, segs):
            return [{"label": "Ad", "conf": 0.5, "multi": True} for _ in segs]
    monkeypatch.setattr(srv, "get_card_seg", lambda: (lambda im, conf=0.4: [dict(seg)]))
    monkeypatch.setattr(srv, "get_value_cls", lambda: FakeCls())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "settings", "view": "card", "seg_value": True}))
        ws.receive_json()
        ws.send_bytes(sorted(FIX.glob("*.jpg"))[0].read_bytes())
        r = ws.receive_json()
    assert r["segs"][0]["value"]["shown"] is False and r["segs"][0]["value"]["multi"] is True
