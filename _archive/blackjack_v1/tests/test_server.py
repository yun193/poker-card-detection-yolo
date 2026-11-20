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
    assert "真牌測試台" in client.get("/").text
    assert "yolov8" in client.get("/api/models").json()["models"]


@pytest.mark.parametrize("img", sorted(FIX.glob("*.jpg")), ids=lambda p: p.name[:9])
def test_ws_detects_dataset_image(client, img):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        ws.send_text(json.dumps({"type": "settings", "model": "yolov8", "smooth": False, "split": 0.5}))
        ws.receive_json()
        ws.send_bytes(img.read_bytes())
        r = ws.receive_json()
    assert "error" not in r, r.get("error")
    assert {c["label"] for c in r["cards"]} == truth(img)
    assert set(r["dealer"] + r["player"]) == truth(img)
    assert r["infer_ms"] > 0 and r["advice"] is not None


def test_ws_bad_frame_keeps_connection(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(b"not a jpeg")
        assert "error" in ws.receive_json()
        ws.send_text(json.dumps({"type": "cmd", "cmd": "reset_shoe"}))
        assert ws.receive_json()["seen"] == 0


def test_capture_validates_truth(client, tmp_path, monkeypatch):
    import base64
    import web.server as srv
    monkeypatch.setattr(srv, "REAL_TEST", tmp_path)
    img = sorted(FIX.glob("*.jpg"))[0]
    b64 = "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()
    ok = client.post("/api/capture", json={"image": b64, "truth": "10H-as"})
    assert ok.status_code == 200 and ok.json()["saved"].startswith("10h-As_")
    assert client.post("/api/capture", json={"image": b64, "truth": "ZZ"}).status_code == 400
