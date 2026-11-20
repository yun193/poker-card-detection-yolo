# 21 點版本封存（2026-11-18）

移出原因：專案暫時專注在撲克牌辨識。此資料夾是移出當下**完整可運作**的快照（38 項測試通過）。

| 檔案 | 內容 |
|---|---|
| `blackjack_vision/blackjack.py` | 點數、基本策略（多副牌 S17 DAS）、莊家分佈遞迴、各動作勝／和／負與 EV |
| `blackjack_vision/advisor.py` | 桌面牌組 → 建議；牌靴記牌與 Hi-Lo 真數 |
| `blackjack_vision/cards.py` | 含 `split_zones`（莊家／玩家分區）與雙區 `StableTable` |
| `blackjack_vision/overlay.py`、`app.py` | OpenCV 視窗版 App |
| `web/` | 含 21 點面板、分區線拖曳、規則設定的網頁前端 |
| `tests/` | `test_blackjack.py`（獨立無限副牌遞迴核對爆牌率等）與對應的前後端測試 |

還原方式：把這裡的 `blackjack_vision/`、`web/`、`tests/`、`app.py` 覆蓋回專案根目錄（`tests/fixtures/` 未封存，沿用現有的），
再跑 `.venv/Scripts/python.exe -m pytest -q`。若之後辨識端的 `cards.py` 有改動，需手動合併 `split_zones` 與 `StableTable`。
