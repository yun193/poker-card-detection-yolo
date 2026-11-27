# 牌角路線封存（2026-11-21）

整張牌路線（card_seg＋value_cls）成為唯一推論路線後，舊的牌角路線移到這裡：

- `blackjack_vision/`、`web/`、`tests/`：封存當下的完整程式（牌角偵測 yolov8＋多尺度融合＋corner_cls 二次分類＋牌角追蹤，與整張牌路線並列的版本）
- `evaluate.py`、`evaluate_b2.py`：牌角路線的真牌評估與 B-2 驗收腳本
- `models/`：yolov8、corner_cls 權重（不入庫，Release v1.0 有備份）

要跑回舊版：把這些檔案複製回專案根目錄的對應位置，`models/` 移回 `models/`。
