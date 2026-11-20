# Poker Card Detection — 用撲克牌辨識學 YOLO

用 Kaggle 免費 GPU 微調 YOLO 系列模型辨識撲克牌（52 類），再在本機（RTX 3050 Ti 4GB）用網頁測試台拿**真牌對鏡頭**實測。
這是一個以**學習 YOLO** 為主的個人專案：從單純的物件偵測出發，一路碰到合成資料到真實畫面的落差（sim-to-real gap），
再嘗試多尺度集成、兩階段分類、實例分割等方法改善。

## 學到了什麼

| 主題 | 內容 |
|---|---|
| YOLO 偵測微調 | Ultralytics YOLOv8n 在 2 萬張合成圖上訓練，合成測試集 mAP50 0.995 |
| 資料集標註的本質 | 資料集標的是**牌角索引**（點數＋花色），不是整張牌 → 一張牌會偵測到兩個對角，需要後處理合併 |
| 合成 vs 真實 | 合成指標近乎滿分，但真牌對鏡頭時遠距、模糊、遮擋、扇形重疊都會掉分 |
| 推論端技巧 | 多尺度集成（416＋832 投票）、依位置追蹤的時間投票＋遲滯，壓掉偶發誤判 |
| 兩階段辨識 | 偵測器找牌角 → 從原始高解析度畫面裁圖 → YOLOv8s-cls 二次分類 |
| 實例分割 | YOLOv8n-seg 分割「整張牌露出的區域」→ 透視轉正 → 裁牌角 → 分類牌值 |
| 自製合成資料 | 把單張牌照片以扇形／散放／疊牌貼到紋理背景上，自動產生分割標註 |

## 架構

```
攝影機畫面
  ├─ 路線 A：YOLOv8 牌角偵測 ──► 牌角二次分類（corner_cls）──► 同牌兩角合併 ──► 時間投票
  └─ 路線 B：YOLOv8-seg 整張牌分割 ──► 旋轉框透視轉正 ──► 裁兩個牌角 ──► 牌值分類（value_cls，52 類＋none）
                                    └─ 與路線 A 的牌角交叉驗證，濾掉背景誤判
```

## 結果（合成驗證集）

| 模型 | 任務 | 指標 |
|---|---|---|
| YOLOv8n | 牌角偵測 | mAP50 0.995／mAP50-95 0.913 |
| YOLOv8s-cls | 牌角二次分類 | top-1 0.9996 |
| YOLOv8n-seg | 整張牌分割 | mask mAP50-95 0.945 |
| YOLOv8s-cls | 轉正牌角牌值分類 | top-1 0.903 |

真牌實測（人工判讀）：單張可讀的牌路線 B 認對約 85%，信心 ≥ 0.6 的全部正確；
目前最大的弱點是**多張牌緊密重疊**時分割框容易把相鄰牌併成一個（約 3 成），連續兩輪改合成器重訓都沒能改善 —
合成資料上的進步不會自動轉移到真實畫面，是這個專案最大的教訓。

## 專案結構

```
blackjack_vision/   核心推論：偵測器後端、牌角合併與追蹤、轉正裁角
web/                FastAPI 伺服器＋網頁測試台（攝影機／圖片模式）
kaggle/             Kaggle 訓練腳本（.py 為原始碼，build_notebooks.py 轉成 .ipynb）與合成器
models.yaml         本機模型登錄表（檔案不存在的模型自動略過）
evaluate.py         真牌集多模型比較（Precision／Recall／F1／FPS）
evaluate_b2.py      路線 B 離線重播評估
tests/              pytest 單元與 API 測試
_archive/           早期加上 21 點提示的版本（已封存）
```

> 套件名稱 `blackjack_vision` 是沿用早期版本的命名。

## 1. 取得模型權重

**直接下載**：到 [Releases v1.0](https://github.com/yun193/poker-card-detection-yolo/releases/tag/v1.0) 下載 `<名稱>_best.pt`，
放到 `models/<名稱>/best.pt`（例：`yolov8_best.pt` → `models/yolov8/best.pt`）。只要 `yolov8` 就能跑基本辨識，其他缺少時自動略過。

```bash
gh release download v1.0 --repo yun193/poker-card-detection-yolo --dir models/_dl
```

**自己訓練**：用 `kaggle/` 的 notebook（Kaggle → Import Notebook → GPU、Internet 開 → Save & Run All），
跑完下載輸出 zip 解壓到 `models/<名稱>/`：

| Notebook | 輸出 | 資料集 |
|---|---|---|
| `train_ultralytics.ipynb` | `models/yolov8/`（也可改 `yolo12n`、`yolo26n`） | [andy8744/playing-cards-object-detection-dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset)（CC0） |
| `train_corner_cls.ipynb` | `models/corner_cls/` | 同上 |
| `train_card_seg.ipynb` | `models/card_seg/` | [gpiosenka/cards-image-datasetclassification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)、[DTD](https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd) |
| `train_value_cls.ipynb` | `models/value_cls/` | 同上 |

T4 上 yolov8n 60 epoch 約 1.5～2 小時。改 `.py` 後執行 `python kaggle/build_notebooks.py` 重新產生 notebook。

## 2. 本機環境

需要 Python 3.12 與 [uv](https://github.com/astral-sh/uv)，以下為 Windows 路徑：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

## 3. 網頁測試台

```bash
.venv/Scripts/python.exe -m uvicorn web.server:app --port 8765
```

Chrome 開 <http://localhost:8765> 並允許攝影機。啟動時會先預熱模型（約 10～15 秒）。

- **攝影機／圖片模式**：框色代表信心度（綠 ≥80%、黃 50–80%、紅 <50%）
- **設定**：模型、信心度門檻、推論解析度、多尺度集成、追蹤記憶、牌角合併距離、牌角二次分類、整張牌框模式
- **存成測試照**（S）存到 `data/real_test/`；**回報誤判**（E）存到 `data/hard_cases/`
- 快捷鍵：空白 暫停、S 存測試照、E 回報誤判、C 清除追蹤
- 除錯監看：啟動前設 `CARD_DEBUG=1`，可從 `/api/debug/latest.jpg`、`/api/debug/history` 看伺服器端影格

## 4. 多模型比較

真牌照片檔名格式 `As-10h-Kd_xxx.jpg`（沒牌寫 `none_xxx.jpg`）放在 `data/real_test/`：

```bash
.venv/Scripts/python.exe evaluate.py
```

輸出 `reports/compare_*.md`，包含合成 mAP、真牌 Precision／Recall／F1、整張全對率與本機 FPS。

## 下一步

- 用真牌與誤判樣本做第二輪微調（縮小 sim-to-real gap）
- 訓練 YOLO12n、YOLO26n，在真牌測試集比較三代模型
- 改善緊密疊牌的分割

## License

程式碼以 [MIT](LICENSE) 授權。資料集依各自授權。
