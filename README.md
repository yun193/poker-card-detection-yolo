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
| 切片推論的取捨 | 畫面只有 1280×720、牌寬中位約 62px，整圖切片（SAHI 式）對整張牌沒幫助、對牌角路線耗時 6 倍只多一點召回；改在**轉正後的牌內**沿四邊切片，找疊牌時後面那張的牌角 |
| 自製合成資料 | 把單張牌照片以扇形／散放／疊牌貼到紋理背景上，自動產生分割標註 |

## 架構

```
攝影機畫面
  └─ YOLOv8-seg 整張牌分割（card_seg）
       └─ 旋轉框透視轉正 ──► 裁左上／右下牌角 ──► 牌值分類（value_cls，52 類＋none）
            ├─ 背景過濾：外觀像牌（四周白邊、幾乎沒有紅以外的彩色）＋至少一角不是 none＋牌面夠白或牌值很有把握
            ├─ 牌內切片：沿轉正牌面四邊掃描，找疊在後面那張牌的牌角
            └─ 跨幀追蹤投票：依位置配對，信心加權、隨時間衰減；附加牌值要連續出現才列出
```

早期的「路線 A」（YOLOv8 牌角偵測＋corner_cls 二次分類＋多尺度集成）在 2026-11-21 封存到 `_archive/corner_route_v2/`：
整張牌路線的主牌值準確度相同（實拍約 96%），而一框多牌值的錯誤附加牌值幾乎都來自路線 A 的牌角偵測。

## 結果（合成驗證集）

| 模型 | 任務 | 指標 |
|---|---|---|
| YOLOv8n | 牌角偵測（已封存） | mAP50 0.995／mAP50-95 0.913 |
| YOLOv8s-cls | 牌角二次分類（已封存） | top-1 0.9996 |
| YOLOv8n-seg | 整張牌分割 | mask mAP50-95 0.945 |
| YOLOv8s-cls | 轉正牌角牌值分類 | top-1 0.903 |

真牌實測（人工判讀）：單張可讀的牌路線 B 認對約 85%，信心 ≥ 0.6 的全部正確；
目前最大的弱點是**多張牌緊密重疊**時分割框容易把相鄰牌併成一個（約 3 成），連續兩輪改合成器重訓都沒能改善 —
合成資料上的進步不會自動轉移到真實畫面，是這個專案最大的教訓。

## 專案結構

```
blackjack_vision/   核心推論：card_seg／value_cls 包裝、轉正裁角與牌內切片、背景過濾、追蹤投票
web/                FastAPI 伺服器＋網頁測試台（攝影機／圖片／重播模式、錄製、回測）
kaggle/             Kaggle 訓練腳本（.py 為原始碼，build_notebooks.py 轉成 .ipynb）與合成器
models.yaml         本機模型登錄表（card_seg、value_cls）
backtest_real.py    實拍影格回測（完整流程重播＋轉正牌面總覽）
tests/              pytest 單元與 API 測試
_archive/           封存：早期 21 點提示版（blackjack_v1）、牌角路線（corner_route_v2）
```

> 套件名稱 `blackjack_vision` 是沿用早期版本的命名。

## 1. 取得模型權重

**直接下載**：到 [Releases v1.0](https://github.com/yun193/poker-card-detection-yolo/releases/tag/v1.0) 下載 `<名稱>_best.pt`，
放到 `models/<名稱>/best.pt`（例：`card_seg_best.pt` → `models/card_seg/best.pt`）。測試台需要 `card_seg` 與 `value_cls` 兩個。

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

- **攝影機／圖片／重播**：已確認的牌以追蹤編號配色、標牌值與信心；確認中的半透明；被背景過濾剔除的框是灰色虛線
- **手牌面板**：大字顯示已確認的手牌，下方逐框列出牌值、信心、穩定度與狀態（沒有牌值時說明原因：信心低／多張?／讀不到牌角）
- **除錯資訊**（D）：框上畫兩個牌角位置、牌內切片命中點、追蹤編號、驗證方式與白底／邊框白底／彩色比例
- **設定**：分割信心、牌值顯示門檻、背景過濾、牌內切片、追蹤投票
- **錄製**（R，攝影機模式）：連續存下每一幀到 `data/recordings/`（上限 60 秒），可在「重播」依原始時間播放，或在「回測」看統計
- **回報誤判**（E）存到 `data/hard_cases/`；**存成測試照**（S）存到 `data/real_test/`，可一鍵逐張評估
- 除錯監看：啟動前設 `CARD_DEBUG=1`，每 2 秒存一幀到 `data/debug_frames/`，`/api/debug/latest.jpg` 看伺服器端畫面

## 4. 實拍影格回測

影格來源：測試台「錄製」（連續影格，可驗證追蹤投票）或 `CARD_DEBUG=1` 監看存檔（每 2 秒一幀，一輪測完改名成 `data/debug_frames_roundN/`）。
換模型或改流程後，用同一批影格依原始時間戳重跑完整流程：

```bash
.venv/Scripts/python.exe backtest_real.py                                         # 全部 DEBUG 輪次
.venv/Scripts/python.exe backtest_real.py --folders "data/recordings/*"           # 錄製的連續影格
.venv/Scripts/python.exe backtest_real.py --seg models/card_seg_seg1/best.pt --no-scan
```

輸出 `reports/backtest_*/`：`summary.md`（框數、剔除數、單／多／無牌值、手牌變動次數、每幀耗時）、`items.json`、
`sheet_N.jpg`（轉正牌面總覽，每張標牌值，用來人工判讀對錯）。測試台的「回測」卡片是同一套流程。

## 下一步

- 用真牌與誤判樣本做第二輪微調（按輪次切分訓練／測試、混合合成資料，避免過擬合）
- value_cls 的 none 類加入「牌中央大花色」裁圖，降低牌內切片把花色讀成牌角的誤判
- 改善緊密疊牌的分割

## License

程式碼以 [MIT](LICENSE) 授權。資料集依各自授權。
