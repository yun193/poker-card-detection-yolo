# 撲克牌辨識＋21 點勝率提示

Kaggle 微調 YOLOv8 / YOLO12 / YOLO26 → 本機攝影機即時辨識真牌 → 疊加 21 點基本策略與各動作勝率。

- 資料集：[andy8744/playing-cards-object-detection-dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset)
  （20k 合成圖、52 類、416×416，CC0）。標註的是**牌角索引**，不是整張牌。
- YOLOv16 不存在，改用 YOLO26（Ultralytics 2025-09）。
- YOLOv4（Darknet）已移除：Kaggle 現行 CUDA 13 環境下多次編譯／訓練失敗。

## 1. Kaggle 端訓練

| notebook | 模型 | 預估時間（T4） |
|---|---|---|
| `kaggle/train_ultralytics.ipynb` | yolov8n / yolo12n / yolo26n（`MODELS` 可改 s 版） | 三個合計約 4~6 h |

步驟：Kaggle → New Notebook → File → Import Notebook → 選 .ipynb →
右側 Add Input 加入資料集、Accelerator 選 GPU、Internet 開 → **Save Version → Save & Run All**。
跑完到 Output 下載 `cards_ultralytics.zip`，**解壓到專案的 `models/`**，結構如下：

```
models/
  yolov8/  best.pt  metrics.json  results.csv ...
  yolo12/  best.pt ...
  yolo26/  best.pt ...
```

改了 `kaggle/*.py` 後執行 `python kaggle/build_notebooks.py` 重新產生 .ipynb。

## 2. 本機環境

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m pytest -q
```

## 3. 真牌測試台（網頁前端，建議）

```bash
.venv/Scripts/python.exe -m uvicorn web.server:app --port 8765
```

瀏覽器開 <http://localhost:8765>，允許攝影機權限（首次載入模型約 10~15 秒，伺服器啟動時會先預熱）。

- **攝影機模式**：即時框出牌角、合併成實體牌，右側顯示莊家／玩家牌面、點數、基本策略與停牌／要牌／加倍的勝和負機率＋EV。
- **圖片模式**：上傳一張或多張照片逐張測（←/→ 切換），沒有鏡頭也能測。
- 在畫面上**拖曳黃線**調整莊家／玩家分區；可切換模型、信心度門檻、穩定視窗、副數與規則。
- **存成測試照**：正解欄預先帶入目前辨識結果，核對修正後按存檔 → `data/real_test/`；
  **評估真牌集**：用目前模型對已存的照片算 Precision / Recall / 全對率，並列出錯誤樣本。
- 快捷鍵：空白 暫停、N 下一局（記牌）、R 重置牌靴、S 存測試照、F 對調分區。

### 3b. OpenCV 視窗版（舊）

```bash
.venv/Scripts/python.exe app.py --model yolo26 --cam 0
```

- 畫面上方是**莊家區**（只露出 1 張明牌），下方是**玩家區**；黃線是分隔線，`[` `]` 移動、`F` 上下對調。
- 面板顯示：雙方牌面、玩家點數（軟／硬）、基本策略建議、停牌／要牌／加倍的勝／和／負機率與 EV（★ 是 EV 最高的動作）。
- `N` 結束一局：桌上的牌計入牌靴，後續勝率會依剩餘牌組調整（附 Hi-Lo 真數）；`R` 重置牌靴。
- `1`~`3` 切換模型、`S` 截圖存到 `data/captures/`（檔名自動帶入辨識結果）。

規則預設：6 副牌、莊家軟 17 停（`--h17` 改為要牌）、莊家偷看底牌（`--no-peek` 改歐式）、BJ 賠 3:2。
勝率計算：莊家分佈依「扣除已知牌後的牌靴」精確遞迴；玩家要牌後依 EV 最大策略繼續。

## 4. 三模型比較

1. 用 App 按 `S` 截圖（建議 50~100 張：不同光線、角度、距離、牌數，含 1~2 張無牌）。
2. 確認／修正檔名，例如 `As-10h-Kd_xxx.jpg`、`none_xxx.jpg`，移到 `data/real_test/`。
3. 執行：

```bash
.venv/Scripts/python.exe evaluate.py
```

輸出 `reports/compare_*.md`：合成 test mAP（Kaggle 端）＋真牌 Precision / Recall / F1 / 整張全對率＋本機推論 ms / FPS。
