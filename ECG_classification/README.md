# ECG_classification — AF 分類模型（訓練、部署、PC 端驗證）

從 PhysioNet/CinC 2017 訓練一個能放進 M55M1 的 INT8 三分類模型（Normal / AF / Other），並提供 PC 端推論與評估工具。

## 檔案

| 檔案 | 用途 |
|---|---|
| `AF_Edge_M55M1_Training.ipynb` | **訓練流水線**（Google Colab，T4）：載入 + 升頻 + 切窗 → record-level 切分 → 多尺度 1D-CNN + ResBlock → 訓練 → TTA / record 投票評估 → 剪枝 → INT8 量化 → Vela → 產 `af_model.h` → 打包 `deploy.zip` |
| `AF_Edge_M55M1_Training_v5_rerun.ipynb` | 同一配方的重跑版（改成從 Google Drive 讀 `data.zip`、存結果到 Drive），用來產生 `docs/report_v5/` 的訓練曲線 |
| `deploy/model_int8.tflite` | INT8 模型（PC 端推論用，455.7 KB） |
| `deploy/model_int8_vela.tflite` | Vela 編譯後的模型（Ethos-U55-256，221 KB） |
| `deploy/af_model.h` | Vela 模型轉成的 C 陣列，供 M55M1 韌體 `#include`。**模型單一來源**，`M55M1_Firmware/NN_ModelInference/af_model.h` 是它的複本 |
| `inference.py [dir 或 .npy]` | PC 端 TFLite 推論：讀 `.npy` 視窗 → band-pass → z-score → INT8 → Normal / AF / Other，多視窗時另印多數投票與平均機率 |
| `tools_eval_tflite_mcu.py <tflite...>` | 在同一個 seed-42 record-level 測試集上，用與 MCU 相同的逐視窗前處理評估任意 INT8 tflite（需 tensorflow、pandas、scikit-learn 與本地資料集） |
| `rr_features.py` | RR 間期 16 維特徵的 Python 參考實作（雙輸入模型選用；與韌體 `rr_features.h` 逐行對應） |
| `read_npy.py` | 畫 `.npy` 視窗波形 |
| `data/` | PhysioNet/CinC 2017 資料集放置處（未收錄，見 `data/README.md`） |

## 資料規格（訓練 = PC 推論 = 韌體，必須一致）

| 項目 | 值 |
|---|---|
| 採樣率 | **500 Hz**（PhysioNet 原始 300 Hz → 升頻 500 Hz，對齊 TriAnswer） |
| 視窗 | **30 s = 15000 樣本**，stride 10 s = 5000 樣本 |
| 前處理 | band-pass（訓練端 0.5–45 Hz 於 300 Hz 整筆記錄；推論端 0.5–40 Hz 於 500 Hz 視窗）→ **z-score** → **INT8** |
| 量化 | 輸入 `[1, 15000, 1]` scale 0.145202 / zp 20；輸出 `[1, 3]` scale 1/256 / zp −128 |
| 分類 | `0 = Normal (N)` / `1 = AF (A)` / `2 = Other (O 與雜訊 ~ 合併)` |
| 切分 | record-level 分層切分，seed 42（同一筆記錄的視窗不跨集）；Test 1024 record（N 610 / A 91 / O 323） |

## 成績（部署模型，PC 端重現 MCU 前處理，同一測試集）

| 層級 | accuracy | macro-F1 |
|---|---|---|
| Window（1485） | 85.79% | 0.8383 |
| **Record（1024）** | **86.43%** | **0.8427** |

AF recall 0.868（漏報 12/91，其中 11 筆判 Other）；Normal 誤判為 AF 2/610。Vela：3079 ops 全在 NPU、SRAM 1246 KiB、177 M MACs。完整表格、訓練曲線、混淆矩陣見 [`../docs/report_v5/`](../docs/report_v5/README.md)。

## PC 端使用

```bash
pip install numpy scipy ai-edge-litert      # 裝不上 ai-edge-litert 時改裝 tensorflow
python inference.py ../captures/ecg_uart     # 用 TriAnswer_Firmware/tools/uart_capture.py 擷取的資料夾
python inference.py path/to/window_0s_30s.npy
AF_MODEL=path/to/other.tflite python inference.py ../captures/ecg_uart   # 指定別的模型
```

`inference.py` 用 `model_content` 載入模型，因為 LiteRT 的 C++ 開檔不吃非 ASCII 路徑。若模型有第二個輸入（RR 特徵），會自動從模型旁的 `rr_norm.npz` 或 `metrics.json` 讀標準化常數。

## 訓練（Colab）

1. 把 `data/` 壓成 `data.zip`（含 `REFERENCE.csv` 與 `training2017/`）。
2. 開 `AF_Edge_M55M1_Training.ipynb`，執行階段選 T4 GPU，依序執行；在上傳 cell 上傳 `data.zip`。
3. 結束後下載 `deploy.zip`，解壓到 `deploy/`，並把 `af_model.h` 複製到 `../M55M1_Firmware/NN_ModelInference/`。
4. 重新建置 M55M1 韌體。若新模型的 Vela SRAM 用量不同，調整 `af_inference.h` 的 `TENSOR_ARENA_SIZE`（或在 `af_model.h` 定義 `AF_MODEL_ARENA_KB`）。

## 訓練配方摘要

多尺度 1D-CNN stem（kernel 75 / 45 / 25）+ ResBlock 堆疊 + GAP + Dense；AdamW、warmup + cosine decay（峰值 1e-3）、batch 64、最多 200 epoch、early stopping patience 40、class weight（N 0.572 / A 3.733 / O 1.016）、增強（幅度縮放、高斯噪音、基線漂移、隨機平移）；訓練後剪枝 20% → 40%（各 8 epoch 微調）→ full-integer INT8 PTQ（量化掉點 0.00）→ Vela `--accelerator-config ethos-u55-256 --optimise Performance`。
