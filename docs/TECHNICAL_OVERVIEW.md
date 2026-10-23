# AF-Edge 技術總覽
## HECG100 → TriAnswer → Nuvoton M55M1 心房顫動（AF）邊緣偵測鏈

本文整合專案所有已驗證的設計、數據與經驗，是 README 之外最完整的技術說明。模型數據包（圖、CSV、原始評估輸出）在 `docs/report_v5/`。

---

## 0. 一頁摘要

本專案建立一條**不含 ESP32**、可離線運作的 AF 即時偵測鏈：

```
HECG100 (WhaleTeq ECG 模擬器)      TriAnswer (Yutech TriBLE nRF52840)         Nuvoton NuMaker-X-M55M1D
播放 PhysioNet/CinC2017 真實 ECG  →  自製韌體：500 Hz 12-bit ADC 採樣         →  Cortex-M55 + Ethos-U55 NPU
經實體電極輸出 (Lead II, ≤5 mV)      硬體 UART 921600 送幀                        band-pass → z-score → INT8 推論
                                     [0xAA][0xBB][LEN][int32 LE × 100]            每 10 s 輸出 Normal / AF / Other
```

| 指標 | 數值 |
|---|---|
| 模型 | AF-Edge **v5**：多尺度 1D-CNN + ResBlock，INT8，Vela 編譯 |
| 測試集成績（1024 record，與訓練無洩漏） | record-level macro-F1 **0.843**，accuracy 86.4% |
| AF 召回 / AF 假警報 | recall **0.868**（漏報 12/91）；Normal 誤判為 AF 僅 2/610（0.3%） |
| NPU 使用率 | **100%**（3079 operators 全在 Ethos-U55，CPU 0） |
| 模型大小 / SRAM | Vela tflite 221 KB；SRAM 1246 KiB；Tensor Arena 實測 1,276,468 bytes |
| 實機全鏈驗證 | AF 記錄 A01210 → 連續 10 視窗判 AF（0.59–0.99）；Normal 記錄 A04565 → 連續 11 視窗判 Normal（0.73–0.96） |
| 可重現性 | v5 原樣重訓一次，macro-F1 0.8395 vs 現役 0.8427（差 0.003） |
| 優化嘗試 | v6 / v6.1 / v7 / v7b 四個新設計皆低於 v5（0.806–0.831），定案維持 v5 |

---

## 1. 專案背景與範圍

- 本專案是一個更大的「AF 偵測 + 閉環 taVNS 刺激」系統的偵測子系統。母系統原本以 ESP32 做 BLE→UART 橋接並有第二台 ESP32 控制刺激器；本專案在**不使用 ESP32** 的約束下，把 TriAnswer 改燒自製韌體、以硬體 UART 直送 M55M1，範圍只到 AF 偵測。
- 不含：ESP32（任何角色）、刺激控制器、刺激器。

### 1.1 開發策略
逐步驗證：每一段先獨立可用（HECG100 連線 → TriAnswer 到 PC → M55M1 開機）再串接，最後全鏈。每段皆有對應驗證腳本（見第 7 節）。

---

## 2. 硬體清單

| 角色 | 型號 / 識別 | 備註 |
|---|---|---|
| ECG 訊號源 | WhaleTeq **HECG100**（FW 1.0.19） | 透過原廠 SDK（`HECG100x64.dll`）以 USB 控制；輸出幅度上限 5 mV |
| 穿戴感測 | Yutech **TriAnswer**（TriBLE nRF52840） | Adafruit nRF52 fork 核心 0.3.15（SoftDevice s140 6.1.1、FreeRTOS、TinyUSB）；使用 CH2 電極埠 |
| 邊緣運算 | Nuvoton **NuMaker-X-M55M1D**（M55M1H2LJAE，Cortex-M55 + Ethos-U55-256 NPU） | Nu-Link2-Me 燒錄 + VCOM；Keil MDK Nuvoton Edition 建置 |
| PC | Windows 11，Python 3.13 | numpy、scipy、wfdb、pyserial、ai-edge-litert（見 `requirements.txt`） |

### 2.1 接線（已實測定案）

**電極（HECG100 → TriAnswer，Lead II）**

| TriAnswer 電極（配戴位置） | HECG100 端子 |
|---|---|
| 右鎖骨（右上） | **RA / R** |
| 左肋（左下） | **LL / F** |
| （選）GND | RL / GND，小幅加分 |

- ECG 子板插 TriAnswer **中排 = CH2 = A3**。原因：子板橫跨左右，插上排會蓋住 D0（UART TX）。
- 極性接反時波形上下顛倒。用 `TriAnswer_Firmware/tools/polarity_check.py A01210` 做互相關判定（corr 正 = 正確、負 = 反接；實測 −0.6 → 對調 → +0.6）。

**TriAnswer → M55M1**

| TriAnswer | M55M1 |
|---|---|
| pin 22 = UART TX / D0（P0.25） | **NU2 排針 `D13` = PA.2 = UART1 RX** |
| GND（pin 4 / 8 / 12） | 任一 GND |

- ⚠️ NU3 排針的 A4/A5（SDA/SCL）是 PB.0/PB.1 = UART2，**接那裡完全收不到**（來源：NuMaker-X-M55M1D UM Rev1.01 §3.3.2 Table 3-3）。
- 3.3 V 邏輯，直接相接，不需準位轉換。

---

## 3. TriAnswer 自製韌體（`TriAnswer_Firmware/TriAnswer_UART_to_M55M1/`）

### 3.1 設計
- 取代原廠 BLE 範例 `TriBLE_Stu_Demo.ino`（Service FF05 / Char AA05 / 8-bit / SysTick 2 ms）。原廠範例隨 Yutech nRF52 核心提供，未收錄。
- **硬體 TIMER4 中斷**每 2 ms 讀 `analogRead(A3)`（12-bit）→ 1024 點環形緩衝 → 每 100 點組一幀走 `Serial1`（921600 8N1）。
- 幀格式與 M55M1 `uart_receiver.h` 完全一致，M55M1 韌體**零修改**：

```
[0xAA][0xBB][LEN_L][LEN_H][int32 LE × 100]     LEN = 400
```

- `MIRROR_TO_USB=1`：同一份幀同時送 USB CDC，PC 可不經任何轉接板直接驗證。
- 狀態燈：紅燈常亮、每幀（200 ms）藍燈閃一次。

### 3.2 關鍵技術發現：取樣率偏慢 2%
原廠範例用 SysTick 計時。loop 等 UART 送幀期間 FreeRTOS 讓 CPU 進入睡眠，Cortex-M SysTick 在睡眠時停擺，實測取樣率只有 ~490 Hz（每幀 4.4 ms / 200 ms ≈ 2%）。改用周邊 TIMER4（睡眠照跑）後韌體自測 **499.9 Hz**（對 32 kHz RTC 晶振）。PC 以 USB 到達時間量測會有 ±1% 抖動，以韌體自報 `rate~` 為準。

### 3.3 編譯 / 燒錄
```
arduino-cli compile --fqbn Yutech:nrf52:TriBLE_nRF52 TriAnswer_Firmware/TriAnswer_UART_to_M55M1
arduino-cli upload  --fqbn Yutech:nrf52:TriBLE_nRF52 -p COMx TriAnswer_Firmware/TriAnswer_UART_to_M55M1
```
arduino-cli 1.2.0（Arduino IDE 內附的也可用）。燒錄走 adafruit-nrfutil 串列 DFU（1200 bps touch），USB VID 0x8787。救磚可走 J-Link。編譯產物約 40 KB。

### 3.4 訊號品質
UART 路徑 **60 Hz 市電占比 0.00**（舊 BLE 路徑曾達 60%；TriAnswer 放桌上實測 60 Hz 僅 ~4 counts）。12-bit 下刻度約 32.6 × 16 ≈ 520 counts/mV（M55M1 端 z-score 不在乎絕對刻度）。

---

## 4. M55M1 韌體（`M55M1_Firmware/NN_ModelInference/`）

### 4.1 架構
以 Nuvoton M55M1BSP 的 `SampleCode/MachineLearning/NN_ModelInference` 範例為基礎（BSP 本身未收錄，見 `M55M1_Firmware/README.md`）。本專案撰寫 / 修改的檔案：

| 檔案 | 功能 |
|---|---|
| `main.cpp` | UART1 初始化（`SET_UART1_RXD_PA2`）、主迴圈、`HandleInferenceResult` 印 `[結果]` 行 |
| `uart_receiver.h` | 幀解析狀態機（WAIT_H0 → WAIT_H1 → LEN_L → LEN_H → DATA），滑動視窗 15000 點、stride 5000 |
| `af_inference.h` | 前處理（band-pass → z-score → INT8）+ TFLite Micro / Ethos-U55 推論 |
| `af_model.h` | Vela 編譯後 INT8 模型（`ECG_classification/deploy/af_model.h` 的複本，後者為模型單一來源） |
| `rr_features.h` | v7 雙輸入用 RR 特徵（`AF_MODEL_RR_FEATURES` 未定義時不啟用；v5 不受影響） |

### 4.2 前處理（與訓練必須一致）
1. 收 15000 點（30 s @ 500 Hz），每 5000 點（10 s）推論一次。
2. **Band-pass 0.5–40 Hz**：`bandpass_0p5_40()` = 去均值 → 前後雙向 DF2T biquad（穩態初始），重現訓練端 `butter(3,[0.5,40]/250,'band') + filtfilt`。PC 端 `tools/validate_c_bandpass.py` 驗證模型輸出與 SciPy filtfilt **逐位元相同**。
3. z-score：`x_norm = (x − mean) / std`。
4. 量化：`x_int8 = round(x_norm / 0.145202) + 20`（in_scale / in_zp 來自模型）。
5. 推論 → 3 個 int8 分數 → `prob = (q + 128) × 0.003906` → argmax。

⚠️ band-pass 是**關鍵步驟**：早期版本的韌體與 PC 推論都漏了這步，在 60 Hz 市電雜訊下 AF 機率會被壓到 0、全判 Other。補上後同一筆 A01210 由 AF 0.00 → 0.98。

### 4.3 記憶體
- `TENSOR_ARENA_SIZE = 1300 KB`，開機回報實際用量 **1,276,468 bytes**。
- `tensor_arena` 與 60 KB `bp_buf`（float[15000]）都放 `.bss.sram.data` section（SRAM01/HyperRAM），否則會擠爆僅 ~48 KB 的 DTCM 預設 .bss（linker L6406E/L6407E）。
- 建置結果：Code 134,858 B / ZI 1,537,080 B；bin 約 370 KB。

### 4.4 建置 / 燒錄
- Keil MDK **Nuvoton Edition**（免費、含 M55、無 code size 限制），AC6 V6.24，DFP `NuMicroM55_DFP`。
- 命令列只建置：`UV4.exe -b NN_ModelInference.uvprojx -j0 -o log`（約 25 s）。
- 燒錄請用 **Keil GUI F8**，或命令列前先在 `KEIL/Nu_Link_Driver.ini` 設 `DisableFirmwareUpdate=1`（詳見 11.2 踩雷）。
- 開機記錄（`tools/serial_monitor.py COMx`，115200）：Ethos-U55-256 init → 模型載入成功 → Tensor Arena 1276468 bytes → 等待 UART 資料。`SD initial fail!!` 無害（模型編在 Flash）。
- 雙 USB：Nu-Link2-Me Type-C（資料線）負責燒錄 + VCOM；HSUSB Type-C 供電。

---

## 5. 模型（AF-Edge v5）

### 5.1 資料集與切分

| 項目 | 值 |
|---|---|
| 來源 | PhysioNet/CinC Challenge 2017（AliveCor 單導程，300 Hz）；本地 `ECG_classification/data/` |
| Record 數 | 8528（N 5076 / A 758 / O 2415 / ~ 279） |
| 分類 | 3 類：Normal(N) / AF(A) / Other(O + 雜訊 ~ 合併) |
| 重採樣 | 300 → **500 Hz**（對齊 TriAnswer） |
| 視窗 | 30 s = 15000 樣本，stride 10 s；總視窗 19044（N 11093 / A 1720 / O 6231） |
| 切分 | **record-level** 分層（同 record 視窗不跨集，防洩漏），seed 42；Train / Val / Test 視窗 14658 / 2085 / 2301；Test **1024 record**（N 610 / A 91 / O 323） |
| 類別權重 | N 0.572 / A 3.733 / O 1.016 |

### 5.2 模型與訓練配方

| 項目 | 值 |
|---|---|
| 架構 | 多尺度 1D-CNN stem（kernel 75 / 45 / 25）+ ResBlock 堆疊 + GAP + Dense；輸入 `[1, 15000, 1]`，輸出 `[1, 3]` |
| 前處理（訓練端） | 整筆 record band-pass 0.5–45 Hz @300 Hz → FFT resample 500 Hz → 整筆 z-score → 切窗（短檔補零） |
| 優化器 | AdamW，warmup + cosine decay，峰值 LR 1e-3，batch 64，最多 200 epoch，early stop patience 40 |
| 增強 | 幅度縮放、高斯噪音、基線漂移、隨機平移 |
| 後處理 | 剪枝 20% → 40%（各 8 epoch Adam 2e-5 微調）→ full-integer INT8 PTQ → Vela（u55-256，`--optimise Performance`） |
| Notebook | `ECG_classification/AF_Edge_M55M1_Training.ipynb`（原版）；`AF_Edge_M55M1_Training_v5_rerun.ipynb`（原樣重跑，用於產生訓練曲線） |

### 5.3 訓練過程（來自 v5 原樣重跑，原版 notebook 未保存輸出）

| epoch | train acc | train loss | val acc | val loss | lr |
|---|---|---|---|---|---|
| 1 | 0.337 | 3.392 | 0.360 | 3.032 | 1.25e-4 |
| 50 | 0.810 | 0.582 | 0.808 | 0.592 | 8.9e-4 |
| 100 | 0.849 | 0.452 | 0.810 | 0.532 | 5.4e-4 |
| **144（best）** | **0.875** | **0.362** | **0.860** | **0.447** | 2.0e-4 |
| 184（stop） | 0.899 | 0.291 | 0.854 | 0.465 | 2.0e-5 |

184 epoch early stop，Colab T4 每 epoch 約 8 s。144 之後 train 續升、val 持平略降，early stopping 正確截斷。逐 epoch 原始資料：`docs/report_v5/history_v5_rerun.csv`。

### 5.4 各階段準確度（notebook 內，test 2301 視窗）

| 階段 | window acc |
|---|---|
| 單模型 FP32（best epoch） | 85.57% |
| TTA | 85.27%（無增益，部署不用） |
| Record-level 投票（1024 record） | 85.64% |
| 剪枝 40% + 微調 | **86.27%** |
| **INT8 PTQ** | **86.27%**（量化掉點 0.00） |

### 5.5 部署模型最終成績（PC 端重現 MCU 前處理 ★ 報告主數據）

評估腳本 `ECG_classification/tools_eval_tflite_mcu.py`：對 test 1485 視窗 / 1024 record 逐視窗做與韌體完全相同的 band-pass → z-score → INT8 推論。現役 `deploy/model_int8.tflite`：

| 層級 | accuracy | macro-F1 |
|---|---|---|
| Window（1485） | 85.79% | 0.8383 |
| **Record（1024）** | **86.43%** | **0.8427** |
| Record，僅 ≥30 s 的 record | — | 0.8525 |

| 類別 | precision | recall | F1 | support |
|---|---|---|---|---|
| Normal | 0.899 | 0.921 | 0.910 | 610 |
| **AF** | 0.806 | **0.868** | 0.836 | 91 |
| Other | 0.811 | 0.755 | 0.782 | 323 |
| macro avg | 0.839 | 0.848 | **0.843** | 1024 |

混淆矩陣（列 = 真實，欄 = 預測）：

| | → Normal | → AF | → Other |
|---|---|---|---|
| Normal (610) | **562** | 2 | 46 |
| AF (91) | 1 | **79** | 11 |
| Other (323) | 62 | 17 | **244** |

解讀：AF 漏報 12/91，其中 11 筆判 Other、只有 1 筆判 Normal；Normal 誤判為 AF 僅 2/610（假警報率 0.3%）。主要混淆在 Normal ↔ Other（46 + 62），屬 CinC2017 Other 類本身邊界模糊（含雜訊、多種非 AF 心律）。

### 5.6 模型規格與 Vela 編譯

| 項目 | v5 現役 |
|---|---|
| INT8 tflite | 455.7 KB |
| Vela tflite / `af_model.h` | 221.0 KB |
| 輸入量化 | scale 0.145202，zp 20 |
| 輸出量化 | scale 0.003906（1/256），zp −128 |
| Operators | CPU **0** / NPU **3079**（100% NPU） |
| SRAM 用量 | 1246 KiB |
| Flash 用量 | 149 KiB |
| MACs / 推論 | 177 M |

### 5.7 可重現性
原樣重跑 v5：record macro-F1 **0.8395**，與現役 0.8427 差 0.003；INT8 465 KB、Vela 224 KB、3062 ops 100% NPU、SRAM 1246 KiB。證明 v5 成績不是單次運氣。

---

## 6. 模型優化探索與定案理由

2026-10-22～23 另設計四版模型，皆以**同一測試集、同一評估腳本、INT8**量測：

| 模型 | 設計 | record macro-F1 | ≥30 s | window acc | 結論 |
|---|---|---|---|---|---|
| **v5（定案）** | 多尺度 CNN + ResBlock，整筆 record 前處理 | **0.8427** | 0.8525 | 85.8% | 現役 |
| v5 重跑 | 同配方重訓 | 0.8395 | 0.8497 | 85.4% | 可重現 |
| v7b | v5 流程 + 16 維 RR 間期特徵雙輸入 + modality dropout 25% | 0.8312 | 0.8475 | 84.7% | RR 歸零消融 0.811（RR 有 +0.017 貢獻）但整體仍輸 |
| v6 | 新主幹（kernel ≤ 64、dilated ResBlock），逐視窗 MCU 前處理 | 0.8182 | 0.8287 | 84.3% | Vela 88 ops / SRAM 531 KiB（較省），但準確度輸 |
| v6.1 | v6 + 輕量增強、SWA | 0.8132 | — | — | 輸 |
| v7 | v5 主幹 + RR 雙輸入，逐視窗 MCU 前處理 | 0.8063 | 0.8081 | 82.4% | RR 歸零消融只剩 0.599（過度依賴 RR） |

**關鍵推論**：v7 用 v5 主幹仍只有 0.806，差距來自**訓練流程**（v5 整筆 record band-pass + FFT resample + 整筆 z-score + 短檔補零 + 剪枝微調），而非主幹架構；v5 式流程比逐視窗流程好約 0.02。

**定案**：四個新設計都未超過 v5（差 0.01–0.04），v5 已接近此資料集與模型規模的合理上限；考量重新部署韌體的風險，**維持 v5 部署、停止優化**。模型集成（v5+v6 平均僅 0.845）不值得雙倍算力，未採用。本 repo 只收錄 v5 的模型產物。

副產物（已就緒但未部署）：韌體 `af_inference.h` 支援雙輸入（`AF_MODEL_RR_FEATURES` 開關、依張量形狀辨認輸入）；`rr_features.py` / `rr_features.h` / notebook Cell 3 三處 R 峰與 16 維 RR 特徵演算法逐行一致（409 視窗 C vs Python 峰集合 0 不一致、特徵最大差 8e-7）。

---

## 7. 驗證紀錄（時間線）

| 日期 | 段落 | 結果 |
|---|---|---|
| 2026-06-28 | PC 端 BLE 路徑 | 發現並修復 band-pass 漏失 bug（AF 0.00 → 0.98） |
| 2026-07-03 | M55M1 韌體補 band-pass | PC 端逐位元驗證；Keil 0 error；上板模型載入成功 |
| 2026-07-03 | AF 穩定度（PC 端，RR 變異最高的 ≥30 s record） | A03346（RR_CV 0.947）AF 0.948、A00137 0.969、A01501 0.655 → 3/3 判 AF |
| 2026-07-03 | 60 Hz 雜訊釐清 | 雜訊 = 60 Hz 市電（TriAnswer 固有），接地僅 −13 pt；在 band-pass 通帶外，AF 不受影響 0.96–0.98 |
| 2026-10-20 | 偵測鏈復刻（新 TriAnswer / 新 M55M1 板 / 新 HECG100，仍經 ESP32） | AF 0.86–0.99、Normal 0.91–0.97 |
| 2026-10-22 | TriAnswer 新韌體 → PC（USB 鏡射） | A01210 三次皆 3/3 AF 0.93–0.99；A04565 3/3 Normal 0.94；60 Hz 占比 0.000；取樣率 499.9 Hz |
| 2026-10-22 | **全鏈 TriAnswer pin22 → M55M1 D13（無 ESP32）** | **A01210 連續 10 視窗 AF 0.59–0.99；切 A04565 過渡 1 視窗 Other 後連續 11 視窗 Normal 0.73–0.96** |
| 2026-10-23 | v5 可重現性 + 數據包 | macro-F1 0.8395 vs 0.8427；`report_v5/` 完成 |

### 7.1 各段驗證指令
```
# 1. HECG100 連線（安全、不輸出訊號）
python HECG_Lab/connect_check.py

# 2. TriAnswer → PC：另一終端播 AF，收 3 個 30 s 視窗後推論
python TriAnswer_Firmware/tools/uart_capture.py COMx 3
python HECG_Lab/run_af.py A01210          # 或 run_normal.py A04565
python ECG_classification/inference.py captures/ecg_uart

# 3. 極性判定
python TriAnswer_Firmware/tools/polarity_check.py A01210

# 4. M55M1 監看（115200）
python M55M1_Firmware/tools/serial_monitor.py COMx

# 5. 全鏈：接好 pin22→D13、共地後播放，看 serial_monitor 的 [結果] 行
python HECG_Lab/run_af.py A01210
python run_hecg_loop.py A04565 A01210 120   # Normal/AF 每 120 s 交替
```

---

## 8. 已知限制與邊界條件

1. **快速 AF 信心偏低**：HR 高（如 A01501 HR 115）時 R-R 短、形態近 Other，仍判 AF 但 Other 洩漏至 ~0.30。模型相對較弱區。
2. **短檔循環拼接假影**：record < 30 s 時播放腳本會在一視窗內循環多次，接縫處節拍不連續被判 Other（A00781 13.4 s 即此例）。屬測試治具限制，非模型漏報；穩定度測試須挑 ≥30 s record。
3. **Normal ↔ Other 混淆**為主要誤差來源（46 + 62 / 1024），反映資料集 Other 類定義寬泛。
4. **60 Hz 市電**為 TriAnswer 固有弱點（BLE 路徑曾占 60%），band-pass 0.5–40 Hz 已濾除；UART 路徑實測接近 0。無需額外 notch。
5. **切換延遲**：Normal ↔ AF 切換時滑動視窗（30 s 視窗 / 10 s stride）需 1 個過渡視窗，期間可能短暫判 Other。
6. 測試資料為 PhysioNet 單導程家用 ECG 經模擬器重播，**非真人臨床試驗**；真人靜止實測僅做過 Normal（0.96，舊 BLE 路徑）。

---

## 9. 數據包與圖檔索引

全部位於 `docs/report_v5/`：

| 檔案 | 內容 | 建議用途 |
|---|---|---|
| `README.md` | 所有表格的原始版本（本文 5.1–5.7 與第 6 節即取自此） | 交叉核對 |
| `fig_training_curves.png` | 重跑 v5 的 train/val acc、loss、LR 曲線（標 best epoch 144） | 「訓練過程」圖 |
| `history_v5_rerun.csv` | 逐 epoch acc / loss / val_acc / val_loss / lr（184 列） | 自製圖表 |
| `fig_cm_v5_deployed_record.png` | **現役 v5** INT8 record-level 混淆矩陣（MCU 前處理） | **主結果圖** |
| `fig_cm_v5_rerun_record.png` | 重跑 v5 混淆矩陣 | 可重現性佐證 |
| `fig_notebook_training_curves_raw.png` | notebook Cell 7 原始曲線（FP32） | 備用 |
| `fig_notebook_cm_record_v5style.png` | notebook Cell 7 混淆矩陣（v5 式前處理、FP32） | 備用 |
| `eval_mcu_v5_vs_rerun.txt` | `tools_eval_tflite_mcu.py` 原始輸出（含 sklearn classification report） | 數字來源 |

其他原始檔：
- 模型：`ECG_classification/deploy/{model_int8.tflite, model_int8_vela.tflite, af_model.h}`
- Notebook：`ECG_classification/AF_Edge_M55M1_Training.ipynb`、`AF_Edge_M55M1_Training_v5_rerun.ipynb`
- M55M1 應用層：`M55M1_Firmware/NN_ModelInference/`
- TriAnswer 韌體：`TriAnswer_Firmware/TriAnswer_UART_to_M55M1/*.ino`

---

## 10. 踩雷紀錄（避免後續工程重蹈）
- **Keil 命令列燒錄降級 Nu-Link 韌體**：`UV4.exe -j0`（無 GUI）會自動接受「ICE 韌體降級」對話框，把 Nu-Link2-Me 從 7829 降到 7746 → `Wait mcu stop error` / `Invalid Chip`，完全燒不進去。預防：GUI F8 燒錄，或先在 `KEIL/Nu_Link_Driver.ini` 設 `DisableFirmwareUpdate=1`。救法：裝最新 Nu-Link_Keil_Driver（V3.24.8062r）升 ICE 韌體。
- **Nu-Link2-Me 拖放燒錄（MSC）對 M55M1 不可靠**（`Erase Timeout`），用 Keil。
- **DTCM 溢位**：大型緩衝須放 `.bss.sram.data`，否則 linker L6406E/L6407E。
- **UART 腳位**：M55M1 NU3 A4/A5 不是 UART1，必須接 NU2 D13（PA.2）。
- **SysTick 取樣偏慢**：見 3.2。
- **HECG100 USB handle 鎖死**：播放腳本被硬殺（SIGTERM）後 handle 約 15 s 才釋放；`play_dataset.py` 已攔 SIGTERM 並有 `guard_single_instance()` 防雙程序。真鎖住時拔插 USB。
- **HECG100 卡死模式**：曾有兩個 `run_af.py` 同時存活 → 之後 connect/start 都回 True 但類比完全不輸出，`stop_output` 救不回，只能 USB + 電源拔掉重插。判斷法：TriAnswer 手指測試有反應但播放是平線。
- **TriAnswer 原廠 SysTick 警告**：原廠核心要求 `SysTick_Handler()` 與 `SysTick_Config()` 同時存在否則當機；本自製韌體完全不用 SysTick，不適用。另外不可用 `while(!Serial)` 等 USB，未插 USB 會卡住（M55M1 現場無 PC）。
- **平線 / 無訊號**：兩次皆為 HECG100 夾子 ↔ 電極接觸不良（手指逐段測試）；空接特徵為 p-p ~25 counts、mean ~1320。
- **M55M1 雙重供電造成 TriAnswer 無訊號（2026-10-23）**：M55M1 同時接 HSUSB 外接電源與 Nu-Link2-Me（PC），TriAnswer 也接同一台 PC 時，TriAnswer 讀值如浮接（mean≈1318、p-p≈18），子板其實已插好。拔掉 HSUSB 外接電源即恢復。推測為兩個電源地之間的電位差經共地線干擾類比前端。規則：全鏈測試時 M55M1 只用 Nu-Link2-Me 供電。
- **`play_dataset.normalize_polarity`** 會把負極值較大的 record 自動反相播放（A04565 翻、A01210 不翻），判極性必須比對實際播出波形（`polarity_check.py` 已套同規則）。
- **litert 不吃中文路徑**：`inference.py` 已改以 `model_content` 載入。
- **`tools_eval_tflite_mcu.py`** 需要 tensorflow（測試時為 2.21）、pandas、scikit-learn，且要有本地資料集；`os.path.relpath` 不吃跨磁碟路徑，tflite 請放在 repo 內。

## 11. 名詞表
| 名詞 | 說明 |
|---|---|
| AF | Atrial Fibrillation，心房顫動 |
| CinC2017 | PhysioNet/Computing in Cardiology Challenge 2017 單導程 ECG 資料集 |
| Lead II | 右臂（RA）→ 左腳（LL）導程向量 |
| Vela | Arm Ethos-U NPU 離線編譯器，將 INT8 tflite 轉為 NPU 命令流 |
| PTQ | Post-Training Quantization，訓練後量化 |
| TTA | Test-Time Augmentation |
| record-level | 同一筆 ECG 記錄的所有視窗機率平均後 argmax，以記錄為單位計分 |
| macro-F1 | 三類 F1 的算術平均（不加權） |
| RR 間期 | 相鄰 R 峰時間差，AF 的主要節律線索 |
| Tensor Arena | TFLite Micro 的靜態工作記憶體 |
| Nu-Link2-Me | NuMaker 板載除錯 / 燒錄器 |

