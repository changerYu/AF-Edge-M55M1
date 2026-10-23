# AF-Edge: HECG100 → TriAnswer → Nuvoton M55M1 心房顫動（AF）邊緣偵測鏈

端到端、可離線運作的 AF 即時偵測系統：ECG 模擬器播放真實心電圖 → 穿戴式感測器以 500 Hz 採樣並經硬體 UART 送出 → Cortex-M55 + Ethos-U55 NPU 上以 INT8 模型每 10 秒判定一次 **Normal / AF / Other**。

```
HECG100 (WhaleTeq ECG 模擬器)      TriAnswer (Yutech TriBLE nRF52840)         Nuvoton NuMaker-X-M55M1D
播放 PhysioNet/CinC2017 真實 ECG  →  自製韌體：500 Hz 12-bit ADC 採樣         →  Cortex-M55 + Ethos-U55 NPU
經實體電極輸出 (Lead II, ≤5 mV)      硬體 UART 921600 送幀                        band-pass → z-score → INT8 推論
                                     [0xAA][0xBB][LEN][int32 LE × 100]            每 10 s 輸出 Normal / AF / Other
```

| 指標 | 數值 |
|---|---|
| 模型 | 多尺度 1D-CNN + ResBlock，INT8，Vela 編譯，**100% 在 NPU 上執行**（3079 ops，CPU 0） |
| 測試集（1024 record，record-level 切分） | macro-F1 **0.843**，accuracy 86.4% |
| AF 召回 / AF 假警報 | recall **0.868**（漏報 12/91）；Normal 誤判為 AF 2/610（0.3%） |
| 模型大小 / 記憶體 | Vela tflite 221 KB；SRAM 1246 KiB；Tensor Arena 實測 1,276,468 bytes |
| 實機全鏈 | AF 記錄連續 10 視窗判 AF（0.59–0.99）；Normal 記錄連續 11 視窗判 Normal（0.73–0.96） |

完整技術細節（硬體接線、韌體設計、模型訓練與評估、限制）見 **[docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md)**；模型數據包（訓練曲線、混淆矩陣、逐 epoch CSV）在 **[docs/report_v5/](docs/report_v5/README.md)**。

## 目錄

| 目錄 | 內容 |
|---|---|
| [`TriAnswer_Firmware/`](TriAnswer_Firmware/README.md) | TriAnswer 自製韌體（Arduino，TIMER4 500 Hz 採樣 → Serial1 送幀）與 PC 端擷取 / 極性檢查工具 |
| [`M55M1_Firmware/`](M55M1_Firmware/README.md) | M55M1 應用層原始碼（UART 幀解析、band-pass、z-score、INT8、NPU 推論）與 PC 端監看 / 驗證工具。需搭配 Nuvoton M55M1BSP 建置 |
| [`ECG_classification/`](ECG_classification/README.md) | 模型訓練 notebook（Colab）、部署模型（`deploy/`）、PC 端推論與評估腳本 |
| [`HECG_Lab/`](HECG_Lab/README.md) | HECG100 ECG 模擬器控制腳本（播放 CinC2017 記錄 / 合成波形 / 緊急停止） |
| `legacy_ble/` | 舊的 BLE 擷取路徑（搭配 TriAnswer 原廠 BLE 韌體時使用），非主要流程 |
| `docs/` | 技術總覽與模型數據包 |
| `run_hecg_loop.py` | Normal ↔ AF 交替播放，用來觀察 M55M1 的即時切換 |

## 本 repo 不包含的東西（需自行取得）

| 項目 | 原因 | 取得方式 |
|---|---|---|
| PhysioNet/CinC 2017 資料集 | 體積大、有自己的授權 | 見 [`ECG_classification/data/README.md`](ECG_classification/data/README.md) |
| WhaleTeq HECG100 SDK（`HECG100x64.dll`） | 第三方 SDK（MIT），只收錄我們修改過的 Python 包裝 | 見 [`HECG_Lab/README.md`](HECG_Lab/README.md) |
| Nuvoton M55M1BSP（`Library/`、`ThirdParty/`、Keil 專案） | 原廠 BSP，只收錄我們撰寫 / 修改的應用層檔案 | 見 [`M55M1_Firmware/README.md`](M55M1_Firmware/README.md) |
| Yutech nRF52 Arduino 核心與原廠 BLE 範例韌體 | 原廠提供 | 見 [`TriAnswer_Firmware/README.md`](TriAnswer_Firmware/README.md) |

## 快速開始（PC 端驗證，不需 M55M1）

```bash
pip install -r requirements.txt

# 1. HECG100 連線測試（安全，不輸出訊號）
python HECG_Lab/connect_check.py

# 2. 燒錄 TriAnswer 韌體後，以 USB 接 PC，收 3 個 30 s 視窗
python TriAnswer_Firmware/tools/uart_capture.py COMx 3

# 3. 另一終端讓 HECG100 播放一筆 AF 記錄（Ctrl+C 停止）
python HECG_Lab/run_af.py A01210

# 4. 推論
python ECG_classification/inference.py captures/ecg_uart      # 預期 AF ≈ 0.9+
```

全鏈（接上 M55M1）：TriAnswer pin 22（UART TX）→ M55M1 NU2 `D13`（PA.2 = UART1 RX），共地；`python M55M1_Firmware/tools/serial_monitor.py COMx` 看 `[結果]` 行。

## 關鍵設計要點

- **前處理必須與訓練一致**：500 Hz、30 s 視窗（15000 點）、stride 10 s；**band-pass 0.5–40 Hz（零相位）→ z-score → INT8**。缺了 band-pass，60 Hz 市電雜訊會讓 AF 機率歸零、全部判成 Other。PC 端 `inference.py` 與韌體 `af_inference.h` 的實作已驗證經模型後逐位元相同。
- **取樣率**：TriAnswer 原廠範例用 SysTick 計時，在 FreeRTOS 睡眠時會停擺而偏慢約 2%；本韌體改用硬體 TIMER4，實測 499.9 Hz。
- **幀格式**：TriAnswer 與 M55M1 共用 `[0xAA][0xBB][LEN_L][LEN_H][int32 LE × N]`，M55M1 端解析器與 PC 端 `uart_capture.py` 同邏輯。
- **電極**：Lead II（右鎖骨電極 → HECG100 RA/R，左肋電極 → LL/F）。極性可用 `polarity_check.py` 互相關判定。
