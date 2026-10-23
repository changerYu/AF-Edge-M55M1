# M55M1_Firmware — Nuvoton NuMaker-M55M1 邊緣端韌體（應用層）

在 Cortex-M55 + Ethos-U55 NPU 上執行 Vela 編譯的 INT8 模型，對 UART 送進來的 ECG 做即時 AF 偵測。

```
UART1 ISR (PA.2) → uart_receiver.h (幀解析, 30 s 視窗) → af_inference.h
   → band-pass 0.5–40 Hz → z-score → INT8 → NPU 推論 → Normal / AF / Other
   → printf 到 Nu-Link VCOM (115200) + LED + UART1 TX 單字元結果碼 'N'/'A'/'O'
```

## 本目錄只收錄我們撰寫 / 修改的檔案

Nuvoton **M55M1BSP**（`Library/`、`ThirdParty/` 的 TFLite Micro 與 ml-embedded-evaluation-kit、Keil / VS Code 專案檔）屬原廠 BSP，**未收錄**。建置方式：

1. 取得 Nuvoton M55M1BSP（GitHub `OpenNuvoton/M55M1BSP`），裡面有範例 `SampleCode/MachineLearning/NN_ModelInference/`。
2. 把本目錄 `NN_ModelInference/` 的 5 個檔案複製進去（覆蓋同名檔）：

| 檔案 | 內容 |
|---|---|
| `main.cpp` | UART1 初始化（`SET_UART1_RXD_PA2` / `SET_UART1_TXD_PA3`）、ISR、主迴圈、結果輸出 |
| `uart_receiver.h` | 幀解析狀態機、15000 點視窗、stride 5000 滑動 |
| `af_inference.h` | 前處理（band-pass → z-score → INT8）與 TFLite Micro / Ethos-U55 推論 |
| `rr_features.h` | RR 間期特徵（雙輸入模型選用；單輸入模型不會編到） |
| `af_model.h` | Vela 編譯後的 INT8 模型 C 陣列（與 `ECG_classification/deploy/af_model.h` 同一份；模型單一來源在那裡） |

3. 專案設定（沿用 BSP 範例即可）：Define `NVT_VECTOR_ON_FLASH, TF_LITE_STATIC_MEMORY, ARM_MATH_DSP, ARM_NPU`；include path 含 `NN_ModelInference/`、`ThirdParty/tflite_micro`、`ThirdParty/ml-embedded-evaluation-kit/...`、`Library/StdDriver/inc/npu`。
4. 用 **Keil MDK Nuvoton Edition**（免費、含 M55、無 code size 限制；AC6）開 `KEIL/NN_ModelInference.uvprojx` 建置燒錄，或 VS Code + NuMicro pack。

建置結果參考：Code ≈ 135 KB、ZI ≈ 1.5 MB（含 1.3 MB Tensor Arena 與 60 KB band-pass 緩衝），0 error。

## 前處理（必須與訓練一致）

1. 收 15000 點（30 s @ 500 Hz），之後每 5000 點（10 s）推論一次。
2. **Band-pass 0.5–40 Hz 零相位**：`bandpass_0p5_40()` = 去均值 → 前向 DF2T biquad ×3 → 反轉 → 再前向 → 反轉，每個 pass 以穩態初始條件起始。重現訓練端 `scipy.signal.butter(3,[0.5,40]/250,'band') + filtfilt`。係數與初始條件由 `tools/gen_bandpass_golden.py` 產生。
3. z-score：`(x − mean) / std`。
4. INT8：`round(x / in_scale) + in_zp`，scale / zero-point 由模型張量讀取。
5. 推論 → 3 個 int8 分數反量化 → argmax。

**band-pass 不可省**：少了它，60 Hz 市電雜訊會把 AF 機率壓到 0、全部判 Other。`tools/validate_c_bandpass.py` 用與 C 相同的逐樣本迴圈證明：經模型後輸出與 SciPy filtfilt 逐位元相同。

## 記憶體

- `TENSOR_ARENA_SIZE` 1300 KB（本模型 Vela SRAM 用量 1246 KiB；開機回報實際 1,276,468 bytes）。
- `tensor_arena`、`bp_buf`（60 KB float）、`ecg_window`（60 KB int32）都放 `.bss.sram.data` section（SRAM01 / HyperRAM 區）。放在預設 `.bss` 會擠爆只有約 48 KB 的 DTCM，linker 報 L6406E / L6407E。

## 接線與燒錄

- TriAnswer UART TX → **NU2 排針 `D13` = PA.2 = UART1 RX**；共地。NU3 的 A4/A5 是 PB.0/PB.1（UART2），接那裡收不到。
- 燒錄 / Log 用 Nu-Link2-Me Type-C（需資料線），VCOM 115200。**全鏈測試時只用這一條 USB 供電**：若再插 HSUSB 外接電源，而 TriAnswer 接同一台 PC，兩個電源地的電位差會讓 TriAnswer 收不到任何 ECG（讀值像浮接）；拔掉 HSUSB 即恢復。
- 開機記錄應見：Ethos-U55 init → `模型載入成功` → `Tensor Arena 使用量: 1276468 bytes` → 等待 UART 資料。`SD initial fail!!` 無害（模型編在 Flash）。
- 燒錄時若 Nu-Link 詢問是否**降級** ICE 韌體，請選 Cancel。Keil 命令列 `UV4.exe -j0` 無視窗模式會自動接受降級，之後板子會 `Wait mcu stop error` 燒不進去；命令列燒錄前先在 `Nu_Link_Driver.ini` 設 `DisableFirmwareUpdate=1`，或直接用 GUI。救法是安裝最新 Nu-Link_Keil_Driver 重新升級 ICE 韌體。
- Nu-Link2-Me 的拖放燒錄隨身碟對 M55M1 不可靠（Erase Timeout），用 Keil。

## tools/（PC 端）

| 檔案 | 用途 |
|---|---|
| `serial_monitor.py [COMx] [--save]` | 監看 Nu-Link VCOM 的推論結果，上色、可存 CSV |
| `verify_model.py` | 印 `ECG_classification/deploy/model_int8.tflite` 的輸入輸出量化參數並做零訊號 smoke test |
| `gen_bandpass_golden.py [window.npy]` | 產 band-pass SOS 係數 / 穩態初始條件，比較 naive 與穩態初始兩種做法對模型輸出的影響，輸出 golden vector 標頭 |
| `validate_c_bandpass.py [window.npy]` | 手刻 C 等價 DF2T 雙向迴圈，驗證經模型後與 filtfilt 一致 |

後兩支預設讀 `<repo>/captures/ecg_uart/window_0s_30s.npy`（用 `TriAnswer_Firmware/tools/uart_capture.py` 擷取）。
