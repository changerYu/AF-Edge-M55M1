# TriAnswer_Firmware — TriAnswer（Yutech TriBLE nRF52840）UART 直送韌體

把 TriAnswer 從原廠 BLE 範例韌體換成自製韌體：**硬體 TIMER4 每 2 ms 讀 ADC（500 Hz，12-bit）→ 環形緩衝 → 每 100 點組一幀走 `Serial1`（921600 8N1）**，直接餵給 M55M1，不用 BLE、不用任何橋接板。

```
[0xAA][0xBB][LEN_L][LEN_H][int32 LE × 100]     LEN = 400，每 200 ms 一幀
```

幀格式與 `M55M1_Firmware/NN_ModelInference/uart_receiver.h` 相同，M55M1 韌體不需為此修改。

## 目錄

| 路徑 | 內容 |
|---|---|
| `TriAnswer_UART_to_M55M1/TriAnswer_UART_to_M55M1.ino` | 韌體原始碼。設定都在檔頂 `#define`（ADC 腳位、解析度、baud、`MIRROR_TO_USB`） |
| `tools/uart_capture.py COMx [N] [out_dir]` | PC 端收幀：驗取樣率、存 30 s 視窗 `.npy`（預設到 `<repo>/captures/ecg_uart`）。`--selftest` 不接硬體自測解析器 |
| `tools/polarity_check.py <record> [dir]` | 擷取視窗 vs 來源記錄的互相關：判 RA/LL 是否接反（corr 正 = 正確、負 = 反接），lag 等速前進 = 取樣率正確 |
| `tools/live_level.py COMx` | 每秒印一行 mean / p-p，邊調整子板與電極邊看訊號有沒有進來 |

## 環境

- Arduino 核心：**Yutech `nrf52` 0.3.15**（Adafruit nRF52 fork：SoftDevice s140 6.1.1、FreeRTOS、TinyUSB）。Boards index 與核心由 Yutech 提供（GitHub `YuTecHealth`），FQBN `Yutech:nrf52:TriBLE_nRF52`。
- 燒錄走 adafruit-nrfutil 串列 DFU（1200 bps touch），arduino-cli `upload` 直接支援。TriAnswer 用 USB 線接 PC，裝置管理員出現 `Yutech TriBLE nRF52840` 的 COM。

```
arduino-cli compile --fqbn Yutech:nrf52:TriBLE_nRF52 TriAnswer_Firmware/TriAnswer_UART_to_M55M1
arduino-cli upload  --fqbn Yutech:nrf52:TriBLE_nRF52 -p COMx TriAnswer_Firmware/TriAnswer_UART_to_M55M1
```

燒錄成功後紅燈常亮、每 200 ms 藍燈閃一次。要還原成原廠 BLE 韌體，燒回核心內建範例 `TriBLE_Tutorial/examples/TriBLE_Stu_Demo`（未收錄，原廠提供）。

## 硬體

- **ECG 子板插中排 = CH2 = A3**。CH1 = A5（上排）、CH3 = A2（下排）。子板橫跨左右兩側插槽，插上排會蓋住 D0（UART TX），因此韌體讀 A3。
- 板接頭 **pin 22 = UART TX / D0（P0.25）**、pin 21 = UART RX / D1（P0.24），即 Arduino `Serial1`，3.3 V 邏輯。GND 在 pin 4 / 8 / 12。
- 接 M55M1：pin 22 → M55M1 **NU2 `D13`（PA.2 = UART1 RX）**，GND 共地。
- 供電：USB（VIN–VUSB 跳帽）或電池。

## 設計要點

- **為什麼不用 SysTick**：原廠範例用 SysTick 計時，但 loop 等 UART 送完幀時 FreeRTOS 讓 CPU 睡眠，Cortex-M 的 SysTick 在睡眠時停擺，實測取樣率只有約 490 Hz（每幀 4.4 ms / 200 ms ≈ 2%）。改用周邊 TIMER4 後韌體自測 499.9 Hz。
- 另有啟動外部 32 MHz 晶振（HFXO）；不跑 BLE 時 SoftDevice 不會幫忙開，預設內部 RC 會偏約 2%。
- `MIRROR_TO_USB=1` 時同一份幀也送 USB CDC，PC 可直接用 `uart_capture.py` 驗證，不需 USB-TTL 轉接。USB 每 5 秒另印一行 `# frames=... rate~... Hz` 狀態（純文字，解析器會略過）。PC 端用 USB 到達時間估的取樣率有 ±1% 抖動，以韌體自報的 `rate~` 為準。
- 不要用 `while(!Serial)` 等 USB，未插 USB 時會卡住。
- ADC 用 12-bit（原廠 8-bit），參考電壓沿用核心預設；M55M1 端做 z-score，不在乎絕對刻度。

## 訊號判讀（12-bit、讀 A3）

| 現象 | 意義 |
|---|---|
| mean ≈ 1320、p-p < 50 | 浮接：子板沒接到 A3（排針接觸不良） |
| mean ≈ 1266、p-p < 100 | 子板已接上，但沒有 ECG 輸入（HECG100 沒播 / 電極夾沒夾好） |
| p-p > 300 | 有 ECG |

**讀值像浮接但子板確實插好時，先檢查 M55M1 的供電**：M55M1 若同時接 HSUSB 外接電源與 Nu-Link2-Me（PC），而 TriAnswer 也接同一台 PC，兩個電源地之間的電位差會經共地線讓 TriAnswer 完全收不到 ECG（mean≈1318、p-p≈18）；拔掉 HSUSB 外接電源即恢復。全鏈測試時 M55M1 只用 Nu-Link2-Me 一條 USB 供電。

平線時另一個常見原因是 HECG100 鱷魚夾與電極扣接觸不良；用手指捏兩顆電極有反應 → TriAnswer 與子板正常，問題在夾子。擷取資料開頭幾秒有接電極的大幅假影，做極性 / 相關分析時要避開。
