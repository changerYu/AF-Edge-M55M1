# HECG_Lab — WhaleTeq HECG100 ECG 模擬器控制

把已知 ECG（PhysioNet/CinC 2017 記錄或合成波形）從 HECG100 的實體電極端子輸出，餵給 TriAnswer，作為可控訊號源驗證整條鏈路。

## 需要自行取得的檔案（第三方 SDK，未收錄）

本目錄依賴 WhaleTeq 官方 **HECG100 SDK 1.0.0.1**（MIT License，授權全文見 `LICENSE-WhaleTeq-SDK.txt`）。請向 WhaleTeq 取得 SDK 後，把下面這個檔案複製到本目錄：

| 檔案 | SDK 內位置 |
|---|---|
| `HECG100x64.dll` | `SDK/HECG100x64.dll` |

`hecg100.py` 是 SDK 內 `Sample Code/python/hecg100.py` 的 Python ctypes 包裝，已收錄於本目錄。與原版相比只有一處修改：模組層級的 `device_lib = cdll.LoadLibrary(...)` 用相對於工作目錄的路徑載入 DLL，在 SDK 目錄以外執行會 `FileNotFoundError`，因此包成 `try/except`（`Device.__init__` 本來就用絕對路徑載 DLL，行為不變）。

## 腳本

| 腳本 | 用途 |
|---|---|
| `connect_check.py` | **先跑這個**。連線並印 Serial / 型號 / 韌體版本，不輸出訊號 |
| `run_af.py [record]` | 播放 CinC2017 **AF** 記錄（預設隨機一筆；全鏈驗證常用 `A01210`），循環播放，Ctrl+C 停止 |
| `run_normal.py [record]` | 播放 CinC2017 **Normal** 記錄（常用 `A04565`） |
| `run_ecg.py [BPM] [mV]` | 數學合成的正常竇性心律。太乾淨規律，模型多半判 Other，只適合驗訊號路徑 |
| `stop_output.py` | 緊急停止輸出（連送 3 次 stop） |
| `inspect_af.py` | 純資料分析：掃所有 AF 記錄算 RR 變異度，列出最不規律的前 10 筆 |
| `play_dataset.py` | 上述播放腳本的共用核心 |

播放腳本需要資料集放在 `../ECG_classification/data/`（見該目錄 README）。

## 電極接法（Lead II）

| TriAnswer 電極（配戴位置） | HECG100 端子 |
|---|---|
| 右鎖骨（右上） | **RA / R** |
| 左肋（左下） | **LL / F** |
| （選）接地 | RL / GND |

- 接反時波形上下顛倒。用 `../TriAnswer_Firmware/tools/polarity_check.py <record>` 做互相關判定。
- HECG100 輸出幅度上限 5 mV，超過時 `start_output` 回 False。

## 行為備註

- `play_dataset.py` 會對「負極值大於正極值 1.1 倍」的記錄自動反相再播放（印 `[POLARITY FLIPPED]`），因為 AliveCor 家用裝置常有反向配戴。比對極性時要以實際播出的波形為準，`polarity_check.py` 已套用同一規則。
- 播放腳本攔截 SIGINT / SIGBREAK / SIGTERM，中斷時會 stop ×3 再釋放 handle。被強制 kill 後 handle 約需 15 秒才釋放，馬上重連會失敗，稍等再試。
- 兩個播放程序同時連 HECG100 會讓裝置卡成「指令回 True 但類比不輸出」，只能斷電重插。`guard_single_instance()` 會在偵測到其他播放程序時拒絕啟動。
- 記錄短於 30 秒時循環播放會在一個推論視窗內產生接縫，接縫處節拍不連續可能被判 Other。做穩定度測試請挑 ≥30 秒的記錄。
- 驗 AF 請用 `run_af.py` 播放真實記錄，不要用 SDK 內建的臨床 AF 波形（形態與訓練分布不同，模型會判 Other）。
