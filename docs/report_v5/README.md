# AF_Edge_v5 技術報告數據包（2026-10-23）

現役模型 v5（`deploy/model_int8.tflite` / `af_model.h`）的完整數字與圖，供技術報告直接貼用。
原始 v5 notebook 沒有保存輸出，因此「訓練過程」數據來自 **2026-10-23 的原樣重跑**（`AF_Edge_M55M1_Training_v5_rerun.ipynb`，
只改 Cell 2 讀 Drive、Cell 12 存 zip，配方完全相同）；「部署模型」數據則是 PC 端直接量現役 `deploy/model_int8.tflite`。
兩者在同一測試集上 record macro-F1 0.8427 vs 0.8395（差 0.003），證明 v5 的成績可重現、不是單次運氣。

本資料夾檔案：

| 檔案 | 內容 |
|---|---|
| `fig_training_curves.png` | 重跑的 train/val accuracy、loss、learning-rate 曲線（標 best epoch 144） |
| `history_v5_rerun.csv` | 逐 epoch acc / loss / val_acc / val_loss / lr（184 列） |
| `fig_cm_v5_deployed_record.png` | **現役 v5** INT8 record-level 混淆矩陣（MCU 前處理） |
| `fig_cm_v5_rerun_record.png` | 重跑 v5 INT8 record-level 混淆矩陣（MCU 前處理） |
| `fig_notebook_training_curves_raw.png` / `fig_notebook_cm_record_v5style.png` | notebook Cell 7 原圖（v5 式前處理、FP32 單模型） |
| `eval_mcu_v5_vs_rerun.txt` | `ECG_classification/tools_eval_tflite_mcu.py` 原始輸出 |

---

## 1. 資料集與切分

| 項目 | 值 |
|---|---|
| 資料來源 | PhysioNet/CinC Challenge 2017（AliveCor 單導程，300 Hz） |
| Record 數 | 8528（N 5076 / A 758 / O 2415 / ~ 279） |
| 分類 | 3 類：Normal(N) / AF(A) / Other(O + ~ 合併) |
| 重採樣 | 300 Hz → **500 Hz**（對齊 TriAnswer） |
| 視窗 | **30 s = 15000 samples**，stride 10 s = 5000 samples |
| 視窗總數 | 19044（N 11093 / A 1720 / O 6231） |
| 切分方式 | **record-level** 分層切分（同一筆 record 的視窗不跨集，防洩漏），seed 42 |
| Train / Val / Test 視窗 | 14658 / 2085 / 2301 |
| Test record 數 | **1024**（N 610 / A 91 / O 323） |
| 類別權重 | N 0.572 / A 3.733 / O 1.016 |

## 2. 模型與訓練設定

| 項目 | 值 |
|---|---|
| 架構 | 多尺度 1D-CNN stem（kernel 75 / 45 / 25）+ ResBlock 堆疊 + GAP + Dense，輸入 `[1, 15000, 1]` |
| 優化器 / LR | AdamW，warmup + cosine decay，峰值 1e-3 |
| Batch / 最大 epoch | 64 / 200 |
| Early stopping | patience 40（監控 val），自動還原最佳權重 |
| 資料增強 | 幅度縮放、高斯噪音、基線漂移、隨機平移（notebook Cell 6） |
| 實際訓練 | **184 epoch 停止，最佳 epoch 144**（val acc 0.8600） |
| 單 epoch 時間 | Colab T4 約 8 s（第 1 epoch 含編譯 134 s） |

訓練曲線（`fig_training_curves.png`）關鍵點：

| epoch | train acc | train loss | val acc | val loss | lr |
|---|---|---|---|---|---|
| 1 | 0.337 | 3.392 | 0.360 | 3.032 | 1.25e-4 |
| 50 | 0.810 | 0.582 | 0.808 | 0.592 | 8.9e-4 |
| 100 | 0.849 | 0.452 | 0.810 | 0.532 | 5.4e-4 |
| **144（best）** | **0.875** | **0.362** | **0.860** | **0.447** | 2.0e-4 |
| 184（stop） | 0.899 | 0.291 | 0.854 | 0.465 | 2.0e-5 |

epoch 144 之後 train 持續上升、val 持平略降（輕微過擬合），early stopping 正確截斷。

## 3. 訓練後各階段準確度（notebook 內評估，test 2301 視窗，v5 式前處理）

| 階段 | window acc | 說明 |
|---|---|---|
| 單模型 FP32 | 85.57% | best epoch 144 權重 |
| TTA（測試時增強）| 85.27% | 多版本平均，對本資料無增益 → 部署不用 TTA |
| Record-level 投票 | 85.64%（1024 record）| 同 record 視窗機率平均後 argmax |
| 剪枝 20% → 40% + 微調 | **86.27%** | 各 8 epoch Adam 2e-5 微調，準確度不降反升 |
| **INT8 PTQ** | **86.27%** | full-integer 量化，**與剪枝後 FP32 掉點 0.00** |

INT8 模型（test 2301 視窗）各類別：

| 類別 | precision | recall | F1 | support |
|---|---|---|---|---|
| Normal | 0.89 | 0.93 | 0.91 | 1346 |
| AF | 0.83 | 0.84 | 0.83 | 225 |
| Other | 0.82 | 0.75 | 0.78 | 730 |
| macro avg | 0.85 | 0.84 | **0.84** | 2301 |

## 4. 部署模型最終成績（PC 端重現 MCU 前處理，INT8，同一測試集）

評估腳本 `tools_eval_tflite_mcu.py`：逐視窗 band-pass 0.5–40 Hz（butter 3 階 filtfilt，= 韌體 `bandpass_0p5_40()`）→ z-score → 用 tflite 的 scale/zp 量化 → INT8 推論，
與 M55M1 韌體 `af_inference.h` 的處理完全一致。Test = 1024 record / 1485 視窗。

| 模型 | window acc | window macro-F1 | record acc | **record macro-F1** | ≥30 s record F1 |
|---|---|---|---|---|---|
| **v5 現役（deploy/）** | 85.79% | 0.8383 | 86.43% | **0.8427** | 0.8525 |
| v5 重跑（deploy_v5_rerun/）| 85.39% | 0.8352 | 86.43% | 0.8395 | 0.8497 |

現役 v5 record-level 各類別（`fig_cm_v5_deployed_record.png`）：

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

AF 漏報僅 12/91（其中 11 筆判成 Other、只有 1 筆判成 Normal）；Normal 誤判成 AF 只有 2/610，AF 假警報率 0.3%。
主要混淆在 Normal ↔ Other（46 + 62），屬資料集本身 Other 類（含雜訊、各種非 AF 心律）邊界模糊。

## 5. 模型規格與 Vela 編譯（Ethos-U55-256，`--optimise Performance`）

| 項目 | v5 現役 | v5 重跑 |
|---|---|---|
| INT8 tflite 大小 | 455.7 KB | 465.2 KB |
| Vela tflite 大小 | 221.0 KB | 223.8 KB |
| 輸入量化 | scale 0.1452, zp 20 | scale 0.1548, zp 11 |
| 輸出量化 | scale 1/256, zp −128 | 同 |
| CPU / NPU operators | **0 / 3079（100% NPU）** | 0 / 3062（100% NPU） |
| SRAM 用量 | 1246 KiB | 1246.06 KiB |
| Flash 用量 | 149 KiB | 157.2 KiB |
| MACs / 推論 | 177 M | 176.9 M |
| SRAM 頻寬 / 推論 | — | 9.96 MB |

M55M1 實機：Tensor Arena 配置 1300 KB，開機回報用量 1276468 bytes（1.22 MB）；韌體 Code 134858 B、ZI 1537080 B（含 60 KB band-pass 緩衝）。

## 6. 實機驗證（HECG100 → TriAnswer → M55M1 全鏈）

| 測試 | 結果 |
|---|---|
| AF 播放 A01210 → M55M1 | 連續 8–10 視窗 AF 0.59–0.99（2026-10-20/22） |
| Normal 播放 A04565 → M55M1 | 過渡 1 視窗後連續 8–11 視窗 Normal 0.73–0.97 |
| AF 穩定度（PC 端，RR_CV 最高的 ≥30 s record）| A03346 0.948、A00137 0.969、A01501 0.655 → 3/3 判 AF |
| 短檔 A00781（13.4 s，循環拼接）| 判 Other（治具假影，非模型漏報） |
| 60 Hz 市電（TriAnswer 固有）| band-pass 0.5–40 Hz 濾除，接地前後 AF 皆 0.96–0.98 |

## 7. 架構 / 流程探索對照（同一測試集、同一腳本、INT8，說明為何定案 v5）

| 模型 | 說明 | record macro-F1 | ≥30 s | window acc |
|---|---|---|---|---|
| **v5（定案）** | 多尺度 CNN + ResBlock，整筆 record 前處理，剪枝 + PTQ | **0.8427** | 0.8525 | 85.8% |
| v5 重跑 | 同配方重新訓練（可重現性檢查） | 0.8395 | 0.8497 | 85.4% |
| v7b | v5 流程 + 16 維 RR 間期特徵雙輸入 + modality dropout | 0.8312 | 0.8475 | 84.7% |
| v6 | 新主幹（kernel ≤ 64、dilated ResBlock），逐視窗 MCU 前處理 | 0.8182 | 0.8287 | 84.3% |
| v6.1 | v6 + 輕量增強、SWA | 0.8132 | — | — |
| v7 | v5 主幹 + RR 雙輸入，逐視窗 MCU 前處理 | 0.8063 | 0.8081 | 82.4% |

結論：四個新設計都沒有超過 v5，差距 0.01–0.04，且 v5 重跑與現役僅差 0.003，顯示 v5 已接近此資料集與模型規模的合理上限；
考量重新燒錄韌體的風險，維持 v5 部署。
