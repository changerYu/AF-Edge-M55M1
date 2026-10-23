# legacy_ble — 舊的 BLE 擷取路徑（非主要流程）

主要流程是 TriAnswer 自製 UART 韌體直送 M55M1（見 `../TriAnswer_Firmware/`）。本目錄保留 TriAnswer 還在跑**原廠 BLE 韌體**時的 PC 端擷取腳本，供需要 BLE 路徑的人參考。

| 腳本 | 用途 |
|---|---|
| `ble_scan.py` | 掃描 BLE 裝置（RSSI 過濾），找 TriAnswer 位址 |
| `ble_services.py` | 連線並列出 GATT services / characteristics |
| `capture_live.py [N] [out_dir]` | 有界擷取：收滿 N 個 30 s 視窗存 `.npy` 後結束，之後丟給 `ECG_classification/inference.py` |
| `test_af_playback.py [record] [N] [out_dir]` | 單一程序內：HECG100 播放指定記錄 + BLE 擷取 + 乾淨停止 |

兩種原廠韌體的封包格式（環境變數 `TRI=old` 切換）：

| | 舊款 TriBLE | 新款 GROUP5 範例 |
|---|---|---|
| ECG GATT | Service `0000a000` / Char `0000a001` | Service `0000ff05` / Char `0000aa05` |
| 封包 | int16 LE、3 通道交錯，取 `samples[0::3]` | uint8、單通道、每 byte 一樣本，20 bytes/pkt |

腳本內的 BLE 位址是範例，請改成 `ble_scan.py` 掃到的自己裝置。BLE 路徑受 60 Hz 市電干擾明顯（曾占總功率 60%），band-pass 可濾除、不影響判定；UART 路徑則幾乎沒有。需要 `pip install bleak`。
