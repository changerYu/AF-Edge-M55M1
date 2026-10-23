"""
即時顯示 TriAnswer 訊號狀態 (每秒一行), 用來邊調整子板/電極邊看有沒有接上。
用法: python TriAnswer_Firmware/tools/live_level.py COMx      (Ctrl+C 結束)

判讀 (12-bit, 讀 A3):
  mean ~1323, p-p < 50  → 浮接: ECG 子板沒接到 A3 (排針接觸不良 / 沒插到底)
  mean ~1266, p-p < 100 → 子板已接上, 但沒有 ECG 輸入 (HECG100 沒播 / 電極線沒接)
  p-p > 300             → 有 ECG
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uart_capture import FrameParser

def main():
    import serial
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    port = sys.argv[1]
    ser = serial.Serial(port, 921600, timeout=0.05); ser.dtr = True
    time.sleep(0.3); ser.reset_input_buffer()
    p = FrameParser(); buf = []; t_rep = time.time()
    print(f'[open] {port}  Ctrl+C 結束')
    try:
        while True:
            d = ser.read(4096)
            if d: buf += p.feed(d)
            if time.time() - t_rep >= 1.0:
                if buf:
                    a = np.array(buf, dtype=float); pp = a.max() - a.min(); m = a.mean()
                    if pp > 300:   tag = '✅ 有 ECG'
                    elif m > 1300: tag = '❌ 浮接 (子板沒接到 A3)'
                    elif pp < 100: tag = '🟡 子板已接上, 無 ECG 輸入'
                    else:          tag = '…'
                    print(f'{time.strftime("%H:%M:%S")}  mean={m:6.0f}  p-p={pp:6.0f}  n={len(a):4d}  {tag}', flush=True)
                else:
                    print(f'{time.strftime("%H:%M:%S")}  (no frames)', flush=True)
                buf = []; t_rep = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == '__main__':
    main()
