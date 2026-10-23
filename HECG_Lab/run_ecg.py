"""
執行: python run_ecg.py [BPM] [mV]
範例: python run_ecg.py 60 1.0   (預設值)
      python run_ecg.py 80 1.5
按 Ctrl+C 停止
"""
import os, sys, time, signal, atexit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ctypes import pointer
from hecg100 import Device, ECGWaveformType

DLL = os.path.join(_HERE, 'HECG100x64.dll')   # 由 WhaleTeq SDK 複製而來, 見 HECG_Lab/README.md

bpm = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
mv  = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

from play_dataset import guard_single_instance  # noqa: E402
guard_single_instance()

device = Device(DLL)
handle = device.connect(-1, 5000)
if not handle:
    print('[FAIL] 沒連到裝置'); sys.exit(1)

_cleaned = False
def cleanup(*_):
    global _cleaned
    if _cleaned: return
    _cleaned = True
    print('\n停止中...', flush=True)
    # 連送 3 次 stop，再等韌體完成停止；裝置偶有「stop_output 回 True 但類比仍輸出」的情況
    for _i in range(3):
        try: device.stop_output(handle)
        except Exception: pass
        time.sleep(0.4)
    time.sleep(0.8)
    try: device.free(handle)
    except Exception: pass
    print('[OK] 已停止 (若仍在輸出, 請手動關 HECG100 電源)', flush=True)

atexit.register(cleanup)
signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))      # Ctrl+C
try: signal.signal(signal.SIGBREAK, lambda *_: sys.exit(0))  # Ctrl+Break (Windows)
except (AttributeError, ValueError): pass

wf = device.get_default_ecg_waveform()
wf.WaveformType  = ECGWaveformType.ECG.value
wf.Frequency     = bpm / 60.0
wf.Amplitude     = mv
wf.STSegment     = 0.0
wf.NoiseAmplitude = 0.0
wf.DriftEnabled  = 0

ok = device.start_output(handle, pointer(wf))
print(f'[輸出中] Normal Sinus Rhythm  {bpm:.0f} BPM, {mv:.1f} mV   (start_output={ok})')
print('按 Ctrl+C 停止')

while True:
    time.sleep(1)
