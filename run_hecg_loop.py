"""
HECG100 循環播放: Normal N 秒 → AF N 秒 → 循環 (預設每段 120 秒 = 2 分鐘)。

設計重點 (避免 HECG100 卡住):
  - 單一連線, 內部切換 record (不反覆 connect/disconnect)。
  - Ctrl+C / 終止時乾淨釋放 (stop_output x3 + free), 不會鎖住 handle。

用法:
  python run_hecg_loop.py                       # Normal=A04565, AF=A01210, 每段 120s
  python run_hecg_loop.py A04565 A01210 120      # 指定 Normal / AF record / 每段秒數

停止: 在該終端按 Ctrl+C (會乾淨釋放)。
搭配: 另開終端跑 M55M1_Firmware/tools/serial_monitor.py COMx 看推論結果。
      切換訊號時 M55M1 會經過 AF<->Normal 的過渡窗 (滑動視窗特性), 屬正常。
"""
import os, sys, time, signal
import numpy as np
from scipy import io as sio
from ctypes import c_double, pointer, cast, c_void_p

_ROOT    = os.path.dirname(os.path.abspath(__file__))
HECG_DIR = os.path.join(_ROOT, "HECG_Lab")
HECG_DLL = os.path.join(HECG_DIR, "HECG100x64.dll")
DATA_DIR = os.path.join(_ROOT, "ECG_classification", "data", "training2017")
sys.path.insert(0, HECG_DIR)
from hecg100 import Device, PLAY_DATA  # noqa: E402  (hecg100.py 與 HECG100x64.dll 見 HECG_Lab/README.md)

FS_REC = 300.0     # PhysioNet 原始取樣率
GAIN   = 1000.0    # .hea: 1000/mV

NORMAL_REC = sys.argv[1] if len(sys.argv) > 1 else "A04565"
AF_REC     = sys.argv[2] if len(sys.argv) > 2 else "A01210"
PHASE_SEC  = int(sys.argv[3]) if len(sys.argv) > 3 else 120


def load(rec):
    """讀 record → mV, 自動翻正極性 (同 test_af_playback)。"""
    val = sio.loadmat(os.path.join(DATA_DIR, rec + ".mat"))["val"].flatten().astype(np.float64)
    sig = val / GAIN
    s = sig - np.median(sig)
    if abs(s.min()) > abs(s.max()) * 1.1:
        sig = -sig
    return sig


def make_pd(sig):
    """建 PLAY_DATA; 回傳 (pd, buf) — buf 需保持存活, 勿被 GC。"""
    buf = (c_double * len(sig))(*sig)
    pd = PLAY_DATA()
    pd.SampleRate = FS_REC
    pd.Size = len(sig)
    pd.Data = cast(buf, c_void_p)
    return pd, buf


# ── 連線 ──
dev = Device(HECG_DLL)
h = dev.connect(-1, 5000)
if not h:
    print("[FAIL] 沒連上 HECG100 (若剛 kill 過, 拔插 USB 再跑一次)")
    sys.exit(1)
print(f"[HECG100] connected. Normal={NORMAL_REC}  AF={AF_REC}  每段 {PHASE_SEC}s", flush=True)

# 兩段訊號各建一次 buffer, 整個程式生命週期保持存活
norm_pd, _norm_buf = make_pd(load(NORMAL_REC))
af_pd,   _af_buf   = make_pd(load(AF_REC))
dev.enable_play_data_loop(h, True)

# ── 終止旗標 (Ctrl+C / SIGTERM) ──
_stop = False
def _handler(signum, frame):
    global _stop
    _stop = True
signal.signal(signal.SIGINT, _handler)
try:
    signal.signal(signal.SIGTERM, _handler)
except Exception:
    pass


def play(pd, label):
    dev.stop_output(h)
    ok = dev.start_play_data(h, pointer(pd))
    print(f"[播放] {label}  (start={ok})  {PHASE_SEC}s ...", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < PHASE_SEC and not _stop:
        time.sleep(0.5)


try:
    while not _stop:
        play(norm_pd, f"Normal {NORMAL_REC}")
        if _stop:
            break
        play(af_pd, f"AF {AF_REC}")
finally:
    print("[HECG100] 停止輸出 + 釋放...", flush=True)
    for _ in range(3):
        try:
            dev.stop_output(h)
        except Exception:
            pass
        time.sleep(0.4)
    try:
        dev.free(h)
    except Exception:
        pass
    print("[HECG100] 已乾淨釋放, 可安全結束。", flush=True)
