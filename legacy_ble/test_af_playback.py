"""
受控實驗:把指定的 PhysioNet record 透過 HECG100 播放(start_play_data) → TriAnswer 收 → 存窗。
用來隔離「HECG100/電極/TriAnswer 類比鏈路是否造成訊號失真」。

用法: python test_af_playback.py [record] [num_windows] [out_dir]
預設: A01210, 3 視窗, captures/ecg_play/
之後: python ECG_classification/inference.py <out_dir>
"""
import os, sys, time, asyncio, shutil
from collections import deque
import numpy as np
from scipy import io as sio

# ── HECG100 ──
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo 根目錄
HECG_DIR = os.path.join(_ROOT, "HECG_Lab")
HECG_DLL = os.path.join(HECG_DIR, "HECG100x64.dll")
DATA_DIR = os.path.join(_ROOT, "ECG_classification", "data", "training2017")
sys.path.insert(0, HECG_DIR)
from hecg100 import Device, PLAY_DATA  # noqa: E402
from ctypes import c_double, pointer, cast, c_void_p  # noqa: E402

# ── TriAnswer ──
from bleak import BleakClient  # noqa: E402
# 裝置設定: 環境變數 TRI=old 用舊款 TriBLE 韌體, 預設新款 GROUP5 韌體。位址請改成自己的裝置
TRI_DEVICES = {
    # 舊款韌體: int16 LE 3 通道交錯, 取 CH0
    "old": dict(addr="F8:5B:B6:4E:26:67", char="0000a001-0000-1000-8000-00805f9b34fb", fmt="int16x3"),
    # 新款 GROUP5 韌體: uint8 單通道 ~502Hz, 約 32.6 counts/mV
    "new": dict(addr="DC:1D:69:3B:9F:26", char="0000aa05-0000-1000-8000-00805f9b34fb", fmt="uint8"),
}
TRI   = TRI_DEVICES[os.environ.get("TRI", "new")]
ADDR  = TRI["addr"]
CHAR  = TRI["char"]
FS    = 500
WIN   = 30 * FS
STRIDE = 10 * FS
FS_REC = 300.0          # PhysioNet 原始取樣率
GAIN   = 1000.0         # .hea: 1000/mV

RECORD  = sys.argv[1] if len(sys.argv) > 1 else "A01210"
NWIN    = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_ROOT, "captures", "ecg_play")
MAX_S   = 30 + (NWIN-1)*10 + 25

if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 載入 record (mV)，自動翻正極性 (同 play_dataset) ──
val = sio.loadmat(os.path.join(DATA_DIR, RECORD + ".mat"))["val"].flatten().astype(np.float64)
sig = val / GAIN
s = sig - np.median(sig)
flipped = abs(s.min()) > abs(s.max()) * 1.1
if flipped: sig = -sig
print(f"[record] {RECORD}  len={len(sig)} ({len(sig)/FS_REC:.1f}s @300Hz)  "
      f"peak={sig.max():.2f}/{sig.min():.2f} mV{'  [FLIPPED]' if flipped else ''}")

# ── capture state ──
buf = deque(maxlen=WIN); total = 0; last = 0; saved = 0; t0 = None
done = asyncio.Event()
def on_ecg(_s, data):
    global total, last, saved, t0
    if t0 is None: t0 = time.monotonic()
    if TRI["fmt"] == "uint8":
        sm = list(data)                      # 新機: 每 byte 一個樣本
    else:
        sm = [int.from_bytes(data[i:i+2], "little", signed=True) for i in range(0, len(data)-1, 2)][0::3]
    for x in sm:
        buf.append(x); total += 1
    if len(buf) >= WIN and total - last >= STRIDE:
        et = total // FS; st = et - WIN // FS
        fn = os.path.join(OUT_DIR, f"window_{st}s_{et}s.npy")
        np.save(fn, np.array(buf, dtype=np.int32))
        last = total; saved += 1
        print(f"  [{saved}/{NWIN}] {os.path.basename(fn)}  (CH0 ~{total/(time.monotonic()-t0):.0f} Hz)", flush=True)
        if saved >= NWIN: done.set()

async def main():
    dev = Device(HECG_DLL); h = dev.connect(-1, 5000)
    if not h: print("[FAIL] 沒連到 HECG100"); sys.exit(1)
    cbuf = (c_double * len(sig))(*sig)        # 保留參考，勿被 GC
    pd = PLAY_DATA(); pd.SampleRate = FS_REC; pd.Size = len(sig); pd.Data = cast(cbuf, c_void_p)
    dev.enable_play_data_loop(h, True)
    ok = dev.start_play_data(h, pointer(pd))
    print(f"[HECG100] 播放 {RECORD} 循環中 (start={ok})", flush=True)
    time.sleep(2)
    try:
        async with BleakClient(ADDR, timeout=20) as cli:
            print(f"[TriAnswer] {ADDR} ({TRI['fmt']}) connected={cli.is_connected}, 擷取 {NWIN} 視窗...", flush=True)
            await cli.start_notify(CHAR, on_ecg)
            try: await asyncio.wait_for(done.wait(), timeout=MAX_S)
            except asyncio.TimeoutError: print(f"  [timeout] 僅 {saved} 視窗", flush=True)
            await cli.stop_notify(CHAR)
    finally:
        print("[HECG100] 停止輸出...", flush=True)
        for _ in range(3):
            try: dev.stop_output(h)
            except Exception: pass
            time.sleep(0.4)
        time.sleep(0.6)
        try: dev.free(h)
        except Exception: pass
    print(f"\nDone. {saved} 視窗 → {OUT_DIR}")
    if saved == 0: print("[FAIL] 沒收到資料"); sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
