"""
Bounded TriAnswer ECG capture (no GUI) — collects a fixed number of 30s windows
then exits cleanly, so it can be piped straight into inference.py.

Based on ble_receive.py (same parsing: int16 LE, CH0 = samples[0::3]),
but headless + auto-stop instead of an infinite plotting loop.

Usage:
    python capture_live.py [num_windows] [out_dir]
Defaults: 3 windows -> ./ecg_live/
"""
import asyncio
import os
import sys
import shutil
import time
from collections import deque
import numpy as np
from bleak import BleakClient

# ===== Config (matches ble_receive.py) =====
# 裝置設定: 環境變數 TRI=old 用舊款 TriBLE 韌體, 預設新款 GROUP5 韌體。位址請改成 ble_scan.py 掃到的裝置
TRI_DEVICES = {
    "old": dict(addr="F8:5B:B6:4E:26:67", char="0000a001-0000-1000-8000-00805f9b34fb", fmt="int16x3"),  # TriBLE 2503-29
    "new": dict(addr="DC:1D:69:3B:9F:26", char="0000aa05-0000-1000-8000-00805f9b34fb", fmt="uint8"),    # GROUP5 FW 0.3.11
}
TRI            = TRI_DEVICES[os.environ.get("TRI", "new")]
DEVICE_ADDRESS = TRI["addr"]
ECG_CHAR_UUID  = TRI["char"]
SAMPLING_RATE  = 500
WINDOW_SIZE    = 30 * SAMPLING_RATE            # 15000 samples = 30s
STRIDE         = 10 * SAMPLING_RATE            # 5000 samples = 10s

NUM_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR     = sys.argv[2] if len(sys.argv) > 2 else "ecg_live"
# Safety cap: 30s fill + (NUM_WINDOWS-1)*10s stride, plus generous margin
MAX_SECONDS = 30 + (NUM_WINDOWS - 1) * 10 + 25

# ===== Prepare output dir (fresh each run) =====
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ===== State =====
ecg_buffer = deque(maxlen=WINDOW_SIZE)
total_samples = 0
last_window_sample = 0
saved = 0
t_first = None
done_evt = asyncio.Event()


def notification_handler(sender, data):
    global total_samples, last_window_sample, saved, t_first
    if t_first is None:
        t_first = time.monotonic()

    if TRI["fmt"] == "uint8":
        ch0 = list(data)                       # 新機: 每 byte 一個樣本, 單通道
    else:
        # 舊機: 2 bytes -> int16 LE; 3 channels interleaved, take CH0
        samples = [int.from_bytes(data[i:i+2], "little", signed=True)
                   for i in range(0, len(data) - 1, 2)]
        ch0 = samples[0::3]

    for s in ch0:
        ecg_buffer.append(s)
        total_samples += 1

    if len(ecg_buffer) >= WINDOW_SIZE and total_samples - last_window_sample >= STRIDE:
        end_t = total_samples // SAMPLING_RATE
        start_t = end_t - WINDOW_SIZE // SAMPLING_RATE
        fname = os.path.join(OUT_DIR, f"window_{start_t}s_{end_t}s.npy")
        np.save(fname, np.array(ecg_buffer, dtype=np.int32))
        last_window_sample = total_samples
        saved += 1
        rate = total_samples / (time.monotonic() - t_first) if t_first else 0
        print(f"  [{saved}/{NUM_WINDOWS}] saved {os.path.basename(fname)}  "
              f"(CH0 rate ~{rate:.0f} Hz)")
        if saved >= NUM_WINDOWS:
            done_evt.set()


async def main():
    print(f"Connecting to {DEVICE_ADDRESS} ...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")
        print(f"Receiving ECG... need {NUM_WINDOWS} window(s); "
              f"first ready at ~30s, then every 10s.")
        await client.start_notify(ECG_CHAR_UUID, notification_handler)
        try:
            await asyncio.wait_for(done_evt.wait(), timeout=MAX_SECONDS)
        except asyncio.TimeoutError:
            print(f"  [timeout] stopped after {MAX_SECONDS}s with {saved} window(s).")
        await client.stop_notify(ECG_CHAR_UUID)

    elapsed = (time.monotonic() - t_first) if t_first else 0
    print(f"\nDone. {saved} window(s) -> {OUT_DIR}/")
    if saved and elapsed:
        print(f"Total CH0 samples: {total_samples} in {elapsed:.1f}s "
              f"(~{total_samples/elapsed:.0f} Hz; expected {SAMPLING_RATE} Hz)")
    if saved == 0:
        print("No windows captured — check the device is worn/streaming.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
