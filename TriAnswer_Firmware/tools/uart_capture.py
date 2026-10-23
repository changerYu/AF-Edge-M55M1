"""
從 TriAnswer 新韌體 (TriAnswer_UART_to_M55M1) 收 ECG 幀，驗證取樣率並存成 30 s 視窗 .npy。

幀格式 (= M55M1 uart_receiver.h): [0xAA][0xBB][LEN_L][LEN_H][int32 LE × N]

用法:
  python uart_capture.py COMx [N視窗] [out_dir]
    COMx    = TriAnswer 的 USB CDC 埠 (MIRROR_TO_USB=1)，或接在 Serial1 上的 USB-TTL 埠
    N視窗   = 要收幾個 30 s 視窗 (預設 3; stride 10 s)
    out_dir = 輸出資料夾 (預設 <repo>/captures/ecg_uart)
之後: python ECG_classification/inference.py <out_dir>

  python uart_capture.py --selftest     # 不接硬體，用合成幀驗證解析器
"""
import os, sys, time, shutil
from collections import deque
import numpy as np

FS           = 500
WINDOW       = 30 * FS
STRIDE       = 10 * FS
H0, H1       = 0xAA, 0xBB


class FrameParser:
    """逐 byte 狀態機，與 M55M1 的 UART_Receiver_ProcessByte 同邏輯。"""
    def __init__(self):
        self.state = 0; self.len = 0; self.buf = bytearray(); self.frames = 0; self.bad = 0

    def feed(self, data: bytes):
        out = []
        for b in data:
            if self.state == 0:
                self.state = 1 if b == H0 else 0
            elif self.state == 1:
                if b == H1: self.state = 2
                else: self.state = 1 if b == H0 else 0
            elif self.state == 2:
                self.len = b; self.state = 3
            elif self.state == 3:
                self.len |= b << 8; self.buf = bytearray(); self.state = 4
                if self.len == 0 or self.len % 4 or self.len > 4096:
                    self.bad += 1; self.state = 0
            else:
                self.buf.append(b)
                if len(self.buf) >= self.len:
                    out.extend(np.frombuffer(bytes(self.buf), dtype='<i4').tolist())
                    self.frames += 1; self.state = 0
        return out


def selftest():
    import struct, random
    p = FrameParser()
    samples = list(range(-5, 1000))
    stream = bytearray(b'# garbage text line\n')
    for i in range(0, len(samples), 100):
        chunk = samples[i:i+100]
        stream += bytes([H0, H1]) + struct.pack('<H', len(chunk)*4) + struct.pack(f'<{len(chunk)}i', *chunk)
        stream += b'\n# frames=1 samples=100\n'   # 模擬韌體的狀態行夾在幀之間
    got = []
    for k in range(0, len(stream), 7):           # 切成不規則小塊餵
        got += p.feed(bytes(stream[k:k+7]))
    assert got == samples, (len(got), got[:5])
    print(f'[selftest OK] {p.frames} frames, {len(got)} samples, bad={p.bad}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        selftest(); return
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    import serial
    port  = sys.argv[1]
    nwin  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    here  = os.path.dirname(os.path.abspath(__file__))
    out   = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, '..', '..', 'captures', 'ecg_uart')
    out   = os.path.abspath(out)
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)

    ser = serial.Serial(port, 921600, timeout=0.2)
    ser.dtr = True
    time.sleep(0.3); ser.reset_input_buffer()   # 丟掉開埠前 OS 緩衝裡的殘渣, 避免第一幀錯位
    print(f'[open] {port}  收 {nwin} 個 30s 視窗 → {out}')
    parser = FrameParser()
    buf = deque(maxlen=WINDOW)
    total = 0; last_win = 0; saved = 0
    t0 = None; t_rep = time.time()
    try:
        while saved < nwin:
            data = ser.read(4096)
            if not data: continue
            s = parser.feed(data)
            if s and t0 is None: t0 = time.time()
            buf.extend(s); total += len(s)
            if time.time() - t_rep >= 2 and t0:
                el = time.time() - t0
                print(f'  samples={total}  rate≈{total/el:.1f} Hz  frames={parser.frames}  bad={parser.bad}  '
                      f'last={s[-1] if s else "-"}', flush=True)
                t_rep = time.time()
            if len(buf) >= WINDOW and total - last_win >= STRIDE:
                st = (total - WINDOW) // FS; et = total // FS
                fn = os.path.join(out, f'window_{st}s_{et}s.npy')
                np.save(fn, np.array(buf, dtype=np.int32))
                saved += 1; last_win = total
                print(f'[save] {os.path.basename(fn)}  ({saved}/{nwin})', flush=True)
    finally:
        ser.close()
    el = time.time() - t0 if t0 else 0
    print(f'\nDone. {saved} 視窗, {total} samples, {el:.1f}s → 平均 {total/el if el else 0:.1f} Hz, bad frames={parser.bad}')
    print(f'下一步: python ECG_classification/inference.py "{out}"')


if __name__ == '__main__':
    main()
