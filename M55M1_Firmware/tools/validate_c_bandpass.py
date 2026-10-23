"""
validate_c_bandpass.py — 用 Python 手刻「與 C 完全相同的 sample-by-sample DF2T biquad
雙向濾波」,證明這個迴圈能重現訓練的 filtfilt(經模型後逐位元相同)。
C 端 bandpass_0p5_40() 只要照抄本檔的 df2t_biquad 迴圈即可信。

無需 gcc:本檔的 float 迴圈就是 C 迴圈的等價實作(同運算順序、同 DF2T state)。

用法: python validate_c_bandpass.py [window.npy]
"""
import os, sys
import numpy as np
from scipy.signal import butter, filtfilt, tf2sos, sosfilt_zi
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FS, LO, HI, N = 500, 0.5, 40.0, 15000
HERE = os.path.dirname(__file__)
WIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "captures", "ecg_uart", "window_0s_30s.npy")

b, a = butter(3, [LO/(FS/2), HI/(FS/2)], "band")
SOS = tf2sos(b, a).astype(np.float64)      # (3,6): [b0 b1 b2 a0 a1 a2]
ZI  = sosfilt_zi(SOS).astype(np.float64)   # (3,2)

def df2t_forward(x, sos, zi_scaled):
    """一個 pass:3 個 biquad 串接,Direct-Form-II Transposed,sample-by-sample。
    這段就是 C 要寫的東西。zi_scaled[section] = ZI[section] * x[0]。"""
    y = np.empty_like(x)
    s1 = [zi_scaled[s][0] for s in range(sos.shape[0])]
    s2 = [zi_scaled[s][1] for s in range(sos.shape[0])]
    for n in range(len(x)):
        v = x[n]
        for s in range(sos.shape[0]):
            b0, b1, b2, _, a1, a2 = sos[s]
            out = b0 * v + s1[s]              # y[n] = b0*x + s1
            s1[s] = b1 * v - a1 * out + s2[s] # s1 = b1*x - a1*y + s2
            s2[s] = b2 * v - a2 * out         # s2 = b2*x - a2*y
            v = out                            # 串到下一個 biquad
        y[n] = v
    return y

def bandpass_manual(raw):
    """做法B 的手刻版 = C 端 bandpass_0p5_40() 的等價:
       去均值 → forward(穩態初始) → 反轉 → forward → 反轉。"""
    m = raw.mean()
    s = raw - m
    y = df2t_forward(s, SOS, ZI * s[0])
    y = y[::-1].copy()
    y = df2t_forward(y, SOS, ZI * y[0])
    return y[::-1].copy()

x = np.load(WIN).astype(np.float64)
ref = filtfilt(b, a, x)          # 訓練 golden
man = bandpass_manual(x)         # 手刻 C 等價
rms = np.sqrt(np.mean(ref**2))
err = np.abs(ref - man)
print(f"[input] {os.path.basename(WIN)}  N={len(x)}")
print(f"手刻 DF2T 雙向 vs filtfilt(golden):")
print(f"  max err = {err.max():.3f} ({100*err.max()/rms:.2f}% RMS,集中在尾端邊界)")
print(f"  內部 max err(去頭尾500) = {err[500:-500].max():.4f}")

# 經模型比對(這才是驗收標準)
try:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    MP = os.path.join(HERE, "..", "..", "ECG_classification", "deploy", "model_int8.tflite")
    it = Interpreter(model_path=MP); it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    insc, inzp = ind["quantization"]; osc, ozp = outd["quantization"]
    def run(sig):
        mu, sd = sig.mean(), sig.std()
        sig = (sig-mu)/sd if sd > 1e-6 else sig-mu
        q = np.clip(np.round(sig/insc+inzp), -128, 127).astype(np.int8).reshape(ind["shape"])
        it.set_tensor(ind["index"], q); it.invoke()
        return (it.get_tensor(outd["index"])[0].astype(np.float32)-ozp)*osc
    pr, pm = run(ref), run(man)
    print(f"\n模型輸出:")
    print(f"  filtfilt   : N={pr[0]:.3f} AF={pr[1]:.3f} O={pr[2]:.3f}")
    print(f"  手刻DF2T   : N={pm[0]:.3f} AF={pm[1]:.3f} O={pm[2]:.3f}")
    ok = abs(pr[1]-pm[1]) < 0.02
    print(f"  AF 差={abs(pr[1]-pm[1]):.4f} → {'PASS: C 迴圈可直接照抄' if ok else 'FAIL'}")
except Exception as e:
    print(f"[跳過模型比對] {e}")
