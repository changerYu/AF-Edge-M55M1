# -*- coding: utf-8 -*-
"""
RR 間期特徵 (雙輸入模型的第二路輸入, 選用) — Python 參考實作。

輸入: 已 band-pass 0.5-40Hz + z-score 的 30 s @ 500 Hz 視窗 (float32, 15000 點),
      即 inference.py 的 preprocess() 輸出 / 韌體 bp_buf 正規化後的內容。
輸出: 16 維特徵 (未標準化)。標準化 (mean/std 由訓練集算, 存在 af_model.h) 在 notebook / 韌體做。

演算法刻意寫成能逐行翻成 C 的形式 (韌體 rr_features.h 必須與此完全一致):
  1. 中央差分 d[i] = x[i+1] - x[i-1]; e = d^2
  2. 75 點 (150 ms) 置中移動平均 m (邊界一律除以 75)
  3. 門檻 thr = THR_FRAC * median(6 段各自的 p99(m)); p99 用 256-bin 直方圖近似 (不排序、抗單段假影)
  4. m > thr 的連續區段各取第一個最大值為能量峰, 再在 ±50 ms 內對齊 |x| 最大值為 R 位置; 兩峰 < 200 ms 取 m 較大者
  5. RR = 峰距 (整數樣本); 只保留 120-1200 樣本 (0.24-2.4 s); pNN50 / 熵分箱都用整數樣本比較
  6. 16 個統計量 (見 FEATURE_NAMES)

用法 (PC 端 golden, 與韌體 [RR] 輸出逐欄比對):
  python rr_features.py ../captures/ecg_uart             # 目錄內所有 .npy
  python rr_features.py path/to/window.npy
"""
import os, sys, glob
import numpy as np

FS = 500
WIN = 15000
MA_LEN = 75            # 150 ms 移動平均 (奇數, 置中)
MA_HALF = MA_LEN // 2  # 37
REFRACTORY = 100       # 200 ms
THR_FRAC = 0.30
NSEG = 6               # 門檻用 6 段 (5 s) p99 的中位數
REFINE = 25            # 能量峰 ±50 ms 內找 |x| 最大值當 R 位置
HIST_BINS = 256
PCT = 0.99             # 以整數 ceil(99n/100) 實作
RR_MIN_SAMP, RR_MAX_SAMP = 120, 1200   # 0.24-2.4 s (250..25 bpm), 以樣本數比較
PNN50_SAMP = 25                        # 50 ms
ENT_LO_SAMP, ENT_BIN_SAMP, ENT_NBINS = 150, 50, 16   # RR 直方圖熵: 0.3 s 起每 0.1 s 一格, 16 格
NUM_FEATURES = 16
FEATURE_NAMES = ['n_peaks', 'rr_mean', 'rr_std', 'rr_cv', 'rmssd', 'pnn50', 'prr20', 'rr_median',
                 'rr_iqr', 'rr_min', 'rr_max', 'rr_range_n', 'rr_entropy', 'mad_n', 'tpr', 'amp_cv']


def _p99_hist(m, lo, hi, mx):
    """m[lo:hi] 的 p99 (256-bin 直方圖近似, bin 以整窗最大值 mx 為尺度; C 同)。"""
    seg = m[lo:hi]
    bins = np.minimum((seg / np.float32(mx) * HIST_BINS).astype(np.int32), HIST_BINS - 1)
    counts = np.bincount(bins, minlength=HIST_BINS)
    target = (99 * (hi - lo) + 99) // 100          # ceil(0.99 n), 整數 (C 同, 避免浮點比較差異)
    cum = 0; b = HIST_BINS - 1
    for k in range(HIST_BINS):
        cum += counts[k]
        if cum >= target:
            b = k; break
    return (b + 1) / HIST_BINS * mx


def detect_r_peaks(xz):
    """xz: float32 (15000,) 已 band-pass+z-score。
    回傳 (peaks int array [對齊 |x| 最大值], amps float32 array [對應的 m 峰值], m)。"""
    x = np.asarray(xz, dtype=np.float32)
    n = len(x)
    d = np.zeros(n, np.float32)
    d[1:n - 1] = x[2:] - x[:-2]                      # 中央差分 (兩端 = 0)
    e = d * d
    m = (np.convolve(e, np.ones(MA_LEN, np.float32), mode='same') / np.float32(MA_LEN)).astype(np.float32)

    mx = float(m.max())
    if mx <= 0.0:
        return np.zeros(0, np.int32), np.zeros(0, np.float32), m
    # 門檻: 分 NSEG 段各算 p99, 取中位數 (單一段的大假影不會把門檻拉高) → thr = THR_FRAC * median_p99
    seg_len = n // NSEG
    p99s = sorted(_p99_hist(m, k * seg_len, (k + 1) * seg_len if k < NSEG - 1 else n, mx) for k in range(NSEG))
    med_p99 = p99s[NSEG // 2] if NSEG % 2 == 1 else 0.5 * (p99s[NSEG // 2 - 1] + p99s[NSEG // 2])
    thr = np.float32(THR_FRAC * med_p99)

    above = m > thr
    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1            # exclusive
    if above[0]: starts = np.concatenate([[0], starts])
    if above[-1]: ends = np.concatenate([ends, [n]])
    peaks, amps = [], []
    ax = np.abs(x)
    for s, t in zip(starts, ends):
        pm = s + int(np.argmax(m[s:t]))             # 能量峰 (第一個最大值; C: 嚴格 > 才更新)
        lo, hi = max(0, pm - REFINE), min(n, pm + REFINE + 1)
        pr = lo + int(np.argmax(ax[lo:hi]))         # 對齊 |x| 最大值 → R (或主要偏折) 位置
        if peaks and pr - peaks[-1] < REFRACTORY:   # 不應期: 取 m 較大者
            if m[pm] > amps[-1]: peaks[-1], amps[-1] = pr, m[pm]
        elif not peaks or pr != peaks[-1]:
            peaks.append(pr); amps.append(m[pm])
    return np.asarray(peaks, np.int32), np.asarray(amps, np.float32), m


def rr_features(xz):
    """回傳 (feat float32[16], peaks)。公式順序與韌體 rr_features.h 一一對應。"""
    peaks, amps, m = detect_r_peaks(xz)
    f = np.zeros(NUM_FEATURES, np.float64)
    npk = len(peaks)
    f[0] = npk
    # 峰值能量振幅 CV (全部峰)
    if npk >= 2:
        a = amps.astype(np.float64)
        am = a.mean(); asd = np.sqrt(((a - am) ** 2).mean())
        f[15] = asd / am if am > 1e-12 else 0.0
    # RR 序列 (只留合理範圍)
    # RR 以「整數樣本數」為主 (門檻/直方圖用整數比較, 避免 C float32 與 Python float64 在邊界不一致)
    rr_s = np.diff(peaks).astype(np.int64) if npk >= 2 else np.zeros(0, np.int64)
    rr_s = rr_s[(rr_s >= RR_MIN_SAMP) & (rr_s <= RR_MAX_SAMP)]
    n = len(rr_s)
    if n < 3:
        return f.astype(np.float32), peaks
    rr = rr_s.astype(np.float64) / FS
    mean = rr.mean()
    std = np.sqrt(((rr - mean) ** 2).mean())
    drr = np.diff(rr)                       # n-1 >= 2
    adrr = np.abs(drr)
    adrr_s = np.abs(np.diff(rr_s))          # 整數樣本差
    f[1] = mean
    f[2] = std
    f[3] = std / mean
    f[4] = np.sqrt((drr ** 2).mean())       # RMSSD
    f[5] = (adrr_s > PNN50_SAMP).mean()     # pNN50 (|ΔRR| > 25 樣本 = 50 ms, 整數比較)
    f[6] = (adrr > 0.2 * mean).mean()       # pRR20%
    s = np.sort(rr)
    f[7] = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    f[8] = s[int(0.75 * (n - 1))] - s[int(0.25 * (n - 1))]
    f[9] = s[0]
    f[10] = s[-1]
    f[11] = (s[-1] - s[0]) / mean
    hb = np.clip((rr_s - ENT_LO_SAMP) // ENT_BIN_SAMP, 0, ENT_NBINS - 1)   # 整數分箱: 0.3 s 起每 0.1 s 一格
    p = np.bincount(hb, minlength=ENT_NBINS) / n
    p = p[p > 0]
    f[12] = -(p * np.log(p)).sum()
    f[13] = adrr.mean() / mean
    if len(drr) >= 2:
        f[14] = (drr[:-1] * drr[1:] < 0).mean()   # turning point ratio
    return f.astype(np.float32), peaks


def format_features(f):
    return ' '.join(f'{v:.4f}' for v in f)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import preprocess
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'ecg_windows')
    files = sorted(glob.glob(os.path.join(target, '*.npy'))) if os.path.isdir(target) else [target]
    print('features:', ' '.join(FEATURE_NAMES))
    for fp in files:
        raw = np.load(fp)
        xz = preprocess(raw).astype(np.float32)
        f, peaks = rr_features(xz)
        hr = 60.0 / f[1] if f[1] > 0 else 0.0
        print(f'{os.path.basename(fp):22s} n_peaks={int(f[0]):3d} HR~{hr:5.1f} CV={f[3]:.3f} RMSSD={f[4]:.3f}s pNN50={f[5]:.2f}')
        print('   [RR]', format_features(f))
