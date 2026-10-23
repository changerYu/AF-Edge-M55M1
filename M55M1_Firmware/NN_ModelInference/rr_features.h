/**
 * @file    rr_features.h
 * @brief   RR 間期特徵 (雙輸入模型的第二路輸入, 選用) — C 端, 與 ECG_classification/rr_features.py 逐行對應
 *
 * 輸入: 已 band-pass 0.5-40Hz + z-score 的 30 s @ 500 Hz 視窗 (float, 15000 點) = af_inference.h 的 bp_buf
 * 輸出: RR_NUM_FEATURES(16) 個未標準化特徵 (標準化 mean/std 由 af_model.h 提供)
 *
 * 演算法 (必須與 Python 參考實作完全一致, 改一邊就要改另一邊):
 *   1. 中央差分 d[i] = x[i+1]-x[i-1] (兩端 0), e = d^2
 *   2. 75 點 (150 ms) 置中移動平均 m (邊界一律除 75)
 *   3. 門檻 thr = THR_FRAC * median(6 段各自的 p99(m)); p99 用 256-bin 直方圖近似
 *   4. m > thr 的連續區段取第一個最大值為能量峰, 在 ±50 ms 內對齊 |x| 最大值為 R 位置; 兩峰 < 200 ms 取 m 較大者
 *   5. RR = 峰距 (整數樣本), 只留 120-1200 樣本 (0.24-2.4 s); pNN50 / 熵分箱用整數樣本比較
 *   6. 16 個統計量 (順序見 rr_features.py FEATURE_NAMES)
 *
 * 驗證: 韌體每視窗印 "[RR] f0 f1 ... f15" (4 位小數), 與 PC 端
 *   python ECG_classification/rr_features.py <擷取資料夾> 的 [RR] 行逐欄比對 (允許末位差)。
 */
#ifndef RR_FEATURES_H
#define RR_FEATURES_H

#include <stdint.h>
#include <math.h>

#define RR_FS            500
#define RR_WIN           15000
#define RR_NUM_FEATURES  16
#define RR_MA_LEN        75
#define RR_MA_HALF       37
#define RR_REFRACTORY    100      // 200 ms
#define RR_THR_FRAC      0.30f
#define RR_NSEG          6
#define RR_REFINE        25       // ±50 ms
#define RR_HIST_BINS     256
#define RR_MIN_SAMP      120      // 0.24 s
#define RR_MAX_SAMP      1200     // 2.40 s
#define RR_PNN50_SAMP    25       // 50 ms
#define RR_ENT_LO_SAMP   150      // 0.30 s
#define RR_ENT_BIN_SAMP  50       // 0.10 s
#define RR_ENT_NBINS     16
#define RR_MAX_PEAKS     160      // 不應期 200 ms → 30 s 最多 150 峰

// 能量包絡緩衝 (60 KB float) — 與 bp_buf 一樣放大 SRAM 區, 否則擠爆 DTCM .bss
static float rr_m[RR_WIN] __attribute__((section(".bss.sram.data")));

// m[lo:hi] 的 p99 (256-bin 直方圖近似, bin 尺度用整窗最大值 mx)
static float rr_p99_hist(const float* m, int lo, int hi, float mx)
{
    static int counts[RR_HIST_BINS];
    for (int k = 0; k < RR_HIST_BINS; k++) counts[k] = 0;
    for (int i = lo; i < hi; i++) {
        int b = (int)(m[i] / mx * (float)RR_HIST_BINS);
        if (b > RR_HIST_BINS - 1) b = RR_HIST_BINS - 1;
        if (b < 0) b = 0;
        counts[b]++;
    }
    int n = hi - lo;
    int target = (99 * n + 99) / 100;           // ceil(0.99 n), 與 Python 相同的整數式
    int cum = 0, b = RR_HIST_BINS - 1;
    for (int k = 0; k < RR_HIST_BINS; k++) {
        cum += counts[k];
        if (cum >= target) { b = k; break; }
    }
    return (float)(b + 1) / (float)RR_HIST_BINS * mx;
}

// 回傳峰數; peaks[] = R 位置 (樣本索引), amps[] = 對應能量峰 m 值
static int rr_detect(const float* x, int n, int* peaks, float* amps)
{
    // 1+2. 能量包絡: m[i] = (1/75) * sum_{j=i-37}^{i+37} d(j)^2, d(j) = x[j+1]-x[j-1] (j 在 [1,n-2] 之外為 0)
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        int lo = i - RR_MA_HALF, hi = i + RR_MA_HALF;
        if (lo < 1) lo = 1;
        if (hi > n - 2) hi = n - 2;
        for (int j = lo; j <= hi; j++) {
            float d = x[j + 1] - x[j - 1];
            acc += d * d;
        }
        rr_m[i] = acc / (float)RR_MA_LEN;
    }
    float mx = rr_m[0];
    for (int i = 1; i < n; i++) if (rr_m[i] > mx) mx = rr_m[i];
    if (mx <= 0.0f) return 0;

    // 3. 門檻: 6 段 p99 的中位數
    float p99s[RR_NSEG];
    int seg_len = n / RR_NSEG;
    for (int k = 0; k < RR_NSEG; k++) {
        int lo = k * seg_len, hi = (k < RR_NSEG - 1) ? (k + 1) * seg_len : n;
        p99s[k] = rr_p99_hist(rr_m, lo, hi, mx);
    }
    for (int i = 1; i < RR_NSEG; i++) {            // 插入排序
        float v = p99s[i]; int j = i - 1;
        while (j >= 0 && p99s[j] > v) { p99s[j + 1] = p99s[j]; j--; }
        p99s[j + 1] = v;
    }
    float med = 0.5f * (p99s[RR_NSEG / 2 - 1] + p99s[RR_NSEG / 2]);   // NSEG 為偶數
    float thr = RR_THR_FRAC * med;

    // 4. 區段掃描
    int np = 0, in_seg = 0, pm = 0; float mv = 0.0f;
    for (int i = 0; i <= n; i++) {
        int above = (i < n) && (rr_m[i] > thr);
        if (above) {
            if (!in_seg) { in_seg = 1; pm = i; mv = rr_m[i]; }
            else if (rr_m[i] > mv) { pm = i; mv = rr_m[i]; }
        } else if (in_seg) {
            in_seg = 0;
            int lo = pm - RR_REFINE, hi = pm + RR_REFINE + 1;
            if (lo < 0) lo = 0;
            if (hi > n) hi = n;
            int pr = lo; float best = fabsf(x[lo]);
            for (int j = lo + 1; j < hi; j++) { float a = fabsf(x[j]); if (a > best) { best = a; pr = j; } }
            if (np > 0 && pr - peaks[np - 1] < RR_REFRACTORY) {
                if (mv > amps[np - 1]) { peaks[np - 1] = pr; amps[np - 1] = mv; }
            } else if (np == 0 || pr != peaks[np - 1]) {
                if (np < RR_MAX_PEAKS) { peaks[np] = pr; amps[np] = mv; np++; }
            }
        }
    }
    return np;
}

// 由峰位置算 16 維特徵 (未標準化)
static void rr_compute_features(const int* peaks, const float* amps, int npk, float* f)
{
    for (int k = 0; k < RR_NUM_FEATURES; k++) f[k] = 0.0f;
    f[0] = (float)npk;
    if (npk >= 2) {                                   // f15: 能量峰振幅 CV
        float am = 0.0f; for (int k = 0; k < npk; k++) am += amps[k]; am /= (float)npk;
        float asd = 0.0f; for (int k = 0; k < npk; k++) { float t = amps[k] - am; asd += t * t; }
        asd = sqrtf(asd / (float)npk);
        f[15] = (am > 1e-12f) ? asd / am : 0.0f;
    }
    static float rr[RR_MAX_PEAKS], srt[RR_MAX_PEAKS], drr[RR_MAX_PEAKS];
    static int   rr_s[RR_MAX_PEAKS];
    int n = 0;
    for (int k = 0; k + 1 < npk; k++) {                // RR 以整數樣本為主 (門檻/分箱用整數比較, 與 Python 一致)
        int v = peaks[k + 1] - peaks[k];
        if (v >= RR_MIN_SAMP && v <= RR_MAX_SAMP) { rr_s[n] = v; rr[n] = (float)v / (float)RR_FS; n++; }
    }
    if (n < 3) return;
    float mean = 0.0f; for (int k = 0; k < n; k++) mean += rr[k]; mean /= (float)n;
    float var = 0.0f;  for (int k = 0; k < n; k++) { float t = rr[k] - mean; var += t * t; }
    float sd = sqrtf(var / (float)n);
    int nd = n - 1;
    float rms = 0.0f, mad = 0.0f; int c50 = 0, c20 = 0;
    for (int k = 0; k < nd; k++) {
        drr[k] = rr[k + 1] - rr[k];
        float a = fabsf(drr[k]);
        rms += drr[k] * drr[k]; mad += a;
        { int ds = rr_s[k + 1] - rr_s[k]; if (ds < 0) ds = -ds; if (ds > RR_PNN50_SAMP) c50++; }   // pNN50 整數比較
        if (a > 0.2f * mean) c20++;
    }
    f[1] = mean;
    f[2] = sd;
    f[3] = sd / mean;
    f[4] = sqrtf(rms / (float)nd);                    // RMSSD
    f[5] = (float)c50 / (float)nd;                    // pNN50
    f[6] = (float)c20 / (float)nd;                    // pRR20%
    for (int k = 0; k < n; k++) srt[k] = rr[k];       // 排序 (插入排序, n ≤ 150)
    for (int i = 1; i < n; i++) { float v = srt[i]; int j = i - 1; while (j >= 0 && srt[j] > v) { srt[j + 1] = srt[j]; j--; } srt[j + 1] = v; }
    f[7] = (n % 2 == 1) ? srt[n / 2] : 0.5f * (srt[n / 2 - 1] + srt[n / 2]);
    f[8] = srt[(int)(0.75f * (float)(n - 1))] - srt[(int)(0.25f * (float)(n - 1))];
    f[9] = srt[0];
    f[10] = srt[n - 1];
    f[11] = (srt[n - 1] - srt[0]) / mean;
    int hist[RR_ENT_NBINS]; for (int k = 0; k < RR_ENT_NBINS; k++) hist[k] = 0;
    for (int k = 0; k < n; k++) {
        int b = (rr_s[k] - RR_ENT_LO_SAMP) / RR_ENT_BIN_SAMP;   // 整數分箱 (負值 → 夾到 0)
        if (rr_s[k] < RR_ENT_LO_SAMP) b = 0;
        if (b < 0) b = 0;
        if (b > RR_ENT_NBINS - 1) b = RR_ENT_NBINS - 1;
        hist[b]++;
    }
    float ent = 0.0f;
    for (int k = 0; k < RR_ENT_NBINS; k++) if (hist[k] > 0) { float p = (float)hist[k] / (float)n; ent -= p * logf(p); }
    f[12] = ent;
    f[13] = (mad / (float)nd) / mean;
    if (nd >= 2) {
        int tp = 0; for (int k = 0; k + 1 < nd; k++) if (drr[k] * drr[k + 1] < 0.0f) tp++;
        f[14] = (float)tp / (float)(nd - 1);
    }
}

// 一站式: z-scored 視窗 → 16 特徵。回傳峰數。
static int rr_features_from_window(const float* xz, int n, float* f)
{
    static int   peaks[RR_MAX_PEAKS];
    static float amps[RR_MAX_PEAKS];
    int np = rr_detect(xz, n, peaks, amps);
    rr_compute_features(peaks, amps, np, f);
    return np;
}

#endif /* RR_FEATURES_H */
