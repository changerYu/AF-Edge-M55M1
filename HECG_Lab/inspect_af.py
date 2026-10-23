"""掃 CinC2017 所有 A 標籤 record, 算 RR variability, 列出 top 10 最不規律的.

需要 wfdb / scipy. 如果你的 venv 沒裝, 在另一個有 wfdb 的環境跑這支就好.
這支不接 HECG100, 純資料分析.

用法: python inspect_af.py
輸出: 印出 top 10 候選 record 名字, 拿去 venv 內跑 `python run_af.py XXXXX`
"""
import os, csv, sys
import numpy as np
import wfdb
from scipy.signal import find_peaks

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, 'ECG_classification', 'data', 'training2017')
REF_CSV  = os.path.join(_ROOT, 'ECG_classification', 'data', 'REFERENCE.csv')


def load_af_records():
    names = []
    with open(REF_CSV) as f:
        for row in csv.reader(f):
            if row and row[1] == 'A':
                names.append(row[0])
    return names


def rr_stats(name):
    """傳回 (n_beats, rr_cv, rr_rmssd_ms, hr_mean) 或 None (短/壞 record)"""
    try:
        rec = wfdb.rdrecord(os.path.join(DATA_DIR, name))
    except Exception:
        return None
    sig = rec.p_signal[:, 0].astype(np.float64)
    fs  = rec.fs

    # 簡易極性翻正 (跟 play_dataset 一致)
    s = sig - np.median(sig)
    if abs(s.min()) > abs(s.max()) * 1.1:
        sig = -sig
        s = -s

    # 簡單 R 波偵測: scipy find_peaks, height 取訊號 70 百分位, distance 至少 250 ms
    height = np.percentile(s, 90) * 0.5
    min_dist = int(fs * 0.25)
    peaks, _ = find_peaks(s, height=height, distance=min_dist)
    if len(peaks) < 5:
        return None

    rr_samples = np.diff(peaks)
    rr_ms = rr_samples / fs * 1000.0
    hr = 60000.0 / rr_ms.mean()
    cv = rr_ms.std(ddof=1) / rr_ms.mean()
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    return len(peaks), cv, rmssd, hr


def main():
    af = load_af_records()
    print(f'共 {len(af)} 筆 A 標籤 record, 開始掃...', flush=True)

    rows = []
    for i, name in enumerate(af):
        st = rr_stats(name)
        if st is None: continue
        n, cv, rmssd, hr = st
        rows.append((name, n, cv, rmssd, hr))
        if (i + 1) % 100 == 0:
            print(f'  已處理 {i + 1}/{len(af)}', flush=True)

    # 用 CV 排, 主要排序; RMSSD 次要
    rows.sort(key=lambda r: (r[2], r[3]), reverse=True)

    print('\n=== Top 10 最不規律的 AF record ===')
    print(f'{"name":<10}{"n_beats":>10}{"RR_CV":>10}{"RMSSD(ms)":>12}{"HR_mean":>10}')
    for name, n, cv, rmssd, hr in rows[:10]:
        print(f'{name:<10}{n:>10}{cv:>10.3f}{rmssd:>12.1f}{hr:>10.1f}')

    print('\n=== 中位數 (參考用) ===')
    mid = rows[len(rows) // 2]
    print(f'{mid[0]:<10}{mid[1]:>10}{mid[2]:>10.3f}{mid[3]:>12.1f}{mid[4]:>10.1f}')

    print('\n=== 拿去測試 ===')
    print('在 venv 終端跑:')
    for r in rows[:5]:
        print(f'  python HECG_Lab/run_af.py {r[0]}')


if __name__ == '__main__':
    main()
