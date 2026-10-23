"""
判定 HECG100 → TriAnswer 擷取訊號的極性 (RA/LL 有沒有接反)。

原理: 把播放的 CinC2017 record 升頻到 500Hz、與擷取視窗做正規化互相關。
⚠️ HECG_Lab/play_dataset.py 播放前會對「負極值 > 正極值×1.1」的 record 自動反相 (normalize_polarity),
   本工具套用同一條規則, 比對的是「實際播出去的波形」。
  corr 為 + → 極性與來源相同 (正確);  corr 為 − → 反相, RA/LL 要對調。
  lag 應每段等速前進 (每 10s 段 +10s), 否則表示取樣率不對或沒對齊。

用法: python polarity_check.py <record> [capture_dir]
  例:  python polarity_check.py A01210 captures/ecg_uart
注意: 視窗開頭幾秒的接電極假影已用 ±5σ 裁掉; record 尾端 (循環接縫) 偶有一段例外屬正常。
"""
import glob, os, sys
import numpy as np
from scipy import io as sio, signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

def bp(x, fs):
    b, a = signal.butter(3, [0.5, 40], btype='band', fs=fs)
    return signal.filtfilt(b, a, x - x.mean())

def main():
    if len(sys.argv) < 2: print(__doc__); sys.exit(1)
    rec = sys.argv[1]
    cap_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'captures', 'ecg_uart')
    src = sio.loadmat(os.path.join(ROOT, 'ECG_classification', 'data', 'training2017', rec + '.mat'))['val'].flatten().astype(float)
    # 與 play_dataset.normalize_polarity 完全相同的規則 (去中位數後比正負極值)
    d = src - np.median(src)
    flipped = abs(d.min()) > abs(d.max()) * 1.1
    if flipped: src = -src
    print(f'[{rec}] play_dataset 播放時{"有" if flipped else "沒有"}自動反相' + (' → 以反相後的波形比對' if flipped else ''))
    s = bp(signal.resample_poly(src, 5, 3), 500); s /= s.std(); tmpl = np.concatenate([s, s, s])
    rec_len = len(s) / 500
    signs = []
    for f in sorted(glob.glob(os.path.join(cap_dir, '*.npy'))):
        raw = np.load(f).astype(float); x = bp(raw, 500)
        fr, P = signal.welch(raw - raw.mean(), fs=500, nperseg=4096)
        hum = P[(fr >= 58) & (fr <= 62)].sum() / P[(fr >= 0.5) & (fr <= 40)].sum()
        print(f'{os.path.basename(f)}  p-p={raw.max()-raw.min():.0f}  60Hz/ECG={hum:.3f}')
        for i in range(len(x) // 5000):
            seg = x[i*5000:(i+1)*5000].copy(); seg /= seg.std(); seg = np.clip(seg, -5, 5); seg /= seg.std()
            c = signal.correlate(tmpl, seg, mode='valid') / len(seg); k = np.argmax(np.abs(c))
            print(f'  seg {i*10:2d}-{i*10+10:2d}s  corr={c[k]:+.3f}  lag={(k/500) % rec_len:5.1f}s')
            if abs(c[k]) >= 0.4: signs.append(np.sign(c[k]))
    if not signs: print('\n[??] 相關太弱, 無法判定 (訊號有進來嗎? record 對嗎?)'); return
    pos = sum(1 for v in signs if v > 0); neg = len(signs) - pos
    print(f'\n正相關段 {pos} / 負相關段 {neg}')
    if pos > neg: print('[OK] 極性與來源相同, 接法正確。')
    else:         print('[INVERTED] 反相, 請把 HECG100 的 RA / LL 對調。')

if __name__ == '__main__':
    main()
