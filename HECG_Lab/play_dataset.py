"""
從 CinC2017 資料集挑一筆指定 label 的 record，餵給 HECG100 循環播放。
用法: from play_dataset import play_label
      play_label('N')        # Normal
      play_label('A')        # AF
      play_label('A', specific_record='A00004')   # 指定特定 record
"""
import os, sys, time, signal, atexit, csv, random
from ctypes import c_double, pointer, cast, c_void_p

import numpy as np
import wfdb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, 'ECG_classification', 'data', 'training2017')
REF_CSV  = os.path.join(_ROOT, 'ECG_classification', 'data', 'REFERENCE.csv')
DLL      = os.path.join(_HERE, 'HECG100x64.dll')   # 由 WhaleTeq SDK 複製而來, 見 HECG_Lab/README.md

sys.path.insert(0, _HERE)
from hecg100 import Device, PLAY_DATA  # noqa: E402

LABEL_NAMES = {'N': 'Normal', 'A': 'AF', 'O': 'Other', '~': 'Noisy'}


def load_records_by_label():
    by_label = {}
    with open(REF_CSV) as f:
        for row in csv.reader(f):
            if not row: continue
            by_label.setdefault(row[1], []).append(row[0])
    return by_label


def normalize_polarity(sig):
    """CinC2017 是 AliveCor 家用裝置錄的, 使用者方向反向時整段訊號會倒過來。
    用「去基線後正/負極值的絕對值」比較: 負方向更大 → 視為倒置, 反相。
    回傳 (sig_out, flipped: bool)
    """
    s = sig - np.median(sig)
    if abs(s.min()) > abs(s.max()) * 1.1:   # 加 1.1 緩衝避免邊界情況亂翻
        return -sig, True
    return sig, False


def load_signal(record_name, auto_flip=True):
    rec = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
    sig = rec.p_signal[:, 0].astype(np.float64)   # 已經是 mV
    flipped = False
    if auto_flip:
        sig, flipped = normalize_polarity(sig)
    return sig, rec.fs, flipped


def guard_single_instance():
    """拒絕在已有其他 HECG100 播放程序存活時啟動。
    兩個程序同時連 HECG100 時, 裝置會卡成「指令回 True 但類比不輸出」, 只能斷電重來。"""
    import subprocess
    pats = ('run_af', 'run_normal', 'run_ecg', 'run_hecg_loop', 'play_dataset', 'test_af_playback')
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, timeout=15).stdout
        import json
        procs = json.loads(out) if out.strip() else []
        if isinstance(procs, dict): procs = [procs]
    except Exception:
        return
    me = {os.getpid(), os.getppid()}   # venv 的 python.exe 是啟動器, 會再開真正的直譯器 → 父子都算自己
    others = [p for p in procs if p.get('ProcessId') not in me
              and p.get('CommandLine') and any(k in p['CommandLine'] for k in pats)]
    if others:
        print('[FAIL] 已有其他 HECG100 播放程序在跑, 拒絕啟動 (避免雙程序搶裝置把 HECG100 卡住):')
        for p in others: print(f"       PID {p['ProcessId']}: {p['CommandLine'][:100]}")
        print('       先結束它們 (或 taskkill /PID <pid> /F), 必要時 HECG100 斷電重插。')
        sys.exit(2)

def play_label(label, specific_record=None, auto_flip=True):
    guard_single_instance()
    by_label = load_records_by_label()
    if label not in by_label:
        print(f'[FAIL] label {label!r} 不在 {list(by_label.keys())}')
        sys.exit(1)

    name = specific_record or random.choice(by_label[label])
    sig, fs, flipped = load_signal(name, auto_flip=auto_flip)
    dur = len(sig) / fs
    flip_note = ' [POLARITY FLIPPED]' if flipped else ''
    print(f'[已選] {name}  類別={LABEL_NAMES.get(label, label)}  '
          f'fs={fs} Hz, 長度={dur:.1f}s, peak={sig.max():.2f}/{sig.min():.2f} mV{flip_note}')

    device = Device(DLL)
    handle = device.connect(-1, 5000)
    if not handle:
        print('[FAIL] 沒連到 HECG100')
        sys.exit(1)

    buf = (c_double * len(sig))(*sig)        # 必須保留參考, 不能被 GC
    pd = PLAY_DATA()
    pd.SampleRate = float(fs)
    pd.Size       = len(sig)
    pd.Data       = cast(buf, c_void_p)

    _cleaned = {'done': False}
    def cleanup(*_):
        if _cleaned['done']: return
        _cleaned['done'] = True
        print('\n停止中...', flush=True)
        for _ in range(3):
            try: device.stop_output(handle)
            except Exception: pass
            time.sleep(0.4)
        time.sleep(0.8)
        try: device.free(handle)
        except Exception: pass
        print('[OK] 已停止 (若仍輸出, 請手動關 HECG100 電源)', flush=True)

    atexit.register(cleanup)
    # 攔 SIGINT(Ctrl+C)/ SIGBREAK / SIGTERM(kill/TaskStop)→ sys.exit → atexit cleanup
    # → 中斷時都會乾淨釋放 (stop x3 + free), 不會鎖住 HECG100 handle。
    for _signame in ('SIGINT', 'SIGBREAK', 'SIGTERM'):
        try:
            signal.signal(getattr(signal, _signame), lambda *_: sys.exit(0))
        except (AttributeError, ValueError):
            pass

    device.enable_play_data_loop(handle, True)
    ok = device.start_play_data(handle, pointer(pd))
    print(f'[輸出中] start_play_data={ok}  循環播放中... 按 Ctrl+C 停止')

    while True:
        time.sleep(1)
