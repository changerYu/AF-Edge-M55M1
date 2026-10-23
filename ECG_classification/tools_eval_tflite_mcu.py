# -*- coding: utf-8 -*-
"""Evaluate one or more INT8 tflite models (single-input or dual-input) on the shared
seed-42 record-level TEST split, using MCU-identical per-window preprocessing (band-pass 0.5-40 + z-score)
and, for dual-input models, RR features standardized with rr_norm.npz found next to the tflite.
Usage: python tools_eval_tflite_mcu.py path/to/model_int8.tflite [more.tflite ...]
(needs tensorflow + scikit-learn + pandas; dataset in data/, see data/README.md)"""
import os, sys, time, json, numpy as np, pandas as pd
from scipy import io as sio
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import tensorflow as tf
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
from rr_features import rr_features, NUM_FEATURES
DATA_DIR = os.path.join(PROJ, "data", "training2017"); REF_FILE = os.path.join(PROJ, "data", "REFERENCE.csv")
WIN, STRIDE, FS_ORIG = 15000, 5000, 300
LABEL_MAP = {'N': 0, 'A': 1, 'O': 2, '~': 2}
BP_B, BP_A = butter(3, [0.5 / 250, 40 / 250], 'band')

def mcu_window(raw500):
    x = filtfilt(BP_B, BP_A, raw500.astype(np.float64)); mu, sd = x.mean(), x.std()
    return ((x - mu) / (sd if sd > 1e-6 else 1.0)).astype(np.float32)
def record_windows(sig300):
    sig = resample_poly(sig300.astype(np.float64), 5, 3); short = len(sig) < WIN
    if short: sig = np.tile(sig, int(np.ceil(WIN / len(sig))))[:WIN]
    wins, s = [], 0
    while s + WIN <= len(sig): wins.append(mcu_window(sig[s:s + WIN])); s += STRIDE
    if (not short) and s < len(sig) and (len(sig) - WIN) % STRIDE != 0: wins.append(mcu_window(sig[-WIN:]))
    return wins, short

ref = pd.read_csv(REF_FILE, header=None, names=['id', 'label'])
keep = []
for _, row in ref.iterrows():
    try: n = sio.loadmat(os.path.join(DATA_DIR, row['id'] + '.mat'))['val'].size
    except Exception: continue
    if n >= FS_ORIG * 5: keep.append((row['id'], LABEL_MAP[row['label']]))
uids = np.array([k[0] for k in keep]); labs = np.array([k[1] for k in keep]); o = np.argsort(uids); uids, labs = uids[o], labs[o]
tr_ids, te_ids = train_test_split(uids, test_size=0.12, random_state=42, stratify=labs)
lab_of = dict(zip(uids, labs))
X, F, y, rid, short = [], [], [], [], []
t0 = time.time()
for r in te_ids:
    sig = sio.loadmat(os.path.join(DATA_DIR, r + '.mat'))['val'].flatten().astype(np.float32)
    w, s = record_windows(sig)
    X.extend(w); F.extend(rr_features(x)[0] for x in w); y.extend([lab_of[r]] * len(w)); rid.extend([r] * len(w)); short.extend([s] * len(w))
X = np.asarray(X, np.float32); F = np.asarray(F, np.float32); y = np.asarray(y); rid = np.asarray(rid); short = np.asarray(short)
print(f"test windows {X.shape}, records {len(te_ids)}  ({time.time()-t0:.0f}s)")

def record_level(P, y, rid):
    recs = np.unique(rid); yt, yp = [], []
    for r in recs: m = rid == r; yt.append(y[m][0]); yp.append(int(np.argmax(P[m].mean(0))))
    return np.array(yt), np.array(yp)
names = ['Normal', 'AF', 'Other']
for path in sys.argv[1:]:
    it = tf.lite.Interpreter(model_content=open(path, 'rb').read(), num_threads=8); it.allocate_tensors()
    ins = it.get_input_details(); out = it.get_output_details()[0]
    ecg = [d for d in ins if len(d['shape']) >= 2 and d['shape'][1] == WIN][0]; rr = [d for d in ins if d is not ecg]
    sc, zp = ecg['quantization']; osc, ozp = out['quantization']
    if rr:
        rr = rr[0]; rs, rz = rr['quantization']
        z = np.load(os.path.join(os.path.dirname(os.path.abspath(path)), 'rr_norm.npz')); mu, sd = z['mean'].astype(np.float32), z['std'].astype(np.float32)
        Fz = np.clip((F - mu) / sd, -5, 5)
    P = np.zeros((len(X), 3), np.float32); t0 = time.time()
    for i in range(len(X)):
        it.set_tensor(ecg['index'], np.clip(np.round(X[i] / sc + zp), -128, 127).astype(np.int8).reshape(1, WIN, 1))
        if rr: it.set_tensor(rr['index'], np.clip(np.round(Fz[i] / rs + rz), -128, 127).astype(np.int8).reshape(rr['shape']))
        it.invoke(); P[i] = (it.get_tensor(out['index'])[0].astype(np.float32) - ozp) * osc
    wp = P.argmax(1); yt, yp = record_level(P, y, rid)
    long = ~np.array([short[rid == r][0] for r in np.unique(rid)])
    print(f"\n===== {os.path.relpath(path, PROJ)}  ({os.path.getsize(path)/1024:.0f} KB, {'dual' if rr else 'single'} input)  {time.time()-t0:.0f}s")
    print(f"window acc {100*(wp==y).mean():.2f}%  window macro-F1 {f1_score(y, wp, average='macro'):.4f}")
    print(f"record acc {100*(yt==yp).mean():.2f}%  record macro-F1 {f1_score(yt, yp, average='macro'):.4f}  (>=30s only {f1_score(yt[long], yp[long], average='macro'):.4f})")
    print(classification_report(yt, yp, target_names=names, digits=4)); print("confusion (rows=true):\n", confusion_matrix(yt, yp))
