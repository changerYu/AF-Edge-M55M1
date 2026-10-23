"""
AF-Edge Inference: Run trained INT8 model on TriAnswer ECG windows.

Usage:
    python inference.py                          # Process all .npy in ecg_windows/
    python inference.py ecg_windows/window_0s_30s.npy  # Process single file
    python inference.py my_folder/               # Process all .npy in custom folder

Input format (from TriAnswer):
    - .npy file, shape=(15000,), dtype=int32
    - 500Hz sampling rate, 30-second window
    - Raw ADC values

Processing pipeline (same as training and the M55M1 firmware):
    1. Load raw ADC → float32
    2. Band-pass 0.5–40 Hz (zero-phase)
    3. Z-score normalize: (x - mean) / std
    4. Quantize to int8: (x_norm / scale + zero_point)
    5. Run TFLite INT8 inference
    6. Output: Normal / AF / Other
"""

import numpy as np
import os
import sys
import re
import glob

# ─── Config ───
RR_INPUT = None            # 雙輸入模型 (選用) 的 RR 特徵輸入 (load_model 時依形狀偵測)
RR_MU = RR_SD = None       # RR 特徵標準化常數 (train 集統計, 隨模型提供)
RR_CLIP = 5.0
MODEL_PATH = os.environ.get("AF_MODEL") or os.path.join(os.path.dirname(__file__), "deploy", "model_int8.tflite")  # 環境變數 AF_MODEL 可指定別的 .tflite
CLASS_NAMES = ["Normal (N)", "AF (A)", "Other (O)"]
EXPECTED_LEN = 15000  # 30s @ 500Hz
FS = 500              # sampling rate (Hz)
BP_LOW, BP_HIGH = 0.5, 40.0  # band-pass cutoffs (Hz), matches training pipeline

# Band-pass filter: training applied 0.5-45 Hz before z-score, so inference must do the same.
# Without it, 60 Hz mains noise from the analog chain breaks AF detection.
# NOTE: the M55M1 firmware implements the same band-pass before z-score/quantize.
try:
    from scipy.signal import butter, filtfilt
    _BP_B, _BP_A = butter(3, [BP_LOW / (FS / 2), BP_HIGH / (FS / 2)], "band")
    def bandpass(sig):
        return filtfilt(_BP_B, _BP_A, sig).astype(np.float32)
except ImportError:
    def bandpass(sig):  # scipy missing -> skip (clean signals still work)
        return sig

# ─── Load TFLite model ───
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

def load_model(model_path):
    # 用 model_content 而非 model_path: litert 的 C++ 開檔不吃非 ASCII 路徑
    with open(model_path, 'rb') as f:
        interp = Interpreter(model_content=f.read())
    interp.allocate_tensors()
    ins = interp.get_input_details()
    inp = [d for d in ins if len(d["shape"]) >= 2 and d["shape"][1] == EXPECTED_LEN][0]   # 依形狀辨認 (韌體同)
    out = interp.get_output_details()[0]
    # 雙輸入模型: 第二個輸入 = RR 特徵; 標準化常數從模型旁的 rr_norm.npz / metrics.json 讀
    global RR_INPUT, RR_MU, RR_SD
    rr = [d for d in ins if d is not inp]
    RR_INPUT = rr[0] if rr else None
    if RR_INPUT is not None:
        mdir = os.path.dirname(os.path.abspath(model_path))
        if os.path.isfile(os.path.join(mdir, "rr_norm.npz")):
            z = np.load(os.path.join(mdir, "rr_norm.npz")); RR_MU, RR_SD = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        elif os.path.isfile(os.path.join(mdir, "metrics.json")):
            import json
            rn = json.load(open(os.path.join(mdir, "metrics.json")))["rr_norm"]; RR_MU, RR_SD = np.array(rn["mean"], np.float32), np.array(rn["std"], np.float32)
        else:
            raise FileNotFoundError(f"雙輸入模型需要 rr_norm.npz 或 metrics.json 放在 {mdir}")
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    return interp, inp, out, in_scale, in_zp, out_scale, out_zp

def preprocess(raw_signal):
    """
    Preprocess raw ADC signal (same pipeline as training).
    raw_signal: 1D array, shape=(15000,), raw ADC int values from TriAnswer.
    Returns: int8 array ready for model, shape=(1, 15000, 1).
    """
    sig = raw_signal.astype(np.float32)

    # Handle wrong length
    if len(sig) < EXPECTED_LEN:
        # Wrap-pad (same as training)
        reps = (EXPECTED_LEN // len(sig)) + 1
        sig = np.tile(sig, reps)[:EXPECTED_LEN]
    elif len(sig) > EXPECTED_LEN:
        sig = sig[:EXPECTED_LEN]

    # Band-pass (same as training) — removes baseline drift + high-freq noise
    sig = bandpass(sig)

    # Z-score normalize (same as training)
    mu, std = np.mean(sig), np.std(sig)
    if std > 1e-6:
        sig = (sig - mu) / std
    else:
        sig = sig - mu

    return sig

def predict(interp, inp_detail, out_detail, in_scale, in_zp, signal_normalized):
    """Run INT8 inference on a single z-score normalized window."""
    # Quantize float → int8
    x_int8 = np.clip(
        np.round(signal_normalized / in_scale + in_zp), -128, 127
    ).astype(np.int8)
    x_int8 = x_int8.reshape(1, EXPECTED_LEN, 1)

    interp.set_tensor(inp_detail["index"], x_int8)
    if RR_INPUT is not None:                              # 雙輸入: RR 特徵 → 標準化 → int8 (== 韌體 af_inference.h 步驟 4b)
        from rr_features import rr_features
        feat, _ = rr_features(signal_normalized.astype(np.float32))
        z = np.clip((feat - RR_MU) / RR_SD, -RR_CLIP, RR_CLIP)
        rs, rz = RR_INPUT["quantization"]
        fq = np.clip(np.round(z / rs + rz), -128, 127).astype(np.int8).reshape(RR_INPUT["shape"])
        interp.set_tensor(RR_INPUT["index"], fq)
    interp.invoke()
    output = interp.get_tensor(out_detail["index"])[0]  # (3,) int8

    # Dequantize output
    out_scale, out_zp = out_detail["quantization"]
    probs = (output.astype(np.float32) - out_zp) * out_scale

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    return class_id, confidence, probs

def process_file(filepath, interp, inp_d, out_d, in_scale, in_zp, out_scale, out_zp):
    """Process a single .npy ECG window file."""
    raw = np.load(filepath)
    sig = preprocess(raw)
    cls_id, conf, probs = predict(interp, inp_d, out_d, in_scale, in_zp, sig)

    return {
        "file": os.path.basename(filepath),
        "class_id": cls_id,
        "class_name": CLASS_NAMES[cls_id],
        "probs": probs,
        "samples": len(raw),
    }

def main():
    # Determine input
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.join(os.path.dirname(__file__), "ecg_windows")

    # Collect .npy files
    if os.path.isfile(target) and target.endswith(".npy"):
        npy_files = [target]
    elif os.path.isdir(target):
        npy_files = sorted(glob.glob(os.path.join(target, "*.npy")),
                          key=lambda f: int(re.search(r"(\d+)s_", os.path.basename(f)).group(1))
                          if re.search(r"(\d+)s_", os.path.basename(f)) else 0)
    else:
        print(f"Error: {target} is not a .npy file or directory")
        sys.exit(1)

    if not npy_files:
        print(f"No .npy files found in {target}")
        sys.exit(1)

    # Load model
    print(f"Model: {MODEL_PATH}")
    print(f"  Size: {os.path.getsize(MODEL_PATH)/1024:.1f} KB")
    interp, inp_d, out_d, in_scale, in_zp, out_scale, out_zp = load_model(MODEL_PATH)
    print(f"  Input:  {inp_d['shape']} (scale={in_scale:.6f}, zp={in_zp})")
    if RR_INPUT is not None:
        print(f"  RR in:  {RR_INPUT['shape']} (scale={RR_INPUT['quantization'][0]:.6f}, zp={RR_INPUT['quantization'][1]})  [雙輸入]")
    print(f"  Output: {out_d['shape']} (scale={out_scale:.6f}, zp={out_zp})")
    print(f"  Classes: {CLASS_NAMES}")
    print()

    # Run inference
    print(f"{'File':<25} {'Result':<15} {'N':>8} {'AF':>8} {'Other':>8}")
    print("─" * 68)

    results = []
    for f in npy_files:
        r = process_file(f, interp, inp_d, out_d, in_scale, in_zp, out_scale, out_zp)
        results.append(r)
        p = r["probs"]
        marker = " [!AF!]" if r["class_id"] == 1 else ""  # Alert for AF
        print(f"{r['file']:<25} {r['class_name']:<15} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f}{marker}")

    # Summary
    print("─" * 68)
    af_count = sum(1 for r in results if r["class_id"] == 1)
    total = len(results)

    if total > 1:
        # Majority vote across all windows
        from collections import Counter
        votes = Counter(r["class_id"] for r in results)
        overall = votes.most_common(1)[0][0]
        # Also average probabilities
        avg_probs = np.mean([r["probs"] for r in results], axis=0)
        avg_cls = int(np.argmax(avg_probs))

        print(f"\n[Summary] Overall ({total} windows):")
        print(f"   Majority vote: {CLASS_NAMES[overall]}")
        print(f"   Avg probs:     N={avg_probs[0]:.4f}  AF={avg_probs[1]:.4f}  O={avg_probs[2]:.4f}")
        print(f"   Avg result:    {CLASS_NAMES[avg_cls]}")

        if af_count > 0:
            print(f"\n   [!AF!]  AF detected in {af_count}/{total} windows!")
        else:
            print(f"\n   [OK] No AF detected")

if __name__ == "__main__":
    main()
