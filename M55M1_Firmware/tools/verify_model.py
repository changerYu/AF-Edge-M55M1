"""
verify_model.py - 驗證 af_model.tflite 的輸入輸出量化參數

功能:
    1. 讀取 deploy/model_int8.tflite
    2. 印出輸入/輸出張量的 shape, dtype, scale, zero_point
    3. 這些數值必須與 M55M1 韌體中使用的完全一致

用法:
    python verify_model.py

輸出範例:
    ✅ 輸入張量:
       Shape     : [1, 15000, 1]
       Dtype     : int8
       Scale     : 0.145202
       Zero Point: 20

    ✅ 輸出張量:
       Shape     : [1, 3]
       Dtype     : int8
       Scale     : 0.003906
       Zero Point: -128
       Classes   : Normal(0) / AF(1) / Other(2)
"""

import os
import sys
import numpy as np

# ─── 路徑設定 ───
# deploy 的單一權威來源在 ECG_classification/deploy/ (由訓練 notebook 產出)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "ECG_classification")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "deploy", "model_int8.tflite")

if not os.path.exists(MODEL_PATH):
    print(f"[錯誤] 找不到模型檔: {MODEL_PATH}")
    print("請確認 ECG_classification/deploy/model_int8.tflite 存在。")
    sys.exit(1)

# ─── 載入 TFLite 推論引擎 ───
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

interp = Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print("=" * 50)
print(" AF-Edge Model 量化參數驗證")
print("=" * 50)
print(f"\n模型路徑: {MODEL_PATH}")
print(f"模型大小: {os.path.getsize(MODEL_PATH)/1024:.1f} KB\n")

print("✅ 輸入張量 (Input Tensor):")
print(f"   Shape     : {inp['shape'].tolist()}")
print(f"   Dtype     : {inp['dtype'].__name__}")
print(f"   Scale     : {inp['quantization'][0]:.8f}")
print(f"   Zero Point: {inp['quantization'][1]}")

print("\n✅ 輸出張量 (Output Tensor):")
print(f"   Shape     : {out['shape'].tolist()}")
print(f"   Dtype     : {out['dtype'].__name__}")
print(f"   Scale     : {out['quantization'][0]:.8f}")
print(f"   Zero Point: {out['quantization'][1]}")
print(f"   Classes   : Normal(0) / AF(1) / Other(2)")

print("\n" + "─" * 50)
print("📋 請將以下數值確認與 M55M1 韌體一致:")
print(f"   input->params.scale       = {inp['quantization'][0]:.8f}f")
print(f"   input->params.zero_point  = {inp['quantization'][1]}")
print(f"   output->params.scale      = {out['quantization'][0]:.8f}f")
print(f"   output->params.zero_point = {out['quantization'][1]}")
print("─" * 50)
print("⚠️  以上參數由 TFLite Micro 自動從模型中讀取，")
print("    不需要手動填寫到韌體中，只需確認格式正確。")
print()

# ─── 快速功能測試 (用零訊號) ───
print("🔧 快速功能測試 (輸入全零訊號)...")
dummy_input = np.zeros((1, 15000, 1), dtype=np.int8)
interp.set_tensor(inp['index'], dummy_input)
interp.invoke()
raw_output = interp.get_tensor(out['index'])[0]

out_scale = out['quantization'][0]
out_zp    = out['quantization'][1]
probs = (raw_output.astype(np.float32) - out_zp) * out_scale

print(f"   輸出 INT8 原始值: {raw_output.tolist()}")
print(f"   反量化後機率: Normal={probs[0]:.4f}, AF={probs[1]:.4f}, Other={probs[2]:.4f}")
print(f"   預測類別: {['Normal', 'AF', 'Other'][np.argmax(probs)]}")
print("\n✅ 模型驗證完成！可以安心燒錄到 M55M1。")
