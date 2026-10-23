/**
 * @file    af_inference.h
 * @brief   AF 心房顫動偵測 - AI 推論主模組
 *
 * 功能說明:
 *   0. 對 ECG 視窗 (15000 點) 做 band-pass 0.5-40Hz (零相位, 見 bandpass_0p5_40)
 *   1. Z-score 正規化 (在濾波後訊號上)
 *   2. 量化為 INT8 格式 (與訓練時 inference.py 完全一致)
 *   3. 呼叫 TFLite Micro 執行 Ethos-U55 NPU 推論
 *   4. 回傳分類結果 (Normal / AF / Other)
 *
 * 使用前提:
 *   - af_model.h 需在 include path (單一來源: ECG_classification/deploy/)
 *   - Keil 專案需加入 TFLite Micro 與 Ethos-U NPU Driver 函式庫
 *   - Tensor Arena 大小見 TENSOR_ARENA_SIZE (依 Vela 報告的 SRAM 用量設定)
 *
 * 前處理公式 (與 ECG_classification/inference.py 完全一致):
 *   x_bp    = bandpass_0.5-40Hz(x)          ← 關鍵! 少了會在 60Hz 雜訊下漏報 AF
 *   x_norm  = (x_bp - mean(x_bp)) / std(x_bp)
 *   x_int8  = clamp(round(x_norm / input_scale + input_zp), -128, 127)
 */

#ifndef AF_INFERENCE_H
#define AF_INFERENCE_H

#include <stdint.h>
#include <math.h>

// TFLite Micro 標頭檔 (由 Keil 專案提供)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 模型標頭檔 (由 ECG_classification/deploy/af_model.h 複製)
#include "af_model.h"
// 雙輸入模型 (選用): af_model.h 若 #define AF_MODEL_RR_FEATURES 16 並提供 af_rr_feat_mean/std[],
// 會啟用 RR 特徵第二輸入; 單輸入模型沒有這個巨集 → 以下 RR 相關程式碼全部不編譯。
#include "rr_features.h"

// ─────────────────────────────────────────────────────
//  設定區
// ─────────────────────────────────────────────────────
#define ECG_WINDOW_SIZE     15000   // 30秒 @ 500Hz
#define NUM_CLASSES         3       // Normal, AF, Other
// Vela Offline Memory Allocation 模型：arena 只需容納 TFLite Micro 本身的管理開銷
// NPU scratch 由 OfflineMemoryAllocation metadata 指定絕對地址，不佔用 arena
#ifdef AF_MODEL_ARENA_KB
#define TENSOR_ARENA_SIZE   (AF_MODEL_ARENA_KB * 1024)   // 由 af_model.h 依 Vela SRAM 用量指定
#else
#define TENSOR_ARENA_SIZE   (1300 * 1024)  // 1.3 MB (本模型 Vela SRAM 用量 1246 KB; 超出 SRAM 的部分落在 HyperRAM)
#endif

// 推論結果代碼
typedef enum {
    ECG_RESULT_NORMAL = 0,
    ECG_RESULT_AF     = 1,
    ECG_RESULT_OTHER  = 2,
    ECG_RESULT_ERROR  = -1
} EcgResult;

// 推論結果詳細資訊
typedef struct {
    EcgResult result;
    float     prob_normal;
    float     prob_af;
    float     prob_other;
    float     confidence;  // 最高類別的機率
} InferenceOutput;

// ─────────────────────────────────────────────────────
//  靜態內部變數
// TFLite Micro 的 Tensor Arena
// 必須放在 NPU 可存取的 on-chip SRAM012 @ 0x20100000
// 使用 .bss.sram.data 映射到 SRAM01_HYPERRAM (aliased SRAM012)
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((section(".bss.sram.data")));

static tflite::MicroInterpreter* interpreter    = nullptr;
static TfLiteTensor*             input_tensor   = nullptr;   // ECG [1,15000,1] int8
static TfLiteTensor*             rr_tensor      = nullptr;   // 選用: RR 特徵 [1,16] int8 (單輸入模型為 nullptr)
static TfLiteTensor*             output_tensor  = nullptr;
static bool                      model_initialized = false;

// ─────────────────────────────────────────────────────
//  函式: 初始化 AI 推論引擎 (在 main 初始化階段呼叫一次)
//  回傳: true = 成功, false = 失敗
// ─────────────────────────────────────────────────────
static bool AF_Inference_Init(void)
{
    tflite::InitializeTarget();

    // 載入模型
    const tflite::Model* model = tflite::GetModel(af_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[錯誤] 模型版本不相符! 模型版本: %d, 預期版本: %d\r\n", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // 建立 Op Resolver
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddEthosU();       // NPU 加速 (Vela 編譯後的 Operator)
    resolver.AddSoftmax();      // 最後一層 Softmax (CPU 執行)
    resolver.AddReshape();      // Reshape 操作
    resolver.AddFullyConnected(); // 預防 CPU fallback
    resolver.AddConv2D();
    resolver.AddAdd();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddConcatenation();  // 雙輸入 concat (預防 CPU fallback)

    // 建立直譯器
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // 配置記憶體
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("[錯誤] AllocateTensors 失敗! Tensor Arena %d bytes 可能不夠\r\n", TENSOR_ARENA_SIZE);
        return false;  // Arena 太小
    }

    // 取得輸入/輸出張量指標 — 依形狀辨認 (tflite 的輸入順序不保證): dims[1]==15000 → ECG, 否則 → RR 特徵
    input_tensor = nullptr; rr_tensor = nullptr;
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
        TfLiteTensor* t = interpreter->input(i);
        if (t->dims->size >= 2 && t->dims->data[1] == ECG_WINDOW_SIZE) input_tensor = t;
        else rr_tensor = t;
    }
    output_tensor = interpreter->output(0);
    if (!input_tensor) { printf("[錯誤] 找不到 15000 點的 ECG 輸入張量\r\n"); return false; }
#ifdef AF_MODEL_RR_FEATURES
    if (!rr_tensor || rr_tensor->dims->data[rr_tensor->dims->size - 1] != AF_MODEL_RR_FEATURES) {
        printf("[錯誤] 模型需要 %d 維 RR 特徵輸入, 但找不到對應張量\r\n", AF_MODEL_RR_FEATURES);
        return false;
    }
    printf("[初始化] 雙輸入: ECG 15000 + RR 特徵 %d (scale=%.6f zp=%d)\r\n",
           AF_MODEL_RR_FEATURES, rr_tensor->params.scale, (int)rr_tensor->params.zero_point);
#else
    if (rr_tensor) printf("[警告] 模型有第二個輸入但 af_model.h 未定義 AF_MODEL_RR_FEATURES\r\n");
#endif

    model_initialized = true;
    return true;
}

// ─────────────────────────────────────────────────────
//  Band-pass 0.5-40Hz (零相位) — 重現訓練/inference.py 的
//  scipy.signal.butter(3,[0.5,40]/(500/2),'band') + filtfilt。
//  做法: 去均值 → forward → reverse → forward → reverse (穩態初始條件)。
//  已在 PC 端驗證: 模型輸出與 filtfilt 逐位元相同 (見 tools/validate_c_bandpass.py)。
//  ⚠️ 缺這道 → 60Hz 市電雜訊下 AF 機率會被壓到 0、全判 Other。
// ─────────────────────────────────────────────────────
#define BP_SECTIONS 3
// SOS: [b0, b1, b2, a1, a2] (a0=1)。來源: tools/gen_bandpass_golden.py
static const float BP_SOS[BP_SECTIONS][5] = {
    { 0.0098534254f,  0.0197067957f,  0.0098533704f, -1.4211172620f, 0.6191826971f},
    { 1.0000000000f, -0.0000004576f, -1.0000116273f, -1.5930745222f, 0.5956466294f},
    { 1.0000000000f, -1.9999939575f,  0.9999939576f, -1.9937758656f, 0.9938156913f},
};
// 每個 pass 起始 state = BP_ZI[section] * 該 pass 首樣本 (穩態初始, 消除邊界暴衝)
static const float BP_ZI[BP_SECTIONS][2] = {
    { 0.1891393542f, -0.1133595156f},
    {-0.1999277408f, -0.1984381868f},
    { 0.0009349604f, -0.0009349547f},
};
// 濾波工作緩衝 (60KB float)。放大 SRAM 區(同 tensor_arena 的 .bss.sram.data),
// 否則會擠爆只有 ~48KB 的 DTCM 預設 .bss 區 → linker L6406E/L6407E。
static float bp_buf[ECG_WINDOW_SIZE] __attribute__((section(".bss.sram.data")));

// 一個方向的雙二階串接掃描 (Direct-Form-II Transposed), in-place
static void bp_forward(float* buf, int n)
{
    float s1[BP_SECTIONS], s2[BP_SECTIONS];
    float x0 = buf[0];
    for (int s = 0; s < BP_SECTIONS; s++) { s1[s] = BP_ZI[s][0]*x0; s2[s] = BP_ZI[s][1]*x0; }
    for (int i = 0; i < n; i++) {
        float v = buf[i];
        for (int s = 0; s < BP_SECTIONS; s++) {
            float out = BP_SOS[s][0]*v + s1[s];
            s1[s] = BP_SOS[s][1]*v - BP_SOS[s][3]*out + s2[s];
            s2[s] = BP_SOS[s][2]*v - BP_SOS[s][4]*out;
            v = out;
        }
        buf[i] = v;
    }
}

static void bp_reverse(float* buf, int n)
{
    for (int i = 0, j = n-1; i < j; i++, j--) { float t = buf[i]; buf[i] = buf[j]; buf[j] = t; }
}

// band-pass 0.5-40Hz 零相位: 讀原始 ADC → 去均值 → 前後雙向濾波 → 寫入 out
static void bandpass_0p5_40(const volatile int32_t* raw, float* out)
{
    float sum = 0.0f;
    for (int i = 0; i < ECG_WINDOW_SIZE; i++) sum += (float)raw[i];
    float mean = sum / ECG_WINDOW_SIZE;
    for (int i = 0; i < ECG_WINDOW_SIZE; i++) out[i] = (float)raw[i] - mean;
    bp_forward(out, ECG_WINDOW_SIZE);
    bp_reverse(out, ECG_WINDOW_SIZE);
    bp_forward(out, ECG_WINDOW_SIZE);
    bp_reverse(out, ECG_WINDOW_SIZE);
}

// ─────────────────────────────────────────────────────
//  函式: 執行 ECG 推論 (每次收集完 15000 點後呼叫)
//
//  參數:
//    raw_signal  - 原始 ADC int32 陣列，長度 ECG_WINDOW_SIZE
//    output      - 推論結果輸出 (呼叫者提供)
//
//  回傳: true = 推論成功, false = 推論失敗
// ─────────────────────────────────────────────────────
static bool AF_Inference_Run(
    const volatile int32_t* raw_signal,
    InferenceOutput* output)
{
    if (!model_initialized || !interpreter || !output) {
        output->result = ECG_RESULT_ERROR;
        return false;
    }

    // ── 步驟 0: Band-pass 0.5-40Hz (⚠️關鍵!必須在 z-score 前, 對齊訓練) ──
    //   之後所有 mean/std/量化都吃濾波後的 bp_buf, 不再直接用 raw_signal。
    bandpass_0p5_40(raw_signal, bp_buf);

    // ── 步驟 1: 計算 Mean (濾波後訊號) ──
    float sum = 0.0f;
    for (int i = 0; i < ECG_WINDOW_SIZE; i++) {
        sum += bp_buf[i];
    }
    float mean = sum / ECG_WINDOW_SIZE;

    // ── 步驟 2: 計算 Std (濾波後訊號) ──
    float var_sum = 0.0f;
    for (int i = 0; i < ECG_WINDOW_SIZE; i++) {
        float diff = bp_buf[i] - mean;
        var_sum += diff * diff;
    }
    float std = sqrtf(var_sum / ECG_WINDOW_SIZE);
    if (std < 1e-6f) std = 1.0f;  // 防止除以零 (全為零的訊號)

    // ── 步驟 3: 取得量化參數 (由模型自動提供) ──
    float   input_scale = input_tensor->params.scale;
    int32_t input_zp    = input_tensor->params.zero_point;

    // ── 步驟 4: Z-score 正規化 (就地寫回 bp_buf, 供量化與 RR 特徵共用) + INT8 量化 → 填入模型輸入 ──
    int8_t* input_data = input_tensor->data.int8;

    for (int i = 0; i < ECG_WINDOW_SIZE; i++) {
        // Z-score 正規化 (濾波後訊號)
        float normalized = (bp_buf[i] - mean) / std;
        bp_buf[i] = normalized;

        // 量化: round(x / scale) + zero_point
        int32_t q = (int32_t)roundf(normalized / input_scale) + input_zp;

        // 限制在 INT8 範圍 [-128, 127]
        if (q > 127)  q = 127;
        if (q < -128) q = -128;

        input_data[i] = (int8_t)q;
    }

#ifdef AF_MODEL_RR_FEATURES
    // ── 步驟 4b (雙輸入模型): RR 特徵 → 標準化 (train mean/std, 夾 ±5) → INT8 → 第二個輸入 ──
    //   與 notebook Cell 3/4 及 ECG_classification/rr_features.py 完全一致
    {
        float feat[RR_NUM_FEATURES];
        int npk = rr_features_from_window(bp_buf, ECG_WINDOW_SIZE, feat);
        float   rr_scale = rr_tensor->params.scale;
        int32_t rr_zp    = rr_tensor->params.zero_point;
        int8_t* rr_data  = rr_tensor->data.int8;
        printf("[RR]");
        for (int k = 0; k < RR_NUM_FEATURES; k++) {
            float z = (feat[k] - af_rr_feat_mean[k]) / af_rr_feat_std[k];
            if (z > 5.0f) z = 5.0f;
            if (z < -5.0f) z = -5.0f;
            int32_t q = (int32_t)roundf(z / rr_scale) + rr_zp;
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            rr_data[k] = (int8_t)q;
            printf(" %.4f", feat[k]);
        }
        printf("  (peaks=%d)\r\n", npk);
    }
#endif

    // ── 步驟 5: 執行推論 (NPU 運算) ──
    if (interpreter->Invoke() != kTfLiteOk) {
        output->result = ECG_RESULT_ERROR;
        return false;
    }

    // ── 步驟 6: 取得輸出並反量化 ──
    int8_t* out_data  = output_tensor->data.int8;
    float   out_scale = output_tensor->params.scale;
    int32_t out_zp    = output_tensor->params.zero_point;

    float probs[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = ((float)out_data[i] - (float)out_zp) * out_scale;
    }

    // ── 步驟 7: 找出最大機率的類別 ──
    int   max_idx = 0;
    float max_prob = probs[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx  = i;
        }
    }

    // ── 步驟 8: 填入輸出結構 ──
    output->result      = (EcgResult)max_idx;
    output->prob_normal = probs[0];
    output->prob_af     = probs[1];
    output->prob_other  = probs[2];
    output->confidence  = max_prob;

    return true;
}

// ─────────────────────────────────────────────────────
//  函式: 取得結果的字串描述 (用於 UART 輸出 Log)
// ─────────────────────────────────────────────────────
static const char* AF_Inference_ResultStr(EcgResult result)
{
    switch (result) {
    case ECG_RESULT_NORMAL: return "Normal";
    case ECG_RESULT_AF:     return "AF";
    case ECG_RESULT_OTHER:  return "Other";
    default:                return "Error";
    }
}

#endif /* AF_INFERENCE_H */
