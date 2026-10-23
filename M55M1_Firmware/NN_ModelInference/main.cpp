/**
 * @file    main.cpp
 * @brief   M55M1 AF 偵測主程式 (Nuvoton M55M1BSP NN_ModelInference 範例的應用層)
 *
 * 使用方式: 見 M55M1_Firmware/README.md。本資料夾的 5 個檔案放進 BSP 的
 *   SampleCode/MachineLearning/NN_ModelInference/ 覆蓋同名檔即可建置。
 *
 * ──────────────────────────────────────────────────────────
 *  系統資料流：
 *
 *  TriAnswer (UART 921600)  →  M55M1 UART1 ISR (PA.2)
 *                                        │
 *                                 uart_receiver.h
 *                                 (幀解析 + 30 s 視窗)
 *                                        │
 *                                 (滿 15000 點時)
 *                                        │
 *                                  af_inference.h
 *                      (band-pass + Z-score + INT8 + NPU 推論)
 *                                        │
 *                              Normal / AF / Other
 *                                        │
 *                          ┌─────────────┼─────────────┐
 *                       UART Log      LED 亮燈    UART1 TX 結果碼
 * ──────────────────────────────────────────────────────────
 *
 * LED 輸出定義 (請依板子修改腳位):
 *   GREEN_LED  = Normal
 *   RED_LED    = AF
 *   YELLOW_LED = Other
 */

// ─────────────────────────────────────────────────────
//  系統標頭檔 (Nuvoton M55M1 BSP 提供)
// ─────────────────────────────────────────────────────
#include "NuMicro.h"      // M55M1 硬體暫存器定義
#include <stdio.h>

// ─────────────────────────────────────────────────────
//  專案模組
// ─────────────────────────────────────────────────────
#include "uart_receiver.h"  // UART 接收 + 幀解析
#include "af_inference.h"   // AI 推論引擎
#include "BoardInit.hpp"    // Nuvoton 板級初始化 (Clock, NPU)

// ─────────────────────────────────────────────────────
//  設定區 - 請依板子實際腳位修改
// ─────────────────────────────────────────────────────
#define LED_GREEN_PIN   PH4   // Normal 指示燈
#define LED_RED_PIN     PH5   // AF 警示燈
#define LED_YELLOW_PIN  PH6   // Other 指示燈

#define ECG_UART        UART1 // 接收 TriAnswer ECG 資料的 UART (RX = PA.2 = NU2 D13)

// ─────────────────────────────────────────────────────
//  函式: UART1 ISR (ECG 資料接收)
// ─────────────────────────────────────────────────────
extern "C" void UART1_IRQHandler(void)
{
    if (UART_GET_INT_FLAG(ECG_UART, UART_INTSTS_RDAINT_Msk))
    {
        while (UART_IS_RX_READY(ECG_UART))
        {
            uint8_t byte = (uint8_t)UART_READ(ECG_UART);
            UART_Receiver_ProcessByte(byte);  // 幀解析狀態機
        }
    }
}

// ─────────────────────────────────────────────────────
//  函式: 初始化 UART (僅 ECG 接收，Log 由 BoardInit 處理)
// ─────────────────────────────────────────────────────
static void UART_Init(void)
{
    SYS_UnlockReg();
    // UART1 (ECG 接收) 時鐘 (使用 HIRC)
    CLK_EnableModuleClock(UART1_MODULE);
    CLK_SetModuleClock(UART1_MODULE, CLK_UARTSEL0_UART1SEL_HIRC, CLK_UARTDIV0_UART1DIV(1));
    
    // UART1 腳位設定 (PA2=RXD, PA3=TXD) - RXD 接 TriAnswer TX
    SET_UART1_RXD_PA2();
    SET_UART1_TXD_PA3();
    
    // LED 腳位設定為輸出
    GPIO_SetMode(PH, BIT4 | BIT5 | BIT6, GPIO_MODE_OUTPUT);
    SYS_LockReg();

    // UART1 for ECG
    UART_Open(ECG_UART, UART_BAUD_RATE);  // UART_BAUD_RATE 定義在 uart_receiver.h
    UART_EnableInt(ECG_UART, UART_INTEN_RDAIEN_Msk);
    NVIC_EnableIRQ(UART1_IRQn);
    NVIC_SetPriority(UART1_IRQn, 1);
}

// ─────────────────────────────────────────────────────
//  函式: 依推論結果控制 LED 與輸出 Log
// ─────────────────────────────────────────────────────
static void HandleInferenceResult(const InferenceOutput* out)
{
    // 先全滅
    LED_GREEN_PIN  = 1;
    LED_RED_PIN    = 1;
    LED_YELLOW_PIN = 1;

    switch (out->result)
    {
    case ECG_RESULT_NORMAL:
        LED_GREEN_PIN = 0;  // 綠燈亮
        printf("[結果] ✅ Normal  N=%.3f AF=%.3f O=%.3f\r\n",
               out->prob_normal, out->prob_af, out->prob_other);
        break;

    case ECG_RESULT_AF:
        LED_RED_PIN = 0;    // 紅燈亮 (警告!)
        printf("[結果] ⚠️  ** AF 偵測! **  N=%.3f AF=%.3f O=%.3f\r\n",
               out->prob_normal, out->prob_af, out->prob_other);
        // TODO: 在這裡加入警報機制 (Buzzer / 藍牙通報 / 顯示器)
        break;

    case ECG_RESULT_OTHER:
        LED_YELLOW_PIN = 0; // 黃燈亮
        printf("[結果] ❓ Other   N=%.3f AF=%.3f O=%.3f\r\n",
               out->prob_normal, out->prob_af, out->prob_other);
        break;

    default:
        printf("[錯誤] 推論失敗！請檢查模型是否正確載入。\r\n");
        break;
    }

    // ── 由 UART1 TX (PA.3 = NU2 D10) 送出單字元結果碼, 供外部控制器使用 (選用) ──
    //   'N'=Normal / 'A'=AF / 'O'=Other, 以 '\n' 結尾。
    //   UART1 全雙工: RX(PA.2) 收 ECG, TX(PA.3) 送結果碼, 不衝突。
    uint8_t code;
    switch (out->result)
    {
    case ECG_RESULT_NORMAL: code = 'N'; break;
    case ECG_RESULT_AF:     code = 'A'; break;
    case ECG_RESULT_OTHER:  code = 'O'; break;
    default:                code = 'O'; break;   // 推論失敗 → 當 Other
    }
    uint8_t frame[2] = { code, (uint8_t)'\n' };
    UART_Write(ECG_UART, frame, 2);   // ECG_UART = UART1
}

// ─────────────────────────────────────────────────────
//  main()
// ─────────────────────────────────────────────────────
int main(void)
{
    // ── 硬體初始化 ──
    BoardInit();  // Nuvoton BSP 提供，包含 NPU 與 printf 的 UART6 初始化

    UART_Init();
    UART_Receiver_Init();  // 初始化 ECG 接收緩衝區

    printf("\r\n");
    printf("========================================\r\n");
    printf("  AF-Edge M55M1 ECG Inference System\r\n");
    printf("  Cortex-M55 + Ethos-U55 NPU\r\n");
    printf("  Model: af_model (Vela INT8, 215KB)\r\n");
    printf("========================================\r\n");
    printf("等待 UART1 傳來 ECG 資料...\r\n\r\n");

    // ── AI 引擎初始化 ──
    printf("[初始化] 載入 AI 模型...\r\n");
    if (!AF_Inference_Init())
    {
        printf("[錯誤] 模型初始化失敗！請確認 af_model.h 是否正確匯入。\r\n");
        while (1);  // 停止執行
    }
    printf("[初始化] ✅ AI 模型載入成功！\r\n");
    printf("[初始化] Tensor Arena 使用量: %u bytes\r\n",
           (unsigned)interpreter->arena_used_bytes());
    printf("\r\n[系統] 開始即時 ECG 推論 (每 10 秒輸出一次結果)...\r\n\r\n");

    // ── 主迴圈 ──
    static uint32_t inference_count = 0;
    InferenceOutput result;

    while (1)
    {
        // 等待 UART ISR 填滿 15000 點資料
        if (inference_ready)
        {
            inference_count++;
            printf("[推論 #%lu] 開始分析 30 秒 ECG...\r\n", inference_count);

            // 執行 AI 推論
            if (AF_Inference_Run(ecg_window, &result))
            {
                HandleInferenceResult(&result);
            }

            // 移動滑動視窗 (保留最後 20 秒，等新的 10 秒資料)
            UART_Receiver_SlideWindow();
        }

        // 可以在這裡加入低功耗等待
        // __WFI();
    }
}
