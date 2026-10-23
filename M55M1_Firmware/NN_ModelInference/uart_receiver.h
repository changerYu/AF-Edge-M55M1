/**
 * @file    uart_receiver.h
 * @brief   M55M1 UART 資料接收模組
 *
 * 功能說明:
 *   接收來自 TriAnswer 的 ECG 資料幀，解析後填入 30 秒視窗緩衝。
 *   當累積滿 15000 個點 (30秒 @ 500Hz) 時，
 *   設定 flag 通知主迴圈執行 AI 推論。
 *
 * UART 腳位 (由 main.cpp 的 UART_Init 設定):
 *   PA.2 = UART1_RX (NU2 D13, 接 TriAnswer TX)
 *   PA.3 = UART1_TX (NU2 D10, 送結果碼, 選用)
 *
 * 幀格式 (與 TriAnswer 韌體一致):
 *   [0xAA][0xBB][LEN_L][LEN_H][int32 LE × N]
 *   LEN = 資料區段的 byte 數 = 樣本數 x 4
 */

#ifndef UART_RECEIVER_H
#define UART_RECEIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// ─────────────────────────────────────────────────────
//  設定區
// ─────────────────────────────────────────────────────
#define ECG_WINDOW_SIZE     15000   // 30秒 @ 500Hz
#define UART_BAUD_RATE      921600
#define FRAME_HEADER_0      0xAA
#define FRAME_HEADER_1      0xBB

// Sliding Window 設定
// 每次推論後，保留最後 10000 點，只需要再等 5000 點 (10秒) 就能再推論
#define WINDOW_STRIDE       5000    // 10 秒步距

// ─────────────────────────────────────────────────────
//  全域共享變數 (由 UART ISR 填入，由 main 讀取)
// ─────────────────────────────────────────────────────
volatile int32_t  ecg_window[ECG_WINDOW_SIZE] __attribute__((section(".bss.sram.data")));  // 當前 30 秒 ECG 視窗
volatile uint32_t ecg_rx_count = 0;             // 目前已接收的樣本數
volatile bool     inference_ready = false;       // 當 = true 時，主迴圈執行推論

// 解析狀態機的內部狀態 (僅 uart_receiver.h 內部使用)
typedef enum {
    PARSE_STATE_WAIT_H0 = 0,  // 等待幀頭第1個 byte 0xAA
    PARSE_STATE_WAIT_H1,      // 等待幀頭第2個 byte 0xBB
    PARSE_STATE_LEN_L,        // 接收長度低位元組
    PARSE_STATE_LEN_H,        // 接收長度高位元組
    PARSE_STATE_DATA          // 接收資料本體
} ParseState;

static ParseState  parse_state  = PARSE_STATE_WAIT_H0;
static uint16_t    frame_len    = 0;  // 本幀資料總 byte 數
static uint16_t    byte_idx     = 0;  // 目前已接收的 byte 數 (在 DATA 狀態用)
static uint8_t     raw_buf[4]   = {0};// 暫存單一 int32 的 4 個 bytes
static uint8_t     raw_buf_idx  = 0;

// ─────────────────────────────────────────────────────
//  函式: 重設接收狀態 (UART 硬體初始化在 main.cpp 的 UART_Init)
// ─────────────────────────────────────────────────────
static inline void UART_Receiver_Init(void)
{
    ecg_rx_count   = 0;
    inference_ready = false;
    parse_state    = PARSE_STATE_WAIT_H0;
}

// ─────────────────────────────────────────────────────
//  函式: 處理單一收到的 byte (在 UART ISR 裡呼叫)
//  狀態機解析器，解析 TriAnswer 傳來的幀格式
// ─────────────────────────────────────────────────────
static inline void UART_Receiver_ProcessByte(uint8_t byte)
{
    switch (parse_state)
    {
    case PARSE_STATE_WAIT_H0:
        if (byte == FRAME_HEADER_0)
            parse_state = PARSE_STATE_WAIT_H1;
        break;

    case PARSE_STATE_WAIT_H1:
        if (byte == FRAME_HEADER_1)
            parse_state = PARSE_STATE_LEN_L;
        else
            parse_state = PARSE_STATE_WAIT_H0;  // 不合法，重新等幀頭
        break;

    case PARSE_STATE_LEN_L:
        frame_len   = (uint16_t)byte;
        parse_state = PARSE_STATE_LEN_H;
        break;

    case PARSE_STATE_LEN_H:
        frame_len  |= ((uint16_t)byte << 8);
        byte_idx    = 0;
        raw_buf_idx = 0;
        parse_state = PARSE_STATE_DATA;
        break;

    case PARSE_STATE_DATA:
        // 將 byte 累積成 4 bytes → 轉成 int32
        raw_buf[raw_buf_idx++] = byte;
        byte_idx++;

        if (raw_buf_idx == 4)
        {
            // 組合成 int32 (Little-Endian)
            int32_t sample;
            memcpy(&sample, raw_buf, 4);
            raw_buf_idx = 0;

            // 放入 ECG 視窗
            if (ecg_rx_count < ECG_WINDOW_SIZE)
            {
                ecg_window[ecg_rx_count++] = sample;

                // 當滿 15000 點時，觸發推論 flag
                if (ecg_rx_count >= ECG_WINDOW_SIZE)
                {
                    inference_ready = true;
                }
            }
        }

        // 這一幀的資料接收完畢，等下一幀
        if (byte_idx >= frame_len)
        {
            parse_state = PARSE_STATE_WAIT_H0;
        }
        break;

    default:
        parse_state = PARSE_STATE_WAIT_H0;
        break;
    }
}

// ─────────────────────────────────────────────────────
//  函式: 推論完成後重設，準備接收下一個滑動視窗
//  Sliding Window: 保留最後 10000 點，移動 5000 點
// ─────────────────────────────────────────────────────
static inline void UART_Receiver_SlideWindow(void)
{
    // 將後半段 (10000 點) 移到前面
    uint32_t keep_count = ECG_WINDOW_SIZE - WINDOW_STRIDE;
    for (uint32_t i = 0; i < keep_count; i++)
    {
        ecg_window[i] = ecg_window[WINDOW_STRIDE + i];
    }
    ecg_rx_count   = keep_count;  // 下次只需要再接收 5000 點
    inference_ready = false;
}

#endif /* UART_RECEIVER_H */
