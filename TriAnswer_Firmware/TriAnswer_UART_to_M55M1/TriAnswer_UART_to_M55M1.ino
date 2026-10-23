/**
 * TriAnswer (Yutech TriBLE nRF52840) → M55M1  直接以硬體 UART 傳 ECG (不用 BLE)。
 *
 * 取代原廠 BLE 範例韌體 (Yutech nRF52 核心內建的 TriBLE_Stu_Demo)。
 * 採樣用硬體 TIMER4 中斷 (2 ms = 500 Hz; SysTick 會在 FreeRTOS 睡眠時停擺而偏慢約 2%)，
 * 資料從硬體 UART (Serial1) 送出，幀格式與 M55M1 uart_receiver.h 完全一致：
 *
 *   [0xAA][0xBB][LEN_L][LEN_H][int32 LE × N]      LEN = N*4,  N = BATCH_SIZE
 *
 * 接線 (TriBLE 母板接頭編號見原廠 Pin_config):
 *   TriBLE pin 22 "UART TX / D0" (P0.25) ──► M55M1 NU2 D13 (PA.2 = UART1 RX)
 *   TriBLE GND (pin 4/8/12)               ──► M55M1 GND
 *   (TriBLE pin 21 UART RX 不用接)
 *   電極：ECG 子板插中排 = CH2 (= A3)。上排雖是 CH1 但子板橫跨左右會蓋住 D0 (UART TX)。
 *
 * 燒錄 (arduino-cli, 核心 Yutech:nrf52 已裝):
 *   arduino-cli compile --fqbn Yutech:nrf52:TriBLE_nRF52 TriAnswer_Firmware/TriAnswer_UART_to_M55M1
 *   arduino-cli upload  --fqbn Yutech:nrf52:TriBLE_nRF52 -p COMx TriAnswer_Firmware/TriAnswer_UART_to_M55M1
 *
 * PC 端驗證 (不需要任何轉接板)：MIRROR_TO_USB=1 時同一份幀也送到 USB CDC，
 *   用 TriAnswer_Firmware/tools/uart_capture.py COMx 收 30 s 視窗 → ECG_classification/inference.py。
 *
 * (本韌體不用 SysTick, 原廠「SysTick_Handler 與 SysTick_Config 必須同時存在」的警告不適用。)
 */

// ───────────── 設定區 ─────────────
#define ECG_PIN            A3        // TriAnswer CH2 = 中排 (子板橫跨左右, 上排會蓋住 D0 UART TX)。CH1 = A5, CH3 = A2
#define SAMPLE_INTERVAL_us 2000      // 500 Hz (= 模型 / M55M1 期望的取樣率)
#define ADC_BITS           12        // 原韌體 8-bit；12-bit 解析度更好，M55M1 端 z-score 不在乎刻度
#define UART_BAUD          921600    // 必須 = M55M1 uart_receiver.h 的 UART_BAUD_RATE
#define BATCH_SIZE         100       // 每幀樣本數 (200 ms 一幀)
#define MIRROR_TO_USB      1         // 1 = 同時把幀送到 USB CDC (Serial) 供 PC 驗證；0 = 只走 Serial1
#define LED_HEARTBEAT      1         // 每幀閃一下藍燈

#define FRAME_HEADER_0     0xAA
#define FRAME_HEADER_1     0xBB

// ───────────── 環形緩衝 (ISR 寫入, loop 讀出) ─────────────
#define RING_SIZE          1024      // 2 的次方，約 2 s 緩衝
static volatile int32_t  ring[RING_SIZE];
static volatile uint32_t ring_head = 0;   // ISR 寫
static volatile uint32_t ring_tail = 0;   // loop 讀
static volatile uint32_t dropped   = 0;   // 緩衝滿時丟掉的樣本數 (正常應為 0)

// ⚠️ 不用 SysTick：loop 在等 UART 傳完時 FreeRTOS 會讓 CPU 睡眠，Cortex-M 的 SysTick 睡眠時停擺，
//    實測取樣率因此掉到 ~490 Hz (每幀 4.4ms / 200ms ≈ 2%)。改用硬體 TIMER4 (周邊，睡眠時照跑)。
extern "C" {
  void TIMER4_IRQHandler(void)
  {
    NRF_TIMER4->EVENTS_COMPARE[0] = 0;
    int32_t v = analogRead(ECG_PIN);
    uint32_t next = (ring_head + 1) & (RING_SIZE - 1);
    if (next != ring_tail) {
      ring[ring_head] = v;
      ring_head = next;
    } else {
      dropped++;
    }
  }
}

// ───────────── 幀組裝 ─────────────
static uint8_t frame[4 + BATCH_SIZE * 4];
static uint32_t frames_sent = 0;
static uint32_t samples_sent = 0;
static uint32_t t_last_report = 0;

static void send_frame(uint32_t n)
{
  frame[0] = FRAME_HEADER_0;
  frame[1] = FRAME_HEADER_1;
  uint16_t len = (uint16_t)(n * 4);
  frame[2] = (uint8_t)(len & 0xFF);
  frame[3] = (uint8_t)(len >> 8);
  for (uint32_t i = 0; i < n; i++) {
    int32_t v = ring[ring_tail];
    ring_tail = (ring_tail + 1) & (RING_SIZE - 1);
    uint8_t* p = &frame[4 + i * 4];
    p[0] = (uint8_t)(v & 0xFF);          // int32 little-endian
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
  }
  Serial1.write(frame, 4 + len);
#if MIRROR_TO_USB
  if (Serial) Serial.write(frame, 4 + len);
#endif
  frames_sent++;
  samples_sent += n;
#if LED_HEARTBEAT
  digitalToggle(LED_BLUE);
#endif
}

void setup()
{
  // 啟動外部 32MHz 晶振 (HFXO) 當 CPU/SysTick 時脈。不跑 BLE 時 SoftDevice 不會幫忙開,
  // 預設走內部 RC (HFINT) 會偏 ~2% (實測 489.9 Hz)。HFXO 後 SysTick 500 Hz 才準。
  NRF_CLOCK->EVENTS_HFCLKSTARTED = 0;
  NRF_CLOCK->TASKS_HFCLKSTART = 1;
  while (NRF_CLOCK->EVENTS_HFCLKSTARTED == 0) { }

  pinMode(LED_RED,  OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  digitalWrite(LED_RED, HIGH);          // 紅燈常亮 = 韌體在跑

  Serial1.begin(UART_BAUD);             // 硬體 UART: TX = D0/P0.25 (pin 22), RX = D1/P0.24 (pin 21)
#if MIRROR_TO_USB
  Serial.begin(115200);                 // USB CDC (baud 無意義)。不等待連線，免得沒插 USB 時卡住
#endif

  pinMode(ECG_PIN, INPUT);
  analogReadResolution(ADC_BITS);       // 參考電壓沿用核心預設 (與原韌體相同)，只改解析度

  // TIMER4: 1 MHz 計數 (16MHz / 2^4), 每 SAMPLE_INTERVAL_us 觸發 COMPARE0 → 自動清零 → 中斷
  NRF_TIMER4->TASKS_STOP  = 1;
  NRF_TIMER4->MODE        = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->BITMODE     = TIMER_BITMODE_BITMODE_32Bit;
  NRF_TIMER4->PRESCALER   = 4;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->CC[0]       = SAMPLE_INTERVAL_us;
  NRF_TIMER4->SHORTS      = TIMER_SHORTS_COMPARE0_CLEAR_Msk;
  NRF_TIMER4->INTENSET    = TIMER_INTENSET_COMPARE0_Msk;
  NVIC_SetPriority(TIMER4_IRQn, 3);
  NVIC_ClearPendingIRQ(TIMER4_IRQn);
  NVIC_EnableIRQ(TIMER4_IRQn);
  NRF_TIMER4->TASKS_START = 1;
  t_last_report = millis();
}

void loop()
{
  uint32_t avail = (ring_head - ring_tail) & (RING_SIZE - 1);
  if (avail >= BATCH_SIZE) {
    send_frame(BATCH_SIZE);
  }

#if MIRROR_TO_USB
  // 每 5 秒在 USB 上印一行狀態 (純文字, 幀解析器會略過非幀頭 bytes)。
  // 只有在 PC 有開 USB 序列埠時才印，不影響 Serial1。
  uint32_t now = millis();
  if (now - t_last_report >= 5000) {
    if (Serial) {
      Serial.print("\n# frames="); Serial.print(frames_sent);
      Serial.print(" samples="); Serial.print(samples_sent);
      Serial.print(" dropped="); Serial.print(dropped);
      Serial.print(" rate~"); Serial.print(samples_sent * 1000.0f / now, 1); Serial.println(" Hz\n");
    }
    t_last_report = now;
  }
#endif
}
