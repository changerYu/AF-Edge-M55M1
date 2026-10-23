"""
serial_monitor.py - 監看 M55M1 透過 UART 傳回電腦的推論結果

功能:
    1. 開啟 M55M1 對應的 COM Port (NuLink 虛擬 UART)
    2. 即時顯示推論結果 (Normal / AF / Other)
    3. 在終端機用顏色區分正常與 AF 警示
    4. 可選擇性將 Log 存到 CSV 檔案

用法:
    # 自動偵測 COM Port
    python serial_monitor.py

    # 指定 COM Port
    python serial_monitor.py COM3

    # 指定 COM Port 並儲存 Log
    python serial_monitor.py COM3 --save

安裝需求:
    pip install pyserial colorama
"""

import sys
import os
import serial
import serial.tools.list_ports
import threading
import argparse
import csv
from datetime import datetime

# 韌體會印中文 + emoji (✅/⚠️); Windows 主控台預設 gbk 會 UnicodeEncodeError。
# 強制 stdout 用 utf-8 且遇到編不出的字元就替換,避免整個監看崩潰。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    USE_COLOR = True
except ImportError:
    USE_COLOR = False
    print("[提示] 安裝 colorama 可以讓輸出更好看: pip install colorama")

# ─────────────────────────────────────────────────────
#  設定區
# ─────────────────────────────────────────────────────
DEFAULT_BAUD = 115200  # Log UART 的鮑率
TIMEOUT_SEC  = 1.0

# ─────────────────────────────────────────────────────
#  自動尋找 M55M1 (NuLink) 的 COM Port
# ─────────────────────────────────────────────────────
def find_nulink_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = (port.description or "").upper()
        # NuLink 通常會出現在描述中
        if "NULINK" in desc or "NUVOTON" in desc or "USB SERIAL" in desc:
            return port.device
    return None

# ─────────────────────────────────────────────────────
#  解析並格式化輸出一行 M55M1 的 Log
# ─────────────────────────────────────────────────────
def format_log_line(line: str) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    prefix = f"[{timestamp}] "

    if "AF 偵測" in line or "AF" in line and "⚠️" in line:
        if USE_COLOR:
            return Fore.RED + Style.BRIGHT + prefix + line.strip() + Style.RESET_ALL
        return "[!AF!] " + prefix + line.strip()
    elif "Normal" in line or "✅" in line:
        if USE_COLOR:
            return Fore.GREEN + prefix + line.strip()
        return "[OK  ] " + prefix + line.strip()
    elif "Other" in line or "❓" in line:
        if USE_COLOR:
            return Fore.YELLOW + prefix + line.strip()
        return "[?   ] " + prefix + line.strip()
    elif "錯誤" in line or "Error" in line or "失敗" in line:
        if USE_COLOR:
            return Fore.MAGENTA + prefix + line.strip()
        return "[ERR ] " + prefix + line.strip()
    else:
        return prefix + line.strip()

# ─────────────────────────────────────────────────────
#  CSV 紀錄工具
# ─────────────────────────────────────────────────────
class CsvLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "Result", "N_Prob", "AF_Prob", "Other_Prob", "RawLine"])
        print(f"[Log] 推論結果將儲存至: {filename}")

    def log(self, line: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        result = ""
        n_prob = af_prob = o_prob = ""

        # 嘗試解析機率值
        import re
        m = re.search(r"N=([\d.]+)\s+AF=([\d.]+)\s+O=([\d.]+)", line)
        if m:
            n_prob, af_prob, o_prob = m.group(1), m.group(2), m.group(3)

        if "AF 偵測" in line:
            result = "AF"
        elif "Normal" in line:
            result = "Normal"
        elif "Other" in line:
            result = "Other"

        if result:
            self.writer.writerow([ts, result, n_prob, af_prob, o_prob, line.strip()])
            self.file.flush()

    def close(self):
        self.file.close()

# ─────────────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="M55M1 AF Inference Serial Monitor")
    parser.add_argument("port", nargs="?", help="COM Port (例: COM3)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--save", action="store_true", help="將結果儲存為 CSV")
    args = parser.parse_args()

    # 決定 COM Port
    com_port = args.port
    if not com_port:
        com_port = find_nulink_port()
        if com_port:
            print(f"[自動偵測] 找到 NuLink UART: {com_port}")
        else:
            print("[錯誤] 無法自動偵測 COM Port，請手動指定。")
            print("可用的 COM Ports:")
            for p in serial.tools.list_ports.comports():
                print(f"   {p.device} - {p.description}")
            sys.exit(1)

    # 初始化 CSV Logger (選用)
    csv_logger = None
    if args.save:
        log_name = f"ecg_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_logger = CsvLogger(log_name)

    # 開啟序列埠
    try:
        ser = serial.Serial(com_port, args.baud, timeout=TIMEOUT_SEC)
    except serial.SerialException as e:
        print(f"[錯誤] 無法開啟 {com_port}: {e}")
        sys.exit(1)

    print(f"\n{'=' * 55}")
    print(f"  AF-Edge M55M1 Serial Monitor")
    print(f"  Port: {com_port}  Baud: {args.baud}")
    print(f"  按 Ctrl+C 結束")
    print(f"{'=' * 55}\n")

    if USE_COLOR:
        print(Fore.GREEN + "  [綠色] = Normal  | " +
              Fore.RED    + "[紅色] = AF 警示  | " +
              Fore.YELLOW + "[黃色] = Other")
        print(Style.RESET_ALL)

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='replace')
            if line.strip():
                formatted = format_log_line(line)
                print(formatted)

                if csv_logger:
                    csv_logger.log(line)

    except KeyboardInterrupt:
        print("\n[系統] 監看結束。")
    finally:
        ser.close()
        if csv_logger:
            csv_logger.close()
            print(f"[Log] 紀錄已儲存至: {csv_logger.filename}")

if __name__ == "__main__":
    main()
