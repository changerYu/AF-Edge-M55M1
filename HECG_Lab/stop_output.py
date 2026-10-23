import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from hecg100 import Device

device = Device(os.path.join(_HERE, 'HECG100x64.dll'))
handle = device.connect(-1, 5000)
if not handle:
    print('[WARN] 沒連到裝置'); sys.exit(1)

# 連續送 3 次停止指令，每次間隔 0.5 秒，確保 USB 真的把指令送到
for i in range(3):
    ok = device.stop_output(handle)
    print(f'  stop_output #{i+1} = {ok}', flush=True)
    time.sleep(0.5)

time.sleep(1.0)        # 讓裝置韌體完成停止
device.free(handle)
print('[OK] device freed, 輸出應已停止')
