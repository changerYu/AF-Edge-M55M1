import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from ctypes import pointer
from hecg100 import Device, MODEL_INFORMATION, HW_INFORMATION

DLL = os.path.join(os.path.dirname(__file__), 'HECG100x64.dll')

device = Device(DLL)
handle = device.connect(-1, 5000)

if not handle:
    print('[FAIL] 沒連到裝置')
    sys.exit(1)

print(f'[OK] 連上裝置, Serial = {device.get_serial_number(handle)}')

mi = MODEL_INFORMATION()
if device.get_device_info(handle, pointer(mi)):
    print(f'  Product       : {mi.ProductName.decode("ascii", errors="ignore")}')
    print(f'  Generation/Model: {mi.GenerationNumber}/{mi.ModelNumber}')
    print(f'  Year/SN       : {mi.Year}/{mi.SerialNumber}')

hw = HW_INFORMATION()
if device.get_HW_info(handle, pointer(hw)):
    print(f'  FW Version    : {hw.FWMajorVersion}.{hw.FWMinorVersion}.{hw.FWBuildVersion}')
    print(f'  HW Version    : {hw.HWMajorVersion}.{hw.HWMinorVersion}')

device.free(handle)
