# hecg100.py — WhaleTeq HECG100 SDK 的 Python ctypes 包裝。
# 來源: WhaleTeq HECG100 SDK 1.0.0.1 "Sample Code/python/hecg100.py"
# Copyright (c) 2025 WHALETEQ Co., LTD — MIT License (全文見同目錄 LICENSE-WhaleTeq-SDK.txt)
# 本專案的唯一修改: 模組層級 device_lib 的 DLL 載入包成 try/except (見下方 NOTE)。
import os
from ctypes import *
import struct
import platform
from enum import Enum
from functools import wraps

def pause():
    input("Press Enter to continue ...")
    
def get_lib_path():
    lib_base = "../../SDK"
    if platform.system() == 'Windows':
        if struct.calcsize("P") == 4:
            lib_name = "HECG100x86.dll"
        else:
            lib_name = "HECG100x64.dll"
    elif platform.system() == 'Linux':
        if struct.calcsize("P") == 4:
            lib_name = "libhecg00x86.so"
        else:
            lib_name = "libhecg100x64.so"
    elif platform.system() == 'Darwin':
        lib_name = "libHECG100.a"

    return os.path.join(lib_base, lib_name)

# NOTE: module-level global is unused (Device.__init__ loads the DLL via an
# explicit path). Guard it so importing this module never crashes when the
# cwd-relative "../../SDK" path doesn't resolve (e.g. connect_check.py).
try:
    device_lib = cdll.LoadLibrary(get_lib_path())
except OSError:
    device_lib = None

ConnectedCallback = CFUNCTYPE(None, c_void_p, c_bool)

def sdk_fn(cfn_name, args_type=None, return_type=None):
    def decorator_function(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cfn = getattr(self.sdk, cfn_name)
            if args_type:
                cfn.argtypes = args_type
            if return_type:
                cfn.restype = return_type
            return cfn(*args, **kwargs)

        return wrapper

    return decorator_function

class ECGWaveformType (Enum):
	Sine = 0
	Triangle = 1
	Square = 2
	RectanglePulse = 3
	TrianglePulse = 4
	ECG = 5

class RespirationMode (Enum):
    Off = 0
    On = 1


class Arrhythmia(Enum):
	NormalSinusRhythm = 0
	SinusBradycardia = 1
	SinusTachycardia = 2
	SinusArrhythmia = 3

	PrematureVentricularContraction = 4
	AtrialPrematureBeat = 5
	
	AtrialFlutter = 6
	AtrialFibrillation = 7
	VentricularFibrillationCoarse = 8
	VentricularFibrillationFine = 9

	FirstDegreeAVBlock = 10
	SecondDegreeAVBlock1 = 11
	SecondDegreeAVBlock2 = 12
	ThirdDegreeAVBlock = 13

	RightBundleBranchBlock = 14
	LeftBundleBranchBlock = 15


class ECG_WAVEFORM (Structure):
    _fields_ = [
        ('WaveformType', c_int),
        ('Frequency', c_double),
        ('Amplitude', c_double),
        ('STSegment', c_double),
        ('DCOffset', c_int),
        ('PulseWidth', c_int),
        ('NoiseAmplitude', c_double),
        ('NoiseFrequency', c_int),
        ('PacingEnabled', c_int),
        ('PacingAmplitude', c_double),
        ('PacingDuration', c_double),
        ('RespirationMode', c_int),
        ('RespirationRate', c_int),
        ('RespirationBaseline', c_int),
        ('RespirationRatio', c_int),
        ('RespirationAmplitude', c_int),
        ('RespirationApneaDuration', c_int),
        ('DriftEnabled', c_int)
    ]

class PLAY_DATA (Structure):
    _fields_ = [
        ('SampleRate', c_double),
        ('Size', c_int),
        ('Data', c_void_p)
    ]

class MODEL_INFORMATION (Structure):
    _fields_ = [
        ('ProductName', c_char * 2),
        ('GenerationNumber', c_ubyte),
        ('ModelNumber', c_ubyte),
        ('Year', c_int),
        ('SerialNumber', c_int),
    ]
    
class HW_INFORMATION (Structure):
    _fields_ = [
        ('FWMajorVersion', c_int),
        ('FWMinorVersion', c_int),
        ('FWBuildVersion', c_int),
        ('HWMajorVersion', c_int),
        ('HWMinorVersion', c_int),
    ]

class Device:
    #
    # Initialization & Cleanup
    #
    def __init__(self, lib_file_path):
        self.sdk = cdll.LoadLibrary(lib_file_path)
        self.sdk.HECGGetSerialNumber.argtypes = [c_void_p]
        self.sdk.HECGGetSerialNumber.restype = c_char_p

    @sdk_fn("HECGInit", [ConnectedCallback], c_void_p)
    def init(self, cb) -> c_void_p:
        pass

    @sdk_fn("HECGConnect", [c_uint, c_uint], c_void_p)
    def connect(self, port_number, milliseconds_timeout) -> c_void_p:
        pass

    @sdk_fn("HECGFree", [c_void_p], None)
    def free(self, handle) -> None:
        pass

    def get_serial_number(self, handle):        
        return self.sdk.HECGGetSerialNumber(handle).decode('ascii')
    
    # device info
    @sdk_fn("HECGGetDeviceInformation", [c_void_p, POINTER(MODEL_INFORMATION)], c_bool)
    def get_device_info(self, handle, modelInfo) -> c_bool:
        pass

    # HW info
    @sdk_fn("HECGGetHWInformation", [c_void_p, POINTER(HW_INFORMATION)], c_bool)
    def get_HW_info(self, handle, hwInfo) -> c_bool:
        pass

    # get version
    @sdk_fn("HECGGetVersion", [c_void_p], c_char_p)
    def get_version(self, handle) -> c_char_p:
        pass

    @sdk_fn("HECGStartOutput", [c_void_p, c_void_p], c_bool)
    def start_output(self, handle, ecg_wavform) -> c_bool:
        pass

    @sdk_fn("HECGStopOutput", [c_void_p], c_bool)
    def stop_output(self, handle) -> c_bool:
        pass

    def get_default_ecg_waveform(self):
        waveform = ECG_WAVEFORM()
        waveform.WaveformType = ECGWaveformType.Square.value
        waveform.Frequency = 1
        waveform.Amplitude = 2.0
        waveform.STSegment = 0.0

        waveform.DCOffset = 0

        waveform.PulseWidth = 100

        waveform.NoiseAmplitude = 0
        waveform.NoiseFrequency = 0

        waveform.PacingEnabled = 0
        waveform.PacingAmplitude = 2
        waveform.PacingDuration = 2

        waveform.RespirationMode = RespirationMode.Off.value
        waveform.RespirationRate = 60
        waveform.RespirationBaseline = 1000
        waveform.RespirationRate = 1
        waveform.RespirationAmplitude = 1
        waveform.RespirationApneaDuration = 10

        waveform.DriftEnabled = 0
        return waveform
    

    @sdk_fn("HECGStartPlayData", [c_void_p, c_void_p], c_bool)
    def start_play_data(self, handle, raw_data) -> c_bool:
        pass

    @sdk_fn("HECGEnablePlayDataLoop", [c_void_p, c_bool], c_bool)
    def enable_play_data_loop(self, handle, enable) -> c_bool:
        pass

    
    @sdk_fn("HECGStartArrhythmia", [c_void_p, c_int], c_bool)
    def start_play_arrhythmia(self, handle, index) -> c_bool:
        pass
