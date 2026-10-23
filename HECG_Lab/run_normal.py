"""輸出 Normal ECG (CinC2017 N 類隨機一筆, 循環播放)
用法: python run_normal.py           # 隨機
      python run_normal.py A00001    # 指定 record
"""
import sys
from play_dataset import play_label
play_label('N', specific_record=sys.argv[1] if len(sys.argv) > 1 else None)
