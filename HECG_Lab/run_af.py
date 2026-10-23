"""輸出 AF (Atrial Fibrillation) ECG (CinC2017 A 類隨機一筆, 循環播放)
用法: python run_af.py             # 隨機
      python run_af.py A00004      # 指定 record
"""
import sys
from play_dataset import play_label
play_label('A', specific_record=sys.argv[1] if len(sys.argv) > 1 else None)
