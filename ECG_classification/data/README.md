# data/ — PhysioNet/CinC Challenge 2017 資料集（未收錄）

訓練 notebook、`HECG_Lab/play_dataset.py`、`tools_eval_tflite_mcu.py` 都從這裡讀資料。請自行下載並放成以下結構：

```
data/
├── REFERENCE.csv          # 8528 筆標籤 (N / A / O / ~)
└── training2017/
    ├── A00001.hea
    ├── A00001.mat
    └── ...                # 8528 筆，300 Hz，單導程
```

下載：PhysioNet「AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017」訓練集（`training2017.zip`，約 1 GB）。`REFERENCE.csv` 使用 v3 版（`REFERENCE-v3.csv` 更名即可）。

資料集有自己的使用條款，請遵守 PhysioNet 的授權規定。本資料夾內容已列入 `.gitignore`。
