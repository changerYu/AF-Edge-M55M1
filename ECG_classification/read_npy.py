import numpy as np
import matplotlib.pyplot as plt
import os
import re

folder = "ecg_windows"

files = [f for f in os.listdir(folder) if f.endswith(".npy")]

# 依照起始時間排序，例如 window_0s_30s.npy → 0
def get_start_time(filename):
    match = re.search(r"window_(\d+)s_(\d+)s\.npy", filename)
    if match:
        return int(match.group(1))
    return 999999

files.sort(key=get_start_time)

for f in files:
    path = os.path.join(folder, f)
    data = np.load(path)

    print(f"\n📂 {f}")
    print("Length:", len(data))
    print("First 5 samples:", data[:10])

    plt.figure()
    plt.plot(data)
    plt.title(f)
    plt.xlabel("Sample")
    plt.ylabel("ECG raw value")
    plt.show()