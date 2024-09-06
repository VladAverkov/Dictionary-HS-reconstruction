import numpy as np
import os
from PIL import Image
import re
import matplotlib.pyplot as plt


folder_path = "/alpha/projects/hyperspectral_data/CAVE/"     
n_samples = 30
H = 512
W = 512
step = 10

data = np.zeros(((H // step + 1) * (W // step + 1) * 31, 31), dtype=float)

f = 0
for i, filename in enumerate(os.listdir(folder_path)):
    if filename == "watercolors_ms":
        continue
    print(i)
    print(filename)
    file_path = os.path.join(folder_path, filename)
    file_path = os.path.join(file_path, filename)
    s = f
    for j, file in enumerate(os.listdir(file_path)):
        if file.endswith(".png"):
            pattern = r'\d+'
            match = re.search(pattern, file)
            spectra_channel = int(match.group()) - 1
            file = os.path.join(file_path, file)

            img = Image.open(file)
            arr = np.array(img)
            H, W = arr.shape
            s = f
            for h in range(0, H, step):
                for w in range(0, W, step):
                    data[s, spectra_channel] = arr[h, w]
                    s += 1
    f = s

np.save("/beta/students/averkov/data/CAVE.npy", data)

data = np.load("/beta/students/averkov/data/CAVE.npy")
print(data.shape)



