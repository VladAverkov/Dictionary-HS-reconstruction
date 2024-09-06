from utils import get_cube
import numpy as np
import os
import matplotlib.pyplot as plt


folder_path = "/alpha/projects/hyperspectral_data/NTIRE"

N = 940
W = 482
H = 512

sample_sz = 10

K = N * (1 + H // sample_sz) * (1 + W // sample_sz)

D = np.empty((31, K), dtype=float)
t = 0
for i, filename in enumerate(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)
    if i >= N:
        break
    if os.path.isfile(file_path):
        cube = get_cube(file_path)
        for j in range(0, H, sample_sz):    
            for k in range(0, W, sample_sz):
                D[:, t] = cube[:, j, k]
                t += 1
    print(i)

np.save("../data/arad_hyperspectrs.npy", D.T)