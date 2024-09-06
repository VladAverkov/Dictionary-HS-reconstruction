import numpy as np
from utils import get_image_cube
import pandas as pd
import matplotlib.pyplot as plt
from ksvd import transform


sens = pd.read_csv("/alpha/projects/hyperspectral_data/sensitivities/Canon 60D.csv")
S = sens[['r', 'g', 'b']].values
S = S.transpose()
S = S[:,:31]


rgb_dict = np.load("../data/arad_rgb_dict.npy")
spectra_dict = np.load("../data/arad_spectra_dict.npy")
print(spectra_dict.shape)
# plt.plot(spectra_dict[0])
# plt.savefig("hyperspectr.png")

path = "/alpha/projects/hyperspectral_data/NTIRE/ARAD_1K_0263.mat"
image, cube = get_image_cube(path)


mse = 0
mange = 0
nse = 0
mae = 0
# N = image.shape[0]
N = 5
for idx in range(N):
    idx = np.random.choice(image.shape[0])
    rgb = image[idx]
    gamma = transform(rgb_dict, rgb, 3)
    # print("pred grb:", gamma.dot(rgb_dict))
    # print("true rgb:", rgb)
    true_spectra = cube[idx]
    pred_spectra = gamma.dot(spectra_dict)
    plt.plot(true_spectra, color="blue")
    plt.plot(pred_spectra, color="red")
    # print("true spectra:",  np.round(true_spectra, 2))
    # print("pred spectra:", np.round(pred_spectra, 2))
    # print("true rgb:", np.round(rgb, 2))
    # print("pred rgb:", np.round(pred_spectra.dot(S.T), 2))
    mse += np.mean((true_spectra - pred_spectra)**2)
    err_mange = pred_spectra.dot(true_spectra) / np.linalg.norm(true_spectra) / np.linalg.norm(pred_spectra)
    mange += np.arccos(err_mange)
    err_nse = np.linalg.norm(true_spectra - pred_spectra, ord=1)
    err_nse /= np.linalg.norm(true_spectra, ord=1)
    err_nse /= np.linalg.norm(pred_spectra, ord=1)
    nse += err_nse
    mae += np.mean(np.abs(true_spectra-pred_spectra) / np.abs(true_spectra))


plt.savefig("hyperspectrs_2.png")
print("mse =", mse / N)
print("mange =", mange / N)
print("nse =", nse / N)
print("mae =", mae / N)