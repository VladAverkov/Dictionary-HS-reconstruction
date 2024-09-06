import numpy as np
import os
from PIL import Image
import re
import matplotlib.pyplot as plt
from ksvd import OMP, ksvd
from utils import get_S, mse, rmse, nse, mange
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import cv2
import joblib



folder_path = "/alpha/projects/hyperspectral_data/CAVE/"     
n_samples = 30
n_atoms = 128       
OMP_sparsity = 15
D_sparsity = 4
maxiter = 5
step = 5
H = 512
W = 512
m = 0
S_size = 15
errors = dict()
pictures = ["real_and_fake_peppers_ms", "balloons_ms",
            "beads_ms", "sponges_ms", "oil_painting_ms",
            "flowers_ms", "cd_ms", "photo_and_face_ms"]

data = np.zeros(((H // step + 1) * (W // step + 1), 31), dtype=float)
rgb_data = np.zeros(((H // step + 1) * (W // step + 1), 3), dtype=float)

S1 = get_S("NikonD5100.npy")
S2 = get_S("Canon 60D.csv")
S3 = get_S("NikonD80.npy")
S4 = get_S("iPhone8.npy")
S5 = get_S("GalaxyS9.npy")
S6 = get_S("RedmiNote9S.npy")
S7 = get_S("GalaxyS7edge.npy")
S8 = get_S("Canon50D.npy")
S9 = get_S("AQUOSR5G.npy")
S10 = get_S("PentaxK-5.npy")
# S10 = get_S("Xperia5_2.npy")

S = np.array(np.concatenate((S1, S2, S3, S4, S5, S6, S7, S8, S9, S10), axis=0), dtype=float)[:S_size, :]
D = np.load('/beta/students/averkov/data/arad_1_D_CAVE.npy') #D.shape = (31, N_atoms)
# D = np.load('/beta/students/averkov/data/ill_ref_D.npy')
print(D.shape)
linreg_model = joblib.load("/beta/students/averkov/data/linreg_CAVE.joblib")

for i, filename in enumerate(os.listdir(folder_path)):
    if filename in pictures:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        file_path = os.path.join(file_path, filename)
        for j, file in enumerate(os.listdir(file_path)):
            if file.endswith(".png"):
                pattern = r'\d+'
                match = re.search(pattern, file)
                spectra_channel = int(match.group()) - 1
                file = os.path.join(file_path, file)

                img = Image.open(file)
                arr = np.array(img)
                H, W = arr.shape

                f = 0
                for h in range(0, H, step):
                    for w in range(0, W, step):
                        data[f, spectra_channel] = arr[h, w]
                        f += 1
            # if file.endswith(".bmp"):
            #     file = os.path.join(file_path, file)
            #     img = Image.open(file, "r")
            #     arr = np.array(img)
            #     H, W, _ = arr.shape
            #     f = 0
            #     for h in range(0, H, step):
            #         for w in range(0, W, step):
            #             rgb_data[f] = arr[h, w]
            #             f += 1

        R = S.dot(D)
        rgb_data = data.dot(S.T)

        X = OMP(R, rgb_data.T, sparsity=OMP_sparsity)
        # rgb_data_pred_arad = R.dot(X)
        # data_pred_linreg = linreg_model.predict(rgb_data)
        data_pred_arad = np.squeeze((D).dot(X)).T
        idx = np.random.choice(data.shape[0])
        plt.plot(data[idx], color='black', label='true spectra')
        plt.plot(data_pred_arad[idx], color='red', label='pred spectra')
        plt.legend()
        plt.xlabel("spectra channel")
        plt.ylabel("intenstity")
        plt.savefig('/beta/students/averkov/arad/arad_1/spectra.png')
        print("rmse arad many cameras =", rmse(data, data_pred_arad) / 63290 * 255)
        print("---------------------------")
        input()
        plt.clf()
        # errors[filename] = rmse(data, data_pred_arad) / 63290 * 255
        # print("rmse linreg =", rmse(data, data_pred_linreg) / 63290 * 255)


        
# for key, error in errors.items():
#     print(key)
#     print(error)
#     print("---------------------------")