from ksvd import OMP, ksvd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_S, get_R, mse, rmse, nse, mange

n_atoms = 100
D_sparsity = 4
S_size = 15
OMP_sparsity = 12
maxiter = 0
path = "Canon 60D.csv"


data = np.load("/beta/students/averkov/data/dataset_NTIRE.npy", allow_pickle=True)
data = data[np.random.choice(data.shape[0], 100000)]

# D, X, _ = ksvd(data, num_atoms=n_atoms, maxiter=5, sparsity=4, approx=False, initial_D=data[:n_atoms, :].T)
# np.save('/beta/students/averkov/data/arad_1_D.npy', D)
D = np.load('/beta/students/averkov/data/arad_1_D.npy') #D.shape = (31, N_atoms)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      



S1 = get_S("Canon 60D.csv")
S2 = get_S("NikonD80.npy")
S3 = get_S("iPhone8.npy")
S4 = get_S("GalaxyS9.npy")
S5 = get_S("RedmiNote9S.npy")
S6 = get_S("GalaxyS7edge.npy")
S7 = get_S("Canon50D.npy")
S8 = get_S("AQUOSR5G.npy")
S9 = get_S("PentaxK-5.npy")
S10 = get_S("Xperia5_2.npy")

S = np.array(np.concatenate((S1, S2, S3, S4, S5, S6, S7, S8, S9, S10), axis=0), dtype=float)[:S_size, :]


R = S.dot(D)

rgb_data = data.dot(S.T) # rgb_data.shape = (data_size, 3)

print("data collected...")

X = OMP(R, rgb_data.T, sparsity=OMP_sparsity) # X.shape = (N_atoms, data_size)

# rgb_data_pred = R.dot(X)
# print("RGB MSE:", mse(rgb_data.T, rgb_data_pred))
# print("OMP is done...")

data_pred = np.squeeze((D).dot(X)).T # data_pred.shape = (data_size, 31)

# # plt.plot(data[0], color='blue')
# # plt.plot(data_pred[0], color='green')
# plt.savefig("/beta/students/averkov/arad/arad_1/spectra.png")

                                                
# print("nse =", nse(data, data_pred))
# print("rmse =", rmse(data, data_pred)[0])
# print("mange =", mange(data, data_pred))
# print("mse =", mse(data, data_pred))

# err, max_err = rmse(data, data_pred)
# # print(np.max(data))
# err = err / np.max(data) * 255
# print(err)




while True:
    idx = np.random.choice(data.shape[0])
    # plt.ylim(0, 1)
    plt.plot(data[idx], color='blue')
    plt.plot(data_pred[idx], color='red')
    plt.savefig("/beta/students/averkov/arad/arad_1/spectra.png")

    input()
    plt.clf()
