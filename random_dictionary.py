import matplotlib.pyplot as plt
import numpy as np
from utils import get_S, get_R, mse, rmse, nse, mange
from ksvd import OMP


n_atoms = 128
size = 10000
S_size = 12
OMP_sparsity = 8
path = "Canon 60D.csv"

data = np.load("/beta/students/averkov/data/dataset_NTIRE.npy", allow_pickle=True)
data = data[np.random.choice(data.shape[0], size=size)]

# D = data[np.random.choice(data.shape[0], 100), :].T
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


R, r_inv = get_R(D, S) # R.shape = (3, N_atoms)
rgb_data = data.dot(S.T) # rgb_data.shape = (data_size, 3)

print("data collected...")

X = OMP(R, rgb_data.T, sparsity=OMP_sparsity) # X.shape = (N_atoms, data_size)

print("OMP is done...")

data_pred = np.squeeze((D).dot(r_inv.dot(X))).T # data_pred.shape = (data_size, 31)


print("nse =", nse(data, data_pred))
print("rmse =", rmse(data, data_pred))
print("mange =", mange(data, data_pred))
print("mse =", mse(data, data_pred))

