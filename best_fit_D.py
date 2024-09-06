import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from utils import get_S, get_R, rmse, mse, nse, mange
from ksvd import OMP

n = 128
size = 50000
OMP_sparsity = 3
path = "Canon 60D.csv"

data = np.load("/beta/students/averkov/data/dataset_NTIRE.npy")
data = data[np.random.choice(data.shape[0], size=size)]
n_data = normalize(data, axis=1)

D = np.zeros((n, 31), dtype=float)


def find_v(D, data):
    M = np.abs(D.dot(data.T))
    maxes = np.max(M, axis=0)
    idx = np.argmin(maxes)
    return data[idx, :]


D[0, :] = n_data[0, :]
for i in range(1, n):
    #print(i)
    v = find_v(D[:i, :], n_data)
    D[i, :] = v
np.save('/beta/students/averkov/data/arad_D.npy', D)
D = np.load('/beta/students/averkov/data/arad_D.npy').T


S = get_S("Canon 60D.csv")
print(S.shape)
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
