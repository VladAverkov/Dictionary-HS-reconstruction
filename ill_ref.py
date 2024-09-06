import numpy as np
from ksvd import ksvd, OMP
import matplotlib.pyplot as plt
from utils import get_S, get_R, get_reflectance, get_illuminants, dec_mul, mse, nse, mange, rmse


size = 50000
n_atoms = 100
sparsity_D = 3
OMP_sparsity = 3
S_size = 10
path = "Canon 60D.csv"

data = np.load("/beta/students/averkov/data/dataset_NTIRE.npy")
data = data[np.random.choice(data.shape[0], size=size)]

illuminants = get_illuminants()
reflectance = get_reflectance()

ill_ref = dec_mul(illuminants, reflectance)


# D, X, _ = ksvd(ill_ref, num_atoms=n_atoms, sparsity=sparsity_D, approx=False, maxiter=10, initial_D=ill_ref[:n_atoms, :].T)
# np.save('/beta/students/averkov/data/ill_ref_D.npy', D)
D = np.load('/beta/students/averkov/data/ill_ref_D.npy') #D.shape = (31, N_atoms)


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