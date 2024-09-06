import numpy as np
import pandas as pd


dict_spectra = np.load("../data/arad_spectra_dict_1.npy")


sens = pd.read_csv("/alpha/projects/hyperspectral_data/sensitivities/Canon 60D.csv")
S = sens[['r', 'g', 'b']].values
S = S.transpose()
S = S[:,:31]

rgb_dict = dict_spectra.dot(S.T)
print(rgb_dict.shape)

np.save("../data/arad_rgb_dict_1.npy", rgb_dict)