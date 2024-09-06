import numpy as np
from ksvd import ksvd
import matplotlib.pyplot as plt
from utils import get_S, get_R


size = 50000
words = 300
path = "Canon 60D.csv"

data = np.load("/beta/students/averkov/data/dataset_NTIRE.npy")
data = data[np.random.choice(data.shape[0], size=size)].T

# dict_model =ksvd(words=words,iteration=5, errGoal=0.5)
# dictionary=dict_model.constructDictionary(data)

# np.save('/beta/students/averkov/data/arad_dictionary.npy', dictionary)
# print("finish dict training")

# D = dictionary

D = np.load('/beta/students/averkov/data/arad_dictionary.npy')

S = get_S(path)

R, r_inv = get_R(D, S)
rgb_data = S.dot(data)

print("data collected...")

dict_model = ksvd(words=300,iteration=10, errGoal=0.2)
dict_model.dictInitialization(R)
X = dict_model.OMP(R, rgb_data, 0.1)

print("OMP is done...")

data_pred = np.squeeze(D.dot(r_inv.dot(X)))
print("mse =", np.mean((data - data_pred)**2))

N = 1
while True:
    idx = np.random.choice(data.shape[1])
    y_true = data[:, idx]
    y_pred = data_pred[:, idx]
    plt.clf()
    plt.plot(y_true, color='blue')
    plt.plot(y_pred, color='red')
    plt.savefig("/beta/students/averkov/arad/spectras.png")
    input()
    print(f"pair {N}")
    N+=1
