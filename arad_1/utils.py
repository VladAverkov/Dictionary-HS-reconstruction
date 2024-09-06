import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import colour_datasets

sens = pd.read_csv("/alpha/projects/hyperspectral_data/sensitivities/Canon 60D.csv")
S = sens[['r', 'g', 'b']].values
S = S.transpose()
S = S[:,:31]

# по номеру картинки от (1 до 950) получаем гиперспектральный куб (сами кубы задаются 0000 до 0950)
def get_cube(path):
    data = h5py.File(path, 'r')
    cube = data["cube"][:]
    return cube

def mse(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred)**2)

def rmse(Y_true, Y_pred):
    A = np.mean((Y_true - Y_pred)**2, axis=1)**0.5
    return np.mean(A)

def mange(Y_true, Y_pred):
    A = np.linalg.norm(Y_true, axis=1)
    B = np.linalg.norm(Y_pred, axis=1)
    C = (Y_true * Y_pred).sum(axis=1)
    D = np.arccos(C / A / B)
    return np.mean(D)

def nse(Y_true, Y_pred):
    A = np.linalg.norm(Y_true, axis=1, ord=1)
    B = np.linalg.norm(Y_pred, axis=1, ord=1)
    C = np.linalg.norm(Y_pred - Y_true, axis=1, ord=1)
    D = C / A / B
    return np.mean(D)



# по гиперспектральному кубу с помощью матрицы чувствительностей получаем обычное трехканальное изображение
def get_image_cube(path):
    cube = get_cube(path)
    d, _, _ = cube.shape
    cube = cube.reshape(d, -1).T
    image = cube.dot(S.T)
    return image, cube


def get_S(path):
    if path[-4:] == ".npy":
        path = "/alpha/projects/hyperspectral_data/sensitivities/" + path
        S = np.load(path)
        S = S.transpose()
        S = S[:,:31]
        return S.astype(float)
    elif path[-4:] == ".csv":
        path = "/alpha/projects/hyperspectral_data/sensitivities/" + path
        sens = pd.read_csv(path)
        S = sens[['r', 'g', 'b']].values
        S = S.transpose()
        S = S[:,:31]
        return S
    else:
        return -1



def get_R(D, S):
    R = S.dot(D)
    r_inv = np.diag(np.array([1 / np.linalg.norm(R[:, j]) for j in range(R.shape[1])]))
    R = normalize(R, axis=0)
    return R, r_inv


def aggregate(v, step):
    n = v.shape[0]
    w = np.zeros(n // step)
    for i in range(n // step):
        w[i] = np.mean(v[i*step:(i+1)*step])
    return w


def get_reflectance():
    dataset = colour_datasets.load("3269918")['X']
    reflectance = np.zeros((1600, 31), dtype=float)
    for i in range(1, 1601):
        v = np.array(dataset[str(i)])[16:326, 1]
        w = aggregate(v, 10)
        reflectance[i-1, :] = w
    return reflectance


def get_illuminants():
    xlsx_file_path = "/beta/students/averkov/arad/illuminations.csv"
    illuminants = pd.read_csv(xlsx_file_path, decimal=',').values[:, 1:]
    return illuminants.T


def dec_mul(A, B):
    n = A.shape[0]
    m = B.shape[0]
    if len(A.shape) != len(B.shape):
        return -1 
    if len(A.shape) > 2:
        return -1
    if len(A.shape) == 1:
        C = np.zeros(m * n)
        t = 0
        for i in range(n):
            for j in range(m):
                C[t] = A[i] * B[j]
                t += 1
    if A.shape[1] != B.shape[1]:
        return -1
    C = np.zeros((m * n, A.shape[1]))
    t = 0
    for i in range(n):
        for j in range(m):
            C[t] = A[i] * B[j]
            t += 1
    return C
