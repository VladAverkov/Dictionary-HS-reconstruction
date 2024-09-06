import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
from tqdm import tqdm

def OMP(D, Y, sparsity):
    gram = (D.T).dot(D)
    Dy = (D.T).dot(Y)
    X = orthogonal_mp_gram(gram, Dy, n_nonzero_coefs=sparsity)
    return X


def ksvd(Data, num_atoms, sparsity, initial_D=None,
    maxiter=10, etol=1e-10, approx=False, debug=True):
    """
        K-SVD for Overcomplete Dictionary Learning
        Author: Alan Yang - Fall 2017

        See:
            M. Aharon, M. Elad and A. Bruckstein, "K-SVD: An 
            Algorithm for Designing Overcomplete Dictionaries 
            for Sparse Representation," in IEEE Transactions
            on Signal Processing, vol. 54, no. 11, pp. 4311-4322, 
            Nov. 2006.
            
            Rubinstein, R., Zibulevsky, M. and Elad, M., 
            "Efficient Implementation of the K-SVD Algorithm 
            using Batch Orthogonal Matching Pursuit Technical 
            Report" - CS Technion, April 2008.
                
        Data:       rows hold training data for dictionary fitting
        num_atoms:  number of dictionary atoms
        sparsity:   max sparsity of signals. Reduces to K-means
                    when sparsity=1
        initial_D:  if given, an initial dictionary. Otherwise, random
                    rows of data are chosen for initial dictionary
        maxiter:    maximum number of iterations
        err_thresh: stopping criteria; minimum residual
        approx:     True if using approximate KSVD update method.
                    Code runs faster if True, but results generally
                    in higher training error.
        
        Returns:
            D:               learned dictionary
            X:               sparse coding of input data
            error_norms:     array of training errors for each iteration
        Task: find best dictionary D to represent Data Y;
              minimize squared norm of Y - DX, constraining
              X to sparse codings.
    """
    # **implemented using column major order**
    Data = Data.T

    assert Data.shape[1] > num_atoms # enforce this for now

    # intialization
    if initial_D is not None: 
        D = initial_D / np.linalg.norm(initial_D, axis=0)
        Y = Data
        X = np.zeros([num_atoms, Data.shape[1]])
    else:
        # randomly select initial dictionary from data
        idx_set = range(Data.shape[1])
        idxs = np.random.choice(idx_set, num_atoms, replace=False)    
        Y = Data[:,np.delete(idx_set, idxs)]
        X = np.zeros([num_atoms, Data.shape[1] - num_atoms])
        D = Data[:,idxs] / np.linalg.norm(Data[:,idxs], axis=0)

    # repeat until convergence or stopping criteria
    error_norms = []
    
    iterator = tqdm(range(1,maxiter+1)) if debug else range(1,maxiter+1)
    for iteration in iterator:
        # sparse coding stage: estimate columns of X
        gram = (D.T).dot(D)
        Dy = (D.T).dot(Y)
        X = orthogonal_mp_gram(gram, Dy, n_nonzero_coefs=sparsity)
        # codebook update stage
        for j in range(D.shape[1]):
            # index set of nonzero components
            index_set = np.nonzero(X[j,:])[0]
            if len(index_set) == 0:
                # for now, replace with some white noise
                if not approx:
                    D[:,j] = np.random.randn(*D[:,j].shape)
                    D[:,j] = D[:,j] / np.linalg.norm(D[:,j])
                continue
            # approximate K-SVD update
            if approx:
                E = Y[:,index_set] - D.dot(X[:,index_set])
                D[:,j] = E.dot(X[j,index_set])     # update D
                D[:,j] /= np.linalg.norm(D[:,j])
                X[j,index_set] = (E.T).dot(D[:,j]) # update X
            else:
                # error matrix E
                E_idx = np.delete(range(D.shape[1]), j, 0)
                E = Y - np.dot(D[:,E_idx], X[E_idx,:])
                U,S,VT = np.linalg.svd(E[:,index_set])
                # update jth column of D
                D[:,j] = U[:,0]
                # update sparse elements in jth row of X    
                X[j,:] = np.array([
                    S[0]*VT[ 0,np.argwhere(index_set==n)[0][0] ]
                    if n in index_set else 0
                    for n in range(X.shape[1])])
        # stopping condition: check error        
        err = np.linalg.norm(Y-D.dot(X),'fro')
        error_norms.append(err)
        if err < etol:
            break
    return D,X, np.array(error_norms)

# coding:utf-8
# import numpy as np
# import scipy as sp
# from sklearn.linear_model import orthogonal_mp_gram


# class ApproximateKSVD(object):
#     def __init__(self, n_components, max_iter=10, tol=1e-6,
#                  transform_n_nonzero_coefs=None):
#         """
#         Parameters
#         ----------
#         n_components:
#             Number of dictionary elements

#         max_iter:
#             Maximum number of iterations

#         tol:
#             tolerance for error

#         transform_n_nonzero_coefs:
#             Number of nonzero coefficients to target
#         """
#         self.components_ = None
#         self.max_iter = max_iter
#         self.tol = tol
#         self.n_components = n_components
#         self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

#     def _update_dict(self, X, D, gamma):
#         for j in range(self.n_components):
#             I = gamma[:, j] > 0
#             if np.sum(I) == 0:
#                 continue

#             D[j, :] = 0
#             g = gamma[I, j].T
#             r = X[I, :] - gamma[I, :].dot(D)
#             d = r.T.dot(g)
#             d /= np.linalg.norm(d)
#             g = r.dot(d)
#             D[j, :] = d
#             gamma[I, j] = g.T
#         return D, gamma

#     def _initialize(self, X):
#         if min(X.shape) < self.n_components:
#             D = np.random.randn(self.n_components, X.shape[1])
#         else:
#             u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
#             D = np.dot(np.diag(s), vt)
#         D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
#         return D

#     def _transform(self, D, X):
#         gram = D.dot(D.T)
#         Xy = D.dot(X.T)

#         n_nonzero_coefs = self.transform_n_nonzero_coefs
#         if n_nonzero_coefs is None:
#             n_nonzero_coefs = int(0.1 * X.shape[1])

#         return orthogonal_mp_gram(
#             gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

#     def fit(self, X):
#         """
#         Parameters
#         ----------
#         X: shape = [n_samples, n_features]
#         """
#         D = self._initialize(X)
#         for i in range(self.max_iter):
#             gamma = self._transform(D, X)
#             e = np.linalg.norm(X - gamma.dot(D))
#             if e < self.tol:
#                 break
#             D, gamma = self._update_dict(X, D, gamma)

#         self.components_ = D
#         return self

#     def transform(self, X):
#         return self._transform(self.components_, X)