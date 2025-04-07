import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

class TCA:
    def __init__(self, kernel_type='rbf', dim=20, lamb=1.0, gamma=1.0):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.A = None

    def kernel(self, X1, X2=None):
        if self.kernel_type == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel_type == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError("Unsupported kernel type: {}".format(self.kernel_type))

    def fit_transform(self, Xs, Xt):
        X = np.vstack((Xs, Xt))
        n_source = Xs.shape[0]
        n_target = Xt.shape[0]
        n_total = n_source + n_target

        # Construct MMD matrix
        e = np.vstack((np.ones((n_source, 1)) / n_source, -np.ones((n_target, 1)) / n_target))
        M = e @ e.T
        M /= np.linalg.norm(M, 'fro')

        # Construct centering matrix
        H = np.eye(n_total) - 1.0 / n_total * np.ones((n_total, n_total))

        # Kernel computation
        K = self.kernel(X)

        # Construct matrices for eigen-decomposition
        A_matrix = K @ M @ K.T + self.lamb * np.eye(n_total)
        B_matrix = K @ H @ K.T

        # Regularize B to ensure positive definiteness
        epsilon = 1e-6
        B_matrix += epsilon * np.eye(n_total)

        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = eigh(A_matrix, B_matrix)
        indices = np.argsort(eigvals)[:self.dim]
        A = eigvecs[:, indices]
        Z = K @ A
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new = Z[:n_source, :]
        Xt_new = Z[n_source:, :]
        return Xs_new, Xt_new