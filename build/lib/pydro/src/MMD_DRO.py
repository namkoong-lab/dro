from .base import *
import numpy as np
import math
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances

"""
we set (X, Y) as the same scale but it may not be in practice.
"""


class MMD_DRO(base_DRO):
    def __init__(self, input_dim, is_regression=2):
        base_DRO.__init__(self, input_dim, is_regression)
        self.eta = 0.1
        self.sampling_method = 'bound'
        self.n_certify_ratio = 1
    def update(self, config = {}):
        if 'eta' in config.keys():
            self.eta = config['eta']
        if 'sampling_method' in config.keys():
            assert (config['sampling_method'] in ['bound', 'hull'])
            self.sampling_method = config['sampling_method']
        if 'n_certify_ratio' in config.keys():
            self.n_certify_ratio = config['n_certify_ratio']

    def matrix_decomp(self, K):
        try:
            L = np.linalg.cholesky(K)
        except:
            # print('warning, K is singular')
            d, v = np.linalg.eigh(K)     #L == U*diag(d)*U'. the scipy function forces real eigs
            d[np.where(d < 0)] = 0       # get rid of small eigs
            L = v @ np.diag(np.sqrt(d))
        return L

    def medium_heuristic(self, X, Y):
        if self.is_regression == 1 or self.is_regression == 2:
            distsqr = euclidean_distances(X, Y, squared = True)
        else:
            distsqr = euclidean_distances(X[:,0:-1], Y[:,0:-1], squared = True)

        kernel_width = np.sqrt(0.5 * np.median(distsqr))

        # in sklearn,
        # kernel is done by K(x, y) = exp(-gamma ||x-y||^2)
        kernel_gamma = 1.0 / (2 * kernel_width ** 2)

        return kernel_width, kernel_gamma

    def cvx_loss(self, theta, zeta):
        if self.is_regression == 1 or self.is_regression == 2:
            loss = (zeta[-1] - theta @ zeta[:-1]) ** 2
        else:
            loss = cp.pos(1 - cp.multiply(zeta[-1], theta @ zeta[:-1]))
        return loss

    def fit(self, X, y):
        if self.is_regression == 0:
            y = 2*y-1
        sample_size, __ = X.shape
        n_certify = int(self.n_certify_ratio * sample_size)

        theta = cp.Variable(self.input_dim)                  # variable \theta

        # constraint on the decision variable

        # KDRO part
        a = cp.Variable(sample_size + n_certify)    # variable \alpha
        f0 = cp.Variable()                       # variable f0
        
        # --------------------------------------------------------------------------------
        # Step 1: generate the sampled support
        # --------------------------------------------------------------------------------
        if self.sampling_method == 'bound':         # sample within certain bound
            # let the samples also live in the intervel I sampled uncertainty w
            zeta = np.random.uniform(-1, 1, size=[n_certify, self.input_dim + 1])
        elif self.sampling_method == 'hull':        # sample using convex hull of empirical data
            # let the samples also live in the intervel I sampled uncertainty w
            # this is equiv. to really do it in multi dimensions, need to sample coeff. from a simplex
            if self.is_regression == 1 or self.is_regression == 2:
                zeta1 = np.random.uniform(np.min(X), np.max(X), size = [n_certify, self.input_dim])
                zeta2 = np.random.uniform(np.min(y), np.max(y), size = [n_certify, 1])
            else:
                zeta1 = np.random.uniform(-1, 1, size = [n_certify, self.input_dim])
                zeta2 = np.random.choice([-1, 1], size = (n_certify, 1))
            zeta = np.concatenate([zeta1, zeta2], axis = 1)
        else:
            raise NotImplementedError

        data = np.concatenate([X, y.reshape(-1, 1)], axis = 1)
        # in practice, we always include the empirical data in the sampled support
        zeta = np.concatenate([data, zeta])

        kernel_width, kernel_gamma = self.medium_heuristic(zeta, zeta)
        
        # --------------------------------------------------------------------------------
        # Step 3: setup objective function and constraints
        # --------------------------------------------------------------------------------
        # evaluate the f, K at the value of zetas
        K = rbf_kernel(zeta, zeta)
        #gamma = kernel_gamma)
        f = a @ K
        constr = []
        for i in range((len(zeta))):
            constr += [self.cvx_loss(theta, zeta[i]) <= f0 + f[i]]

        obj = f0 + cp.sum(f[0:sample_size]) / sample_size + self.eta * cp.norm(a.T @ self.matrix_decomp(K))
        opt = cp.Problem(cp.Minimize(obj), constr)
    

        opt.solve(solver = cp.MOSEK)
        self.theta = theta.value #, obj.value, a.value, f0.value, kernel_gamma, zeta

        