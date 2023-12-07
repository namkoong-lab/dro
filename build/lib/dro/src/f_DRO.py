from .base import *
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score

class chi2_DRO(base_DRO):
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.eps = 0

    def update(self, config = {}):
        if 'eps' in config.keys():
            self.eps = config['eps']
    
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        eta = cp.Variable()
        
        loss = math.sqrt(1 + self.eps) / math.sqrt(sample_size) * cp.norm(cp.pos(self.cvx_loss(X, y, theta) - eta), 2) + eta
        
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params
    
    def worst_distribution(self, X, y):
        # Eq (8) in p6 of https://jmlr.org/papers/volume20/17-750/17-750.pdf
        ## only change |n p - 1|^2 \leq 2 \rho to |np - 1|^2 \leq n * eps^2
        self.fit(X,y)

        sample_size, __ = X.shape
        per_loss = self.loss(X, y)
        prob = cp.Variable(sample_size, nonneg = True)
        cons = [cp.sum(prob) == 1, cp.sum_squares(sample_size * prob - 1) <= sample_size * self.eps]
        problem = cp.Problem(cp.Maximize(prob @ per_loss), cons)
        problem.solve(solver = cp.MOSEK)
        return {'sample_pts': [X, y], 'weight': prob.value}



class CVaR_DRO(base_DRO):
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.alpha = 1

    def update(self, config = {}):
        if 'alpha' in config.keys():
            self.alpha = config['alpha']
            assert (self.alpha > 0 and self.alpha <= 1)
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        eta = cp.Variable()
        loss = cp.sum(cp.pos(self.cvx_loss(X, y, theta)- eta)) / (sample_size * self.alpha) + eta
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value
        self.threshold_val  = eta.value
        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["threshold"] = self.threshold_val.tolist()
        return model_params

    def worst_distribution(self, X, y):
        sample_size, __ = X.shape
        per_loss = self.loss(X, y)
        index = np.array(np.where(per_loss > self.threshold_val))[0]
        return {'sample_pts': [X[index], y[index]], 'weight': np.ones(len(index)) / len(index)}



class Marginal_CVaR_DRO(base_DRO):
    """
    We follow Equation (27):
    https://arxiv.org/pdf/2007.13982.pdf
    with parameter L and p.
    """
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.alpha = 1
        self.control_name = None
        self.p = 2
        self.L = 10

    def update(self, config = {}):
        if 'control_name' in config.keys():
            assert all(0 <= x <= self.input_dim - 1 for x in config['control_name'])
            self.control_name = config['control_name']
        if 'L' in config.keys():
            assert (config['L'] > 0)
            self.L = config['L']
        if 'p' in config.keys():
            assert (config['p'] >= 1)
            self.p = config['p']
        if 'alpha' in config.keys():
            assert (config['alpha'] > 0 and config['alpha'] <= 1)
            self.alpha = config['alpha']

    def fit(self, X, y):
        sample_size, __ = X.shape
        if self.control_name is not None:
            control_X = X[:,self.control_name]
        else:
            control_X = X
        dist = np.power(squareform(pdist(control_X)), self.p - 1)
        b_var = cp.Variable((sample_size, sample_size), nonneg = True)
        s = cp.Variable(sample_size, nonneg = True)
        eta = cp.Variable()
        theta = cp.Variable(self.input_dim)
        cons = [s >= self.cvx_loss(X, y, theta) - (cp.sum(b_var, axis = 1)  - cp.sum(b_var, axis = 0)) / sample_size - eta]

        cost = cp.sum(s) / (self.alpha * sample_size) + self.L ** (self.p - 1) * cp.sum(cp.multiply(dist, b_var)) / (sample_size ** 2)
        prob = cp.Problem(cp.Minimize(cost + eta), [s >=0, b_var >= 0] + cons)
        print("begin fitting")
        solver_options = {
           'MSK_IPAR_PRESOLVE_USE': 1,
           'MSK_DPAR_BASIS_TOL_X': 1e-5,  
            'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-5,  
            'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-5,  
            'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-5
        }
        prob.solve(solver = cp.MOSEK)
        print("end fitting")
        self.theta = theta.value
        self.b_val = b_var.value
        self.threshold_val = eta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b_val.tolist()
        model_params["threshold"] = self.threshold_val.tolist()
        return model_params

    def worst_distribution(self, X, y):
        # we follow Eq (27) of https://arxiv.org/pdf/2007.13982.pdf there
        sample_size, __ = X.shape
        per_loss = self.loss(X, y)
        perturb_loss = per_loss - (np.sum(self.b_val, axis = 1) - np.sum(self.b_val, axis = 0)) / sample_size
        indices = np.where(perturb_loss > self.threshold_val)[0]
        return {'sample_pts': [X[indices], y[indices]], 'weight': np.ones(len(indices)) / len(indices)}




class TV_DRO(base_DRO):
    # use a surrogate instead of putting the max but max over the empirical distribution
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.eps = 0

    def update(self, config = {}):
        if 'eps' in config.keys():
            self.eps = config['eps']
            assert (0 < self.eps < 1)
    
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        eta = cp.Variable()
        u = cp.Variable()
        if self.is_regression == 1 or self.is_regression == 2:
            loss = cp.sum(cp.pos(cp.power(X@theta - y, 2) - eta)) / (sample_size * (1 - self.eps)) + eta
            cons = [u >= cp.sum_squares(X[i] @ theta - y[i]) for i in range(sample_size)]
        else:
            # svm
            newy = 2*y-1
            loss = cp.sum(cp.pos(cp.pos(1 - cp.multiply(newy, X @ theta)) - eta)) / (sample_size * (1 - self.eps)) + eta
            cons = [u >= 1 - cp.multiply(newy[i], X[i] @ theta) for i in range(sample_size)]
            cons += [u >= 0]
        problem = cp.Problem(cp.Minimize(loss * (1 - self.eps) + self.eps * u), cons)
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value
        self.threshold_val = eta.value
        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["threshold"] = self.threshold_val.tolist()
        return model_params

    def worst_distribution(self, X, y):
        sample_size, __ = X.shape
        per_loss = self.loss(X, y)
        max_indice = np.argmax(per_loss)
        indices = np.where(per_loss > self.threshold_val)[0]
        total_indices = np.concatenate((np.array([max_indice]), indices)) 
        return {'sample_pts': [X[total_indices], y[total_indices]], 'weight': np.concatenate((np.array([self.eps]), (1 - self.eps) * np.ones(len(indices)) / len(indices)))}


class KL_DRO(base_DRO):
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.eps = 0

    def update(self, config = {}):
        if 'eps' in config.keys():
            self.eps = config['eps']
    
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        eta = cp.Variable(nonneg = True)
        t = cp.Variable()
        per_loss = cp.Variable(sample_size)
        epi_g = cp.Variable(sample_size)
        cons = [cp.sum(epi_g) <= sample_size]
        for i in range(sample_size):
            cons.append(cp.constraints.exponential.ExpCone(per_loss[i] - t, eta, epi_g[i]))
        cons.append(per_loss >= self.cvx_loss(X, y, theta))

        loss = t + eta * self.eps
        problem = cp.Problem(cp.Minimize(loss), cons)
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value
        self.dual_variable = eta.value
        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["dual"] = self.dual_variable
        return model_params

    def worst_distribution(self, X, y):
        sample_size, __ = X.shape
        per_loss = self.loss(X, y)
        weight = np.exp(per_loss / self.dual_variable)
        weight = weight / np.sum(weight)
        return {'sample_pts': [X, y], 'weight': weight}

    
