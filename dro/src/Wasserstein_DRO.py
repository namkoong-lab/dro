from .base import *
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm

class Wasserstein_DRO(base_DRO):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9004785
    #fix the uncertainty in Y
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.eps = 0
        self.p = 1
        self.kappa = 1

    def update(self, config = {}):
        if 'cost_matrix' in config.keys():
            self.cost_matrix = config['cost_matrix']
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'eps' in config.keys():
            self.eps = config['eps']
        # the following two are only used in SVM-wasserstein
        if 'p' in config.keys():
            self.p = config['p']
        if 'kappa' in config.keys():
            self.kappa = config['kappa']
        
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        t1 = cp.Variable(nonneg = True)
        t2 = cp.Variable(nonneg = True)
        y2 = cp.Variable()
        if self.is_regression == 1 or self.is_regression == 2:
            cons = [t1 >= cp.norm(X @ theta - y) / math.sqrt(sample_size)]
            if self.eps > 0:
                theta_transform = cp.Variable(self.input_dim)
                cons += [t2 >= math.sqrt(self.eps) * cp.norm(theta_transform), theta_transform == self.cost_inv_transform @ theta]
            final_loss = t1 + t2

        else:
            newy = 2*y - 1
            if not (self.is_regression is 0):
                # logistic
                cons = [t1 >= cp.sum(cp.logistic(-cp.multiply(newy, X @ theta))) / sample_size]
                if self.eps > 0:
                    theta_transform = cp.Variable(self.input_dim)
                    cons += [t2 >= self.eps * cp.norm(theta_transform), theta_transform == self.cost_inv_transform @ theta]
                final_loss = t1 + t2
            else:
                # svm https://jmlr.org/papers/volume20/17-633/17-633.pdf 
                # we do not fix to be the l2 loss (coro 15)
                s = cp.Variable(sample_size)
                cons = [s >= 1 - cp.multiply(newy, X@theta), s >= 0, s >= 1 + cp.multiply(newy, X@theta) - t1 * self.kappa]
                if self.p == 1:
                    dual_norm = 'inf'
                else:
                    dual_norm = 1 / (1 - 1 / self.p)
                final_loss = cp.sum(s) / sample_size + self.eps * cp.norm(theta, dual_norm)

        problem = cp.Problem(cp.Minimize(final_loss), cons)
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params

    def worst_distribution(self, X, y):
        # REQUIRED TO BE CALLED after solving the DRO problem
        # return a dict {"sample_pts": [np.array([pts_num, input_dim]), np.array(pts_num)], 'weight': np.array(pts_num)}

        if self.is_regression == 1 or self.is_regression == 2:
            return NotImplementedError
        else:
            newy = 2*y - 1
            sample_size, __ = X.shape
            if self.p == 1:
                dual_norm = np.inf
            else:
                dual_norm = 1 / (1 - 1 / self.p)
            norm_theta = np.linalg.norm(self.theta, ord = dual_norm)
            if self.kappa == 1000000:
            # not change y, we directly consider RMK 5.2 in https://arxiv.org/pdf/2308.05414.pdf, here norm_theta is lambda* there.
                new_X = np.zeros((sample_size, self.input_dim))
                for i in range(sample_size):
                    var_x = cp.Variable(self.input_dim)
                    obj = 1 - newy[i] * var_x @ self.theta - norm_theta * cp.sum_squares(var_x - X[i])
                    problem = cp.Problem(cp.Maximize(obj))
                    problem.solve(solver = cp.MOSEK)
                    
                    if 1 - newy[i] * var_x.value @ self.theta < 0:
                        new_X[i] = X[i]
                    else:
                        new_X[i] = var_x.value
                return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}
            
            else:
                # for general situations if we can change y, we apply Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf (SVM / logistic loss)
                #eta is the theta in eq(27)
                eta = cp.Variable(nonneg = True)
                alpha = cp.Variable(sample_size, nonneg = True)

                # svm / logistic L = 1
                L = 1
                dual_loss = L * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self.loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self.loss(X, y_flip))) / sample_size
                cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps]
                problem = cp.Problem(cp.Maximize(dual_loss), cons)
                problem.solve(solver = cp.MOSEK)
                weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
                X = np.concatenate((X, X))
                y = np.concatenate((y, y_flip))
                return {'sample_pts': [X, y], 'weight': weight}


class Wasserstein_DRO_satisficing(base_DRO):
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        # target that the robust error need to achieve
        self.target_ratio = 1 / 0.8
        self.eps = 100
        self.p = 1
        self.kappa = 1

    def update(self, config = {}):
        if 'cost_matrix' in config.keys():
            self.cost_matrix = config['cost_matrix']
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'target_ratio' in config.keys():
            assert (config['target_ratio'] >= 1)
            self.target_ratio = config['target_ratio']
        # the following two are only used in SVM-wasserstein
        if 'p' in config.keys():
            self.p = config['p']
        if 'kappa' in config.keys():
            self.kappa = config['kappa']
    
    def fit(self, X, y):
        iter_num = 10
        # determine the empirical obj
        self.eps = 0
        if self.is_regression == 0:
            y = 2 * y - 1
        empirical_rmse = self.fit_oracle(X, y)
        TGT = self.target_ratio * empirical_rmse
        # print('tgt', TGT)
        self.eps = 100
        assert (self.fit_oracle(X, y) > TGT)
        eps_lower, eps_upper = 0, self.eps      
        # binary search and find the maximum eps, such that RMSE + eps theta <= tau  
        for i in range(iter_num):
            self.eps = (eps_lower + eps_upper)/2
            if self.fit_oracle(X, y) > TGT:
                eps_upper = self.eps
            else:
                eps_lower = self.eps
        
        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params
            

    def fit_oracle(self, X, y):
        # same as the Wasserstein_DRO formulation above
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        t1 = cp.Variable(nonneg = True)
        t2 = cp.Variable(nonneg = True)
        y2 = cp.Variable()
        if self.is_regression == 1 or self.is_regression == 2:
            cons = [t1 >= cp.norm(X @ theta - y) / math.sqrt(sample_size)]
            if self.eps > 0:
                theta_transform = cp.Variable(self.input_dim)
                cons += [t2 >= math.sqrt(self.eps) * cp.norm(theta_transform), theta_transform == self.cost_inv_transform @ theta]
            final_loss = t1 + t2

        else:
            if not (self.is_regression is 0):
                # logistic
                cons = [t1 >= cp.sum(cp.logistic(-cp.multiply(y, X @ theta))) / sample_size]
                if self.eps > 0:
                    theta_transform = cp.Variable(self.input_dim)
                    cons += [t2 >= self.eps * cp.norm(theta_transform), theta_transform == self.cost_inv_transform @ theta]
                final_loss = t1 + t2
            else:
                # svm https://jmlr.org/papers/volume20/17-633/17-633.pdf 
                # we do not fix to be the l2 loss (coro 15)
                s = cp.Variable(sample_size)
                cons = [s >= 1 - cp.multiply(y, X@theta), s >= 0, s >= 1 + cp.multiply(y, X@theta) - t1 * self.kappa]
                if self.p == 1:
                    dual_norm = 'inf'
                else:
                    dual_norm = 1 / (1 - 1 / self.p)
                final_loss = cp.sum(s) / sample_size + self.eps * cp.norm(theta, dual_norm)
                
        problem = cp.Problem(cp.Minimize(final_loss), cons)
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value

        return problem.value
        
    
    def worst_distribution(self, X, y):
        # REQUIRED TO BE CALLED after solving the DRO problem
        # return a dict {"sample_pts": [np.array([pts_num, input_dim]), np.array(pts_num)], 'weight': np.array(pts_num)}

        if self.is_regression == 1 or self.is_regression == 2:
            return NotImplementedError
        else:
            sample_size, __ = X.shape
            if self.p == 1:
                dual_norm = np.inf
            else:
                dual_norm = 1 / (1 - 1 / self.p)
            norm_theta = np.linalg.norm(self.theta, ord = dual_norm)
            if self.kappa == 1000000:
                y = 2*y-1
            # not change y, we directly consider RMK 5.2 in https://arxiv.org/pdf/2308.05414.pdf, here norm_theta is lambda* there.
                new_X = np.zeros((sample_size, self.input_dim))
                for i in range(sample_size):
                    var_x = cp.Variable(self.input_dim)
                    obj = 1 - y[i] * var_x @ self.theta - norm_theta * cp.sum_squares(var_x - X[i])
                    problem = cp.Problem(cp.Maximize(obj))
                    problem.solve(solver = cp.MOSEK)
                    
                    if 1 - y[i] * var_x.value @ self.theta < 0:
                        new_X[i] = X[i]
                    else:
                        new_X[i] = var_x.value
                return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}
            
            else:
                # for general situations if we can change y, we apply Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf (SVM / logistic loss)
                #eta is the theta in eq(27)
                y_flip = -y
                eta = cp.Variable(nonneg = True)
                alpha = cp.Variable(sample_size, nonneg = True)

                # svm / logistic L = 1
                L = 1
                dual_loss = L * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self.loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self.loss(X, y_flip))) / sample_size
                cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps]
                problem = cp.Problem(cp.Maximize(dual_loss), cons)
                problem.solve(solver = cp.MOSEK)
                weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
                X = np.concatenate((X, X))
                y = np.concatenate((y, y_flip))
                return {'sample_pts': [X, y], 'weight': weight}

        

class Wasserstein_DRO_aug(base_DRO):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9004785
    #fix the uncertainty in Y
    def __init__(self, input_dim, is_regression):
        base_DRO.__init__(self, input_dim, is_regression)
        self.cost_matrix = np.eye(input_dim + 1)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))

        self.eps = 0
    def update(self, config = {}):
        if 'cost_matrix' in config.keys():
            self.cost_matrix = config['cost_matrix']
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'eps' in config.keys():
            self.eps = config['eps']

        
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        t1 = cp.Variable(nonneg = True)
        t2 = cp.Variable(nonneg = True)
        y2 = cp.Variable()
        if self.is_regression == 1 or self.is_regression == 2:
            cons = [t1 >= cp.norm(X @ theta - y) / math.sqrt(sample_size)]
            if self.eps > 0:
                theta_transform = cp.Variable(self.input_dim + 1)
                cons += [t2 >= math.sqrt(self.eps) * cp.norm(theta_transform), theta_transform == self.cost_inv_transform @ cp.hstack([theta, -1])]
        else:
            y = 2*y - 1
            cons = [t1 >= cp.sum(cp.logistic(-cp.multiply(y, X @ theta))) / sample_size]
            if self.eps > 0: 
                theta_transform = cp.Variable(self.input_dim + 1)
                cons += [t2 >= self.eps * cp.norm(theta_transform), theta_transform == self.cost_inv_transform @ cp.hstack([theta, -1])]
        final_loss = t1 + t2
        problem = cp.Problem(cp.Minimize(final_loss), cons)
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params





        

        
