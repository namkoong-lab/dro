from .base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm
from typing import Dict, Any
import warnings

class WassersteinDROError(Exception):
    """Base exception class for errors in Chi-squared DRO model."""
    pass


class WassersteinDRO(BaseLinearDRO):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9004785
    # our distance metric is set as:
    # d((X_1, Y_1), (X_2, Y_2)) = (|cost_inv_transform @ (X_1 - X_2)|_p)^{square} + kappa |Y_1 - Y_2|, where cost_inv_transform = cost_matrix^{-1/2}
    # If we set kappa to be large enough to infinite, this is equivalent to saying that we do not allow changes in Y.
    # only linear regression is allow for the squared loss, i.e., the parameter: square = 2. 
    def __init__(self, input_dim: int, model_type: str):
        BaseLinearDRO.__init__(self, input_dim, model_type)

        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.eps = 0
        self.p = 1
        self.kappa = 1
        

    def update(self, config: Dict[str, Any]) -> None:
        if 'cost_matrix' in config.keys():
            cost_matrix = config['cost_matrix']
            if not isinstance(cost_matrix, np.ndarray) or cost_matrix != (self.input_dim, self.input_dim) or np.all(np.linalg.eigvals(cost_matrix) > 0):
                raise WassersteinDROError("Cost Adjust Matrix 'cost matrix' must be a PD matrix")
            self.cost_matrix = cost_matrix
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'eps' in config.keys():
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise WassersteinDROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)
        if 'p' in config.keys():
            p = config['p']
            if not isinstance(p, (float, int)) or p < 1:
                raise WassersteinDROError("Norm parameter 'p' must be float and larger than 1.")
            self.p = float(p)

        if 'kappa' in config.keys():
            kappa = config['kappa']
            if not isinstance(kappa, (float, int)) or kappa < 0:
                raise WassersteinDROError("Y-Robustness parameter 'kappa' must be a non-negative float.")
            elif kappa == 0 and self.model_type == 'ols':
                warnings.warn("DR-OLS does not support changes of Y in the ambiguity set")
            self.kappa = float(kappa)

        
    def penalization(self, theta):
        """
        corresponding to lambda in the standard WDRO regularization.
        """
        if self.p == 1:
            dual_norm = 'inf'
        else:
            dual_norm = 1 / (1 - 1 / self.p)
        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta, dual_norm)
        elif self.model_type == 'lad':
            # due to the \|\theta, -1\|_dual_norm penalization
            return cp.max(cp.norm(self.cost_inv_transform @ theta, dual_norm), 1 / self.kappa)




        
    def fit(self, X, y):
        sample_size, __ = X.shape
        theta = cp.Variable(self.input_dim)
        if self.model_type == 'ols':
            final_loss = cp.norm(X @ theta - y) / math.sqrt(sample_size) + math.sqrt(self.eps) * self.penalization(theta)

        else:
            if self.model_type in ['svm', 'logistic']:
                newy = 2*y - 1
                s = cp.Variable(sample_size)
                cons = [s >= self._cvx_loss(X, newy), s >= self._cvx_loss(X, -newy) - self.penalization(theta) * self.kappa]
                final_loss = cp.sum(s) / sample_size + self.eps * self.penalization(theta)
            else:
                # model type == 'lad' for general p.
                final_loss = cp.sum(self._cvx_loss(X, y)) / sample_size + self.eps * self.penalization(theta)

        problem = cp.Problem(cp.Minimize(final_loss), [])
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params
    def lipschitz_norm(self):
        """
        output the Lipschitz norm of the loss function
        it can be overridden externally
        """
        if self.model_type in ['svm', 'logistic', 'lad']:
            return 1
        

    def worst_distribution(self, X, y):
        # REQUIRED TO BE CALLED after solving the DRO problem
        # return a dict {"sample_pts": [np.array([pts_num, input_dim]), np.array(pts_num)], 'weight': np.array(pts_num)}

        if self.model_type == 'ols':
            return NotImplementedError
        else: # linear classification or regression with Lipschitz norm
            newy = 2*y - 1
            sample_size, __ = X.shape
            if self.p == 1:
                dual_norm = np.inf
            else:
                dual_norm = 1 / (1 - 1 / self.p)
            norm_theta = np.linalg.norm(self.theta, ord = dual_norm)
            # we denote the following case when we do not change Y.
            if self.kappa == 1000000:
            #TODO: unify OLS into the standard framework through lambda from Gao Rui's original paper.
            # \max ell(\theta;(X, Y)) - \lambda d((X, Y), (X_i, Y_i))
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
                if self.model_type in ['svm', 'logistic']:
                    # for general situations if we can change y, we apply Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf (SVM / logistic loss)
                    #eta is the theta in eq(27)
                    #gamma = 0 in that equation.
                    eta = cp.Variable(nonneg = True)
                    alpha = cp.Variable(sample_size, nonneg = True)

                    # svm / logistic L = 1
                    dual_loss = self.lipschitz_norm() * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self._loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self._loss(X, -y))) / sample_size
                    cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps]
                    problem = cp.Problem(cp.Maximize(dual_loss), cons)
                    problem.solve(solver = cp.MOSEK)
                    weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))

                    X = np.concatenate((X, X))
                    y = np.concatenate((y, -y))
                    return {'sample_pts': [X, y], 'weight': weight}
                elif self.model_type == 'lad':
                    # since we want the asymptotic with respect to n
                    gamma = 1 / sample_size
                    weight = np.zeros(sample_size + 1)
                    weight[1:-1] = np.ones(sample_size - 1) / sample_size
                    weight[0] = (1 - gamma) / sample_size
                    weight[-1] = gamma / sample_size
                    # solve the following perturbation problem
                    X_star = cp.Variable(self.input_dim)
                    y_star = cp.Variable()
                    cons = [cp.norm(sqrtm(self.cost_matrix) @ X_star, self.p) + self.kappa * cp.abs(y_star) <= 1]
                    dual_loss = cp.dot(self.theta, X_star) - y_star
                    problem = cp.Problem(cp.Maximize(dual_loss), cons)
                    problem.solve(solver = cp.MOSEK)
                    new_X = X[0] + self.eps * sample_size / gamma * X_star.value
                    new_y = y[0] + self.eps * sample_size / gamma * y_star.value
                    worst_X = np.vstack((X, new_X))
                    worst_y = np.hstack((y, new_y))
                    return {'sample_pts': [worst_X, worst_y], 'weight': weight}





class Wasserstein_DRO_satisficing(BaseLinearDRO):
    def __init__(self, input_dim: int, model_type: str):
        BaseLinearDRO.__init__(self, input_dim, model_type)
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
    

        if self.p == 1:
            dual_norm = 'inf'
        else:
            dual_norm = 1 / (1 - 1 / self.p)

        sample_size, __ = X.shape
        empirical_rmse = self.fit_oracle(X, y)
        TGT = self.target_ratio * empirical_rmse
        theta = cp.Variable(self.input_dim)
        cons = [TGT >= cp.sum(self._cvx_loss(X, y)) * sample_size]
        if self.model_type == 'lad':
            obj = cp.norm(cp.hstack([theta, -1]), dual_norm)
        # TODO: check it is approximation or exact
        elif self.model_type in ['ols', 'svm', 'logistic']:
            obj = cp.norm(theta, dual_norm)


        problem = cp.Problem(cp.Minimize(obj), cons)
        problem.solve(solver = cp.MOSEK)
        self.theta = theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params


    def fit_depreciate(self, X, y):
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

        






        

        
