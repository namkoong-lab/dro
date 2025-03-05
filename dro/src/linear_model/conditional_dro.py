from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any, Optional
import cvxpy as cp

class ConditionalCVaRDROError(Exception):
    """Exception class for errors in Marginal CVaR DRO model."""
    pass


class ConditionalCVaRDRO(BaseLinearDRO):
    """
    Conditional CVaR DRO model following Theorem 2 in:
    with alpha(x) to be beta^T x for simplicity
    alpha corresponds to Gamma in the paper;

    Reference: <https://arxiv.org/pdf/2209.01754.pdf>
    """
    def __init__(self, input_dim: int, is_regression: int):
        BaseLinearDRO.__init__(self, input_dim, is_regression)
        self.alpha = 1
        self.control_name = None
    
    def update(self, config = {}):
        if 'control_name' in config.keys():
            assert all(0 <= x <= self.input_dim - 1 for x in config['control_name'])
            self.control_name = config['control_name']
        if 'alpha' in config.keys():
            assert (config['alpha'] >= 1)
            self.alpha = config['alpha']
    
    def fit(self, X, y):
        sample_size, __ = X.shape
        if self.control_name is not None:
            control_X = X[:,self.control_name]
        else:
            control_X = X
        
        theta = cp.Variable(self.input_dim)

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        beta = cp.Variable(len(control_X[0]))
        cost = cp.sum(self._cvx_loss(X, y, theta, b)) / (self.alpha) + (self.alpha - 1/self.alpha) * cp.sum(cp.pos(self._cvx_loss(X, y, theta, b) - control_X @ beta)) + (1 - 1 / self.alpha) * cp.sum(control_X @ beta) 

        prob = cp.Problem(cp.Minimize(cost / sample_size))
        
        prob.solve(solver = self.solver)
        self.theta = theta.value

        model_params = {}
        model_params['theta'] = self.theta.reshape(-1).tolist()
        model_params["b"]: self.b


        return model_params
