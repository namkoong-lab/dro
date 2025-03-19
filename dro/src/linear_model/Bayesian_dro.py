from dro.src.linear_model.base import BaseLinearDRO
import pandas as pd
import numpy as np
import cvxpy as cp
import pymc3 as pm
from scipy.stats import multivariate_normal
from typing import Dict, Any

# methods that centered at parametric distribution
## we set mixture Gaussian as the parametric distribution.

class BayesianDROError(Exception):
    """Base exception class for errors in BayesianDRO model."""
    pass

class Bayesian_DRO(BaseLinearDRO):
    """
    
    This model minimizes a Bayesian version for regression and other types of losses

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for KL-DRO.
        distance_type (str, default = 'KL'): distance type in DRO model, default = 'KL'. Also support 'chi2'.

    
    Reference: <https://epubs.siam.org/doi/10.1137/21M1465548>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', eps: float = 0.0):
        """
        Initialize the Bayesian-DRO model with specified input dimension and model type.

        Args:

        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)        
        self.eps = eps
        self.posterior_param_num = 1
        self.posterior_sample_ratio = 1
        self.distribution_class = 'Gaussian'
        self.distance_type = 'KL'

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.
        
        """

        if 'distribution_class' in config.keys():
            self.distribution_class = config['distribution_class']
        if 'eps' in config.keys():
            self.eps = config['eps']
        if 'posterior_sample_ratio' in config.keys():
            self.posterior_sample_ratio = config['posterior_sample_ratio']
        if 'posterior_param_num' in config.keys():
            assert (config['posterior_param'] < 100)
            self.posterior_param_num = config['posterior_param_num']
    
    def resample(self, X, y):
        """
        obtain a new X, y with self.posterior_param_num * sample_size * input_dim
        """
        sample_size, __ = X.shape
        if self.model_class == 'Gaussian':
            stacked_data = np.hstack((X, y.reshape(-1, 1)))
            cov = np.cov(stacked_data.T)
            with pm.Model() as model:
                mu = pm.Normal('mu', mu = 0, sd = 10, shape = self.input_dim + 1)
                likelihood = pm.MvNormal('likelihood', mu = mu, cov = np.eye(self.input_dim + 1), observed = stacked_data)
                # Sample from the posterior distribution using MCMC
                # step = pm.Metropolis()
                trace = pm.sample(int(self.posterior_sample_ratio* sample_size), tune = int(0.05 * sample_size), cores = 1)


            posterior_samples_mu = trace['mu']
            random_sample_indices = np.random.choice(range(self.posterior_param_num), size = self.posterior_param_num, replace = False)
            sampled_mu_vectors = posterior_samples_mu[random_sample_indices]

            X, y = [], []
            for i in range(self.posterior_param_num):
                print(f"sample lool {i}")
                new_data = np.random.multivariate_normal(mean = sampled_mu_vectors[i], cov = cov, size = int(self.posterior_sample_ratio * sample_size))
                X.append(new_data[:, 0:-1])
                y.append(new_data[:,-1])                
        else:
            raise NotImplementedError

        return X, y


    def fit(self, X, y):
        X, y = self.resample(X, y)

        sample_size = self.posterior_sample_ratio * sample_size

        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        eta = cp.Variable(nonneg = True)
        t = cp.Variable(self.posterior_param_num)
        per_loss = cp.Variable([self.posterior_param_num, sample_size])
        epi_g = cp.Variable([self.posterior_param_num, sample_size])
        cons = []

        cons.append(per_loss >= self._cvx_loss(X, y, theta, b))
                # if self.is_regression == True:
                #     cons.append(per_loss[i][j] >= (y[i][j] - X[i][j] @ theta) ** 2)
                # else:
                #     # svm loss
                #     cons.append(per_loss[i][j] >= cp.pos(1 - cp.multiply(y[i][j], X[i][j] @ theta)))

        for i in range(self.posterior_param_num):
            cons += [cp.sum(epi_g[i]) <= sample_size]
        for i in range(self.posterior_param_num):
            for j in range(sample_size):
                cons.append(cp.constraints.exponential.ExpCone(per_loss[i][j] - t[i], eta, epi_g[i][j]))

        loss = cp.sum(t) / self.posterior_param_num + eta * self.eps
        problem = cp.Problem(cp.Minimize(loss), cons)

    
        try:
            problem.solve(solver = self.solver, verbose=True)
            self.theta = theta.value

        except cp.SolverError as e:
            raise BayesianDROError(f"Optimization failed to solve using {self.solver}.") from e

        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params['b'] = self.b
        return model_params

