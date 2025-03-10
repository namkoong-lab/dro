from dro.src.linear_model.base import BaseLinearDRO
import pandas as pd
import numpy as np
import cvxpy as cp
import pymc3 as pm
from scipy.stats import multivariate_normal

# methods that centered at parametric distribution
## we set mixture Gaussian as the parametric distribution.

class Bayesian_KL_DRO(BaseLinearDRO):
    # https://arxiv.org/abs/2112.08625
    def __init__(self, input_dim, is_regression):
        BaseLinearDRO.__init__(self, input_dim, is_regression)
        self.eps = 0.5
        self.posterior_param_num = 1
        self.posterior_sample_ratio = 1
        self.model_class = 'Gaussian'

    
    def update(self, config = {}):
        if 'model_class' in config.keys():
            self.model_class = config['model_class']
        if 'eps' in config.keys():
            self.eps = config['eps']
        if 'posterior_sample_ratio' in config.keys():
            self.posterior_sample_ratio = config['posterior_sample_ratio']
        if 'posterior_param_num' in config.keys():
            assert (config['posterior_param'] < 100)
            self.posterior_param_num = config['posterior_param_num']

    def fit(self, X, y):
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
        # we obtain a new X, y with self.posterior_param_num * sample_size * input_dim
        sample_size = self.posterior_sample_ratio * sample_size

        theta = cp.Variable(self.input_dim)
        eta = cp.Variable(nonneg = True)
        t = cp.Variable(self.posterior_param_num)
        per_loss = cp.Variable([self.posterior_param_num, sample_size])
        epi_g = cp.Variable([self.posterior_param_num, sample_size])
        cons = []
        for i in range(self.posterior_param_num):
            cons += [cp.sum(epi_g[i]) <= sample_size]
        for i in range(self.posterior_param_num):
            for j in range(sample_size):
                cons.append(cp.constraints.exponential.ExpCone(per_loss[i][j] - t[i], eta, epi_g[i][j]))
                if self.is_regression == True:
                    cons.append(per_loss[i][j] >= (y[i][j] - X[i][j] @ theta) ** 2)
                else:
                    # svm loss
                    cons.append(per_loss[i][j] >= cp.pos(1 - cp.multiply(y[i][j], X[i][j] @ theta)))
        loss = cp.sum(t) / self.posterior_param_num + eta * self.eps
        problem = cp.Problem(cp.Minimize(loss), cons)
        problem.solve(solver = self.solver, verbose=True)
        self.theta = theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params

