from dro.src.linear_model.base import BaseLinearDRO
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.stats import invwishart, multivariate_normal
from typing import Dict, Any

# methods that centered at parametric distribution
## we set mixture Gaussian as the parametric distribution.

class BayesianDROError(Exception):
    """Base exception class for errors in BayesianDRO model."""
    pass

class BayesianDRO(BaseLinearDRO):
    """
    
    This model minimizes a Bayesian version for regression and other types of losses

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for KL-DRO.
        distance_type (str, default = 'KL'): distance type in DRO model, default = 'KL'. Also support 'chi2'.
        distribution_class (str, default = 'Gaussian'): parametric distribution class, default = 'Gaussian' (used in 'lad' and 'ols'). Also support 'Exponential' (used in 'newsvendor'). 
        posterior_param_num (int: default = 1). The number of posterior parameters sampled.
        posterior_sample_ratio (float): The number of samples in MC sample relative to the total sample number for each posterior parameter.
    
    Reference: <https://epubs.siam.org/doi/10.1137/21M1465548>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', eps: float = 0.0):
        """
        Initialize the Bayesian-DRO model with specified input dimension and model type.

        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)        
        self.eps = eps
        self.posterior_param_num = 1
        self.posterior_sample_ratio = 1
        self.distribution_class = 'Gaussian'
        self.distance_type = 'KL'

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing 'distance_type', 'distribution_class', 'eps', 'posterior_sample_ratio', 'posterior_param_num' keys for robustness and optimization 
        Raises:
            BayesianDROError: If any of the configs does not fall into its domain.
        """
        if 'distance_type' in config.keys():
            distance_type = config['distance_type']
            if distance_type not in ['chi2', 'KL']:
                raise BayesianDROError("Distance type can only be chosen from 'KL' and 'chi2'.")
            self.distance_type = distance_type

        if 'distribution_class' in config.keys():
            
            dist_class = config['distribution_class']
            
            self.distribution_class = dist_class
        if 'eps' in config.keys():
            self.eps = config['eps']
        if 'posterior_sample_ratio' in config.keys():
            self.posterior_sample_ratio = config['posterior_sample_ratio']
        if 'posterior_param_num' in config.keys():
            assert (config['posterior_param_num'] < 100)
            self.posterior_param_num = config['posterior_param_num']
    

    



    def resample(self, X, y):
        """Obtain Monte Carlo samples from the data,
        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Return:
            List[np.ndarray]: resampled arrays of X 
            List[np.ndarray]: resampled arrays of Y
        """
        sample_size, __ = X.shape
        if self.distribution_class == 'Gaussian':
            mu_prior = np.zeros(self.input_dim + 1)  # Prior mean
            kappa_prior = 1.0  # Prior strength (low = weak prior)
            psi_prior = np.eye(self.input_dim + 1)  # Prior scale matrix
            nu_prior = self.input_dim + 2  # Prior degrees of freedom
            n_samples, n_features = X.shape
            stacked_data = np.hstack((X, y.reshape(-1, 1)))  # Stack X and y together

            # Compute sample mean and scatter matrix
            sample_mean = np.mean(stacked_data, axis=0)
            scatter_matrix = np.cov(stacked_data.T, bias=True) * n_samples  # Scatter matrix S

            # Posterior updates
            kappa_post = kappa_prior + n_samples
            nu_post = nu_prior + n_samples
            mu_post = (kappa_prior * mu_prior + n_samples * sample_mean) / kappa_post
            psi_post = psi_prior + scatter_matrix + (kappa_prior * n_samples / kappa_post) * np.outer((sample_mean - mu_prior), (sample_mean - mu_prior))

            cov = invwishart.rvs(df = nu_post, scale = psi_post, size = self.posterior_sample_ratio * sample_size)
            mu = np.array([multivariate_normal.rvs(mean = mu_post, cov = sigma / kappa_post) for sigma in cov])

            X, y = [], []
            for i in range(self.posterior_param_num):
                print(f"sample lool {i}")

                new_data = np.random.multivariate_normal(mean = mu[i], cov = cov[i], size = int(self.posterior_sample_ratio * sample_size))
                X.append(new_data[:, 0:-1])
                y.append(new_data[:,-1])                
        elif self.distribution_class == 'Exponential':
            # use the gamma(1, 1) as the conjugate prior
            prior_shape, prior_rate = 1, 1
            update_shape = prior_shape + len(y)
            update_rate = prior_rate + sum(y)

            X, y = [], []
            sampled_theta = np.random.gamma(update_shape, 1 / update_rate, self.posterior_param_num)
            for i in range(self.posterior_param_num):
                new_data = np.random.exponential(1 / sampled_theta[i], int(self.posterior_sample_ratio * sample_size))
                X.append(np.ones((int(self.posterior_sample_ratio * sample_size), 1)))
                y.append(new_data)

        else:
            raise NotImplementedError

        return X, y


    def fit(self, X, y, resample = True):
        """Fit the model using CVXPY to solve the bayesian robust optimization problem with KL constraint.

        Args:                
            resample (bool, default = True): Whether to conduct posterior sample.
            if resample == True:
                X (np.ndarray): Original feature matrix with shape (n_samples, n_features).
                y (np.ndarray): Original target vector with shape (n_samples,).
            else:
                X (List[np.ndarray]): Processed feature matrix, each with shape (n_samples, n_features).
                y (List[np.ndarray]): Processed target vector, each with shape (n_samples,).
        
        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' keys.

        Raises:
            BayesianDROError: If the optimization problem fails to solve.
        """

        if resample == True:
            sample_size, __ = X.shape
            X, y = self.resample(X, y)

        else:
            sample_size = len(X[0])

        assert len(X) == self.posterior_param_num
    
        sample_size = self.posterior_sample_ratio * sample_size

        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        t = cp.Variable(self.posterior_param_num)
        cons = []


        if self.distance_type == 'chi2':
            eta = cp.Variable(self.posterior_param_num)
            for i in range(self.posterior_param_num):
                cons.append(np.sqrt(1 + self.eps) / np.sqrt(sample_size) * 
                cp.norm(cp.pos(self._cvx_loss(X[i], y[i], theta, b) - eta[i]), 2) + eta[i] <= t[i])
            loss = cp.sum(t)


        elif self.distance_type == 'KL':
            u = cp.Variable(self.posterior_param_num)
            eta = cp.Variable(nonneg = True)
            per_loss = cp.Variable([self.posterior_param_num, sample_size])
            epi_g = cp.Variable([self.posterior_param_num, sample_size])

            for i in range(self.posterior_param_num):
                cons.append(per_loss[i] >= self._cvx_loss(X[i], y[i], theta, b))
                    # if self.is_regression == True:
                    #     cons.append(per_loss[i][j] >= (y[i][j] - X[i][j] @ theta) ** 2)
                    # else:
                    #     # svm loss
                    #     cons.append(per_loss[i][j] >= cp.pos(1 - cp.multiply(y[i][j], X[i][j] @ theta)))

            for i in range(self.posterior_param_num):
                cons += [cp.sum(epi_g[i]) <= sample_size * eta]
                cons += [t[i] + eta * self.eps <= u[i]]
            for i in range(self.posterior_param_num):
                for j in range(sample_size):
                    cons.append(cp.constraints.exponential.ExpCone(per_loss[i][j] - t[i], eta, epi_g[i][j]))
            loss = cp.sum(u)

        problem = cp.Problem(cp.Minimize(loss), cons)

    
        try:
            problem.solve(solver = self.solver)
            self.theta = theta.value

        except cp.SolverError as e:
            raise BayesianDROError(f"Optimization failed to solve using {self.solver}.") from e

        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params['b'] = self.b
        return model_params

    def _cvx_loss(self, X: cp.Expression, y: cp.Expression, theta: cp.Expression, b: cp.Expression) -> cp.Expression:
        """
        Augment the CVXPY loss expression for the BayesianDRO model
        """
        if self.model_type == 'svm':
            return cp.pos(1 - cp.multiply(y, X @ theta + b))
        elif self.model_type == 'logistic':
            return cp.logistic(-cp.multiply(y, X @ theta + b))
        elif self.model_type == 'ols':
            return cp.power(y - X @ theta - b, 2)
        elif self.model_type == 'lad':
            return cp.abs(y - X @ theta - b)
        elif self.model_type == 'newsvendor':
            return 1 * cp.pos(y - X @ theta) + cp.pos(X @ theta - y)
        else:
            raise NotImplementedError("CVXPY loss not implemented for the specified model_type value.")



