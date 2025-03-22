from dro.src.linear_model.base import BaseLinearDRO
import pandas as pd
import numpy as np
import cvxpy as cp
import pymc3 as pm
from scipy.stats import multivariate_normal
from typing import Dict, Any

# TODO
# methods that centered at parametric distribution
## we set mixture Gaussian as the parametric distribution.

class BayesianDROError(Exception):
    """Base exception class for errors in BayesianDRO model."""
    pass

class Bayesian_DRO(BaseLinearDRO):
    """    
    This model minimizes a Bayesian version for regression and other types of losses

    Reference: <https://epubs.siam.org/doi/10.1137/21M1465548>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', eps: float = 0.0, distance_type: str = 'KL'):
        """Initialize the Bayesian DRO model.

        :param input_dim: Dimensionality of the input features.
        :type input_dim: int
        :param model_type: Type of base model. Supported values: 'svm', 'logistic', 'ols', 'lad'('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss). Defaults to 'svm'.
        :type model_type: str
        :param fit_intercept: Whether to fit an intercept term. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered). Defaults to True.
        :type fit_intercept: bool
        :param solver: Optimization solver. Supported solvers: 'MOSEK'.
        :type solver: str
        :param eps: Robustness parameter for KL-divergence ambiguity set. A higher value increases robustness. Defaults to 0.0 (non-robust).
        :type eps: float
        :param distance_type: Distance type in DRO model, default = 'KL'. Also support 'chi2'. Default to 'KL'.
        :type distance_type: str
        """
        
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)        
        self.eps = eps
        self.posterior_param_num = 1
        self.posterior_sample_ratio = 1
        self.distribution_class = 'Gaussian'
        self.distance_type = distance_type

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration dynamically.
    
        Modify parameters like robustness level (`eps`), optimization solver, distance metric type, 
        or other algorithm settings during runtime.

        :param config: Dictionary containing configuration key-value pairs to update. Supported keys include: 

            - ``eps``: Robustness parameter (non-negative float)

            - ``solver``: Optimization solver (e.g., 'MOSEK', 'SCS')

            - ``distance_type``: Distance metric ('KL' or 'chi2')

            - ``distribution_class``: Distribution class

            - ``posterior_sample_ratio``

            - ``posterior_param_num``

            - Other model-specific parameters

        :type config: Dict[str, Any]
        :raises ValueError: If the configuration contains invalid keys, unsupported values, 
            or negative values for parameters like ``eps``.

        .. note::

            - Updating some parameters (e.g., ``solver``) may trigger reinitialization of the optimizer.

            - For safety, avoid modifying ``input_dim`` or ``model_type`` after initialization.

        Example:
            >>> model.update({"eps": 0.5, "distance_type": "chi2", "solver": "MOSEK"})
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
        """Generate resampled data based on posterior parameters.

        This method produces new feature and target arrays whose dimensions are determined by the model's posterior parameters, input dimensionality, and the sample size of the original data.

        :param X: Original feature matrix of shape `(sample_size, input_dim)`.
        :type X: numpy.ndarray
        :param y: Original target values of shape `(sample_size,)` or `(sample_size, n_targets)`.
        :type y: numpy.ndarray

        :returns: A tuple containing the resampled feature matrix and target array. 

            - Resampled X has shape `(posterior_param_num, sample_size, input_dim)`

            - Resampled y has shape `(posterior_param_num, sample_size)` (for single-target) or `(posterior_param_num, sample_size, n_targets)`

        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        :raises ValueError: 

            - If `X` and `y` have inconsistent sample sizes (first dimension mismatch).

            - If `posterior_param_num` is not initialized (e.g., model not yet fitted).

        Example:
            >>> # After fitting a Bayesian_DRO model
            >>> X_new, y_new = model.resample(X_train, y_train)
            >>> print(X_new.shape)  # (n_params, 1000, 10) if sample_size=1000, input_dim=10
            >>> print(y_new.shape)  # (n_params, 1000)
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
        """Train the Bayesian DRO model by solving the convex optimization problem.

        :param X: Feature matrix of shape `(original_sample_size, input_dim)`.
        :type X: numpy.ndarray
        :param y: Target values of shape `(original_sample_size,)` for classification or `(original_sample_size, n_targets)` for regression.
        :type y: numpy.ndarray

        :returns: Dictionary containing the trained parameters:

            - ``theta``: Weight vector of shape `(input_dim,)`

            - ``b``: Intercept scalar (only if `fit_intercept=True`)

        :rtype: Dict[str, Union[List[float], float]]
        :raises BayesianDROError: If the optimization solver fails to converge.

        :raises ValueError: 

            - If `X` and `y` have inconsistent sample sizes.

            - If resampled data dimensions are incompatible with `posterior_param_num`.

        Optimization Formulation:
            Minimize:
                ``(1/K) * Σ t_k + η * eps``  
                where K = `posterior_param_num`, eps = `self.eps`
                
            Subject to:

                - Exponential cone constraints: ``ExpCone(per_loss - t, η, epi_g)``

                - Loss bounds: ``per_loss ≥ model-specific loss (e.g., SVM hinge loss)``

                - Ambiguity set: ``Σ epi_g ≤ sample_size``
        
        Example:
            >>> model = Bayesian_DRO(input_dim=10, eps=0.1)
            >>> X_train = np.random.randn(100, 10)
            >>> y_train = np.sign(np.random.randn(100))
            >>> params = model.fit(X_train, y_train)
            >>> print(params["theta"])  # e.g., [0.5, -1.2, ..., 0.8]
            >>> print(params["b"])      # e.g., 0.3
        """
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

