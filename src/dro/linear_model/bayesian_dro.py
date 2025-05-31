from .base import BaseLinearDRO
from .base import ParameterError, InstallError
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
        
        if input_dim <= 0:
            raise ParameterError("Input dimension must be a positive integer.")
        if model_type not in {'svm', 'logistic', 'ols', 'lad', 'newsvendor'}:
            raise ParameterError(f"Unsupported model_type: {model_type}. Default supported types are svm, logistic, ols, lad, newsvendor. Please define your personalized loss.")
        self.input_dim = input_dim
        self.model_type = model_type
        self.fit_intercept = fit_intercept
        self.b = 0

        if solver not in cp.installed_solvers():
            raise InstallError(f"Unsupported solver {solver}. It does not exist in your package. Please change the solver or install {solver}.")
        self.solver = solver      
        self.eps = eps
        self.posterior_param_num = 1
        self.posterior_sample_ratio = 1
        self.distribution_class = 'Gaussian'
        if distance_type not in ['chi2', 'KL']:
            raise BayesianDROError("Distance type can only be chosen from 'KL' and 'chi2'.")
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
        if 'distance_type' in config.keys():
            distance_type = config['distance_type']
            if distance_type not in ['chi2', 'KL']:
                raise BayesianDROError("Distance type can only be chosen from 'KL' and 'chi2'.")
            self.distance_type = distance_type

        if 'distribution_class' in config.keys():
            
            dist_class = config['distribution_class']
            if dist_class not in ['Gaussian', 'Exponential']:
                raise BayesianDROError("Distributoin class can only be chosen from 'Gaussian' and 'Exponential'.")            
            self.distribution_class = dist_class

        if 'eps' in config.keys():
            self.eps = config['eps']
        if 'posterior_sample_ratio' in config.keys():
            self.posterior_sample_ratio = config['posterior_sample_ratio']
        if 'posterior_param_num' in config.keys():
            assert (config['posterior_param_num'] < 100)
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
            >>> # After fitting a BayesianDRO model
            >>> X_new, y_new = model.resample(X_train, y_train)
            >>> print(X_new.shape)  # (n_params, 1000, 10) if sample_size=1000, input_dim=10
            >>> print(y_new.shape)  # (n_params, 1000)
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
            >>> model = BayesianDRO(input_dim=10, eps=0.1)
            >>> X_train = np.random.randn(100, 10)
            >>> y_train = np.sign(np.random.randn(100))
            >>> params = model.fit(X_train, y_train)
            >>> print(params["theta"])  # e.g., [0.5, -1.2, ..., 0.8]
            >>> print(params["b"])      # e.g., 0.3
        """

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise BayesianDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")

        if self.model_type in {'svm', 'logistic'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise BayesianDROError("classification labels not in {-1, +1}")

        sample_size = X.shape[0]
        X, y = self.resample(X, y)        
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



