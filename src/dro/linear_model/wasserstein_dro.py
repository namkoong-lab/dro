from .base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm
from typing import Dict, Any
import warnings
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import Nystroem



class WassersteinDROError(Exception):
    """Base exception class for errors in Wasserstein DRO model."""
    pass


class WassersteinDRO(BaseLinearDRO):
    r"""Wasserstein Distributionally Robust Optimization (WDRO) model
    
    This model minimizes a Wasserstein-robust loss function for both regression and classification.

    The Wasserstein distance is defined as the minimum probability coupling of two distributions for the distance metric:

    .. math::
        d((X_1, Y_1), (X_2, Y_2)) = (\|\Sigma^{1/2} (X_1 - X_2)\|_p)^{square} + \kappa |Y_1 - Y_2|,

    where parameters are:

        - :math:`\Sigma`: cost matrix, (a PSD Matrix);

        - :math:`\kappa`;

        - :math:`p`;

        - square (notation depending on the model type), where square = 2 for 'svm', 'logistic', 'lad'; square = 1 for 'ols'.

    Reference:

    [1] OLS: <https://www.cambridge.org/core/journals/journal-of-applied-probability/article/robust-wasserstein-profile-inference-and-applications-to-machine-learning/4024D05DE4681E67334E45D039295527>

    [2] LAD / SVM / Logistic: <https://jmlr.org/papers/volume20/17-633/17-633.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', 
                 fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """Initialize Mahalanobis-Wasserstein DRO model.

        :param input_dim: Dimension of feature space. Must satisfy :math:`\text{input\_dim} \geq 1`
        :type input_dim: int

        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)

        :type model_type: str

        :param fit_intercept: Whether to learn intercept term :math:`b`.
            Set to ``False`` for pre-centered data. Defaults to True.
        :type fit_intercept: bool

        :param solver: Convex optimization solver. Valid options:
        
                - ``'MOSEK'`` (commercial, recommended)
        
        :type solver: str

        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :raises ValueError:

            - If input_dim < 1

            - If unsupported solver is selected

        Example:
            >>> model = WassersteinDRO(
            ...     input_dim=5,
            ...     model_type='svm',
            ...     solver='MOSEK'
            ... )
            >>> model.cost_matrix.shape  # (5, 5)

        .. note::
            - Changing ``cost_matrix`` after initialization requires calling ``update()``
        """
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        
        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.eps = 0 
        self.p = 1  
        self.kappa = 'inf' 

    def update(self, config: Dict[str, Any]) -> None:
        """Update Wasserstein-DRO model parameters dynamically.
        
        :param config: Configuration dictionary with keys:

            - ``'cost_matrix'``: Mahalanobis metric matrix :math:`\Sigma^{-1} \succ 0`

                - Shape: (input_dim, input_dim)

                - Type: numpy.ndarray

            - ``'eps'``: Wasserstein radius :math:`\epsilon \geq 0`

            - ``'p'``: Wasserstein order :math:`p \geq 1` or ``'inf'``

            - ``'kappa'``: Y-ambiguity radius :math:`\kappa \geq 0` or ``'inf'``

        :type config: dict[str, Any]

        :raises ValueError:

            - If cost_matrix is not positive definite

            - If eps < 0

            - If p < 1 and p ≠ 'inf'

            - If kappa < 0 and kappa ≠ 'inf'

        :raises TypeError:

            - If cost_matrix is not numpy array

            - If numeric parameters are not float/int
        
    
        Example:
            >>> model = WassersteinDRO(input_dim=3)
            >>> new_config = {
            ...     'eps': 0.5,
            ...     'p': 2,
            ...     'cost_matrix': np.diag([1, 2, 3])
            ... }
            >>> model.update(new_config)
            >>> model.p  # 2.0
        """

        if 'cost_matrix' in config:
            cost_matrix = config['cost_matrix']
            if not isinstance(cost_matrix, np.ndarray):
                raise TypeError("cost_matrix must be numpy.ndarray")
            if self.kernel == 'linear' and cost_matrix.shape != (self.input_dim, self.input_dim):
                raise ValueError(f"cost_matrix must have shape ({self.input_dim}, {self.input_dim})")
            if not np.all(np.linalg.eigvals(cost_matrix) > 0):
                raise ValueError("cost_matrix must be positive definite")
            
            self.cost_matrix = cost_matrix
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))

        if 'eps' in config:
            eps = config['eps']
            if not isinstance(eps, (float, int)):
                raise TypeError("eps must be numeric")
            if eps < 0:
                raise ValueError(f"eps must be ≥ 0, got {eps}")
            self.eps = float(eps)

        if 'p' in config:
            p = config['p']
            if p != 'inf' and (not isinstance(p, (float, int)) or p < 1):
                raise ValueError(f"p must be ≥1 or 'inf', got {p}")
            self.p = float(p) if p != 'inf' else 'inf'

        if 'kappa' in config:
            kappa = config['kappa']
            if kappa != 'inf' and (not isinstance(kappa, (float, int)) or kappa < 0):
                raise ValueError(f"kappa must be ≥0 or 'inf', got {kappa}")
            if kappa != 'inf' and self.model_type == 'ols':
                warnings.warn("Y-ambiguity is disabled for OLS models", UserWarning)
            self.kappa = float(kappa) if kappa != 'inf' else 'inf'
     
    def _penalization(self, theta: cp.Expression) -> float:
        """
        Module for computing the regularization part in the standard Wasserstein DRO problem.

        Args:
            theta (:py:class:`cvxpy.Expression`): Feature vector with shape (n_feature,).
        
        Returns:
            Float: Regularization term part.

        """
        if self.kernel != 'linear':
            if self.n_components is not None:
                nystrom = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                Phi_X = nystrom.fit_transform(self.support_vectors_)
                theta_K = sqrtm(Phi_X.T @ Phi_X) @ theta
            else:
                theta_K = sqrtm(pairwise_kernels(self.support_vectors_, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)) @ theta

        else:
            theta_K = theta

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type == 'lad':
            if self.kappa == 'inf':
                return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
            else:
                return cp.maximum(cp.norm(self.cost_inv_transform @ theta_K, dual_norm), 1 / self.kappa)


        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the WDRO problem.

        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param Y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type Y: numpy.ndarray

        :returns: Dictionary containing trained parameters:
        
            - ``theta``: Weight vector of shape `(n_features,)`
            
            - ``b``
            
        :rtype: Dict[str, Any]
        
        .raises: WassersteinDROError: If the optimization problem fails to solve.
        """
        if self.model_type in {'logistic', 'svm'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise WassersteinDROError("classification labels not in {-1, +1}")
        
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise WassersteinDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise WassersteinDROError("Input X and target y must have the same number of samples.")

        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
                self.cost_matrix = np.eye(sample_size)
                self.cost_inv_transform = np.eye(sample_size)
            else:
                theta = cp.Variable(self.n_components)
                self.cost_matrix = np.eye(self.n_components)
                self.cost_inv_transform = np.eye(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0


        lamb_da = cp.Variable()
        cons = [lamb_da >= self._penalization(theta)]
        if self.model_type == 'ols':
            final_loss = cp.norm(X @ theta + b - y) / math.sqrt(sample_size) + math.sqrt(self.eps) * lamb_da

        else:
            if self.model_type in ['svm', 'logistic']:
                s = cp.Variable(sample_size)
                cons += [s >= self._cvx_loss(X, y, theta, b)]
                if self.kappa != 'inf':
                    cons += [s >= self._cvx_loss(X, -y, theta, b) - lamb_da * self.kappa]
                final_loss = cp.sum(s) / sample_size + self.eps * lamb_da
            else:
                final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + self.eps * lamb_da

        problem = cp.Problem(cp.Minimize(final_loss), cons)
        try:
            problem.solve(solver = self.solver)
        except cp.error.SolverError as e:
            raise WassersteinDROError(f"Optimization failed to solve using {self.solver}.") from e
        
        if theta.value is None:
            raise WassersteinDROError("Optimization did not converge to a solution.")

        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
        return model_params
    
    def _distance_compute(self, X_1: cp.Expression, X_2: np.ndarray, Y_1: cp.Expression, Y_2: float) -> cp.Expression:
        """
        Computing the distance between two points (X_1, Y_1), (X_2, Y_2) under our defined metric in cvxpy problem

        Args:
            X_1 (:py:class:`cvxpy.expressions.expression.Expression`): Input feature-1 (n_feature,);
            X_2 (np.ndarray): Input feature-2 (n_feature,);
            Y_1 (:py:class:`cvxpy.expressions.expression.Expression`): Input label-1;
            Y_2 (float): Input label-2;

        Returns:
            :py:class:`cvxpy.expressions.expression.Expression`: Distance value
            
        Raises:
            WassersteinDROError: If the dimensions of two input feature are different.
        """
        if X_1.shape[-1] != X_2.shape[-1]:
            raise WassersteinDROError(f"two input feature dimensions are different.")
        # if Y_1 != Y_2 and self.kappa != 'inf':
        #     warnings.warn("Despite labels are different, we do not count their difference since we do not allow change in Y.")

        component_X = cp.norm(sqrtm(self.cost_matrix) @ (X_1 - X_2), self.p)
        if self.model_type == 'ols':
            component_X = component_X ** 2

        if self.kappa != 'inf':
            component_Y = self.kappa * cp.abs(Y_1 - Y_2)

        else:
            # change of Y is not allowed
            component_Y = 0
        return component_X + component_Y
        

    def _lipschitz_norm(self):
        """
        Computing the Lipschitz norm of the loss function

        Returns:
            Float: the size of the Lipschitz norm of the loss function

        """
        if self.model_type in ['svm', 'logistic', 'lad']:
            return 1
        else:
            return np.inf
    
    def worst_distribution(self, X: np.ndarray, y: np.ndarray, 
                      compute_type: str = 'asymp', gamma: float = 0) -> Dict[str, Any]:
        """Compute worst-case distribution under Wasserstein ambiguity set.

        :param X: Input feature matrix. Shape: (n_samples, n_features)
            Must satisfy ``n_features == input_dim``
        :type X: numpy.ndarray

        :param y: Target vector. Shape: (n_samples,)

            - Classification: binary labels (-1/1)

            - Regression: continuous values

        :type y: numpy.ndarray

        :param compute_type: Computation methodology. Options:

            - ``'asymp'``: Asymptotic approximation (faster, less accurate)
                *Supported models*: ``['svm', 'logistic', 'lad']``

            - ``'exact'``: Exact dual solution (slower, precise)

        :type compute_type: str
        :param gamma: Regularization parameter for asymptotic method. 
            Must satisfy :math:`\gamma > 0` when ``compute_type='asymp'``.
            Defaults to 0.
        :type gamma: float

        :return: Dictionary containing:

            - ``'sample_pts'``: Worst-case sample locations. Shape: (m, n_features)

            - ``'weights'``: Probability weights. Shape: (m,) with :math:`\sum w_i = 1`

        :rtype: dict[str, Any]

        :raises ValueError:

            - If ``compute_type='asymp'`` with ``model_type='ols'``

            - If ``compute_type='asymp'`` and ``kappa == 'inf'``

            - If gamma ≤ 0 when required

        :raises TypeError:

            - If input dimensions mismatch
        
        
        Example:
            >>> X, y = np.random.randn(100, 3), np.random.randint(0,2,100)
            >>> model = WassersteinDRO(model_type='svm', input_dim=3)
            >>> wc_dist = model.worst_distribution(X, y, 'asymp', gamma=0.1)
            >>> wc_dist['weights'].sum()  # Approximately 1.0
        
        .. note::
            - Asymptotic method ignores curvature regularization (κ=infty)
            - Exact method requires ``solver='MOSEK'`` for conic constraints

        Reference of Worst-case Distribution:

        [1] SVM / Logistic / LAD: Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf, where eta is the theta in eq(27) and gamma = 0 in that equation.

        [2] In all cases, we use a reduced dual case (e.g., Remark 5.2 of https://arxiv.org/pdf/2308.05414) to compute their worst-case distribution.

        [3] General Worst-case Distributions can be found in: https://pubsonline.informs.org/doi/abs/10.1287/moor.2022.1275, where norm_theta is lambda* here.

        """

        if self.model_type == 'ols' and compute_type == 'asymp':
            warnings.warn("OLS does not support the corresponding computation method.")
        elif self.kappa == 'inf' and compute_type == 'asymp' and self.model_type in ['svm', 'logistic']:
            raise WassersteinDROError("The corresponding computation method do not support kappa = infty!")
        
        if not isinstance(gamma, (float, int)) or gamma < 0:
            raise WassersteinDROError("Worst-case parameter 'gamma' must be a non-negative float.")
        
        sample_size, __ = X.shape


        self.fit(X, y)
        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        norm_theta = np.linalg.norm(self.cost_inv_transform @ self.theta, ord = dual_norm)

        if compute_type == 'exact':
            if self.model_type == 'ols':
                # dual_norm_parameter = np.linalg.norm(self.cost_inv_transform @ self.theta, dual_norm) ** 2
                # new_X = np.zeros((sample_size, self.input_dim))
                # for i in range(sample_size):
                #     var_x = cp.Variable(self.input_dim)
                #     var_y = cp.Variable()
                #     # TODO: modify or remove
                #     obj = (y[i] - self.theta @ var_x - self.b) ** 2 - dual_norm_parameter * self._distance_compute(var_x, X[i], var_y, y[i])
                #     problem = cp.Problem(cp.Maximize(obj))
                #     problem.solve(solver = self.solver)
                #     new_X[i] = var_x.value
                # return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}
                raise WassersteinDROError("exact does not work for ols")

            else: # linear classification or regression with Lipschitz norm
                # we denote the following case when we do not change Y.
                new_X = np.zeros((sample_size, self.input_dim))
                new_y = np.zeros(sample_size)
                if self.model_type == 'svm':
                    for i in range(sample_size):
                        var_x = cp.Variable(self.input_dim)
                        var_y = cp.Variable()
                        # TODO: modify or remove, does not work.
                        obj = 1 - y[i] * (var_x @ self.theta + self.b) - norm_theta * self._distance_compute(var_x, X[i], var_y, y[i])
                        problem = cp.Problem(cp.Maximize(obj))
                        problem.solve(solver = self.solver)
                        
                        if 1 - y[i] * (var_x.value @ self.theta + self.b) < 0:
                            new_X[i] = X[i]
                            new_y[i] = y[i]
                        else:
                            new_X[i] = var_x.value
                            new_y[i] = var_y.value if var_y.value is not None else y[i]
                            # print('test', var_x.value, X[i])
                    return {'sample_pts': [new_X, new_y], 'weight': np.ones(sample_size) / sample_size}

                # elif self.model_type in ['lad','logistic']:
                #     for i in range(sample_size):
                #         var_x = cp.Variable(self.input_dim)
                #         var_y = cp.Variable()
                #         # TODO: modify or remove, does not work.
                #         obj = self._cvx_loss(var_x, y[i], self.theta, self.b) - norm_theta * self._distance_compute(var_x, X[i], var_y, y[i])
                #         problem = cp.Problem(cp.Maximize(obj))
                #         problem.solve(solver = self.solver)
                #         new_X[i] = var_x.value
                #         new_y[i] = var_y.value if var_y.value is not None else y[i]
                #     return {'sample_pts': [new_X, new_y], 'weight': np.ones(sample_size) / sample_size}
                
                else:
                    raise WassersteinDROError(f"{self.model_type} not supported!")
      
        elif compute_type == 'asymp':
            # in the following cases, we take gamma = 1 / sample_size since we want the asymptotic with respect to n
            if self.model_type in ['svm', 'logistic']:
            # Theorem 20 in https://jmlr.org/papers/volume20/17-633/17-633.pdf, where eta refers to theta in their equation, and eta_gamma refers to eta(\gamma)
                gamma = min(min(gamma, self.eps), 1)
                #min(min(1 / math.sqrt(sample_size), self.eps), 1)
                eta = cp.Variable(nonneg = True)
                alpha = cp.Variable(sample_size, nonneg = True)

                # svm / logistic L = 1
                dual_loss = self._lipschitz_norm() * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self._loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self._loss(X, -y))) / sample_size
                cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps - gamma]
                problem = cp.Problem(cp.Maximize(dual_loss), cons)
                problem.solve(solver = self.solver)
                eta_gamma = gamma / (eta.value + self.kappa - self.eps + gamma + 1)
                weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
                weight = np.hstack((weight, eta_gamma / sample_size))
                weight[0] = weight[0] * (1 - eta_gamma)
                weight[sample_size] = weight[sample_size] * (1 - eta_gamma)
                # print(alpha.value, eta_gamma, eta.value)
                # solve the following perturbation problem
                X_star = cp.Variable(self.input_dim)
                cons = [cp.norm(sqrtm(self.cost_matrix) @ X_star, self.p) <= 1]
                problem = cp.Problem(cp.Maximize(X_star @ self.theta), cons)
                problem.solve(solver = self.solver)
                if eta_gamma != 0:
                    new_X = X[0] + X_star.value * sample_size * eta.value / eta_gamma
                else:
                    new_X = X[0]
                new_y = y[0]

                X = np.concatenate((X, X))
                X = np.vstack((X, new_X))
                y = np.concatenate((y, -y))
                y = np.hstack((y, new_y))
                return {'sample_pts': [X, y], 'weight': weight}

            elif self.model_type == 'lad':
            # Theorem 9 in https://jmlr.org/papers/volume20/17-633/17-633.pdf
                gamma = gamma
                weight = np.zeros(sample_size + 1)
                weight[1:-1] = np.ones(sample_size - 1) / sample_size
                weight[0] = (1 - gamma) / sample_size
                weight[-1] = gamma / sample_size
                # solve the following perturbation problem
                X_star = cp.Variable(self.input_dim)
                if self.kappa not in [np.inf, 'inf']:
                    y_star = cp.Variable()
                    cons = [cp.norm(sqrtm(self.cost_matrix) @ X_star, self.p) + self.kappa * cp.abs(y_star) <= 1]
                else:
                    y_star = 0
                    cons = [cp.norm(sqrtm(self.cost_matrix) @ X_star, self.p) <= 1]
                dual_loss = self.theta @ X_star - y_star
                problem = cp.Problem(cp.Maximize(dual_loss), cons)
                problem.solve(solver = self.solver)
                new_X = X[0] + self.eps * sample_size / gamma * X_star.value
                if self.kappa not in [np.inf, 'inf']:
                    new_y = y[0] + self.eps * sample_size / gamma * y_star.value
                else:
                    new_y = y[0]
                worst_X = np.vstack((X, new_X))
                worst_y = np.hstack((y, new_y))
                return {'sample_pts': [worst_X, worst_y], 'weight': weight}

            else:
                raise WassersteinDROError(f"We do not support {self.model_type} for asypm!")
        else:
            raise WassersteinDROError("We do not support the computation type. The computation type can only be 'asymp' or 'exact'.")


class WassersteinDROSatisificingError(Exception):
    """Base exception class for errors in Wasserstein DRO (Robust Satisficing) model."""
    pass


class WassersteinDROsatisficing(BaseLinearDRO):
    """
    Robust satisficing version of Wasserstein DRO

    This model minimizes the subject to (approximated version) of the robust satisficing constraint of Wasserstein DRO. The Wasserstein Distance is defined as the minimum probability coupling of two distributions for the distance metric: 

    .. math::
        d((X_1, Y_1), (X_2, Y_2)) = (\|\Sigma^{1/2} (X_1 - X_2)\|_p)^{square} + \kappa |Y_1 - Y_2|,

    Reference: <https://pubsonline.informs.org/doi/10.1287/opre.2021.2238>

    """
    def __init__(self, input_dim: int, model_type: str = 'svm', 
                fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """Initialize Robust satisficing version of Wasserstein DRO.

        :param input_dim: Feature space dimension. Must satisfy :math:`d \geq 1`

        :type input_dim: int

        :param model_type: Base model architecture. Supported:

            - ``'svm'``

            - ``'logistic'``

            - ``'ols'``

            - ``'lad'``

        :type model_type: str
        :param fit_intercept: Whether to learn intercept :math:`b`.
            Disable for standardized data. Defaults to True.
        :type fit_intercept: bool

        :param solver: Convex optimization solver. Options:

            - ``'MOSEK'`` (commercial, recommended)

        :type solver: str

        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :raises ValueError:

            - If input_dim < 1

            - If invalid solver selected

        Initialization Defaults:
            1. Cost matrix initialized as identity :math:`I_d`
            2. Target ratio :math:`\tau = 1/0.8` (20% performance margin)
            3. Wasserstein order :math:`p=1` (earth mover's distance)

        Example:
            >>> model = WassersteinDROsatisficing(
            ...     input_dim=5,
            ...     model_type='svm',
            ...     solver='ECOS'
            ... )
            >>> model.cost_matrix.shape  # (5, 5)

        """
        # Parameter validation
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        
        # Initialize metric components
        self.cost_matrix = np.eye(input_dim)  
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.target_ratio = 1 / 0.8  
        self.eps = 0  
        self.p = 1 
        self.kappa = 1  

    def update(self, config: Dict[str, Any]) -> None:
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
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        sample_size = len(X)
        if self.model_type in {'logistic', 'svm'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise WassersteinDROSatisificingError("classification labels not in {-1, +1}")
    
        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
                self.cost_matrix = np.eye(sample_size)
                self.cost_inv_transform = np.eye(sample_size)
            else:
                theta = cp.Variable(self.n_components)
                self.cost_matrix = np.eye(self.n_components)
                self.cost_inv_transform = np.eye(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)
        
    
        if self.kernel != 'linear':
            if self.n_components is not None:
                nystrom = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                Phi_X = nystrom.fit_transform(self.support_vectors_)
                theta_K = sqrtm(Phi_X.T @ Phi_X) @ theta
            else:
                theta_K = sqrtm(pairwise_kernels(self.support_vectors_, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)) @ theta

        else:
            theta_K = theta

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1

        sample_size, __ = X.shape
        empirical_rmse = self.fit_oracle(X, y)
        TGT = self.target_ratio * empirical_rmse
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        cons = [TGT >= cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size]
        if self.model_type == 'lad':
            if self.kappa == 'inf':
                obj = cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
            else:
                obj = cp.maximum(cp.norm(self.cost_inv_transform @ theta_K, dual_norm), 1 / self.kappa)
        elif self.model_type in ['ols', 'svm', 'logistic']:
            obj = cp.norm(self.cost_inv_transform @ theta_K, dual_norm)

        problem = cp.Problem(cp.Minimize(obj), cons)
        problem.solve(solver = self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
    
        return model_params



    # def fit_depreciate(self, X, y):
    #     """
    #     Find the best epsilon that matches the desired robust objective via bisection (depreciated)

    #     Args:
    #         X (np.ndarray): Input feature matrix with shape (n_samples, n_features).

    #         y (np.ndarray): Target vector with shape (n_samples,).

    #     Returns:
    #         Dict[str, Any]: Model parameters dictionary with 'theta' key.

    #     """

    #     warnings.warn("The bisection search is depreciated for Robust Satisficing Wasserstein DRO.")
    #     iter_num = 1
    #     # determine the empirical obj
    #     self.eps = 0
    #     empirical_rmse = self.fit_oracle(X, y)
    #     TGT = self.target_ratio * empirical_rmse
    #     # print('tgt', TGT)
    #     self.eps = 100
    #     assert (self.fit_oracle(X, y) > TGT)
    #     eps_lower, eps_upper = 0, self.eps      
    #     # binary search and find the maximum eps, such that RMSE + eps theta <= tau  
    #     for i in range(iter_num):
    #         self.eps = (eps_lower + eps_upper)/2
    #         if self.fit_oracle(X, y) > TGT:
    #             eps_upper = self.eps
    #         else:
    #             eps_lower = self.eps
        
    #     model_params = {}
    #     model_params["theta"] = self.theta.reshape(-1).tolist()
    #     return model_params
    
    def _penalization(self, theta: cp.Expression) -> float:
        """
        Module for computing the regularization part in the standard Wasserstein DRO problem.

        Args:
            theta (:py:class:`cvxpy.Expression`): Feature vector with shape (n_feature,).
        
        Returns:
            Float: Regularization term part.

        """
        if self.kernel != 'linear':
            if self.n_components is not None:
                nystrom = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                Phi_X = nystrom.fit_transform(self.support_vectors_)
                theta_K = sqrtm(Phi_X.T @ Phi_X) @ theta
            else:
                theta_K = sqrtm(pairwise_kernels(self.support_vectors_, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)) @ theta

        else:
            theta_K = theta

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type == 'lad':
            if self.kappa == 'inf':
                return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
            else:
                return cp.maximum(cp.norm(self.cost_inv_transform @ theta_K, dual_norm), 1 / self.kappa)
            

    def fit_oracle(self, X, y):
        """
        Depreciated, find the optimal that given the ambiguity constraint.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).

            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            float: robust objective value

        """

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise WassersteinDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise WassersteinDROError("Input X and target y must have the same number of samples.")


        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
                self.cost_matrix = np.eye(sample_size)
                self.cost_inv_transform = np.eye(sample_size)
            else:
                theta = cp.Variable(self.n_components)
                self.cost_matrix = np.eye(self.n_components)
                self.cost_inv_transform = np.eye(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)
            
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0


        lamb_da = cp.Variable()
        cons = [lamb_da >= self._penalization(theta)]
        if self.model_type == 'ols':
            final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + math.sqrt(self.eps) * lamb_da

        else:
            if self.model_type in ['svm', 'logistic']:
                s = cp.Variable(sample_size)
                cons += [s >= self._cvx_loss(X, y, theta, b)]
                if self.kappa != 'inf':
                    cons += [s >= self._cvx_loss(X, -y, theta, b) - lamb_da * self.kappa]
                final_loss = cp.sum(s) / sample_size + self.eps * lamb_da
            else:
                # model type == 'lad' for general p.
                final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + self.eps * lamb_da

        problem = cp.Problem(cp.Minimize(final_loss), cons)

        problem.solve(solver = self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b

        return problem.value
        
    
    def worst_distribution(self, X, y):
        raise Warning("We do not compute worst case distribution for robust satisficing model since the distribution constraint is set to be held for any distribution.")

        # REQUIRED TO BE CALLED after solving the DRO problem
        # return a dict {"sample_pts": [np.array([pts_num, input_dim]), np.array(pts_num)], 'weight': np.array(pts_num)}

        # if self.is_regression == 1 or self.is_regression == 2:
        #     return NotImplementedError
        # else:
        #     sample_size, __ = X.shape
        #     if self.p == 1:
        #         dual_norm = np.inf
        #     else:
        #         dual_norm = 1 / (1 - 1 / self.p)
        #     norm_theta = np.linalg.norm(self.theta, ord = dual_norm)
        #     if self.kappa == 1000000:
        #     # not change y, we directly consider RMK 5.2 in https://arxiv.org/pdf/2308.05414.pdf, here norm_theta is lambda* there.
        #         new_X = np.zeros((sample_size, self.input_dim))
        #         for i in range(sample_size):
        #             var_x = cp.Variable(self.input_dim)
        #             obj = 1 - y[i] * var_x @ self.theta - norm_theta * cp.sum_squares(var_x - X[i])
        #             problem = cp.Problem(cp.Maximize(obj))
        #             problem.solve(solver = self.solver)
                    
        #             if 1 - y[i] * var_x.value @ self.theta < 0:
        #                 new_X[i] = X[i]
        #             else:
        #                 new_X[i] = var_x.value
        #         return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}
            
        #     else:
        #         # for general situations if we can change y, we apply Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf (SVM / logistic loss)
        #         #eta is the theta in eq(27)
        #         y_flip = -y
        #         eta = cp.Variable(nonneg = True)
        #         alpha = cp.Variable(sample_size, nonneg = True)

        #         # svm / logistic L = 1
        #         L = 1
        #         dual_loss = L * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self.loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self.loss(X, y_flip))) / sample_size
        #         cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps]
        #         problem = cp.Problem(cp.Maximize(dual_loss), cons)
        #         problem.solve(solver = self.solver)
        #         weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
        #         X = np.concatenate((X, X))
        #         y = np.concatenate((y, y_flip))
        #         return {'sample_pts': [X, y], 'weight': weight}

        






        

        
