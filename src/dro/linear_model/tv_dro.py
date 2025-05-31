from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class TVDROError(Exception):
    """Base exception class for errors in Total Variation (TV) DRO model."""
    pass

class TVDRO(BaseLinearDRO):
    """Total Variation Distributionally Robust Optimization (TV-DRO) model.

    Implements DRO with TV ambiguity set defined as:

    .. math::
        \\mathcal{P} = \{ Q \, | \, \\text{TV}(Q, P) \\leq \\epsilon \}

    where :math:`\\text{TV}` is the total variation distance.

    """

    def __init__(self, input_dim: int, model_type: str = 'svm', 
                 fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear', 
                 eps: float = 0.0):
        """Initialize TV-constrained DRO model.

        :param input_dim: Feature space dimension. Must be ≥ 1
        :type input_dim: int

        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)

        :type model_type: str
        :param fit_intercept: Whether to learn intercept term :math:`b`.
            Disable for pre-centered data. Defaults to True.
        :type fit_intercept: bool
        :param solver: Convex optimization solver. Recommended:
            - ``'MOSEK'`` (commercial)
        :type solver: str

        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :param eps: TV ambiguity radius. Special cases:

            - 0: Standard empirical risk minimization

            - >0: Controls distributional robustness

        :type eps: float

        :raises ValueError:

            - If input_dim < 1

            - If eps < 0

        Example:
            >>> model = TVDRO(
            ...     input_dim=5,
            ...     model_type='svm',
            ...     eps=0.1
            ... )
            >>> model.eps  # 0.1

        """
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")
        if eps < 0:
            raise ValueError(f"eps must be ≥ 0, got {eps}")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        self.eps = eps
        self.threshold_val = None  #: Decision boundary threshold (set during fitting)
        self.kernel = kernel

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.

        :param config: Dictionary containing configuration updates. Supported keys:

            - ``eps``: Robustness parameter controlling the size of the chi-squared ambiguity set (must be ≥ 0)
            
        :type config: Dict[str, Any]

        :raises TVDROError: If 'eps' is not in the valid range (0, 1).
        """

        if 'eps' in config:
            eps = config['eps']
            if not (0 < eps < 1):
                raise TVDROError("Robustness parameter 'eps' must be in the range (0, 1).")
            self.eps = float(eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the robust optimization problem with TV constraint.

        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param Y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type Y: numpy.ndarray

        :returns: Dictionary containing trained parameters:
        
            - ``theta``: Weight vector of shape `(n_features,)`
            
            - ``threshold``
            
            - ``b``
            
        :rtype: Dict[str, Any]
        
        .raises: TVDROError: If the optimization problem fails to solve.
        """
        if self.model_type in {'logistic', 'svm'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise TVDROError("classification labels not in {-1, +1}")

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise TVDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise TVDROError("Input X and target y must have the same number of samples.")

        # Define optimization variables
        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
            else:
                theta = cp.Variable(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable()
        u = cp.Variable()

        # Set up loss function and constraints based on model type
        # Loss for regression models
        loss = cp.sum(cp.pos(self._cvx_loss(X,y, theta, b) - eta)) / (sample_size * (1 - self.eps)) + eta
        
        # constraints = [u >= self._cvx_loss(X[i],y[i], theta, b) for i in range(sample_size)]

        loss_vector = self._cvx_loss(X, y, theta, b)
        constraints = [u >= cp.max(loss_vector)]


        # Define the objective with the total variation constraint
        objective = loss * (1 - self.eps) + self.eps * u
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:
            problem.solve(solver=self.solver)
            self.theta = theta.value
            self.threshold_val = eta.value
        except cp.SolverError as e:
            raise TVDROError("Optimization failed to solve using MOSEK.") from e

        if self.theta is None or self.threshold_val is None:
            raise TVDROError("Optimization did not converge to a solution.")
        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.tolist(), "threshold": self.threshold_val, "b": self.b}

    def worst_distribution(self, X: np.ndarray, y: np.ndarray, precision: float = 1e-5) -> Dict[str, Any]:
        """Compute the worst-case distribution based on TV constraint.

        :param X: Feature matrix of shape `(n_samples, n_features)`. 
            Must match the model's `input_dim` (n_features).
        :type X: numpy.ndarray
        :param y: Target vector of shape `(n_samples,)`. For regression tasks, continuous values 
            are expected; for classification, ±1 labels.
        :type y: numpy.ndarray

        :returns: Dictionary containing:

            - ``sample_pts``: Original data points as a tuple ``(X, y)``

            - ``weight``: Worst-case probability weights of shape `(n_samples,)`

        :rtype: Dict[str, Any]

        .raises: TVDROError: If the worst-case distribution calculation fails.
        """
        self.fit(X,y)
        # Calculate the per-sample loss with current theta
        per_loss = self._loss(X, y)

        if self.threshold_val is None:
            raise TVDROError("Threshold value is not set. Ensure 'fit' method has been called.")

        # Identify samples with loss greater than the threshold and compute weights
        max_index = np.argmax(per_loss)
        indices = np.where(per_loss + precision > self.threshold_val)[0]
        total_indices = np.concatenate(([max_index], indices))

        # Calculate the weight for worst-case distribution
        weight = np.concatenate(([self.eps], (1 - self.eps) * np.ones(len(indices)) / len(indices)))

        return {'sample_pts': [X[total_indices], y[total_indices]], 'weight': weight}
