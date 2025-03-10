from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class TVDROError(Exception):
    """Base exception class for errors in Total Variation (TV) DRO model."""
    pass

class TVDRO(BaseLinearDRO):
    """Total Variation (TV) Distributionally Robust Optimization (DRO) model.

    This model minimizes a robust loss function subject to a total variation constraint.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for TV-DRO.
        threshold_val (float): Threshold value from the optimization.
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', eps: float = 0.0):
        """
        Initialize the TV-DRO model with specified input dimension and model type.

        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols').
            eps (float): Ambiguity size for the TV constraint (default is 0.0).
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.eps = eps
        self.threshold_val = None  # Stores threshold value after fit

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing 'eps' key for robustness parameter.

        Raises:
            TVDROError: If 'eps' is not in the valid range (0, 1).
        """
        if 'eps' in config:
            eps = config['eps']
            if not (0 < eps < 1):
                raise TVDROError("Robustness parameter 'eps' must be in the range (0, 1).")
            self.eps = float(eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the robust optimization problem with TV constraint.

        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' and 'threshold' keys.

        Raises:
            TVDROError: If the optimization problem fails to solve.
        """
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise TVDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise TVDROError("Input X and target y must have the same number of samples.")

        # Define optimization variables
        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable()
        u = cp.Variable()

        # Set up loss function and constraints based on model type
        if self.model_type in {'ols', 'logistic'}:
            # Loss for regression models
            loss = (cp.sum(cp.pos(self._cvx_loss(X,y, theta, b) - eta)) / 
                    (sample_size * (1 - self.eps)) + eta)
            constraints = [u >= cp.sum_squares(X[i] @ theta - y[i]) for i in range(sample_size)]
        else:
            # Loss for SVM
            loss = (cp.sum(cp.pos(self._cvx_loss(X,y, theta, b) - eta)) /
                    (sample_size * (1 - self.eps)) + eta)
            constraints = [u >= 1 - cp.multiply(y[i], X[i] @ theta) for i in range(sample_size)]
            constraints += [u >= 0]

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

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).
            precision (float): The perturbation amount in case when the loss is too insignificant to count.

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            TVDROError: If the worst-case distribution calculation fails.
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
