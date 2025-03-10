from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class CVaRDROError(Exception):
    """Base exception class for errors in CVaR DRO model."""
    pass

class CVaRDRO(BaseLinearDRO):
    """Conditional Value-at-Risk (CVaR) Distributionally Robust Optimization (DRO) model.

    This model minimizes a robust loss function for both regression and classification tasks
    under a CVaR constraint.

    Attributes:
    input_dim (int): Dimensionality of the input features.
    model_type (str): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression).
    fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
    alpha (float): Risk level for the CVaR constraint.
    threshold_val (float): Threshold value from the optimization.

    Reference: <https://www.risk.net/journal-risk/2161159/optimization-conditional-value-risk>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm',fit_intercept: bool = True, solver: str = 'MOSEK', alpha: float = 1.0):
        """
        Initialize the CVaR DRO model with specified input dimension, model type, and alpha parameter.

        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'
            alpha (float): Risk level for CVaR (default is 1.0).
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.alpha = alpha
        self.threshold_val = None  # Stores threshold value after fit

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing 'alpha' key for CVaR risk level.

        Raises:
            CVaRDROError: If 'alpha' is not in the valid range (0, 1].
        """
        if 'alpha' in config:
            alpha = config['alpha']
            if not (0 < alpha <= 1):
                raise CVaRDROError("Risk parameter 'alpha' must be in the range (0, 1].")
            self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the robust optimization problem under CVaR constraint.

        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' and 'threshold' keys.

        Raises:
            CVaRDROError: If the optimization problem fails to solve.
        """
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise CVaRDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise CVaRDROError("Input X and target y must have the same number of samples.")

        # Define optimization variables
        theta = cp.Variable(self.input_dim)
        eta = cp.Variable()

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        # Define the CVaR loss function
        loss = cp.sum(cp.pos(self._cvx_loss(X, y, theta, b) - eta)) / (sample_size * self.alpha) + eta
        problem = cp.Problem(cp.Minimize(loss))

        try:
            problem.solve(solver=self.solver)
            self.theta = theta.value
            self.threshold_val = eta.value

        except cp.SolverError as e:
            raise CVaRDROError("Optimization failed to solve using MOSEK.") from e

        if self.theta is None or self.threshold_val is None:
            raise CVaRDROError("Optimization did not converge to a solution.")
        
        

        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.tolist(), "threshold": self.threshold_val, "b":self.b}

    def worst_distribution(self, X: np.ndarray, y: np.ndarray, precision: float = 1e-5) -> Dict[str, Any]:
        """Compute the worst-case distribution based on CVaR constraint.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).
            precision (float): The perturbation amount in case when the loss is too insignificant to count.

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            CVaRDROError: If the worst-case distribution calculation fails.
        """
        self.fit(X,y)
        
        # Calculate per-sample loss with the current theta
        per_loss = self._loss(X, y)

        if self.threshold_val is None:
            raise CVaRDROError("Threshold value is not set. Ensure 'fit' method has been called.")

        # Identify samples with loss above the threshold
        indices = np.where(per_loss + precision > self.threshold_val)[0]
        # Compute weights for the worst-case distribution
        weight = np.ones(len(indices)) / len(indices) if len(indices) > 0 else np.array([])

        return {'sample_pts': [X[indices], y[indices]], 'weight': weight}
