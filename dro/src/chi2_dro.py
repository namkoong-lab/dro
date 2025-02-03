from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score

class Chi2DROError(Exception):
    """Base exception class for errors in Chi-squared DRO model."""
    pass

class Chi2DRO(BaseLinearDRO):
    """Chi-squared Distributionally Robust Optimization (DRO) model.

    This model minimizes a chi-squared robust loss function for both regression and classification.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., svm for SVM, logistic for Logistic Regression, linear for Linear Regression).
        eps (float): Robustness parameter for DRO.
    """

    def __init__(self, input_dim: int, model_type: str):
        super().__init__(input_dim, model_type)
        self.eps = 0.0

    def update(self, config: Dict[str, Any] = {}):
        """Update the model configuration.

        Args:
            config (dict): Configuration dictionary containing 'eps' key for robustness parameter.

        Raises:
            Chi2DROError: If 'eps' is provided but is not a non-negative float.
        """
        if 'eps' in config:
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise Chi2DROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the robust optimization problem.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' key.

        Raises:
            Chi2DROError: If the optimization problem fails to solve.
        """
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise Chi2DROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise Chi2DROError("Input X and target y must have the same number of samples.")

        theta = cp.Variable(self.input_dim)
        eta = cp.Variable()

        loss = (np.sqrt(1 + self.eps) / np.sqrt(sample_size) * 
                cp.norm(cp.pos(self._cvx_loss(X, y, theta) - eta), 2) + eta)
        
        problem = cp.Problem(cp.Minimize(loss))
        try:
            problem.solve(solver=cp.MOSEK)
        except cp.error.SolverError as e:
            raise Chi2DROError("Optimization failed to solve using MOSEK.") from e

        if theta.value is None:
            raise Chi2DROError("Optimization did not converge to a solution.")

        self.theta = theta.value
        return {"theta": self.theta.reshape(-1).tolist()}

    def evaluate(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, fast: True):
        """Fast Evaluate the true model performance for the obtained theta efficiently"""
        sample_num, __ = X.shape
        errors = (predictions - y) ** 2
        if self.model_type == 'ols':
            predictions = self.predict(X)
            cov_inv = np.linalg.pinv(np.cov(X.T))
            grad_square = np.dot(X.T, errors * X)
            bias = 2 * np.trace(cov_inv @ grad_square)/ len(sample_num ** 2)
        return np.mean(errors) + bias



    def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute the worst-case distribution.

        Reference: Eq (8) in p6 of https://jmlr.org/papers/volume20/17-750/17-750.pdf. And only change |n p - 1|^2 \leq 2 \rho to |np - 1|^2 \leq n * eps^2

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            Chi2DROError: If the worst-case distribution optimization fails.
        """
        self.fit(X, y)

        sample_size, _ = X.shape
        
        per_loss = self._loss(X, y)
        prob = cp.Variable(sample_size, nonneg=True)
        
        constraints = [
            cp.sum(prob) == 1,
            cp.sum_squares(sample_size * prob - 1) <= sample_size * self.eps
        ]
        
        problem = cp.Problem(cp.Maximize(prob @ per_loss), constraints)
        try:
            problem.solve(solver=cp.MOSEK)
        except cp.error.SolverError as e:
            raise Chi2DROError("Optimization failed to solve for worst-case distribution.") from e

        if prob.value is None:
            raise Chi2DROError("Worst-case distribution optimization did not converge to a solution.")

        return {'sample_pts': [X, y], 'weight': prob.value}

