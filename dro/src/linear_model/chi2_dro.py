from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class Chi2DROError(Exception):
    """Base exception class for errors in Chi-squared DRO model.
    :meta private:
    """
    pass

class Chi2DRO(BaseLinearDRO):
    """Chi-squared Distributionally Robust Optimization (chi2-DRO) model.

    This model minimizes a chi-squared robust loss function for both regression and classification.

    Args:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        fit_intercept (bool): Whether to calculate the intercept for this model, default = True. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for DRO.

    Reference: <https://www.jmlr.org/papers/volume20/17-750/17-750.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK'):
        """
        Initialize the Chi2-DRO model with specified input dimension and model type.

        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool): Whether to calculate the intercept for this model, default = True. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str): Optimization solver to solve the problem, default = 'MOSEK'
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
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
        """Fit the model using CVXPY to solve the Chi2 distributionally robust optimization problem.

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
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable()

        loss = (np.sqrt(1 + self.eps) / np.sqrt(sample_size) * 
                cp.norm(cp.pos(self._cvx_loss(X, y, theta, b) - eta), 2) + eta)
        
        problem = cp.Problem(cp.Minimize(loss))
        try:
            problem.solve(solver=self.solver)
        except cp.error.SolverError as e:
            raise Chi2DROError(f"Optimization failed to solve using {self.solver}.") from e

        if theta.value is None:
            raise Chi2DROError("Optimization did not converge to a solution.")

        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.reshape(-1).tolist(), "b": self.b}





    def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute the worst-case distribution.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            Chi2DROError: If the worst-case distribution optimization fails.

        Reference: <https://jmlr.org/papers/volume20/17-750/17-750.pdf> (Equation (8))
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
            problem.solve(solver=self.solver)
        except cp.error.SolverError as e:
            raise Chi2DROError("Optimization failed to solve for worst-case distribution.") from e

        if prob.value is None:
            raise Chi2DROError("Worst-case distribution optimization did not converge to a solution.")

        return {'sample_pts': [X, y], 'weight': prob.value}

