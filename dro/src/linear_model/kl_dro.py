from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class KLDROError(Exception):
    """Base exception class for errors in KL-DRO model."""
    pass

class KLDRO(BaseLinearDRO):
    """Kullback-Leibler divergence-based Distributionally Robust Optimization (KL-DRO) model.

    This model minimizes a KL-robust loss function for both regression and classification.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for KL-DRO.
        dual_variable (Optional[float]): Dual variable value from the optimization problem.

    Reference: <https://optimization-online.org/wp-content/uploads/2012/11/3677.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', eps: float = 0.0):
        """
        Initialize the KL-DRO model with specified input dimension and model type.

        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'
            eps (float): Ambiguity size for the KL constraint (default is 0.0).
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)        
        self.eps = eps
        self.dual_variable = None  # To store dual variable value after fit

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing 'eps' key for robustness parameter.

        Raises:
            KLDROError: If 'eps' is provided but is not a non-negative float.
        """
        if 'eps' in config:
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise KLDROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the robust optimization problem with KL constraint.

        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' and 'dual' keys.

        Raises:
            KLDROError: If the optimization problem fails to solve.
        """
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise KLDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise KLDROError("Input X and target y must have the same number of samples.")

        # Define variables for optimization
        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable(nonneg=True)
        t = cp.Variable()
        per_loss = cp.Variable(sample_size)
        epi_g = cp.Variable(sample_size)

        # Constraints for KL-DRO
        constraints = [cp.sum(epi_g) <= sample_size * eta]
        for i in range(sample_size):
            constraints.append(cp.constraints.exponential.ExpCone(per_loss[i] - t, eta, epi_g[i]))
        constraints.append(per_loss >= self._cvx_loss(X, y, theta, b))

        # Define loss objective for minimization
        loss = t + eta * self.eps
        problem = cp.Problem(cp.Minimize(loss), constraints)

        try:
            problem.solve(solver=self.solver)
            self.theta = theta.value
            self.dual_variable = eta.value
        except cp.SolverError as e:
            raise KLDROError("Optimization failed to solve using MOSEK.") from e

        if self.theta is None or self.dual_variable is None:
            raise KLDROError("Optimization did not converge to a solution.")

        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.tolist(), "dual": self.dual_variable, "b": self.b}

    def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute the worst-case distribution based on KL divergence.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            KLDROError: If the worst-case distribution optimization fails.
        """
        self.fit(X, y)  # Fit model to obtain theta and dual variable

        # Calculate the loss with current theta
        per_loss = self._loss(X, y)
        
        if self.dual_variable is None:
            raise KLDROError("Dual variable is not set. Ensure 'fit' method has been called.")

        # Calculate weights for the worst-case distribution
        weight = np.exp(per_loss / self.dual_variable)
        weight /= np.sum(weight)  # Normalize weights

        return {'sample_pts': [X, y], 'weight': weight}
