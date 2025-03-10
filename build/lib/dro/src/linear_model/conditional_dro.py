from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class ConditionalCVaRDROError(Exception):
    """Exception class for errors in Marginal CVaR DRO model."""
    pass


class ConditionalCVaRDRO(BaseLinearDRO):
    """Y|X (ConditionalShiftBased) Conditional Value-at-Risk (Conditional-CVaR) Distributionally Robust Optimization (DRO) model that only allow likelihood ratio changes in Y|X.

    This model minimizes a robust loss function for both regression and classification tasks
    under a CVaR constraint only for the distribution of Y|X.    

    Conditional CVaR DRO model following Theorem 2 in:
    with alpha(x) to be beta^T x for simplicity
    alpha corresponds to Gamma in the paper.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Type of model (e.g., 'svm', 'logistic', 'ols').
        alpha (float): Risk level for CVaR.
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        alpha (float, default = 1): Risk level for CVaR.
        control_name (Optional[list[int]]): Indices of the control features for conditional DRO.


    Reference: <https://arxiv.org/pdf/2209.01754.pdf>
    """
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK'):
        """
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.

        
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.alpha = 1
        self.control_name = None
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update model configuration with parameters for Conditional DRO.
        
        Args:
            config (Dict[str, Any]): Dictionary containing optional keys: 'control_name', 'alpha'.
        
        Raises:
            ConditionalCVaRDROError: If any parameter is invalid.
        """
        if 'control_name' in config:
            control_name = config['control_name']
            if not all(0 <= x < self.input_dim for x in control_name):
                raise ConditionalCVaRDROError("All indices in 'control_name' must be in the range [0, input_dim - 1].")
            self.control_name = control_name

        if 'alpha' in config:
            alpha = config['alpha']
            if not (0 < alpha <= 1):
                raise ConditionalCVaRDROError("Parameter 'alpha' must be in the range (0, 1].")
            self.alpha = float(alpha)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY for solving the robust optimization problem with Conditional DRO.

        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta', 'b' keys.

        Raises:
            ConditionalCVaRDROError: If the optimization fails to solve or dimensions mismatch.
        """
        sample_size, feature_size = X.shape

        if feature_size != self.input_dim:
            raise ConditionalCVaRDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise ConditionalCVaRDROError("Input X and target y must have the same number of samples.")


        if self.control_name is not None:
            control_X = X[:,self.control_name]
        else:
            control_X = X
        
        theta = cp.Variable(self.input_dim)

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        beta = cp.Variable(len(control_X[0]))
        cost = cp.sum(self._cvx_loss(X, y, theta, b)) / (1 / self.alpha) + (1 / self.alpha - self.alpha) * cp.sum(cp.pos(self._cvx_loss(X, y, theta, b) - control_X @ beta)) + (1 - self.alpha) * cp.sum(control_X @ beta) 

        prob = cp.Problem(cp.Minimize(cost / sample_size))
        
        try:
            prob.solve(solver = self.solver)
            self.theta = theta.value
        except cp.error.SolverError as e:
            raise ConditionalCVaRDROError(f"Optimization failed to solve using {self.solver}.") from e

        if self.theta is None:
            raise ConditionalCVaRDROError("Optimization did not converge to a solution.")

        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params['theta'] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b


        return model_params
