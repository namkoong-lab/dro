from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform

class MarginalCVaRDROError(Exception):
    """Exception class for errors in Marginal CVaR DRO model."""
    pass

class MarginalCVaRDRO(BaseLinearDRO):
    """Marginal-X Conditional Value-at-Risk (Conditional-CVaR) Distributionally Robust Optimization (DRO) model that only allow likelihood ratio changes in X.

    This model minimizes a robust loss function for both regression and classification tasks
    under a CVaR constraint only for the marginal distribution of X.
    
    
    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Type of model (e.g., 'svm', 'logistic', 'ols').
        alpha (float): Risk level for CVaR.
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        control_name (Optional[list[int]]): Indices of the control features for marginal DRO.
        p (int): Power parameter for the distance measure.
        L (float): Scaling parameter for the marginal robustness.
        threshold_val (float): Threshold value from the optimization.


    Reference:
    [1] <https://pubsonline.informs.org/doi/10.1287/opre.2022.2363>
    [2] The specific model follows Equation (27) in:
    https://arxiv.org/pdf/2007.13982.pdf with parameters L and p.
    """
    
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', alpha: float = 1.0, L: float = 10.0, p: int = 2):
        """
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
            alpha (float, default = 1): Risk level for CVaR.
            p (int, default = 2): Power parameter for the distance measure.
            L (float, default = 10): Scaling parameter for the marginal robustness.
        """

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.alpha = alpha
        self.control_name = None
        self.L = L
        self.p = p
        self.threshold_val = None
        self.B_val = None

    def update(self, config: Dict[str, Any]) -> None:
        """Update model configuration with parameters for Marginal DRO.
        
        Args:
            config (Dict[str, Any]): Dictionary containing optional keys: 'control_name', 'L', 'p', 'alpha'.
        
        Raises:
            MarginalCVaRDROError: If any parameter is invalid.
        """
        if 'control_name' in config:
            control_name = config['control_name']
            if not all(0 <= x < self.input_dim for x in control_name):
                raise MarginalCVaRDROError("All indices in 'control_name' must be in the range [0, input_dim - 1].")
            self.control_name = control_name
        
        if 'L' in config:
            L = config['L']
            if L <= 0:
                raise MarginalCVaRDROError("Parameter 'L' must be positive.")
            self.L = float(L)
        
        if 'p' in config:
            p = config['p']
            if p < 1:
                raise MarginalCVaRDROError("Parameter 'p' must be >= 1.")
            self.p = int(p)
        
        if 'alpha' in config:
            alpha = config['alpha']
            if not (0 < alpha <= 1):
                raise MarginalCVaRDROError("Parameter 'alpha' must be in the range (0, 1].")
            self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY for solving the robust optimization problem with Marginal DRO.

        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta', 'b', 'B', and 'threshold' keys.

        Raises:
            MarginalCVaRDROError: If the optimization fails to solve or dimensions mismatch.
        """
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise MarginalCVaRDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise MarginalCVaRDROError("Input X and target y must have the same number of samples.")

        # Select control features for calculating distances
        control_X = X[:, self.control_name] if self.control_name else X
        dist = np.power(squareform(pdist(control_X)), self.p - 1)

        # Define CVXPY variables
        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        eta = cp.Variable()
        B_var = cp.Variable((sample_size, sample_size), nonneg=True)
        s = cp.Variable(sample_size, nonneg=True)

        # Define constraints for optimization problem
        cons = [
            s >= self._cvx_loss(X, y, theta, b) - (cp.sum(B_var, axis=1) - cp.sum(B_var, axis=0)) / sample_size - eta,
            B_var >= 0
        ]

        # Define the cost function
        cost = (
            cp.sum(s) / (self.alpha * sample_size) +
            self.L ** (self.p - 1) * cp.sum(cp.multiply(dist, B_var)) / (sample_size ** 2)
        )

        # Set up and solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost + eta), cons)
        
        try:
            # solver_options = {
            #     'MSK_IPAR_PRESOLVE_USE': 1,
            #     'MSK_DPAR_BASIS_TOL_X': 1e-5,
            #     'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-5,
            #     'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-5,
            #     'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-5
            # }
            problem.solve(solver=self.solver)
            #**solver_options)

            # Extract optimization results
            self.theta = theta.value
            self.B_val = B_var.value
            self.threshold_val = eta.value
        except cp.error.SolverError as e:
            raise MarginalCVaRDROError(f"Optimization failed to solve using {self.solver}.") from e

        if self.theta is None or self.threshold_val is None or self.B_val is None:
            raise MarginalCVaRDROError("Optimization did not converge to a solution.")

        if self.fit_intercept == True:
            self.b = b.value

        return {
            "theta": self.theta.tolist(),
            "B": self.B_val.tolist(),
            "b": self.b,
            "threshold": self.threshold_val
        }

    # def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    #     """Compute the worst-case distribution based on Marginal CVaR constraint.

    #     Args:
    #         X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
    #         y (np.ndarray): Target vector with shape (n_samples,).

    #     Returns:
    #         Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

    #     Raises:
    #         MarginalCVaRDROError: If required parameters are not set.
    #     """
    #     self.fit(X,y)
    #     sample_size, _ = X.shape
        
    #     if self.B_val is None or self.threshold_val is None:
    #         raise MarginalCVaRDROError("Model parameters are not set. Ensure 'fit' method has been called.")

    #     # Compute per-sample perturbed loss
    #     per_loss = self._loss(X, y)
    #     perturb_loss = per_loss - (np.sum(self.B_val, axis=1) - np.sum(self.B_val, axis=0)) / sample_size

    #     # Identify samples exceeding threshold for worst-case distribution
    #     indices = np.where(perturb_loss > self.threshold_val)[0]
    #     weight = np.ones(len(indices)) / len(indices) if len(indices) > 0 else np.array([])

    #     return {'sample_pts': [X[indices], y[indices]], 'weight': weight}
