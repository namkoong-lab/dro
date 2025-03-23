import cvxpy as cp
import numpy as np
from typing import Dict, Any
from dro.src.linear_model.base import BaseLinearDRO, DataValidationError

class HRDROError(Exception):
    """Base exception class for errors in HR DRO model."""
    pass

class HR_DRO_LR(BaseLinearDRO):
    """Holistic Robust DRO Linear Regression (HR_DRO_LR) model.
    
    This model supports HR DRO with additional robustness constraints for linear regression and binary classification. (Theorem 7)

    Args:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator ('svm' for SVM, 'lad' for Linear Regression for LAD), default = 'svm'.
        fit_intercept (bool): Whether to calculate the intercept for this model, default = True. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str): Optimization solver to solve the problem, default = 'MOSEK'.
        r (float): DRO robustness parameter for statistical error
        alpha (float): DRO robustness parameter for misspecification
        epsilon (float): DRO robustness parameter for noise
        epsilon_prime (float): DRO robustnes parameter for determining the uncertainty region.

    Reference: <https://arxiv.org/abs/2207.09560>
    """
    
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', r: float = 1.0, alpha: float = 0.0, 
                 epsilon: float = 0.5):
        """
        Initialize the ORWDRO model with specified input dimension and model type.
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'lad').
            fit_intercept (bool): Whether to calculate the intercept for this model, default = True. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str): Optimization solver to solve the problem, default = 'MOSEK'
        """
        if model_type in ['ols', 'logistic']:
            raise HRDROError("HR DRO does not support OLS, logistic")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.r = r
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_prime = epsilon

    def update(self, config: Dict[str, Any] = {}):
        """Update model configuration based on the provided dictionary.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing 'r', 'alpha', 'epsilon', 'epsilon_prime' keys for robustness parameter.

        Raises:
            HRDROError: If any of the configs does not fall into its domain.
        """
        if "r" in config.keys():
            r = config["r"]
            if not isinstance(r, (float, int)) or r < 0:
                raise HRDROError("Robustness parameter of statistical error 'r' must be a non-negative float.")
            self.r = float(r)

        if "alpha" in config.keys():
            alpha = config["alpha"]
            if not isinstance(alpha, (float, int)) or alpha < 0 or alpha > 1:
                raise HRDROError("Robustness parameter of misspecification 'alpha' must be between 0 and 1.")
        
        if "epsilon" in config.keys():
            epsilon = config["epsilon"]
            if not isinstance(r, (float, int)) or epsilon < 0:
                raise HRDROError("Robustness parameter of statistical error 'epsilon' must be a non-negative float.")
            self.epsilon = float(epsilon)

        if "epsilon_prime" in config.keys():
            epsilon_prime = config["epsilon_prime"]
            if not isinstance(r, (float, int)) or epsilon < 0 or epsilon_prime < self.epsilon:
                raise HRDROError("Robustness parameter of statistical error 'epsilon' must be a non-negative float and larger than epsilon prime")
            self.epsilon_prime = float(epsilon_prime)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Fit model to data using CVXPY by solving the DRO optimization problem.
        
        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' key.

        """
        T, feature_dim = X.shape
        if feature_dim != self.input_dim:
            raise DataValidationError(f"Expected input with {self.input_dim} features, got {feature_dim}.")

        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        w = cp.Variable(T)
        lambda_ = cp.Variable(nonneg=True)
        beta = cp.Variable(nonneg=True)
        eta = cp.Variable()
        temp = cp.Variable()

        # Define the objective function
        objective = cp.Minimize((1 / T) * cp.sum(w) + lambda_ * (self.r - 1) + beta * self.alpha + eta)

        # Constraints setup based on the model type
        constraints = []
        ## Appendix D.1 in https://arxiv.org/pdf/2207.09560v4
        if self.model_type in {'lad'}:
            for t in range(T):
                constraints.extend([
                    temp >= cp.abs(theta.T @ X[t] + b - Y[t]),
                    w[t] >= cp.rel_entr(lambda_, (eta - cp.abs(theta.T @ X[t] + b - Y[t]) - self.epsilon * cp.norm(theta, 2))),
                    w[t] >= cp.rel_entr(lambda_, (eta - temp - self.epsilon_prime * cp.norm(theta, 2))) - beta,
                    eta >= cp.abs(theta.T @ X[t]  + b - Y[t]) + self.epsilon_prime * cp.norm(theta, 2)
                ])
            
        ## Appendix D.2 in https://arxiv.org/pdf/2207.09560v4
        elif self.model_type == 'svm':
            constraints.append(eta >= 1e-6)  # Stability constraint for eta
            for t in range(T):
                constraints.extend([
                    temp <= Y[t] * (theta.T @ X[t]),
                    w[t] >= cp.rel_entr(lambda_, eta),
                    w[t] >= cp.rel_entr(lambda_, (eta - 1 + Y[t] * (theta.T @ X[t] + b) - self.epsilon * cp.norm(theta, 2))),
                    w[t] >= cp.rel_entr(lambda_, (eta - 1 + temp - self.epsilon_prime * cp.norm(theta, 2))) - beta,
                    eta >= 1 - Y[t] * (theta.T @ X[t] + b) + self.epsilon * cp.norm(theta, 2)
                ])

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver)
        except cp.error.SolverError as e:
            raise HRDROError(f"Optimization failed to solve using {self.solver}.") from e

        if theta.value is None:
            raise HRDROError("Optimization did not converge to a solution.")

        # Store fitted parameters
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        self.w = w.value
        self.lambda_ = lambda_.value
        self.beta = beta.value
        self.eta = eta.value

        # Return model parameters in dictionary format
        return {"theta": self.theta.tolist(), "b": self.b}
