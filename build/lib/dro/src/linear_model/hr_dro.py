import cvxpy as cp
import numpy as np
from typing import Dict, Any
from dro.src.linear_model.base import BaseLinearDRO, DataValidationError

class HRDROError(Exception):
    """Base exception class for errors in HR DRO model."""
    pass

class HR_DRO_LR(BaseLinearDRO):
    """Holistic Robust DRO Linear Regression (HR_DRO_LR) model.
    
    This model supports DRO with additional robustness constraints for linear regression and binary classification.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str, default = 'svm'): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression for OLS, 'lad' for Linear Regression for LAD).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        r (float): DRO robustness parameter.
        alpha (float): Parameter for marginal robustness.
        epsilon (float): Tolerance for primary DRO constraints.
        epsilon_prime (float): Tolerance for secondary DRO constraints.

    Reference: <https://arxiv.org/abs/2207.09560>
    """
    
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', r: float = 1.0, alpha: float = 1.0, 
                 epsilon: float = 0.5, epsilon_prime: float = 1.0):
        """
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.r = r
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime

    def update(self, config: Dict[str, Any] = {}):
        # TODO: Add hyper-parameter checking
        """Update model configuration based on the provided dictionary."""
        self.r = config.get("r", self.r)
        self.alpha = config.get("alpha", self.alpha)
        self.epsilon = config.get("epsilon", self.epsilon)
        self.epsilon_prime = config.get("epsilon_prime", self.epsilon_prime)

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
        else:
            # TODO: check whether ols works for HR-DRO.
            raise NotImplementedError("Model type not supported for HR_DRO_LR.")

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver, mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8}, verbose=True)
        except cp.error.SolverError as e:
            raise HRDROError("Optimization failed to solve using MOSEK.") from e

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
