import cvxpy as cp
import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, Any
from .base import BaseLinearDRO, DataValidationError

class HR_DRO_LR(BaseLinearDRO):
    """Holistic Robust DRO Linear Regression (HR_DRO_LR) model.
    
    This model supports DRO with additional robustness constraints for linear regression and binary classification.

    Attributes:
        r (float): DRO robustness parameter.
        alpha (float): Parameter for marginal robustness.
        epsilon (float): Tolerance for primary DRO constraints.
        epsilon_prime (float): Tolerance for secondary DRO constraints.
    """
    
    def __init__(self, input_dim: int, model_type: str = 'linear', r: float = 1.0, alpha: float = 1.0, 
                 epsilon: float = 0.5, epsilon_prime: float = 1.0):
        super().__init__(input_dim, model_type)
        self.r = r
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime

    def update(self, config: Dict[str, Any] = {}):
        """Update model configuration based on the provided dictionary."""
        self.r = config.get("r", self.r)
        self.alpha = config.get("alpha", self.alpha)
        self.epsilon = config.get("epsilon", self.epsilon)
        self.epsilon_prime = config.get("epsilon_prime", self.epsilon_prime)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Fit model to data by solving the DRO optimization problem."""
        T, feature_dim = X.shape
        if feature_dim != self.input_dim:
            raise DataValidationError(f"Expected input with {self.input_dim} features, got {feature_dim}.")

        theta = cp.Variable(self.input_dim)
        w = cp.Variable(T)
        lambda_ = cp.Variable(nonneg=True)
        beta = cp.Variable(nonneg=True)
        eta = cp.Variable()
        temp = cp.Variable()

        # Define the objective function
        objective = cp.Minimize((1 / T) * cp.sum(w) + lambda_ * (self.r - 1) + beta * self.alpha + eta)

        # Constraints setup based on the model type
        constraints = []
        if self.model_type in {'linear', 'logistic'}:
            for t in range(T):
                constraints.extend([
                    temp >= cp.abs(theta.T @ X[t] - Y[t]),
                    w[t] >= cp.rel_entr(lambda_, (eta - cp.abs(theta.T @ X[t] - Y[t]) - self.epsilon * cp.norm(theta, 2))),
                    w[t] >= cp.rel_entr(lambda_, (eta - temp - self.epsilon_prime * cp.norm(theta, 2))) - beta,
                    eta >= cp.abs(theta.T @ X[t] - Y[t]) + self.epsilon_prime * cp.norm(theta, 2)
                ])
        elif self.model_type == 'svm':
            Y = 2 * Y - 1  # Convert labels to {-1, 1}
            constraints.append(eta >= 1e-6)  # Stability constraint for eta
            for t in range(T):
                constraints.extend([
                    temp <= Y[t] * (theta.T @ X[t]),
                    w[t] >= cp.rel_entr(lambda_, eta),
                    w[t] >= cp.rel_entr(lambda_, (eta - 1 + Y[t] * (theta.T @ X[t]) - self.epsilon * cp.norm(theta, 2))),
                    w[t] >= cp.rel_entr(lambda_, (eta - 1 + temp - self.epsilon_prime * cp.norm(theta, 2))) - beta,
                    eta >= 1 - Y[t] * (theta.T @ X[t]) + self.epsilon * cp.norm(theta, 2)
                ])
        else:
            raise NotImplementedError("Model type not supported for HR_DRO_LR.")

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8}, verbose=True)

        # Store fitted parameters
        self.theta = theta.value
        self.w = w.value
        self.lambda_ = lambda_.value
        self.beta = beta.value
        self.eta = eta.value

        # Return model parameters in dictionary format
        return {"theta": self.theta.tolist()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output based on input data and model parameters."""
        scores = X @ self.theta
        if self.model_type == 'linear':
            return scores
        else:
            return np.where(scores >= 0, 1, 0) if self.model_type == 'svm' else np.where(scores >= 0.5, 1, 0)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute the performance metrics (accuracy, F1 score for classification; MSE for regression)."""
        predictions = self.predict(X)
        if self.model_type == 'linear':
            mse = np.mean((predictions - y) ** 2)
            return {"mse": mse}
        else:
            accuracy = np.mean(predictions == y)
            f1 = f1_score(y, predictions, average='macro')
            return {"accuracy": accuracy, "f1": f1}