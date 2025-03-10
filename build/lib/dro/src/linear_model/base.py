import numpy as np
import cvxpy as cp
import warnings
from torch.autograd import grad
from sklearn.metrics import f1_score
from typing import Optional, Tuple, Union

class DROError(Exception):
    """Base exception class for all errors in DRO models."""
    pass


class ParameterError(DROError):
    """Exception raised for invalid parameter configurations."""
    pass

class InstallError(DROError):
    """Exception raised for packages / solvers not installed."""
    pass

class DataValidationError(DROError):
    """Exception raised for invalid data format or dimensions."""
    pass

class BaseLinearDRO:
    """Base class for Linear Distributionally Robust Optimization (DRO) models.
    
    This class supports both regression and binary classification tasks. To ensure convex optimization,
    this class only supports linear models, e.g., SVM, Linear Regression, and Logistic Regression.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str, default = 'svm'): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression for OLS, 'lad' for Linear Regression for LAD).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'
        theta (np.ndarray): Model parameters.

    """
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK'):
        if input_dim <= 0:
            raise ParameterError("Input dimension must be a positive integer.")
        if model_type not in {'svm', 'logistic', 'ols', 'lad'}:
            warnings.warn(f"Unsupported model_type: {model_type}. Default supported types are svm, logistic, ols, lad. Please define your personalized loss.", UserWarning)
        self.input_dim = input_dim
        self.model_type = model_type
        self.theta = np.zeros(self.input_dim)
        self.fit_intercept = fit_intercept
        self.b = 0

        if solver not in cp.installed_solvers():
            raise InstallError(f"Unsupported solver {solver}. It does not exist in your package. Please change the solver or install {solver}.")
        self.solver = solver
        

    def update(self, config: dict):
        """Update model parameters based on configuration."""
        pass  # Method to be implemented in subclasses if needed

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model to data by solving an optimization problem."""
        pass  # Method to be implemented in subclasses

    def evaluate(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, fast: True):
        """Evaluate the true model performance for the obtained theta efficiently"""
        pass

    def load(self, config: dict):
        """Load model parameters from a configuration dictionary."""
        try:
            theta = np.array(config['theta'])
            if theta.shape != (self.input_dim,):
                raise DataValidationError(f"Theta must have shape ({self.input_dim},)")
            self.theta = theta
        except KeyError as e:
            raise ParameterError("Config must contain 'theta' key.") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output based on input data and model parameters."""
        if X.shape[1] != self.input_dim:
            raise DataValidationError(f"Expected input with {self.input_dim} features, got {X.shape[1]}.")

        scores = X @ self.theta + self.b
        if self.model_type == 'ols':
            return scores

        preds = np.where(scores >= (0 if self.model_type == 'svm' else 0.5), 1, 0)
        return preds

    def score(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Union[float, Tuple[float, float]]:
        """Compute accuracy and F1 score for classification tasks, or MSE for regression."""
        predictions = self.predict(X)
        if weights is not None:
            if weights.shape[0] != y.shape[0]:
                raise DataValidationError("Weights must have the same number of elements as y.")
            weights = weights / weights.sum()
        
        if self.model_type == 'ols':
            mse = np.average((predictions - y) ** 2, weights = weights)
            return mse
        elif self.model_type == 'lad':
            lad = np.average(np.abs(predictions - y), weights = weights)
            return lad
        else:
            accuracy = np.average(predictions == y, weights=weights)
            f1 = f1_score(y, predictions, average='macro')
            return accuracy, f1
    


    def _loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the loss for the current model parameters."""

        if self.model_type == 'svm':
            return np.maximum(1 - y * (X @ self.theta + self.b), 0)
        elif self.model_type in 'logistic':
            return np.log(1 + np.exp(-np.multiply(y, X @ self.theta + self.b)))
        elif self.model_type == 'ols':
            return (y - X @ self.theta - self.b) ** 2
        elif self.model_type == 'lad':
            return np.abs(y - X @ self.theta - self.b)
        else:
            raise NotImplementedError("Loss function not implemented for the specified model_type value.")

    def _cvx_loss(self, X: cp.Expression, y: cp.Expression, theta: cp.Expression, b: cp.Expression) -> cp.Expression:
        """Define the CVXPY loss expression for the model."""
        assert X.shape[-1] == self.input_dim, "Mismatch between feature and input dimension."

        if self.model_type == 'svm':
            return cp.pos(1 - cp.multiply(y, X @ theta + b))
        elif self.model_type == 'logistic':
            return cp.logistic(-cp.multiply(y, X @ theta + b))
        elif self.model_type == 'ols':
            return cp.power(y - X @ theta - b, 2)
        elif self.model_type == 'lad':
            return cp.abs(y - X @ theta - b)
        else:
            raise NotImplementedError("CVXPY loss not implemented for the specified model_type value.")






if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 1, 1, 0])
    model = BaseLinearDRO(input_dim=2, model_type='svm')
    
    try:
        acc, f1 = model.score(X, y)
        print(f"Accuracy: {acc}, F1 Score: {f1}")
    except DROError as e:
        print(f"Error: {e}")
