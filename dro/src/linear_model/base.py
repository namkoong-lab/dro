import numpy as np
import cvxpy as cp
import warnings
from sklearn.metrics import f1_score
from typing import Optional, Tuple, Union
import numpy as np

class DROError(Exception):
    """Base exception class for all errors in DRO models.
    
    :meta private:
    """
    pass


class ParameterError(DROError):
    """Exception raised for invalid parameter configurations.
    
    :meta private:
    """
    pass

class InstallError(DROError):
    """Exception raised for packages / solvers not installed.
    
    :meta private:
    """
    pass

class DataValidationError(DROError):
    """Exception raised for invalid data format or dimensions.
    
    :meta private:
    """
    pass

class BaseLinearDRO:
    """Base class for Linear Distributionally Robust Optimization (DRO) models.
    
    This class supports both regression and binary classification tasks. To ensure convex optimization,
    this class only supports linear models, e.g., SVM, Linear Regression, and Logistic Regression.

    """
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK'):
        """
        :param input_dim: Dimensionality of the input features.
        :type input_dim: int
        :param model_type: Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression for OLS, 'lad' for Linear Regression for LAD), default = 'svm'.
        :type model_type: str
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered), default = True.
        :type fit_intercept: bool
        :param solver: Optimization solver to solve the problem, default = 'MOSEK'.
        :type solver: str 
        """
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
        """Update model parameters based on configuration.
        
        :param config: The model configuration
        :type config: dict
        """
        pass  # Method to be implemented in subclasses if needed

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model to data by solving an optimization problem.
        
        :param X: input covariates
        :type X: :py:class:`numpy.ndarray` 
        :param y: input labels
        :type y: :py:class:`numpy.ndarray` 
        """
        pass  # Method to be implemented in subclasses

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Evaluate the true model performance for the obtained model
        
        :param X: input covariates for evaluation
        :type y: :py:class:`numpy.ndarray` 
        :param y: input labels for evaluation
        :type y: :py:class:`numpy.ndarray` 
        """
        pass

    def load(self, config: dict):
        """Load model parameters from a configuration dictionary.
        
        :param config: The model configuration to load
        :type config: dict
        """
        try:
            theta = np.array(config['theta'])
            if theta.shape != (self.input_dim,):
                raise DataValidationError(f"Theta must have shape ({self.input_dim},)")
            self.theta = theta
        except KeyError as e:
            raise ParameterError("Config must contain 'theta' key.") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output based on input data and model parameters.
        
        :param X: Input covariates
        :type X: :py:class:`numpy.ndarray` 
        :returns: the predicted labels
        :rtype: :py:class:`numpy.ndarray` 
        """
        if X.shape[1] != self.input_dim:
            raise DataValidationError(f"Expected input with {self.input_dim} features, got {X.shape[1]}.")

        scores = X @ self.theta + self.b
        if self.model_type in ['ols', 'lad']:
            return scores

        preds = np.where(scores >= (0 if self.model_type == 'svm' else 0.5), 1, 0)
        return preds

    def score(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Union[float, Tuple[float, float]]:
        """Compute accuracy and F1 score for classification tasks, or MSE for regression.
        
        :param X: Input feature matrix of shape (n_samples, n_features)
        :type X: :py:class:`numpy.ndarray` 
        :param y: Target labels/values of shape (n_samples,)
        :type y: :py:class:`numpy.ndarray` 
        :param weights: Sample weight vector of shape (n_samples,), None indicates equal weights
        :type weights: Optional[:py:class:`numpy.ndarray`], defaults to None

        :returns: 

            - For classification: Tuple containing (accuracy, F1-score)

            - For regression: Mean Squared Error (MSE)
            
        :rtype: Union[float, Tuple[float, float]]

        :raises ValueError: If task type is not properly configured

        Example:
            >>> Classification model: Returns (0.95, 0.93)
            >>> Regression model: Returns 3.1415
        """
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
        """Compute the loss values for individual samples using current model parameters.

        :param X: Input feature matrix of shape (n_samples, n_features)
        :type X: :py:class:`numpy.ndarray`
        :param y: Target values of shape (n_samples,)
        :type y: :py:class:`numpy.ndarray`

        :returns: Loss values for each sample in the batch, shape (n_samples,)
        :rtype: :py:class:`numpy.ndarray`

        :raises ValueError: If input dimensions are incompatible

        Example:
            >>> model = BaseLinearDRO()
            >>> X = np.array([[1, 2], [3, 4]])
            >>> y = np.array([0, 1])
            >>> losses = model._loss(X, y)
            >>> losses.shape
            (2,)
        """
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
        """Construct the convex loss expression for optimization using CVXPY.
        
        :param X: Feature matrix expression of shape (n_samples, n_features)
        :type X: :py:class:`cvxpy.expressions.expression.Expression`
        :param y: Target values expression of shape (n_samples,)
        :type y: :py:class:`cvxpy.expressions.expression.Expression`
        :param theta: Model parameters vector expression of shape (n_features,)
        :type theta: :py:class:`cvxpy.expressions.expression.Expression`
        :param b: Intercept term scalar expression
        :type b: :py:class:`cvxpy.expressions.expression.Expression`

        :returns: Loss expression for the optimization problem
        :rtype: :py:class:`cvxpy.expressions.expression.Expression`

        :raises ValueError: If expression dimensions are incompatible
        :raises TypeError: If any input is not a CVXPY Expression

        """
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

    def evaluate(self, X: np.ndarray, y: np.ndarray, fast: bool = True):
        """Fast evaluate the true model performance for the obtained theta efficiently from data unbiased"""
        sample_num, __ = X.shape
        predictions = self.predict(X)
        if self.model_type == 'ols':
            errors = (predictions - y) ** 2
            if self.fit_intercept == True:
                X_intercept = np.ones(sample_num).reshape(-1, 1)
                X = np.hstack((X, X_intercept))
            cov_inv = np.linalg.pinv(np.cov(X.T))
            grad_square = np.multiply(errors.reshape(-1, 1), X).T @ np.multiply(errors.reshape(-1, 1), X)
            
            bias = 2 * np.trace(cov_inv @ grad_square)/(sample_num ** 3)
            print(bias)
        return np.mean(errors) + bias






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
