import numpy as np
import cvxpy as cp
import warnings
from sklearn.metrics import f1_score
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import Nystroem

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
    this class only supports linear models or kernelized models, e.g., SVM, Linear Regression, and Logistic Regression.

    """
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """
        :param input_dim: Dimensionality of the input features.
        :type input_dim: int
        :param model_type: Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression for OLS, 'lad' for Linear Regression for LAD), default = 'svm'.
        :type model_type: str
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered), default = True.
        :type fit_intercept: bool
        :param solver: Optimization solver to solve the problem, default = 'MOSEK'.
        :type solver: str 
        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str
        """
        if input_dim <= 0:
            raise ParameterError("Input dimension must be a positive integer.")
        if model_type not in {'svm', 'logistic', 'ols', 'lad'}:
            raise ParameterError(f"Unsupported model_type: {model_type}. Default supported types are svm, logistic, ols, lad. Please define your personalized loss.")
        self.input_dim = input_dim
        self.model_type = model_type
        self.kernel = kernel
        self.kernel_gamma = 'scale'
        self.n_components = None

        if self.kernel == 'linear':
            self.theta = np.zeros(self.input_dim)
        self.fit_intercept = fit_intercept
        self.b = 0

        if solver not in cp.installed_solvers():
            raise InstallError(f"Unsupported solver {solver}. It does not exist in your package. Please change the solver or install {solver}.")
        self.solver = solver

    def update_kernel(self, config: dict):
        """Update model class (kernel parameters) based on configuration
        
        :param config: Configuration dictionary with keys:

            - ``metric``: The metric to use when calculating kernel between instances in a feature array. 
            - ``kernel_gamma``: Parameter gamma of the pairwise kernel specified by metric.
            - ``n_components``: The number of 

        :type config: dict[str, Any]

        :raises ValueError:
            - If ``kernel_gamma`` is not in scale, auto;
            - If ``n_components`` is not an int;
            - If ``metric`` is 
        """

        if 'metric' in config:
            kernel = config['metric']
            if kernel in ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']:
                self.kernel = kernel
            else:
                raise ValueError("not predefined kernel")
        if 'kernel_gamma' in config:
            kernel_gamma = config['kernel_gamma']
            if not isinstance(kernel_gamma, (float, int)) and (kernel_gamma not in ['scale', 'auto']):
                raise TypeError("gamma must be float or 'scale' or 'auto'")
            elif isinstance(kernel_gamma, (float, int)) and kernel_gamma <= 0:
                raise ValueError("Float gamma must be non-negative")

            if kernel_gamma == 'auto':
                self.kernel_gamma = 1 / self.input_dim
            else:
                self.kernel_gamma = kernel_gamma

        if 'n_components' in config:
            n_components = config['n_components']
            if not isinstance(n_components, (int)):
                raise TypeError("n_components should be int")
            self.n_components = n_components
        

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


    def load(self, config: dict):
        """Load model parameters from a configuration dictionary.
        
        :param config: The model configuration to load
        :type config: dict
        """
        try:
            theta = np.array(config['theta'])
            if self.kernel == 'linear':
                if theta.shape != (self.    input_dim,):
                    raise DataValidationError(f"Theta must have shape ({self.input_dim},) in linear models")
            else:
                # require training data X as the support vectors.
                self.support_vectors_ = np.array(config['support_vectors'])
                if theta.shape != self.support_vectors_.shape[0]:
                    raise DataValidationError(f"Theta must match the dimension of the support vectors ({self.support_vectors_.shape[0]},) in kernel models")
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

        if self.kernel == 'linear':
            scores = X @ self.theta + self.b
        else:
            if self.n_components is None:
                K = pairwise_kernels(X, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)
            else:
                # nystroem = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                # K = nystroem.fit(self.support_vectors_).transform(X)
                K = self.nystroem_transformer.transform(X)
            scores = K.dot(self.theta) + self.b
        if self.model_type in ['ols', 'lad']:
            return scores

        preds = np.where(scores >= (0 if self.model_type == 'svm' else 0.5), 1, -1)
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
            >>> y = np.array([-1, 1])
            >>> losses = model._loss(X, y)
            >>> losses.shape
            (2,)
        """
        if self.kernel == 'linear':
            inner_product = X @ self.theta
        else:
            if self.n_components is None:
                K = pairwise_kernels(X, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)
            else:
                # nystroem = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                # K = nystroem.fit(self.support_vectors_).transform(X)
                K = self.nystroem_transformer.transform(X)
            inner_product = K @ self.theta
        if self.model_type == 'svm':
            return np.maximum(1 - y * (inner_product + self.b), 0)
        elif self.model_type in 'logistic':
            return np.log(1 + np.exp(-np.multiply(y, inner_product + self.b)))
        elif self.model_type == 'ols':
            return (y - inner_product - self.b) ** 2
        elif self.model_type == 'lad':
            return np.abs(y - inner_product - self.b)
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
        if self.kernel == 'linear':
            inner_product = X @ theta
        else:
            if self.n_components is None:
                K = pairwise_kernels(X, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)
            else:
                self.nystroem_transformer = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                K = self.nystroem_transformer.fit_transform(X)
                #K = nystroem.fit(self.support_vectors_).transform(X)
            inner_product = K @ theta

        if self.model_type == 'svm':
            return cp.pos(1 - cp.multiply(y, inner_product + b))
        elif self.model_type == 'logistic':
            return cp.logistic(-cp.multiply(y, inner_product + b))
        elif self.model_type == 'ols':
            return cp.power(y - inner_product - b, 2)
        elif self.model_type == 'lad':
            return cp.abs(y - inner_product - b)
        else:
            raise NotImplementedError("CVXPY loss not implemented for the specified model_type value.")

    def evaluate(self, X: np.ndarray, y: np.ndarray, fast: bool = True) -> float:
        """Evaluate model performance with bias-corrected mean squared error (MSE) or binary cross entropy error (BCE).
        
        Specifically designed for OLS / Logistic models to compute an unbiased performance estimate 
        by adjusting for the covariance structure of the features. This implementation 
        accelerates evaluation by avoiding full retraining.

        :param X: Feature matrix of shape `(n_samples, n_features)`. 
            Must match the model's `input_dim` (n_features).
        :type X: numpy.ndarray
        :param y: Target values of shape `(n_samples,)`.
        :type y: numpy.ndarray

        :return: Bias-corrected MSE for OLS models. Returns raw MSE for other model types.
        :rtype: float
        :raises ValueError: 
            - If `X` and `y` have inconsistent sample sizes
            - If `X` feature dimension ≠ `input_dim`
        :raises LinAlgError: If feature covariance matrix is singular (non-invertible)

        
        Example:
            >>> model = Chi2DRO(input_dim=5, model_type='ols')
            >>> X_test = np.random.randn(50, 5)
            >>> y_test = np.random.randn(50)
            >>> score = model.evaluate(X_test, y_test)  
            >>> print(f"Corrected MSE: {score:.4f}")

        .. note::
            - Currently, bias correction is only implemented for ``model_type='ols'``; other model types return the raw MSE.
            - Covariance matrix computation uses pseudo-inverse to handle high-dimensional data (``n_features > n_samples``).
        """
        sample_num, __ = X.shape
        if self.model_type == 'ols' and self.kernel == 'linear':
            predictions = self.predict(X)

            errors = (predictions - y) ** 2
            if self.fit_intercept == True:
                X_intercept = np.ones(sample_num).reshape(-1, 1)
                X = np.hstack((X, X_intercept))
            cov_inv = np.linalg.pinv(np.cov(X.T))
            grad_square = np.multiply(errors.reshape(-1, 1), X).T @ np.multiply(errors.reshape(-1, 1), X)
            
            bias = 2 * np.trace(cov_inv @ grad_square)/(sample_num ** 3)
    
        elif self.model_type == 'logistic' and self.kernel == 'linear':
            errors = self._loss(X, y)

            Z = np.hstack([X, np.ones((sample_num, 1))])  # Z_i = (X_i, 1)
            theta_full = np.append(self.theta, self.b)  # full theta ∈ R^{d+1}
            
            u = Z @ theta_full                         # shape (n,)
            s = 1 / (1 + np.exp(y * u))                # sigmoid(-Y u), shape (n,)
            
            # Compute I(θ) = sum_i s_i(1 - s_i) * Z_i Z_i^T
            I = sum(s_i * (1 - s_i) * np.outer(z_i, z_i) for s_i, z_i in zip(s, Z))

            # Compute J(θ) = sum_i s_i^2 * Z_i Z_i^T
            J = sum(s_i**2 * np.outer(z_i, z_i) for s_i, z_i in zip(s, Z))

            # Solve I^{-1} J efficiently via linear solve
            I_inv = np.linalg.inv(I)
            IJ = I_inv @ J
            bias = np.trace(IJ) / sample_num


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
