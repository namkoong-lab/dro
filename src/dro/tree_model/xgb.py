from typing import Dict, Any, Union, Optional,Tuple, Literal
import numpy as np
import math
import cvxpy as cp
import xgboost as xgb
from xgboost import DMatrix
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import f1_score

class KLDRO_XGB:
    """XGBoost model with KL-Divergence Distributionally Robust Optimization (DRO)
    
    :param float eps: KL divergence constraint parameter (ε > 0, default: 0.1)
    :param str kind: Task type ('classification' or 'regression', default: classification)
    :raises ValueError: If invalid parameters are provided
    :raises TypeError: If inputs have incorrect types
    
    .. note::
        Requires XGBoost configuration via :meth:`update` before training
    """

    def __init__(self, eps: float = 1e-1, kind: Literal['classification', 'regression'] = 'classification') -> None:
        """Initialize KL-DRO XGBoost model
        
        :param float eps: Robustness parameter (must be > 0)
        :param str kind: Task type specification
        :raises ValueError: For eps <= 0 or invalid task type
        """
        if eps <= 0:
            raise ValueError(f"Invalid eps={eps}. Must be positive")
        if kind not in ('classification', 'regression'):
            raise ValueError(f"Invalid task type: {kind}. Choose 'classification' or 'regression'")
        
        self.eps = eps
        self.kind = kind
        self.config: Dict[str, Any] = {}
        self.model: Optional[xgb.Booster] = None

    def update(self, config: Dict[str, Any]) -> None:
        """Update XGBoost training configuration
        
        :param dict config: XGBoost parameters dictionary
        :raises KeyError: If missing 'num_boost_round' parameter
        :raises TypeError: For non-dictionary input
        """
        if not isinstance(config, dict):
            raise TypeError(f"Expected dictionary, got {type(config)}")
        if 'num_boost_round' not in config:
            raise KeyError("Configuration must contain 'num_boost_round'")
        if "eps" in config:
            self.eps = config["eps"]
        self.config = config

    def loss(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute base loss values
        
        :param numpy.ndarray preds: Model predictions (n_samples,)
        :param numpy.ndarray labels: True labels (n_samples,)
        :return: Loss values array (n_samples,)
        :rtype: numpy.ndarray
        :raises NotImplementedError: For unsupported task types
        """
        if self.kind == "classification":
            return -labels * np.log(preds + 1e-8) - (1 - labels) * np.log(1 - preds + 1e-8)
        elif self.kind == "regression":
            return (preds.reshape(-1) - labels.reshape(-1))**2
        
    def _kl_dro_loss(self, preds: np.ndarray, dtrain: DMatrix, epsilon: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute KL-DRO gradients and Hessians
        
        :param numpy.ndarray preds: Raw model predictions
        :param DMatrix dtrain: Training data container
        :param float epsilon: Robustness parameter
        :return: Tuple of (gradients, hessians)
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises RuntimeError: For invalid DMatrix contents
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"Expected DMatrix, got {type(dtrain)}")
        if dtrain.get_label() is None:
            raise RuntimeError("DMatrix missing labels")
            
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid transform

        loss = self.loss(preds, labels)
        lambda_param = 1.0 / epsilon
        
        # Numerical stability for exponential
        scaled_loss = loss / lambda_param
        max_loss = np.max(scaled_loss)
        log_sum_exp = max_loss + np.log(np.sum(np.exp(scaled_loss - max_loss)))
        
        # Compute softmax weights
        softmax_weights = np.exp(scaled_loss - log_sum_exp)
        softmax_weights /= np.sum(softmax_weights)
        softmax_weights *= softmax_weights.size  # Weight scaling
        
        # Calculate gradients
        grad = softmax_weights * (preds - labels)
        hess = softmax_weights * (preds * (1.0 - preds))
        
        return grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train KL-DRO XGBoost model
        
        :param numpy.ndarray X: Feature matrix (n_samples, n_features)
        :param numpy.ndarray y: Target values (n_samples,)
        :raises ValueError: For invalid input shapes
        :raises RuntimeError: For configuration or training errors
        """
        X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float32)
        if not self.config:
            raise RuntimeError("Configuration missing - call update() first")
            
        try:
            dtrain = DMatrix(X, label=y)
            config = self.config.copy()
            num_round = config.pop("num_boost_round")
            if "eps" in config:
                config.pop('eps')
            self.model = xgb.train(
                config,
                dtrain,
                num_boost_round=num_round,
                obj=lambda p, d: self._kl_dro_loss(p, d, self.eps)
            )
        except xgb.core.XGBoostError as e:
            raise RuntimeError(f"XGBoost training failed: {str(e)}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions
        
        :param numpy.ndarray X: Input features (n_samples, n_features)
        :return: Model predictions
        :rtype: numpy.ndarray
        :raises NotFittedError: If model is untrained
        """
        if self.model is None:
            raise NotFittedError("Model not trained - call fit() first")
            
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        dtest = DMatrix(X)
        y_pred = self.model.predict(dtest)
        
        if self.kind == "classification":
            return (y_pred > 0.5).astype(int)
        return y_pred

    def score(self, X, y):
        """Testing function
        """
        if self.kind == "classification":
            y_pred = self.predict(X)
            acc = (y_pred.reshape(-1) == y.reshape(-1)).mean()
            f1 = f1_score(y.reshape(-1), y_pred.reshape(-1), average='macro')
            return acc, f1
        else:
            y_pred = self.predict(X)
            return np.mean((y_pred.reshape(-1) - y.reshape(-1)) ** 2)


class Chi2DRO_XGB:
    """XGBoost model with Chi2-Divergence Distributionally Robust Optimization (DRO)
    
    :param float eps: Chi2 divergence constraint parameter (ε > 0, default: 0.1)
    :param str kind: Task type ('classification' or 'regression', default: classification)
    :raises ValueError: If invalid parameters are provided
    :raises TypeError: If inputs have incorrect types
    
    .. note::
        Requires XGBoost configuration via :meth:`update` before training
    """

    def __init__(self, eps: float = 1e-1, kind: Literal['classification', 'regression'] = 'classification') -> None:
        """Initialize Chi2-DRO XGBoost model
        
        :param float eps: Robustness parameter (must be > 0)
        :param str kind: Task type specification
        :raises ValueError: For eps <= 0 or invalid task type
        """
        if eps <= 0:
            raise ValueError(f"Invalid eps={eps}. Must be positive")
        if kind not in ('classification', 'regression'):
            raise ValueError(f"Invalid task type: {kind}. Choose 'classification' or 'regression'")
        
        self.eps = eps
        self.kind = kind
        self.config: Dict[str, Any] = {}
        self.model: Optional[xgb.Booster] = None

    def update(self, config: Dict[str, Any]) -> None:
        """Update XGBoost training configuration
        
        :param dict config: XGBoost parameters dictionary
        :raises KeyError: If missing 'num_boost_round' parameter
        :raises TypeError: For non-dictionary input
        """
        if not isinstance(config, dict):
            raise TypeError(f"Expected dictionary, got {type(config)}")
        if 'num_boost_round' not in config:
            raise KeyError("Configuration must contain 'num_boost_round'")
        if "eps" in config:
            self.eps = config["eps"]
        self.config = config

    def loss(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute base loss values
        
        :param numpy.ndarray preds: Model predictions (n_samples,)
        :param numpy.ndarray labels: True labels (n_samples,)
        :return: Loss values array (n_samples,)
        :rtype: numpy.ndarray
        :raises NotImplementedError: For unsupported task types
        """
        if self.kind == "classification":
            return -labels * np.log(preds + 1e-8) - (1 - labels) * np.log(1 - preds + 1e-8)
        elif self.kind == "regression":
            return (preds.reshape(-1) - labels.reshape(-1))**2
        
    def _chi2_dro_loss(self, preds: np.ndarray, dtrain: DMatrix, epsilon: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Chi2-DRO gradients and Hessians
        
        :param numpy.ndarray preds: Raw model predictions
        :param DMatrix dtrain: Training data container
        :param float epsilon: Robustness parameter
        :return: Tuple of (gradients, hessians)
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises RuntimeError: For invalid DMatrix contents
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"Expected DMatrix, got {type(dtrain)}")
        if dtrain.get_label() is None:
            raise RuntimeError("DMatrix missing labels")
            
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid transform

        loss = self.loss(preds, labels)

        n = len(loss)
        v = loss - np.mean(loss)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            softmax_weights = np.ones(n)

        softmax_weights = 1 + math.sqrt(n * epsilon) / norm_v * v

        if not np.all(softmax_weights >= 0):
            p = cp.Variable(n)
            objective = cp.Maximize(p @ loss)
            constraints = [cp.sum(p) == 1, p >= 0, cp.sum_squares(p - (1/n)) <= epsilon / n]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver = 'MOSEK')

            softmax_weights = np.array(p.value) * n
        
        # Calculate gradients
        grad = softmax_weights * (preds - labels)
        hess = softmax_weights * (preds * (1.0 - preds))
        
        return grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train KL-DRO XGBoost model
        
        :param numpy.ndarray X: Feature matrix (n_samples, n_features)
        :param numpy.ndarray y: Target values (n_samples,)
        :raises ValueError: For invalid input shapes
        :raises RuntimeError: For configuration or training errors
        """
        X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float32)
        if not self.config:
            raise RuntimeError("Configuration missing - call update() first")
            
        try:
            dtrain = DMatrix(X, label=y)
            config = self.config.copy()
            num_round = config.pop("num_boost_round")
            if "eps" in config:
                config.pop("eps")
            self.model = xgb.train(
                config,
                dtrain,
                num_boost_round=num_round,
                obj=lambda p, d: self._chi2_dro_loss(p, d, self.eps)
            )
        except xgb.core.XGBoostError as e:
            raise RuntimeError(f"XGBoost training failed: {str(e)}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions
        
        :param numpy.ndarray X: Input features (n_samples, n_features)
        :return: Model predictions
        :rtype: numpy.ndarray
        :raises NotFittedError: If model is untrained
        """
        if self.model is None:
            raise NotFittedError("Model not trained - call fit() first")
            
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        dtest = DMatrix(X)
        y_pred = self.model.predict(dtest)
        
        if self.kind == "classification":
            return (y_pred > 0.5).astype(int)
        return y_pred

    def score(self, X, y):
        """Testing function
        """
        if self.kind == "classification":
            y_pred = self.predict(X)
            acc = (y_pred.reshape(-1) == y.reshape(-1)).mean()
            f1 = f1_score(y.reshape(-1), y_pred.reshape(-1), average='macro')
            return acc, f1
        else:
            y_pred = self.predict(X)
            return np.mean((y_pred.reshape(-1) - y.reshape(-1)) ** 2)



class CVaRDRO_XGB:
    """XGBoost model with Conditional Value-at-Risk (CVaR) Distributionally Robust Optimization (DRO)
    
    :param float eps: constraint parameter (1 > ε > 0, default: 0.2)
    :param str kind: Task type ('classification' or 'regression', default: classification)
    :raises ValueError: If invalid parameters are provided
    :raises TypeError: If inputs have incorrect types
    
    .. note::
        Requires XGBoost configuration via :meth:`update` before training
    """

    def __init__(self, eps: float = 2e-1, kind: Literal['classification', 'regression'] = 'classification') -> None:
        """Initialize CVaR-DRO XGBoost model
        
        :param float eps: Robustness parameter (must be > 0)
        :param str kind: Task type specification
        :raises ValueError: For eps <= 0 or eps >= 1 or invalid task type
        """
        if eps <= 0 or eps>=1:
            raise ValueError(f"Invalid eps={eps}. Must be between 0 and 1")
        if kind not in ('classification', 'regression'):
            raise ValueError(f"Invalid task type: {kind}. Choose 'classification' or 'regression'")
        
        self.eps = eps
        self.kind = kind
        self.config: Dict[str, Any] = {}
        self.model: Optional[xgb.Booster] = None

    def update(self, config: Dict[str, Any]) -> None:
        """Update XGBoost training configuration
        
        :param dict config: XGBoost parameters dictionary
        :raises KeyError: If missing 'num_boost_round' parameter
        :raises TypeError: For non-dictionary input
        """
        if not isinstance(config, dict):
            raise TypeError(f"Expected dictionary, got {type(config)}")
        if 'num_boost_round' not in config:
            raise KeyError("Configuration must contain 'num_boost_round'")
        if "eps" in config:
            self.eps = config["eps"]
        self.config = config

    def loss(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute base loss values
        
        :param numpy.ndarray preds: Model predictions (n_samples,)
        :param numpy.ndarray labels: True labels (n_samples,)
        :return: Loss values array (n_samples,)
        :rtype: numpy.ndarray
        :raises NotImplementedError: For unsupported task types
        """
        if self.kind == "classification":
            return -labels * np.log(preds + 1e-8) - (1 - labels) * np.log(1 - preds + 1e-8)
        elif self.kind == "regression":
            return (preds.reshape(-1) - labels.reshape(-1))**2
        raise NotImplementedError(f"Unsupported task type: {self.kind}")

    def _cvar_dro_loss(self, preds: np.ndarray, dtrain: DMatrix, epsilon: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CVaR-DRO gradients and Hessians
        
        :param numpy.ndarray preds: Raw model predictions
        :param DMatrix dtrain: Training data container
        :param float epsilon: Robustness parameter
        :return: Tuple of (gradients, hessians)
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises RuntimeError: For invalid DMatrix contents
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"Expected DMatrix, got {type(dtrain)}")
        if dtrain.get_label() is None:
            raise RuntimeError("DMatrix missing labels")
            
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid transform

        loss = self.loss(preds, labels)
        
        var = np.percentile(loss, self.eps * 100)
        grad = preds - labels
        grad = np.where(loss > var, grad / (1 - self.eps), grad)
        hess = (preds * (1 - preds))
        hess = np.where(loss > var, hess / (1 - self.eps), hess)
        return grad, hess
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train CVaR-DRO XGBoost model
        
        :param numpy.ndarray X: Feature matrix (n_samples, n_features)
        :param numpy.ndarray y: Target values (n_samples,)
        :raises ValueError: For invalid input shapes
        :raises RuntimeError: For configuration or training errors
        """
        X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float32)
        if not self.config:
            raise RuntimeError("Configuration missing - call update() first")
            
        try:
            dtrain = DMatrix(X, label=y)
            config = self.config.copy()
            num_round = config.pop("num_boost_round")
            if "eps" in config:
                config.pop('eps')
            
            self.model = xgb.train(
                config,
                dtrain,
                num_boost_round=num_round,
                obj=lambda p, d: self._cvar_dro_loss(p, d, self.eps)
            )
        except xgb.core.XGBoostError as e:
            raise RuntimeError(f"XGBoost training failed: {str(e)}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions
        
        :param numpy.ndarray X: Input features (n_samples, n_features)
        :return: Model predictions
        :rtype: numpy.ndarray
        :raises NotFittedError: If model is untrained
        """
        if self.model is None:
            raise NotFittedError("Model not trained - call fit() first")
            
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        dtest = DMatrix(X)
        y_pred = self.model.predict(dtest)
        
        if self.kind == "classification":
            return (y_pred > 0.5).astype(int)
        return y_pred
    
    def score(self, X, y):
        """Testing function
        """
        if self.kind == "classification":
            y_pred = self.predict(X)
            acc = (y_pred.reshape(-1) == y.reshape(-1)).mean()
            f1 = f1_score(y.reshape(-1), y_pred.reshape(-1), average='macro')
            return acc, f1
        else:
            y_pred = self.predict(X)
            return np.mean((y_pred.reshape(-1) - y.reshape(-1)) ** 2)
        





class NotFittedError(RuntimeError):
    """Exception raised for untrained model usage
    
    :param str message: Custom error message
    """
    def __init__(self, message: str = ""):
        super().__init__(f"{message}Model not trained")