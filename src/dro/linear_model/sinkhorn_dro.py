from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Union, Tuple, Any
from sklearn.metrics import f1_score
from .base import BaseLinearDRO


class SinkhornDROError(Exception):
    """Base exception class for errors in Sinkhorn DRO model."""
    pass


class LinearModel(nn.Module):
    """PyTorch Linear Model for regression tasks.
    
    Args:
        linear (nn.Linear): Linear layer implementing y = xA^T + b
    """
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear model."""
        return self.linear(x)


class SinkhornLinearDRO(BaseLinearDRO):
    """Sinkhorn Distributionally Robust Optimization with Linear Models.

    Reference: <https://arxiv.org/abs/2109.11926>
    """

    def __init__(self,
                input_dim: int,
                model_type: str = "svm",
                fit_intercept: bool = True, 
                reg_param: float = 1e-3,
                lambda_param: float = 1e2,
                output_dim: int = 1, 
                max_iter: int = 1000,
                learning_rate: float = 1e-2,
                k_sample_max: int = 5,
                device: str = "cpu"):
        
        """Initialize Sinkhorn Distributionally Robust Optimization model.

        :param input_dim: Dimension of input feature space (d)
        :type input_dim: int

        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Support Vector Machine (hinge loss)

            - ``'logistic'``: Logistic Regression (cross-entropy loss)

            - ``'ols'``: Ordinary Least Squares (L2 loss)

            - ``'lad'``: Least Absolute Deviation (L1 loss)

        :type model_type: str
        :param fit_intercept: Whether to learn bias term :math:`b`. 
            Disable for pre-centered data. Defaults to True.
        :type fit_intercept: bool
        :param reg_param: Entropic regularization strength :math:`\epsilon` 
            controlling transport smoothness. Must be > 0. Defaults to 1e-3.
        :type reg_param: float
        :param lambda_param: Loss scaling factor :math:`\lambda` 
            balancing Wasserstein distance and loss. Must be > 0. Defaults to 1e2.
        :type lambda_param: float
        :param output_dim: Dimension of model output. 
            1 for regression/binary classification. Defaults to 1.
        :type output_dim: int
        :param max_iter: Maximum number of Sinkhorn iterations. 
            Should be ≥ 100. Defaults to 1e3.
        :type max_iter: int
        :param learning_rate: Step size for gradient-based optimization. 
            Typical range: [1e-4, 1e-1]. Defaults to 1e-2.
        :type learning_rate: float
        :param k_sample_max: Maximum level for Multilevel Monte Carlo sampling. 
            Higher values improve accuracy but increase computation. Defaults to 5.
        :type k_sample_max: int
        :param device: Computation device. 
            Supported: ``'cpu'`` or ``'cuda'``. Defaults to 'cpu'.
        :type device: str

        :raises ValueError: 

            - If any parameter violates numerical constraints (ε ≤ 0, λ ≤ 0, etc.)

            - If model_type is not in supported set

        Example:
            >>> model = SinkhornDRO(
            ...     input_dim=10,
            ...     model_type='svm',
            ...     reg_param=0.01,
            ...     lambda_param=50.0
            ... )
            >>> print(model.device)  # 'cpu'

        .. note::
            - Setting ``device='torch.device(cuda)'`` requires PyTorch with GPU support (CUDA-enabled)
            - It is recommended to retain the default k_sample_max=5 to balance accuracy and computational efficiency

        """
        super().__init__(input_dim, model_type, fit_intercept)
        
        if reg_param <= 0 or lambda_param <= 0:
            raise ValueError("Regularization parameters must be positive")

        self.model = LinearModel(input_dim, output_dim, fit_intercept)
        self.reg_param = reg_param
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.k_sample_max = k_sample_max
        self.device = device
        self.model.to(self.device)
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update hyperparameters for Sinkhorn optimization.
        
        :param config: Dictionary containing parameter updates. Valid keys:

            - ``'reg'``: Entropic regularization strength (ε > 0)

            - ``'lambda'``: Loss scaling factor (λ > 0) 

            - ``'k_sample_max'``: Maximum MLMC sampling level (integer ≥ 1)

        :type config: dict[str, Any]

        :raises ValueError: If any parameter value violates type or range constraints
        
        Example:
            >>> model.update({
            ...     'reg': 0.01,
            ...     'lambda': 50.0,
            ...     'k_sample_max': 3
            ... })  # Explicit type conversion handled internally
        
        .. note::
            For GPU-accelerated computation, specify it during initialization instead of this function
        """
        if "reg" in config:
            if not isinstance(config["reg"], (float, int)) or config["reg"] <= 0:
                raise ValueError("reg must be a positive number")
            self.reg_param = float(config["reg"])

        if "lambda" in config:
            if not isinstance(config["lambda"], (float, int)) or config["lambda"] <= 0:
                raise ValueError("lambda must be a positive number")
            self.lambda_param = float(config["lambda"])
            
        if "k_sample_max" in config:
            if not isinstance(config["k_sample_max"], int) or config["k_sample_max"] <= 0:
                raise ValueError("k_sample_max must be positive integer")
            self.k_sample_max = int(config["k_sample_max"])


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the optimized Sinkhorn DRO model.
        
        :param X: Input feature matrix. Should have shape (n_samples, n_features)
            where n_features must match model's input dimension.
            Supported dtype: float32/float64
        :type X: numpy.ndarray

        :return: Model predictions. Shape depends on task:

            - Regression: (n_samples, 1)

            - Classification: (n_samples,) with 0/1 labels

        :rtype: numpy.ndarray

        :raises ValueError: 

            - If input dimension mismatch occurs (n_features != model_dim)

            - If input contains NaN/Inf values
        
        
        Example:
            >>> X_test = np.random.randn(10, 5).astype(np.float32)
            >>> preds = model.predict(X_test)  # Shape: (10, 1) for regression
        
        .. note::
            - Input data is automatically converted to PyTorch tensors
            - For large datasets (>1M samples), use batch prediction
        """
        if X.shape[1] != self.model.linear.in_features:
            raise ValueError(f"Dimension mismatch: model expects {self.model.linear.in_features}, got {X.shape[1]}")
        if np.isnan(X).any():
            raise ValueError(f"Input contains NaNs")

        X_tensor = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> Union[float, Tuple[float, float]]:
        """Evaluate model performance on given data.
        
        :param X: Input feature matrix. Shape: (n_samples, n_features)
            Must match model's expected input dimension
        :type X: numpy.ndarray

        :param y: Target values. Shape requirements:

            - Regression: (n_samples,) or (n_samples, 1)

            - Classification: (n_samples,) with binary labels (0/1)

        :type y: numpy.ndarray

        :return: Performance metrics:

            - Regression: Mean Squared Error (MSE) as float

            - Classification: Tuple of (accuracy%, macro-F1 score) in [0,1] range

        :rtype: Union[float, Tuple[float, float]]

        Example:    
            >>> # Regression
            >>> X_reg, y_reg = np.random.randn(100,5), np.random.randn(100)
            >>> model = SinkhornDRO(model_type='ols')
            >>> mse = model.score(X_reg, y_reg)  # e.g. 0.153
            
            >>> # Classification  
            >>> X_clf, y_clf = np.random.randn(100,5), np.random.randint(0,2,100)
            >>> model = SinkhornDRO(model_type='svm')
            >>> acc, f1 = model.score(X_clf, y_clf)  # e.g. (0.92, 0.89)
        
        .. note::
            - For regression tasks, outputs are not thresholded
            - Computation uses all available samples (no mini-batching)
        """
        predictions = self.predict(X).flatten()
        if self.model_type in ["ols", "lad"]:
            return np.mean((predictions - y.flatten()) ** 2)
        else:
            predictions[predictions<0] = -1
            predictions[predictions>=0] = 1
            acc = np.mean(predictions.round() == y.flatten())
            f1 = f1_score(y, predictions.round(), average='macro')
            return acc, f1

    def fit(self, X: np.ndarray, y: np.ndarray, optimization_type: str = "SG") -> Dict[str, np.ndarray]:
        """Train the Sinkhorn DRO model with specified optimization strategy.
        
        :param X: Training feature matrix. Shape: (n_samples, n_features)
            Should match model's input dimension (n_features == input_dim)
        :type X: numpy.ndarray

        :param y: Target values. Shape requirements:
        
            - Regression: (n_samples,) continuous values
            
            - Classification: (n_samples,) binary labels (0/1)
        
        :type y: numpy.ndarray

        :param optimization_type: Optimization algorithm selection (Defaults to 'SG'):

            - ``'SG'``: Standard Stochastic Gradient (baseline)

            - ``'MLMC'``: Multilevel Monte Carlo acceleration

            - ``'RTMLMC'``: Real-Time MLMC with adaptive sampling

        :type optimization_type: str

        :return: Learned parameters containing:

            - ``'theta'``: Model coefficients (n_features,)

            - ``'bias'``: Intercept term (if fit_intercept=True)

        :rtype: dict[str, numpy.ndarray]

        :raises ValueError:

            - If X/y have mismatched sample counts (n_samples)

            - If optimization_type not in {'SG', 'MLMC', 'RTMLMC'}

            - If input_dim ≠ X.shape[1]

        Example:
            >>> model = SinkhornDRO(input_dim=5, model_type='svm')
            >>> params = model.fit(X_train, y_train, 'MLMC')
            >>> print(params['weights'])  # [-0.12, 1.45, ...]

        """
        if self.model_type in {'logistic'}:
            is_valid = np.all((y == 0) | (y == 1))
            if not is_valid:
                raise SinkhornDROError("classification labels not in {0, +1} for Sinkhorn-DRO on LR")
        elif self.model_type in {'svm'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise SinkhornDROError("classification labels not in {-1, +1} for Sinkhorn-DRO on SVM")
        

        self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y)

        try:
            if optimization_type == "SG":
                self._sg_optimizer(dataloader)
            elif optimization_type == "MLMC":
                self._mlmc_optimizer(dataloader)
            elif optimization_type == "RTMLMC":
                self._rtmlmc_optimizer(dataloader)
            else:
                raise SinkhornDROError(f"Invalid optimization type: {optimization_type}")
        except RuntimeError as e:
            raise SinkhornDROError(f"Training failed: {str(e)}") from e

        return self._extract_parameters()

    #region Private Methods
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to device tensor."""
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input dimensions."""
        if X.shape[0] != y.shape[0]:
            raise SinkhornDROError("X and y must have same number of samples")
        if X.shape[1] != self.model.linear.in_features:
            raise SinkhornDROError(f"Expected {self.model.linear.in_features} features, got {X.shape[1]}")

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Create PyTorch DataLoader from numpy data."""
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y).reshape(-1, 1)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

    def _extract_parameters(self) -> Dict[str, np.ndarray]:
        """Extract model parameters as numpy arrays."""
        weights = self.model.linear.weight.detach().cpu().numpy().flatten()
        bias = self.model.linear.bias.detach().cpu().numpy()
        return {"theta": weights, "bias": bias}

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                     m: int, lambda_reg: float) -> torch.Tensor:
        """Compute Sinkhorn loss for given batch."""
        # TODO: double check.
        if self.model_type == 'ols':
            residuals = (predictions - targets) ** 2 / lambda_reg
        elif self.model_type == 'lad':
            residuals = torch.abs(predictions - targets)/ lambda_reg
        elif self.model_type == 'logistic':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            residuals = criterion((predictions + 1)/2, (targets + 1) / 2) / lambda_reg
        elif self.model_type == 'svm':
            residuals = torch.clamp(1 - targets * predictions, min = 0) / lambda_reg
        residual_matrix = residuals.view(m, -1)
        return torch.mean(torch.logsumexp(residual_matrix, dim=0)-math.log(m)) * lambda_reg

    def _sg_optimizer(self, dataloader: DataLoader) -> None:
        """Stochastic Gradient optimization."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.reg_param

        for epoch in range(self.max_iter):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                m = 2 ** self.k_sample_max

                # Generate perturbed samples
                noise = torch.randn((m, *data.shape), device=self.device) * math.sqrt(self.reg_param)
                noisy_data = (data + noise).view(-1, data.shape[1])
                repeated_target = target.repeat(m, 1)

                # Forward + backward pass
                optimizer.zero_grad()
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                loss.backward()
                optimizer.step()

    def _mlmc_optimizer(self, dataloader: DataLoader) -> None:
        """Multilevel Monte Carlo optimization."""
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.reg_param
        n_levels = [2 ** (k+1) for k in range(self.k_sample_max)]

        for epoch in range(self.max_iter):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss_total = 0.0

                for k in range(self.k_sample_max):
                    m = 2 ** k
                    subset_size = n_levels[-k-1]
                    # sub_data, sub_target = data[:subset_size], target[:subset_size]
                    subset_size = min(n_levels[-k-1], data.size(0)) 
                    sub_data, sub_target = data[:subset_size], target[:subset_size]
                    # Multilevel sample generation
                    noise = torch.randn((m, subset_size, data.shape[1]), device=self.device)
                    # noisy_data = sub_data + noise.view(-1, data.shape[1]) * math.sqrt(self.reg_param)
                    expanded_data = sub_data.repeat(m, 1) 
                    noisy_data = expanded_data + noise.view(-1, data.shape[1]) * math.sqrt(self.reg_param)
                    repeated_target = sub_target.repeat(m, 1)

                    # Loss computation
                    predictions = self.model(noisy_data)
                    residuals = (predictions - repeated_target) ** 2 / lambda_reg
                    residual_matrix = residuals.view(m, subset_size)

                    if k == 0:
                        loss = torch.mean(torch.logsumexp(residual_matrix, dim=0)) * lambda_reg
                    else:
                        # Multilevel correction
                        half_m = m // 2
                        loss_high = torch.logsumexp(residual_matrix[:half_m], dim=0)
                        loss_low = torch.logsumexp(residual_matrix[half_m:], dim=0)
                        loss = (loss_high.mean() - 0.5 * loss_low.mean()) * lambda_reg

                    loss_total += loss

                loss_total.backward()
                optimizer.step()

    def _rtmlmc_optimizer(self, dataloader: DataLoader) -> None:
        """Randomized Truncated MLMC optimization."""
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.reg_param
        level_probs = np.array([0.5 ** k for k in range(self.k_sample_max)])
        level_probs /= level_probs.sum()

        for epoch in range(self.max_iter):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                # Random level selection
                k = np.random.choice(self.k_sample_max, p=level_probs)
                m = 2 ** k

                # Perturbed sample generation
                noise = torch.randn((m, data.shape[0], data.shape[1]), device=self.device)
                # noisy_data = data + noise.view(-1, data.shape[1]) * math.sqrt(self.reg_param)
                expanded_data = data.repeat(m, 1)  
                noisy_data = expanded_data + noise.view(-1, data.shape[1]) * math.sqrt(self.reg_param)
                repeated_target = target.repeat(m, 1)

                # Loss computation with probability weighting
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                (loss / level_probs[k]).backward()
                optimizer.step()

