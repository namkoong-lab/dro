from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Union, Tuple, Any
from sklearn.metrics import f1_score
from dro.src.linear_model.base import BaseLinearDRO


class SinkhornDROError(Exception):
    """Base exception class for errors in Sinkhorn DRO model."""
    pass


class LinearModel(nn.Module):
    """PyTorch Linear Model for regression tasks.
    
    Attributes:
        linear (nn.Linear): Linear layer implementing y = xA^T + b
    """
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear model."""
        return self.linear(x)


class SinkhornLinearDRO(BaseLinearDRO):
    """Sinkhorn Distributionally Robust Optimization with Linear Models.

    Implements three optimization approaches:
    - SG (Standard Stochastic Gradient)
    - MLMC (Multilevel Monte Carlo)
    - RTMLMC (Randomized Truncated MLMC)

    Attributes:
        model (LinearModel): Underlying PyTorch linear model
        reg_param (float): Regularization parameter for Wasserstein distance
        lambda_param (float): Loss scaling parameter
        max_iter (int): Maximum training iterations
        device (torch.device): Computation device (CPU/GPU)

    Reference: <https://arxiv.org/abs/2109.11926>
    """

    def __init__(self,
                 input_dim: int,
                 reg_param: float = 1e-3,
                 lambda_param: float = 1e2,
                 output_dim: int = 1, 
                 max_iter: int = 1e3,
                 learning_rate: float = 1e-2,
                 k_sample_max: int = 5,
                 model_type: str = "svm",
                 device = "cpu"):
        """Initialize Sinkhorn DRO model.

        Args:
            input_dim: Dimension of input features
            reg_param: Regularization strength (ε in paper)
            lambda_param: Loss scaling parameter (λ in paper)
            max_iter: Maximum training iterations
            learning_rate: Learning rate for optimizer
            k_sample_max: Maximum level for MLMC sampling
            model_type: Model type identifier

        Raises:
            SinkhornDROError: For invalid parameter values
        """
        super().__init__(input_dim, model_type)
        
        if reg_param <= 0 or lambda_param <= 0:
            raise SinkhornDROError("Regularization parameters must be positive")

        self.model = LinearModel(input_dim, output_dim)
        self.reg_param = reg_param
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.k_sample_max = k_sample_max
        self.device = device
        self.model.to(self.device)

    def update(self, config: Dict[str, Any]) -> None:
        """Update model hyperparameters.

        Args:
            config: Dictionary containing hyperparameters to update

        Raises:
            SinkhornDROError: For invalid parameter types/values
        """
        if "reg" in config:
            if not isinstance(config["reg"], (float, int)) or config["reg"] <= 0:
                raise SinkhornDROError("reg must be a positive float")
            self.reg_param = float(config["reg"])

        if "lambda" in config:
            if not isinstance(config["lambda"], (float, int)) or config["lambda"] <= 0:
                raise SinkhornDROError("lambda must be a positive float")
            self.lambda_param = float(config["lambda"])
        
        if "k_sample_max" in config:
            if not isinstance(config["k_sample_max"], (int)) or config["k_sample_max"] <= 0:
                raise SinkhornDROError("k_sample_max must be a positive integer")
            self.k_sample_max = float(config["k_sample_max"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate model predictions.

        Args:
            X: Input data matrix (n_samples, n_features)

        Returns:
            Model predictions as numpy array

        Raises:
            SinkhornDROError: For input dimension mismatch
        """
        if X.shape[1] != self.model.linear.in_features:
            raise SinkhornDROError(f"Expected {self.model.linear.in_features} features, got {X.shape[1]}")

        X_tensor = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> Union[float, Tuple[float, float]]:
        """Calculate model performance metrics.

        Args:
            X: Input features
            y: Target values

        Returns:
            MSE for regression tasks, (accuracy, F1) for classification
        """
        predictions = self.predict(X).flatten()
        if self.model_type in ["ols", "lad"]:
            return np.mean((predictions - y.flatten()) ** 2)
        else:
            acc = np.mean(predictions.round() == y.flatten())
            f1 = f1_score(y, predictions.round(), average='macro')
            return acc, f1

    def fit(self, X: np.ndarray, y: np.ndarray, optimization_type: str = "SG") -> Dict[str, np.ndarray]:
        """Train model using specified optimization method.

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            optimization_type: Optimization method (SG/MLMC/RTMLMC)

        Returns:
            Dictionary containing learned parameters

        Raises:
            SinkhornDROError: For invalid optimization type or input mismatch
        """
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
        residuals = (predictions - targets) ** 2 / lambda_reg
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
                    sub_data, sub_target = data[:subset_size], target[:subset_size]

                    # Multilevel sample generation
                    noise = torch.randn((m, subset_size, data.shape[1]), device=self.device)
                    noisy_data = sub_data + noise.view(-1, data.shape[1]) * math.sqrt(self.reg_param)
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
                noisy_data = data + noise.view(-1, data.shape[1]) * math.sqrt(self.reg_param)
                repeated_target = target.repeat(m, 1)

                # Loss computation with probability weighting
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                (loss / level_probs[k]).backward()
                optimizer.step()
    #endregion


if __name__ == "__main__":
    """Example usage with synthetic regression data."""
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, Ridge

    # Data generation
    X, y = make_regression(n_samples=1000, n_features=10, noise=1, random_state=42)
    # Model training
    dro_model = SinkhornLinearDRO(input_dim=10, output_dim=1, k_sample_max=4, reg_param=.001, lambda_param=100, max_iter=1000, model_type='ols')
    params = dro_model.fit(X, y, optimization_type="SG")
    print("Sinkhorn DRO Parameters:", params)
    print(dro_model.score(X, y))

    # Baseline comparison
    lr_model = Ridge()
    lr_model.fit(X, y)
    print("Sklearn Coefficients:", lr_model.coef_)
    print(lr_model.score(X,y))