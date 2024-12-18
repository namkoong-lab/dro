import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import math
from sklearn.metrics import f1_score
from typing import Dict, Union, Tuple


class LinearModel(nn.Module):
    """
    Simple Linear Model for regression or classification tasks.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert numpy array or torch tensor to a torch tensor with float32 dtype.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        raise TypeError("Input should be either numpy array or torch tensor.")


class SinkhornLinearDRO:
    """
    Sinkhorn Distributionally Robust Optimization (DRO) with Linear Models.
    """
    def __init__(self, 
                 input_dim: int, 
                 reg_: float = 1.0, 
                 lambda_: float = 1.0, 
                 output_dim: int = 1, 
                 max_iter: int = 50, 
                 learning_rate: float = 1e-2, 
                 k_sample_max: int = 5, 
                 is_regression: bool = True):
        self.model = LinearModel(input_dim, output_dim)
        self.lambda_ = lambda_
        self.reg_ = reg_
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.k_sample_max = k_sample_max
        self.is_regression = is_regression

    def update(self, config: Dict[str, Union[float, int]]) -> None:
        """
        Update model parameters based on a configuration dictionary.
        """
        self.reg_ = config.get("reg", self.reg_)
        self.lambda_ = config.get("lambda", self.lambda_)
        self.k_sample_max = config.get("k", self.k_sample_max)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input data.
        """
        X_tensor = to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor).cpu()
            if not self.is_regression:
                pred = (pred > 0.5).float()
        return pred.numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> Union[float, Tuple[float, float]]:
        """
        Calculate the accuracy or mean squared error depending on the task type.
        """
        predictions = self.predict(X).flatten()
        if self.is_regression:
            return np.mean((predictions - y.flatten()) ** 2)
        else:
            accuracy = np.mean(predictions == y.flatten())
            f1 = f1_score(y, predictions, average='macro')
            return accuracy, f1

    def fit(self, X: np.ndarray, y: np.ndarray, optimization_type: str = "SG") -> Dict[str, np.ndarray]:
        """
        Train the model using the specified optimization method.
        """
        X_tensor = to_tensor(X)
        Y_tensor = to_tensor(y).reshape(-1, 1)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        if optimization_type == "SG":
            self._sdro_sg_solver(dataloader)
        elif optimization_type == "MLMC":
            self._sdro_mlmc_solver(dataloader)
        elif optimization_type == "RTMLMC":
            self._sdro_rtmlmc_solver(dataloader)
        else:
            raise NotImplementedError(f"Optimization type {optimization_type} is not supported.")

        theta = self.model.linear.weight.detach().cpu().numpy()
        bias = self.model.linear.bias.detach().cpu().numpy()

        return {"theta": theta, "bias": bias}

    def _sdro_sg_solver(self, dataloader: DataLoader) -> None:
        """
        SGD-based Sinkhorn DRO solver for regression problems.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        lambda_reg = self.lambda_ * self.reg_

        for epoch in range(self.max_iter):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                batch_size, dim = data.shape

                # Generate stochastic samples
                m = 2 ** self.k_sample_max
                data_noise = data + torch.randn((m, batch_size, dim), device=device) * math.sqrt(self.reg_)
                data_noise_flat = data_noise.reshape(-1, dim)
                target_repeated = target.repeat(m, 1).reshape(-1, 1)

                # Forward pass
                predictions = self.model(data_noise_flat)
                residuals = (predictions - target_repeated) ** 2 / lambda_reg

                # Compute Sinkhorn loss
                residual_matrix = residuals.reshape(m, batch_size)
                loss = torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _sdro_mlmc_solver(self, dataloader: DataLoader) -> None:
        """
        MLMC-based Sinkhorn DRO solver (multi-level Monte Carlo).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        lambda_reg = self.lambda_ * self.reg_
        n_ell_hist = np.array([2 ** (k + 1) for k in range(self.k_sample_max)])

        for epoch in range(self.max_iter):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                batch_size, dim = data.shape
                optimizer.zero_grad()

                loss_sum = 0.0

                for k in range(self.k_sample_max):
                    m = 2 ** k
                    n_ell = n_ell_hist[-k - 1]
                    data_ell = data[:n_ell]
                    target_ell = target[:n_ell]

                    # Generate noisy data
                    data_noise = data_ell + torch.randn((m, n_ell, dim), device=device) * math.sqrt(self.reg_)
                    data_noise_flat = data_noise.reshape(-1, dim)
                    target_repeated = target_ell.repeat(m, 1).reshape(-1, 1)

                    # Forward pass
                    predictions = self.model(data_noise_flat)
                    residuals = (predictions - target_repeated) ** 2 / lambda_reg
                    residual_matrix = residuals.reshape(m, n_ell)

                    # Compute multi-level loss
                    if k == 0:
                        loss_k = torch.mean(torch.logsumexp(residual_matrix, dim=0)) * lambda_reg
                    else:
                        m_half = m // 2
                        residual_half = residual_matrix[:m_half]
                        residual_remain = residual_matrix[m_half:]
                        loss_k_1 = torch.mean(torch.logsumexp(residual_matrix, dim=0)) * lambda_reg
                        loss_k_2 = (torch.mean(torch.logsumexp(residual_half, dim=0)) +
                                    torch.mean(torch.logsumexp(residual_remain, dim=0))) * lambda_reg
                        loss_k = loss_k_1 - 0.5 * loss_k_2

                    loss_sum += loss_k

                # Backward pass and optimization
                loss_sum.backward()
                optimizer.step()


    def _sdro_rtmlmc_solver(self, dataloader: DataLoader) -> None:
        """
        RTMLMC-based Sinkhorn DRO solver (randomized truncated MLMC).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        lambda_reg = self.lambda_ * self.reg_
        k_sample_max = self.k_sample_max

        # Define probabilities for truncated geometric sampling
        elements = np.arange(k_sample_max)
        probabilities = 0.5 ** elements
        probabilities /= probabilities.sum()

        for epoch in range(self.max_iter):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                batch_size, dim = data.shape
                optimizer.zero_grad()

                # Sample K from the truncated geometric distribution
                k = np.random.choice(elements, p=probabilities)
                m = 2 ** k

                # Generate noisy data
                data_noise = data + torch.randn((m, batch_size, dim), device=device) * math.sqrt(self.reg_)
                data_noise_flat = data_noise.reshape(-1, dim)
                target_repeated = target.repeat(m, 1).reshape(-1, 1)

                # Forward pass
                predictions = self.model(data_noise_flat)
                residuals = (predictions - target_repeated) ** 2 / lambda_reg
                residual_matrix = residuals.reshape(m, batch_size)

                # Compute loss for the sampled level
                if m == 1:
                    loss = torch.mean(torch.logsumexp(residual_matrix, dim=0)) * lambda_reg
                else:
                    m_half = m // 2
                    residual_half = residual_matrix[:m_half]
                    residual_remain = residual_matrix[m_half:]
                    loss_1 = torch.mean(torch.logsumexp(residual_matrix, dim=0)) * lambda_reg
                    loss_2 = (torch.mean(torch.logsumexp(residual_half, dim=0)) +
                            torch.mean(torch.logsumexp(residual_remain, dim=0))) * lambda_reg
                    loss = loss_1 - 0.5 * loss_2

                # Scale loss by probability of the sampled level
                loss *= 1 / probabilities[k]

                # Backward pass and optimization
                loss.backward()
                optimizer.step()



if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    # Generate synthetic data
    sample_size, feature_size = 1000, 10
    X, y = make_regression(n_samples=sample_size, n_features=feature_size, noise=1, random_state=42)

    # Fit Sinkhorn DRO model
    method = SinkhornLinearDRO(input_dim=feature_size, reg_=1.0, lambda_=1000.0, max_iter=2000)
    params = method.fit(X, y)
    print("Sinkhorn DRO Parameters:", params)

    # Fit and compare with sklearn LinearRegression
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    print("Sklearn LinearRegression Coefficients:", model.coef_)
