import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
from typing import Optional, Tuple, Union

class DROError(Exception):
    """Base exception class for all errors in DRO models."""
    pass

class ParameterError(DROError):
    """Exception raised for invalid parameter configurations."""
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
        model_type (str): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'linear' for Linear Regression).
        theta (np.ndarray): Model parameters.
    """
    def __init__(self, input_dim: int, model_type: str = 'svm'):
        if input_dim <= 0:
            raise ParameterError("Input dimension must be a positive integer.")
        if model_type not in {'svm', 'logistic', 'linear', 'lad'}:
            raise ParameterError(f"Unsupported model_type: {model_type}. Supported types are svm, logistic regression, or l1/l2 linear regression.")
        
        self.input_dim = input_dim
        self.model_type = model_type
        self.theta = np.zeros(self.input_dim)

    def update(self, config: dict):
        """Update model parameters based on configuration."""
        pass  # Method to be implemented in subclasses if needed

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model to data by solving an optimization problem."""
        pass  # Method to be implemented in subclasses

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

        scores = X @ self.theta
        if self.model_type == 'linear':
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
        
        if self.model_type == 'linear':
            mse = np.average((predictions - y) ** 2, weights=weights)
            return mse
        else:
            accuracy = np.average(predictions == y, weights=weights)
            f1 = f1_score(y, predictions, average='macro')
            return accuracy, f1

    def _loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the loss for the current model parameters."""

        if self.model_type == 'svm':
            new_y = 2 * y - 1
            return np.maximum(1 - new_y * (X @ self.theta), 0)
        elif self.model_type in {'logistic', 'linear'}:
            return (y - X @ self.theta) ** 2
        elif self.model_type == 'lad':
            return np.abs(y - X @ self.theta)
        else:
            raise NotImplementedError("Loss function not implemented for the specified model_type value.")

    def _cvx_loss(self, X: cp.Expression, y: cp.Expression, theta: cp.Expression) -> cp.Expression:
        """Define the CVXPY loss expression for the model."""
        assert X.shape[-1] == self.input_dim, "Mismatch between feature and input dimension."

        if self.model_type == 'svm':
            new_y = 2 * y - 1
            return cp.pos(1 - cp.multiply(new_y, X @ theta))
        elif self.model_type in {'logistic', 'linear'}:
            return cp.power(y - X @ theta, 2)
        elif self.model_type == 'lad':
            return cp.abs(y - X @ theta)
        else:
            raise NotImplementedError("CVXPY loss not implemented for the specified model_type value.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_FRAC = 0.8
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, num_classes)
        
    def forward(self, X, **kwargs):
        return self.dense0(X.float())


class BaseNNDRO:
    """Base class for Neural Network Distributionally Robust Optimization (DRO) models.
    
    This class supports both regression and binary classification tasks. 

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'linear' for Linear Regression).
        theta (np.ndarray): Model parameters.
    """
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.model = MLP(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("accuracy", correct / total)

        return correct / total





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
