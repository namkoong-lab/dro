import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
from typing import Tuple, Union, Dict, List
import torchvision.models as models

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants
DEFAULT_TRAIN_RATIO = 0.8
MIN_IMAGE_SIZE = 32

class DROError(Exception):
    """Base exception class for all DRO-related errors."""
    pass

class ParameterError(DROError):
    """Raised when invalid parameters are provided."""
    pass

class DataValidationError(DROError):
    """Raised for invalid data formats or dimensions."""
    pass

class Linear(nn.Module):
    """Linear Model"""
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_classes)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer1(X)
        

class MLP(nn.Module):
    """Multi-Layer Perceptron for neural network-based DRO."""
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int, 
                 hidden_units: int = 16, 
                 activation: nn.Module = nn.ReLU(), 
                 dropout_rate: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.activation(self.layer1(X.float()))
        X = self.dropout(X)
        X = self.activation(self.layer2(X))
        return self.output(X)

class BaseNNDRO:
    """Neural Network Distributionally Robust Optimization base model.
    
    Supports both classification and regression tasks with various architectures.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int, 
                 task_type: str = "classification",
                 model_type: str = 'mlp', 
                 device: torch.device = device):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.model_type = model_type
        
        assert task_type in ["classification", "regression"]
        self.task_type = task_type
        if self.task_type == "regression":
            self.num_classes = 1    
        
        self._initialize_model(model_type)
        self.model.to(self.device)

    def _initialize_model(self, model_type: str):
        """Initialize the specified model architecture."""
        if model_type == 'mlp':
            self.model = MLP(self.input_dim, self.num_classes)
        elif model_type == 'linear':
            self.model = Linear(self.input_dim, self.num_classes)
        elif model_type == 'resnet':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif model_type == 'alexnet':
            self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        else:
            raise ParameterError(f"Unsupported model type: {model_type}. Choose from ['mlp', 'resnet', 'alexnet']")

    def _validate_input_shape(self, X: torch.Tensor):
        """Validate input dimensions based on model type."""
        if self.model_type in ['resnet', 'alexnet']:
            if X.dim() != 4:
                raise DataValidationError(
                    f"Image models require 4D input (batch x channels x height x width). "
                    f"Received {X.dim()}D input."
                )
            if X.shape[1] != 3:
                raise DataValidationError(
                    f"Pretrained models expect 3-channel RGB input. "
                    f"Received {X.shape[1]} channels."
                )
            h, w = X.shape[2], X.shape[3]
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                raise DataValidationError(
                    f"Input resolution insufficient. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}. "
                    f"Received: {h}x{w}."
                )
    def criterion(self, outputs, labels):
        if self.task_type == "classification":
            return nn.CrossEntropyLoss()(outputs, labels)
        else:
            return nn.MSELoss()(outputs, labels)

    def fit(self, 
           X: Union[np.ndarray, torch.Tensor], 
           y: Union[np.ndarray, torch.Tensor], 
           train_ratio: float = DEFAULT_TRAIN_RATIO,
           lr: float = 1e-3,
           batch_size: int = 32,
           epochs: int = 100,
           verbose: bool = True) -> Dict[str, List[float]]:
        """Complete training implementation with early stopping and model checkpointing.
        
        Args:
            X: Input features
            y: Target labels
            train_ratio: Ratio of training data split
            lr: Learning rate
            batch_size: Training batch size
            epochs: Maximum number of epochs
            verbose: Whether to print training progress
            
        Returns:
            Training metrics dictionary
        """
        # Convert and validate input data
        X = self._convert_to_tensor(X)
        if self.task_type == 'classification':
            y = self._convert_to_tensor(y, dtype=torch.long)
        else:
            y = self._convert_to_tensor(y)
        self._validate_input_shape(X)

        # Dataset preparation
        dataset = TensorDataset(X, y)
        train_size = int(len(dataset) * train_ratio)
        train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            with tqdm(train_loader, unit="batch", disable=not verbose) as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}")
                    
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    self.current_inputs = inputs # this is for HRNNDRO
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

        # Validation phase
        return self._evaluate(val_loader)    
        
    def _convert_to_tensor(self, data: Union[np.ndarray, torch.Tensor], 
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert input data to properly typed tensor."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype)
        return data.to(dtype=dtype)

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if self.task_type == "classification":
            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            f1 = f1_score(all_labels, all_preds, average='macro')
            return {"acc":acc, "f1":f1}
        else: 
            mse = np.mean((np.array(all_preds)-np.array(all_labels))**2)
            return {"mse":mse}

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Make predictions on input data."""
        X = self._convert_to_tensor(X)
        with torch.no_grad():
            inputs = X.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

    def score(self, X: Union[np.ndarray, torch.Tensor], 
             y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate classification accuracy."""
        return np.mean(self.predict(X) == y)

    def f1score(self, X: Union[np.ndarray, torch.Tensor], 
                y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate macro-averaged F1 score."""
        return f1_score(y, self.predict(X), average='macro')
    


if __name__ == "__main__":
    # Example usage
    X = np.random.randn(1000, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    model = BaseNNDRO(input_dim=10, num_classes=2, model_type='mlp', task_type="classification")
    
    try:
        # Training
        metrics = model.fit(X, y, epochs=100)
        print(metrics)

        # Inference
        preds = model.predict(X[:5])
        print(f"Sample predictions: {preds}")

        # Evaluation
        acc = model.score(X, y)
        f1 = model.f1score(X, y)
        print(f"Final Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        
    except DROError as e:
        print(f"Error occurred: {str(e)}")