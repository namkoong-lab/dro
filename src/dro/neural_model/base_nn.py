import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
from typing import Optional, Union, Dict
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

class Linear(torch.nn.Module):
    """Fully-connected neural layer for classification/regression.

    Implements the linear transformation:

    .. math::
        Y = XW^\\top + b

    where:

        - :math:`X \in \mathbb{R}^{N \\times d}`: input features

        - :math:`W \in \mathbb{R}^{K \\times d}`: weight matrix

        - :math:`b \in \mathbb{R}^K`: bias term
        
        - :math:`Y \in \mathbb{R}^{N \\times K}`: output logits

    :param input_dim: Dimension of input features :math:`d`. Must be ≥ 1.
    :type input_dim: int
    :param num_classes: Number of output classes :math:`K`. 
        Use 1 for binary classification.
    :type num_classes: int

    Example::
        >>> model = Linear(input_dim=5, num_classes=3)
        >>> x = torch.randn(32, 5)  # batch_size=32
        >>> y = model(x)
        >>> y.shape
        torch.Size([32, 3])

    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥1, got {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be ≥1, got {num_classes}")
            
        self.layer1 = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer.
        
        :param X: Input tensor of shape :math:`(N, d)`
            where :math:`N` = batch size
        :type X: torch.Tensor
        :return: Output logits of shape :math:`(N, K)`
        :rtype: torch.Tensor
        """
        return self.layer1(X)


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron with dropout regularization.
    
    Implements the forward computation:

    .. math::
        h_1 &= \sigma(W_1 x + b_1) \\\\
        h_2 &= \sigma(W_2 h_1 + b_2) \\\\
        y &= W_o h_2 + b_o

    where:
        - :math:`\sigma`: activation function (default: ReLU)
        - :math:`p \in [0,1)`: dropout probability
        - :math:`x \in \mathbb{R}^d`: input features
        - :math:`y \in \mathbb{R}^K`: output logits

    :param input_dim: Input feature dimension :math:`d \geq 1`
    :type input_dim: int
    :param num_classes: Output dimension :math:`K \geq 1`
    :type num_classes: int
    :param hidden_units: Hidden layer dimension :math:`h \geq 1`, 
        defaults to 16
    :type hidden_units: int
    :param activation: Nonlinear activation module, 
        defaults to :py:class:`torch.nn.ReLU`
    :type activation: torch.nn.Module
    :param dropout_rate: Dropout probability :math:`p \in [0,1)`, 
        defaults to 0.1
    :type dropout_rate: float
    :param num_layers: Number of layers,
        defaults to 2
    :type num_layers: int

    Example::
        >>> model = MLP(
        ...     input_dim=64,
        ...     num_classes=10,
        ...     hidden_units=32,
        ...     activation=nn.GELU(),
        ...     dropout_rate=0.2
        ... )
        >>> x = torch.randn(128, 64)  # batch_size=128
        >>> y = model(x)
        >>> y.shape
        torch.Size([128, 10])

    """
    def __init__(self, input_dim: int, num_classes: int, 
                hidden_units: int = 16, 
                activation: torch.nn.Module = nn.ReLU(),
                dropout_rate: float = 0.1,
                num_layers: int = 2):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥1, got {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be ≥1, got {num_classes}")
        if hidden_units < 1:
            raise ValueError(f"hidden_units must be ≥1, got {hidden_units}")
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate ∈ [0,1), got {dropout_rate}")

        self.num_layers = num_layers
        self.nonlin = activation
        self.dropout = nn.Dropout(dropout_rate)
        layers = [nn.Linear(input_dim, hidden_units), activation, self.dropout]
        for _ in range(num_layers - 2):  
            layers.extend([nn.Linear(hidden_units, hidden_units), activation, self.dropout])
        
        layers.append(nn.Linear(hidden_units, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propagation with dropout regularization.
        
        :param X: Input tensor of shape :math:`(N, d)`
            where :math:`N` = batch size
        :type X: torch.Tensor
        :return: Output logits of shape :math:`(N, K)`
        :rtype: torch.Tensor
        
        """
        return self.model(X.float())


class BaseNNDRO:
    """Neural Network-based Distributionally Robust Optimization (DRO) framework.
    
    Implements the core DRO optimization objective:

    .. math::
        \\min_{\\theta} \\sup_{Q \\in \\mathcal{B}_\\epsilon(P)} \\mathbb{E}_Q[\\ell(f_\\theta(X), y)]

    where:

        - :math:`f_\theta`: Parametric neural network

        - :math:`\mathcal{B}_\epsilon(P)`: Wasserstein ambiguity set

    :ivar robust_obj: Final-epoch estimate of the empirical optimization
        objective, or ``None`` before fitting.

    """

    def __init__(self, 
                input_dim: int, 
                num_classes: int, 
                task_type: str = "classification",
                model_type: str = 'mlp', 
                device: torch.device = torch.device("cpu")):
        """Initialize neural DRO framework.

        :param input_dim: Input feature dimension :math:`d \geq 1`
        :type input_dim: int
        :param num_classes: Output dimension:
            - Classification: :math:`K \geq 2` (number of classes)
            - Regression: Automatically set to 1
        :type num_classes: int
        :param task_type: Learning task type. Supported:
            - ``'classification'``: Cross-entropy loss
            - ``'regression'``: MSE loss
        :type task_type: str
        :param model_type: Neural architecture type. Supported:

            - ``'mlp'``: Multi-Layer Perceptron (default)

            - ``linear``

            - ``resnet``

            - ``alexnet``

        :type model_type: str
        :param device: Target computation device, defaults to CPU
        :type device: torch.device
        :raises ValueError:
            - If input_dim < 1
            - If classification task with num_classes < 2
            - If unsupported model_type

        Example (Classification)::
            
            >>> model = BaseNNDRO(
            ...     input_dim=64,
            ...     num_classes=10,
            ...     task_type="classification",
            ...     model_type="mlp",
            ...     device=torch.device("cuda")
            ... )

        Example (Regression)::
        
            >>> model = BaseNNDRO(
            ...     input_dim=8,
            ...     num_classes=5,  # Auto-override to 1
            ...     task_type="regression"
            ... )

        """
        # Parameter validation
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥1, got {input_dim}")
        if task_type == "classification" and num_classes < 2:
            raise ValueError(f"num_classes must be ≥2 for classification, got {num_classes}")
        if not task_type in {"classification", "regression"}:
            raise ValueError(f"task_type must be classification or regression, got {task_type}")

        self.input_dim = input_dim
        self.num_classes = num_classes if task_type == "classification" else 1
        self.device = device
        self.model_type = model_type
        self.task_type = task_type
        self.robust_obj = None

        self._initialize_model(model_type)
        self.model.to(self.device)

    def _initialize_model(self, model_type: str):
        """Initialize the specified model architecture.
        
        .param model_type: Supported:

            - ``mlp``

            - ``linear``

            - ``resnet``

            - ``alexnet``

        .type model_type: str
        """
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
    
    def _criterion(self, outputs, labels):
        if self.task_type == "classification":
            return nn.CrossEntropyLoss()(outputs, labels)
        else:
            return nn.MSELoss()(outputs.reshape(-1), labels.reshape(-1))

    def update_user_mode(self, 
            input_dim: int, 
            num_classes: int, 
            model: torch.nn.Module, 
            task_type: str = "classification",
            device: torch.device = torch.device("cpu")):
        """Update user's own model

        :param input_dim: Input feature dimension :math:`d \geq 1`
        :type input_dim: int
        :param num_classes: Output dimension:
            - Classification: :math:`K \geq 2` (number of classes)
            - Regression: Automatically set to 1
        :type num_classes: int
        :param model: User's own model
        :type model: torch.nn.Module 
        :param task_type: Learning task type. Supported:
            - ``'classification'``: Cross-entropy loss
            - ``'regression'``: MSE loss
        :type task_type: str
        :param device: Target computation device, defaults to CPU
        :type device: torch.device
        """
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = model 
        self.model_type = "custom"  # Input shape is now the custom module's responsibility.
        self.task_type = task_type
        self.device = device
        self.model.to(self.device)

    def fit(self,
        X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        verbose: bool = True,
        *,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        scheduler_step: str = "epoch",
    ) -> Dict[str, float]:
        """Train a neural DRO model with arrays or user-supplied data loaders.
        
        :param X: Input feature matrix/tensor. Shape: 

            - :math:`(N, d)` where :math:`N` = total samples

            - Supports both numpy arrays and torch tensors

        :type X: Union[numpy.ndarray, torch.Tensor], optional

        :param y: Target labels. Shape:

            - Classification: :math:`(N,)` (class indices). Note that y in {0,1} here.

            - Regression: :math:`(N,)` or :math:`(N, 1)`

        :type y: Union[numpy.ndarray, torch.Tensor], optional
        :param train_ratio: Train-validation split ratio :math:`\in (0,1)`, 
            defaults to 0.8
        :type train_ratio: float
        :param lr: Learning rate :math:`\eta > 0`, defaults to 1e-3
        :type lr: float
        :param batch_size: Mini-batch size :math:`B \geq 1`, 
            defaults to 32
        :type batch_size: int
        :param epochs: Maximum training epochs :math:`T \geq 1`, 
            defaults to 100
        :type epochs: int
        :param verbose: Whether to print epoch-wise metrics, 
            defaults to True
        :type verbose: bool
        :param train_loader: Caller-provided training loader. Supply together
            with val_loader instead of X and y. Each batch must contain an
            input tensor and a target tensor.
        :type train_loader: torch.utils.data.DataLoader, optional
        :param val_loader: Caller-provided validation loader.
        :type val_loader: torch.utils.data.DataLoader, optional
        :param optimizer: PyTorch optimizer built from the current model's
            parameters. If omitted, Adam is constructed using lr.
        :type optimizer: torch.optim.Optimizer, optional
        :param scheduler: Learning-rate scheduler bound to optimizer. Requires
            an explicit optimizer. ReduceLROnPlateau receives the mean
            training loss after each epoch.
        :param scheduler_step: Step the scheduler after each epoch (default)
            or after each batch. ReduceLROnPlateau supports epoch only.
        :type scheduler_step: str

        :return: Validation metrics: acc and f1 for classification, or mse
            for regression.
        :rtype: Dict[str, float]

        :raises ValueError:

            - If input dimensions mismatch

            - If train_ratio ∉ (0,1)

            - If learning rate ≤ 0

        Example (Classification)::
            
            >>> X, y = np.random.randn(1000, 64), np.random.randint(0,2,1000)
            >>> model = BaseNNDRO(input_dim=64, num_classes=2)
            >>> metrics = model.fit(X, y, lr=5e-4, epochs=50)
            >>> print(metrics['acc'])

        Example (Regression)::
        
            >>> X = torch.randn(500, 8)
            >>> y = X @ torch.randn(8,1) + 0.1*torch.randn(500,1)
            >>> model = BaseNNDRO(input_dim=8, num_classes=1, task_type='regression')
            >>> model.fit(X, y.squeeze(), batch_size=64)

        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if scheduler_step not in {"epoch", "batch"}:
            raise ValueError("scheduler_step must be 'epoch' or 'batch'")
        if (train_loader is None) != (val_loader is None):
            raise ValueError("train_loader and val_loader must be supplied together")

        if train_loader is None:
            if X is None or y is None:
                raise ValueError("X and y are required when loaders are not supplied")
            if not 0 < train_ratio < 1:
                raise ValueError("train_ratio must be between 0 and 1")
            if batch_size < 1:
                raise ValueError("batch_size must be at least 1")
            if self.task_type == "classification":
                num_labels = len(np.unique(y))
                if num_labels > self.num_classes:
                    raise DataValidationError("input class number is larger than num_classes")

            X = self._convert_to_tensor(X)
            y = self._convert_to_tensor(
                y, dtype=torch.long if self.task_type == "classification" else torch.float32
            )
            self._validate_input_shape(X)
            dataset = TensorDataset(X, y)
            train_size = int(len(dataset) * train_ratio)
            if train_size == 0 or train_size == len(dataset):
                raise ValueError("training and validation splits must both be nonempty")
            train_set, val_set = random_split(
                dataset, [train_size, len(dataset) - train_size]
            )
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size)
        elif X is not None or y is not None:
            raise ValueError("X and y cannot be combined with external loaders")

        if optimizer is None:
            if scheduler is not None:
                raise ValueError("pass an optimizer when using a scheduler")
            if lr <= 0:
                raise ValueError("lr must be positive")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")

        model_parameters = {id(param) for param in self.model.parameters()}
        optimizer_parameters = {
            id(param) for group in optimizer.param_groups for param in group["params"]
        }
        if not model_parameters.intersection(optimizer_parameters):
            raise ValueError("optimizer is not bound to the current model")
        if scheduler is not None:
            if not callable(getattr(scheduler, "step", None)):
                raise TypeError("scheduler must provide a step() method")
            if getattr(scheduler, "optimizer", optimizer) is not optimizer:
                raise ValueError("scheduler is not bound to the supplied optimizer")
            if (scheduler_step == "batch" and
                    isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                raise ValueError("ReduceLROnPlateau requires scheduler_step='epoch'")

        try:
            batches_per_epoch = len(train_loader)
        except TypeError:
            batches_per_epoch = None
        if batches_per_epoch == 0:
            raise ValueError("train_loader must not be empty")
        total_batches = epochs * batches_per_epoch if batches_per_epoch is not None else None
        self.robust_obj = None

        with tqdm(total=total_batches, unit="batch", disable=not verbose) as pbar:
            for _ in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                batch_count = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    self.current_inputs = inputs
                    loss = self._criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if scheduler is not None and scheduler_step == "batch":
                        scheduler.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                    pbar.update(1)

                if batch_count == 0:
                    raise ValueError("train_loader must not be empty")
                avg_epoch_loss = epoch_loss / batch_count
                self.robust_obj = float(avg_epoch_loss)
                pbar.set_postfix(loss=avg_epoch_loss)
                if scheduler is not None and scheduler_step == "epoch":
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_epoch_loss)
                    else:
                        scheduler.step()

        return self._evaluate(val_loader)
        
    def _convert_to_tensor(self, data: Union[np.ndarray, torch.Tensor], 
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert input data to properly typed tensor."""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype)
        return data.to(dtype=dtype)

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                if self.task_type == "classification":
                    preds = outputs.argmax(dim=1)
                else:
                    preds = outputs.reshape(-1)
                all_preds.extend(preds.cpu().numpy().reshape(-1))
                all_labels.extend(labels.cpu().numpy().reshape(-1))

        if not all_labels:
            raise ValueError("val_loader must not be empty")
        
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
            if self.task_type == "classification":
                outputs = outputs.argmax(dim=1)
            return outputs.reshape(-1).cpu().numpy()

    def score(self, X: Union[np.ndarray, torch.Tensor], 
             y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate classification accuracy and macro-F1 score."""
        return self.acc(X, y), self.f1score(X, y)

    def acc(self, X: Union[np.ndarray, torch.Tensor], 
             y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate classification accuracy."""
        return np.mean(self.predict(X) == y)

    def f1score(self, X: Union[np.ndarray, torch.Tensor], 
                y: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate macro-averaged F1 score."""
        return f1_score(y, self.predict(X), average='macro')
