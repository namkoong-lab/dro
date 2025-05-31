from .base_nn import device, MLP, BaseNNDRO
from .fdro_utils import RobustLoss
import torch 
import numpy as np 


class Chi2NNDRO(BaseNNDRO):
    """Chi-square Divergence-based Neural DRO Model.
    """

    def __init__(self, input_dim: int, num_classes: int, 
                task_type: str = "classification", 
                model_type: str = 'mlp', 
                device: torch.device = device,
                size: float = 0.1, 
                reg: float = 0.1,
                max_iter: int = 100):
        r"""Initialize Chi-square DRO neural model.

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

        :param size: Chi-square divergence radius :math:`\rho \geq 0` controlling distributional robustness. Larger values increase conservativeness. Defaults to 0.1.
        
        :type size: float

        :param reg: Regularization coefficient. Defaults to 0.1.
        
        :type reg: float

        :param max_iter: Maximum Sinkhorn iterations. Defaults to 100.

        :type max_iter: int


        :raises ValueError:

            - If size < 0

            - If reg ≤ 0

            - If max_iter < 1

        """
        if size < 0:
            raise ValueError(f"size must be ≥0, got {size}")
        if reg < 0:
            raise ValueError(f"reg must be >=0, got {reg}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be ≥1, got {max_iter}")

        super().__init__(input_dim, num_classes, task_type, model_type, device)
        self.size = size    
        self.reg = reg      
        self.max_iter = max_iter  

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.layer_num = config["layer_num"]
        self.model = MLP(self.input_dim, self.num_classes, hidden_units=config["hidden_size"], dropout_rate=config["dropout_ratio"], num_layers=self.layer_num).to(self.device)

        self.size = config["size"]
        self.reg = config["reg"]

        
    def _criterion(self, outputs, labels):
        is_regression = False
        if self.task_type == "regression":
            is_regression = True
        loss = RobustLoss(
            geometry='chi-square',
            size=self.size,
            reg=self.reg,
            max_iter=self.max_iter,
            is_regression=is_regression)
        return loss(outputs, labels)
    
class CVaRNNDRO(BaseNNDRO):
    def __init__(self, input_dim: int, num_classes: int, task_type: str = "classification", model_type: str = 'mlp', device: torch.device = device, size: float = 0.1, reg: float = 0.1, max_iter: int = 100):
        r"""Initialize Conditional Value-at-Risk DRO neural model.

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

        :param size: CVaR ratio :math:`1 \geq \rho \geq 0` controlling distributional robustness. Smaller values increase conservativeness. Defaults to 0.1.
        
        :type size: float

        :param reg: Regularization coefficient. Defaults to 0.1.
        
        :type reg: float

        :param max_iter: Maximum Sinkhorn iterations. Defaults to 100.

        :type max_iter: int


        :raises ValueError:

            - If size < 0

            - If reg ≤ 0

            - If max_iter < 1

        """

        if size < 0 or size > 1:
            raise ValueError(f"size must be between 0 and 1, got {size}")
        if reg <= 0:
            raise ValueError(f"reg must be >0, got {reg}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be ≥1, got {max_iter}")
        
        super().__init__(input_dim, num_classes, task_type, model_type, device)
        self.reg = reg 
        self.size = size 
        self.max_iter = max_iter
    
    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.layer_num = config["layer_num"]
        self.model = MLP(self.input_dim, self.num_classes, hidden_units=config["hidden_size"], dropout_rate=config["dropout_ratio"], num_layers=self.layer_num).to(self.device)
        self.size = config["size"]
        self.reg = config["reg"]
        

    def _criterion(self, outputs, labels):
        is_regression = False
        if self.task_type == "regression":
            is_regression = True
        loss = RobustLoss(
            geometry='cvar',
            size=self.size,
            reg=self.reg,
            max_iter=self.max_iter,
            is_regression=is_regression)
        return loss(outputs, labels)
    
# if __name__ == "__main__":
#     # Example usage
#     X = np.random.randn(1000, 10)  # 100 samples, 10 features
#     y = np.random.randint(0, 2, 1000)  # Binary classification

#     model = CVaRNNDRO(input_dim=10, num_classes=2, model_type='mlp', task_type="classification", reg=0.0, size=1.0, max_iter=100)
    
#     try:
#         # Training
#         metrics = model.fit(X, y, epochs=100)
#         print(metrics)

#         # Inference
#         preds = model.predict(X[:5])
#         print(f"Sample predictions: {preds}")

#         # Evaluation
#         acc = model.score(X, y)
#         f1 = model.f1score(X, y)
#         print(f"Final Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        
#     except DROError as e:
#         print(f"Error occurred: {str(e)}")