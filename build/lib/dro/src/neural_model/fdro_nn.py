from dro.src.neural_model.base_nn import device, DROError, BaseNNDRO
from dro.src.neural_model.fdro_utils import RobustLoss
import torch 
import numpy as np 

class Chi2NNDRO(BaseNNDRO):
    def __init__(self, input_dim: int, num_classes: int, task_type: str = "classification", model_type: str = 'mlp', device: torch.device = device, size: float = 0.1, reg: float = 0.1, max_iter: int = 100):
        super().__init__(input_dim, num_classes, task_type, model_type, device)
        self.reg = reg 
        self.size = size 
        self.max_iter = max_iter
        

    def criterion(self, outputs, labels):
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
        super().__init__(input_dim, num_classes, task_type, model_type, device)
        self.reg = reg 
        self.size = size 
        self.max_iter = max_iter
        

    def criterion(self, outputs, labels):
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


if __name__ == "__main__":
    # Example usage
    X = np.random.randn(1000, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    model = CVaRNNDRO(input_dim=10, num_classes=2, model_type='mlp', task_type="classification", reg=0.0, size=1.0, max_iter=100)
    
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