import torch
import torchattacks
import numpy as np
from dro.src.neural_model.base_nn import BaseNNDRO, DROError

class WNNDRO(BaseNNDRO):
    """Wasserstein Distributionally Robust Optimization Model
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        task_type: "classification" or "regression"
        model_type: Base model architecture ('mlp', 'resnet', etc)
        epsilon: Adversarial perturbation bound (ε ≥ 0)
        adversarial_params: 
            - adversarial_steps: Number of PGD steps
            - adversarial_step_size: PGD step size
            - advresarial_norm: Adversarial norm ("l2" or "l-inf")
            - adversarial_method: Defense method ("PGD" or "FFGSM")
        device: Computation device
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: str = "classification",
        model_type: str = "mlp",
        epsilon: float = 0.1,
        adversarial_steps: int = 10,
        adversarial_step_size: float = 0.02,
        adversarial_norm: str = 'l2',
        adversarial_method: str = 'PGD',
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # Initialize base model
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            task_type=task_type,
            model_type=model_type,
            device=device
        )
        
        # Robustness parameters
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.adversarial_step_size = adversarial_step_size
        self.adversarial_norm = adversarial_norm
        self.adversarial_method = adversarial_method
        
        self._init_adversarial_attack()
        
    def _init_adversarial_attack(self):
        """Initialize adversarial attack generator"""
        attack_cls = {
            'PGD': torchattacks.PGDL2 if self.adversarial_norm == 'l2' else torchattacks.PGD,
            'FFGSM': torchattacks.FFGSM
        }.get(self.adversarial_method, torchattacks.PGD)

        self.attack = attack_cls(
            self.model,
            eps=self.epsilon,
            alpha=self.adversarial_step_size,
            steps=self.adversarial_steps
        )

    def criterion(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute robust loss with dynamic batch handling"""
        # Generate adversarial examples
        if self.epsilon > 0:
            inputs = self.current_inputs  # Saved during forward pass see BaseNNDRO.fit()
            adv_inputs = self.attack(inputs, labels)
            outputs = self.model(adv_inputs)

        if self.task_type == "classification":
            losses = torch.nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        else:
            losses = torch.nn.MSELoss(reduction="none")(outputs, labels)
        
        return torch.mean(losses)
    

if __name__ == "__main__":
    # Example usage
    X = np.random.randn(1000, 3, 64, 64)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    model = WNNDRO(input_dim=3*64*64, num_classes=2, model_type='alexnet', task_type="classification")
    
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