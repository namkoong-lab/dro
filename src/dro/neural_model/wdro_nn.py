import torch
import torchattacks
import numpy as np
from .base_nn import BaseNNDRO, DROError

class WNNDRO(BaseNNDRO):
    r"""Wasserstein Neural DRO with Adversarial Robustness. 
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
        """Initialize Wasserstein DRO model with adversarial training.

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

        :param epsilon: Dual parameter (parameter of the L2-penalty during adversarial training) :math:`\epsilon \geq 0` controlling distributional robustness.
            Larger values increase model conservativeness. Defaults to 0.1.
        :type epsilon: float
        :param adversarial_steps: Number of PGD attack iterations :math:`T_{adv} \geq 1`.
            Defaults to 10.
        :type adversarial_steps: int
        :param adversarial_step_size: PGD step size :math:`\eta_{adv} > 0`.
            Defaults to 0.02.
        :type adversarial_step_size: float

        :param adversarial_norm: Adversarial perturbation norm type. Options:

            - ``'l2'``: :math:`\ell_2`-ball constraint

            - ``'l-inf'``: :math:`\ell_\infty`-ball constraint

        :type adversarial_norm: str

        :param adversarial_method: Adversarial example generation method. Options:

            - ``'PGD'``: Projected Gradient Descent (default)

            - ``'FGSM'``: Fast Gradient Sign Method

        :type adversarial_method: str
        
        
        :raises ValueError:
            - If epsilon < 0
            - If adversarial_steps < 1
            - If adversarial_step_size ≤ 0
            - If invalid norm/method type

        Example (MNIST)::
            >>> model = WNNDRO(
            ...     input_dim=784,
            ...     num_classes=10,
            ...     epsilon=0.5,
            ...     adversarial_norm="l-inf",
            ...     adversarial_steps=7
            ... )
        
        """
        # Parameter validation
        if epsilon < 0:
            raise ValueError(f"epsilon must be ≥0, got {epsilon}")
        if adversarial_steps < 1:
            raise ValueError(f"adversarial_steps must be ≥1, got {adversarial_steps}")
        if adversarial_step_size <= 0:
            raise ValueError(f"adversarial_step_size must be >0, got {adversarial_step_size}")

        super().__init__(input_dim, num_classes, task_type, model_type, device)
        
        # Initialize adversarial components
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
    
    def _loss(self, outputs, labels):
        if self.task_type == "classification":
            losses = torch.nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        else:
            losses = torch.nn.MSELoss(reduction="none")(outputs, labels)
        return losses

    def _criterion(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute robust loss with dynamic batch handling"""
        # Generate adversarial examples
        if self.epsilon > 0:
            inputs = self.current_inputs  # Saved during forward pass see BaseNNDRO.fit()
            adv_inputs = self.attack(inputs, labels)
            outputs = self.model(adv_inputs)

        losses = self._loss(outputs, labels)
        
        return torch.mean(losses)