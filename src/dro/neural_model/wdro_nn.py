import torch
import numpy as np
from typing import Optional, Tuple
from .base_nn import BaseNNDRO, DROError
from ._attack_utils import make_norm_ball_attacks, run_norm_ball_attack
from ._certificates import wasserstein_radius

class WNNDRO(BaseNNDRO):
    r"""Neural adversarial training with norm-ball or transport-penalty objectives.

    ``norm_ball`` uses fixed-radius attacks. ``transport_penalty`` uses the
    squared Euclidean transport cost from Sinha et al. (2018).
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
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        *,
        robust_objective: str = 'norm_ball',
        gamma: float = 1.0,
        input_bounds: Optional[Tuple[float, float]] = None
    ):
        r"""Initialize a neural adversarial training model.

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

        :param epsilon: Radius :math:`\epsilon \geq 0` of the per-example
            :math:`\ell_2` or :math:`\ell_\infty` attack ball in ``norm_ball``
            mode. Zero disables norm-ball perturbations. Defaults to 0.1.
        :type epsilon: float
        :param adversarial_steps: Number of inner attack iterations :math:`T_{adv} \geq 1`
            for PGD or transport-penalty ascent. FGSM uses one step.
            Defaults to 10.
        :type adversarial_steps: int
        :param adversarial_step_size: Inner attack step size :math:`\eta_{adv} > 0`
            for PGD or transport-penalty ascent. FGSM uses ``epsilon``.
            Defaults to 0.02.
        :type adversarial_step_size: float

        :param adversarial_norm: Perturbation norm in ``norm_ball`` mode. Options:

            - ``'l2'``: :math:`\ell_2`-ball constraint

            - ``'l-inf'``: :math:`\ell_\infty`-ball constraint

        :type adversarial_norm: str

        :param adversarial_method: Attack method in ``norm_ball`` mode. Options:

            - ``'PGD'``: Projected Gradient Descent (default)

            - ``'FGSM'``: Fast Gradient Sign Method (requires ``'l-inf'``)

            - ``'FFGSM'``: Fast FGSM with a random start (requires ``'l-inf'``)

        :type adversarial_method: str

        :param robust_objective: ``'norm_ball'`` (default) or
            ``'transport_penalty'``. The latter approximately optimizes the
            penalized squared-Wasserstein objective of Sinha et al. (2018).
        :type robust_objective: str
        :param gamma: Positive coefficient of the squared Euclidean transport
            cost in ``transport_penalty`` mode. Larger values discourage
            perturbations. Defaults to 1.0.
        :type gamma: float
        :param input_bounds: Optional lower and upper input bounds for
            ``transport_penalty`` mode. For images in [0, 1], use ``(0, 1)``.
            Defaults to no clipping.
        :type input_bounds: Optional[Tuple[float, float]]
        
        
        :raises ValueError:
            - If epsilon < 0
            - If adversarial_steps < 1
            - If adversarial_step_size ≤ 0
            - If robust_objective, gamma, input_bounds, or norm-ball attack
              configuration is invalid

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
        if robust_objective not in {'norm_ball', 'transport_penalty'}:
            raise ValueError(f"Unsupported robust_objective: {robust_objective}")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma must be finite and >0, got {gamma}")
        if input_bounds is not None:
            if (len(input_bounds) != 2 or not np.all(np.isfinite(input_bounds))
                    or input_bounds[0] >= input_bounds[1]):
                raise ValueError("input_bounds must be a finite (lower, upper) pair")

        super().__init__(input_dim, num_classes, task_type, model_type, device)
        
        # Initialize adversarial components
        self.epsilon = epsilon                  
        self.adversarial_steps = adversarial_steps      
        self.adversarial_step_size = adversarial_step_size  
        self.adversarial_norm = adversarial_norm        
        self.adversarial_method = adversarial_method    
        self.robust_objective = robust_objective
        self.gamma = gamma
        self.input_bounds = input_bounds

        self.attack = None
        self._tabular_attack = None
        if self.robust_objective == 'norm_ball':
            self._init_adversarial_attack()
        
    def _init_adversarial_attack(self):
        """Initialize adversarial attack generator"""
        self.attack, self._tabular_attack = make_norm_ball_attacks(
            self.model, self.adversarial_method, self.adversarial_norm,
            self.epsilon, self.adversarial_step_size, self.adversarial_steps,
        )

    def update_user_mode(
        self,
        input_dim: int,
        num_classes: int,
        model: torch.nn.Module,
        task_type: str = "classification",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Replace the network and bind norm-ball attacks to the new network."""
        super().update_user_mode(input_dim, num_classes, model, task_type, device)
        if self.robust_objective == 'norm_ball':
            self._init_adversarial_attack()

    def _transport_attack(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Approximately maximize loss minus gamma times squared input transport."""
        origin = inputs.detach()
        adversarial = origin.clone()
        was_training = self.model.training
        self.model.eval()  # Keep dropout and batch statistics fixed during ascent.
        try:
            for _ in range(self.adversarial_steps):
                adversarial = adversarial.detach().requires_grad_(True)
                with torch.enable_grad():
                    losses = self._loss(self.model(adversarial), labels)
                    transport_cost = (adversarial - origin).flatten(1).square().sum(dim=1)
                    objective = (losses - self.gamma * transport_cost).sum()
                    gradient = torch.autograd.grad(objective, adversarial)[0]
                adversarial = adversarial.detach() + self.adversarial_step_size * gradient.detach()
                if self.input_bounds is not None:
                    adversarial = adversarial.clamp(*self.input_bounds)
        finally:
            self.model.train(was_training)
        return adversarial.detach()
    
    def _loss(self, outputs, labels):
        if self.task_type == "classification":
            losses = torch.nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        else:
            losses = torch.nn.MSELoss(reduction="none")(
                outputs, labels.reshape_as(outputs)
            ).reshape(outputs.shape[0], -1).mean(dim=1)
        return losses

    def _criterion(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute robust loss with dynamic batch handling"""
        if self.robust_objective == 'transport_penalty':
            inputs = self.current_inputs
            adv_inputs = self._transport_attack(inputs, labels)
            adv_losses = self._loss(self.model(adv_inputs), labels)
            transport_cost = (adv_inputs - inputs).flatten(1).square().sum(dim=1)
            # The original input is always feasible for the inner maximization.
            clean_losses = self._loss(outputs, labels)
            penalized_losses = adv_losses - self.gamma * transport_cost
            return torch.maximum(clean_losses, penalized_losses).mean()

        # Generate adversarial examples
        if self.epsilon > 0:
            inputs = self.current_inputs  # Saved during forward pass see BaseNNDRO.fit()
            adv_inputs = run_norm_ball_attack(
                self.attack, self._tabular_attack, inputs, labels
            )
            outputs = self.model(adv_inputs)

        losses = self._loss(outputs, labels)
        
        return torch.mean(losses)

    def certify_radius(
        self, transport_budget, adversarial_losses, transport_costs,
        certified_inner_errors, *, transport_penalty=None,
    ):
        r"""Certify an empirical squared-cost Wasserstein budget.

        Supply per-example losses, squared Euclidean costs, and **proven
        global** inner-maximization error bounds for the same adversarial
        examples, final network, labels, and admissible support. The method
        does not infer error bounds from attack iterations. ``transport_budget``
        is distinct from the ``norm_ball`` constructor's ``epsilon``.

        ``transport_penalty`` defaults to ``gamma`` in ``transport_penalty``
        mode; in ``norm_ball`` mode it must be supplied explicitly. Returns
        ``None`` when the achieved loss cannot be certified at a nonnegative
        budget. Otherwise returns ``epsilon_cert``, the certified loss bound,
        the upper bound at the input budget, and their gap in a dictionary.
        The result concerns the empirical reference distribution only.
        """
        if transport_penalty is None:
            if self.robust_objective != 'transport_penalty':
                raise ValueError("transport_penalty is required in norm_ball mode")
            transport_penalty = self.gamma
        return wasserstein_radius(
            transport_budget, adversarial_losses, transport_costs,
            certified_inner_errors, transport_penalty,
        )
