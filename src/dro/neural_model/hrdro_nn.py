import cvxpy as cp
import torch
import warnings
import numpy as np
from typing import Optional, Dict, Union
from torch.utils.data import DataLoader
from .base_nn import BaseNNDRO, DROError
from ._attack_utils import make_norm_ball_attacks, run_norm_ball_attack

class HRNNDRO(BaseNNDRO):
    """Holistic Robust DRO.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: str = "classification",
        model_type: str = "mlp",
        alpha: float = 0.1,
        r: float = 0.01,
        epsilon: float = 0.1,
        learning_approach: str = "HD",
        adversarial_params: Optional[dict] = None,
        train_batch_size: int = 64,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """Initialize Holistic Robust DRO.

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

        :param alpha: Robustness regularization coefficient :math:`\\alpha > 0`,
            controls model conservativeness. Defaults to 0.1.
        
        :type alpha: float
        
        :param r: Statistical error radius :math:`r > 0` for distribution shift,
            defaults to 0.01.
        
        :type r: float
        
        :param epsilon: Contamination ratio :math:`\\epsilon \in [0,1)`,
            proportion of adversarial samples. Defaults to 0.1.
        
        :type epsilon: float

        :param learning_approach: Robust optimization method. Options:

            - ``"HR"``: Huberian Robust optimization

            - ``"HD"``: Hybrid DRO (default)

        :type learning_approach: str

        :param adversarial_params: Adversarial training configuration.
            ``norm`` is ``'l2'`` or ``'l-inf'``; ``method`` is ``'PGD'``,
            ``'FGSM'``, or ``'FFGSM'``. FGSM and FFGSM require ``'l-inf'``.
            Torchattacks clips attack inputs to [0, 1]. Defaults to
            ``{"steps":10, "step_size":0.02, "norm":"l2", "method":"PGD"}``.

        :type adversarial_params: dict

        :param train_batch_size: Training batch size :math:`B \geq 1`,
            defaults to 64.

        :type train_batch_size: int
        
        
        :raises ValueError:
            - If alpha ≤ 0
            - If r ≤ 0
            - If ε ∉ [0,1)
            - If invalid learning_approach

        Example (CIFAR-10)::
            >>> adv_params = {"steps":5, "step_size":0.03, "norm":"l2", "method":"PGD"}
            >>> model = HRNNDRO(
            ...     input_dim=3072, 
            ...     num_classes=10,
            ...     alpha=0.5,
            ...     epsilon=0.2,
            ...     adversarial_params=adv_params
            ... )
        """
        # Parameter validation
        if alpha <= 0:
            raise ValueError(f"alpha must be >0, got {alpha}")
        if r <= 0:
            raise ValueError(f"r must be >0, got {r}")
        if not 0 <= epsilon < 1:
            raise ValueError(f"epsilon in [0,1), got {epsilon}")
            
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            task_type=task_type,
            model_type=model_type,
            device=device
        )
        
        # Robustness parameters
        self.alpha = max(alpha, 1e-6)
        self.r = max(r, 1e-6)
        self.epsilon = epsilon
        self.train_batch_size = train_batch_size
        self.learning_approach = learning_approach.upper()
        self.numerical_eps = 1e-6

        # Adversarial training setup
        self.adversarial_params = adversarial_params or {
            'steps': 10,
            'step_size': 0.02,
            'norm': 'l2',
            'method': 'PGD'
        }
        self._init_adversarial_attack()
        self._init_optimization_problem()


    def _init_adversarial_attack(self):
        """Initialize adversarial attack generator"""
        params = self.adversarial_params
        self.attack, self._tabular_attack = make_norm_ball_attacks(
            self.model, params['method'], params['norm'], self.epsilon,
            params['step_size'], params['steps'],
        )

    def update_user_mode(
        self,
        input_dim: int,
        num_classes: int,
        model: torch.nn.Module,
        task_type: str = "classification",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Replace the network and bind both attacks to the new network."""
        super().update_user_mode(input_dim, num_classes, model, task_type, device)
        self._init_adversarial_attack()

    def _init_optimization_problem(self):
        """Initialize CVXPY optimization problem"""
        N = self.train_batch_size
        self.p = cp.Variable(N+1, nonneg=True)
        self.loss_param = cp.Parameter(N)
        self.worst_case = cp.Parameter()

        if self.learning_approach == "HR":
            self._init_hr_constraints(N)
        else:
            self._init_hd_constraints(N)

    def _init_hr_constraints(self, N: int):
        """Initialize HR problem constraints"""
        q = cp.Variable(N+1, nonneg=True)
        s = cp.Variable(N, nonneg=True)
        t = cp.Variable(N)

        constraints = [
            cp.sum(self.p) == 1,
            cp.sum(q) == 1,
            cp.sum(t) <= self.r,
            cp.sum(s) <= self.alpha,
            cp.sum(s) + q[-1] == self.p[-1],
            self.p[:-1] + s == q[:-1],
            cp.ExpCone(-t, (1/N)*np.ones(N), q[:-1])
        ]

        self.problem = cp.Problem(
            cp.Maximize(self.p[:N] @ self.loss_param + self.p[-1] * self.worst_case),
            constraints
        )

    def _init_hd_constraints(self, N: int):
        """Initialize HD problem constraints"""
        q = cp.Variable(N+1, nonneg=True)
        s = cp.Variable(N, nonneg=True)
        t = cp.Variable(N+1)

        constraints = [
            cp.sum(self.p) == 1,
            cp.sum(q) == 1,
            cp.sum(t) <= self.r,
            cp.sum(s) <= self.alpha,
            q[:-1] + s == (1/N)*np.ones(N),
            cp.ExpCone(-t, q, self.p)
        ]

        self.problem = cp.Problem(
            cp.Maximize(self.p[:N] @ self.loss_param + self.p[-1] * self.worst_case),
            constraints
        )

    def _criterion(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute robust loss with dynamic batch handling"""
        # Generate adversarial examples
        if self.epsilon > 0:
            inputs = self.current_inputs  # Saved during forward pass
            adv_inputs = run_norm_ball_attack(
                self.attack, self._tabular_attack, inputs, labels
            )
            outputs = self.model(adv_inputs)

        if self.task_type == "classification":
            losses = torch.nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        else:
            losses = (outputs.reshape(-1) - labels.reshape(-1)).square()
        
        # Dynamic batch size handling
        batch_size = labels.size(0)
        if batch_size != self.train_batch_size:
            warnings.warn(f"Batch size changed from {self.train_batch_size} to {batch_size}, reinitializing problem")
            self.train_batch_size = batch_size
            self._init_optimization_problem()

        # Solve optimization problem
        self.loss_param.value = losses.detach().cpu().numpy()
        self.worst_case.value = losses.max().item()
        
        try:
            self.problem.solve(solver=cp.CLARABEL, verbose=False)
        except cp.SolverError:
            self.loss_param.value += self.numerical_eps
            self.problem.solve(solver=cp.SCS, verbose=False)

        # Get optimal weights
        weights = torch.from_numpy(self.p.value).to(self.device)
        return (weights[:batch_size] * losses).sum() + weights[-1] * losses.max()

    def fit(
        self,
        X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        train_ratio: float = 0.8,
        lr: float = 1e-3,
        batch_size: int = None,
        epochs: int = 100,
        verbose: bool = True,
        *,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        scheduler_step: str = "epoch",
    ) -> Dict[str, float]:
        """Fit HR-DRO using arrays or caller-supplied training components."""
        if batch_size is None:
            batch_size = self.train_batch_size

        if train_loader is None and val_loader is None and X is not None and y is not None:
            X = self._convert_to_tensor(X)
            y = self._convert_to_tensor(
                y, dtype=torch.long if self.task_type == "classification" else torch.float32
            )
            self._validate_input_shape(X)

        return super().fit(
            X=X,
            y=y,
            train_ratio=train_ratio,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
        )

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Adversarial-robust prediction"""
        X_tensor = self._convert_to_tensor(X).to(self.device)
        self.current_inputs = X_tensor  # Store for adversarial generation
        
        # Generate adversarial examples if needed
        if self.epsilon > 0:
            dummy_targets = torch.zeros(X_tensor.size(0), dtype=torch.long, device=self.device)
            X_tensor = run_norm_ball_attack(
                self.attack, self._tabular_attack, X_tensor, dummy_targets
            )
        
        return super().predict(X_tensor)
    

# if __name__ == "__main__":
#     # Example usage
#     X = np.random.randn(1000, 3, 64, 64)  # 100 samples, 10 features
#     y = np.random.randint(0, 2, 1000)  # Binary classification

#     model = HRNNDRO(input_dim=3*64*64, num_classes=2, model_type='alexnet', task_type="classification")
    
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
