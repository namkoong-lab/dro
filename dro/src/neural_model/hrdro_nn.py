import cvxpy as cp
import torch
import torchattacks
import warnings
import numpy as np
from typing import Optional, Dict, Union, List
from dro.src.neural_model.base_nn import BaseNNDRO, DROError

class HRNNDRO(BaseNNDRO):
    """Huberian Robust Distributionally Robust Optimization Model
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        task_type: "classification" or "regression"
        model_type: Base model architecture ('mlp', 'resnet', etc)
        alpha: Robustness to distribution shift (α > 0)
        r: Robustness to statistical error (r > 0)
        epsilon: Adversarial perturbation bound (ε ≥ 0)
        learning_approach: Robust optimization method ("HR" or "HD")
        adversarial_params: Dictionary containing:
            - steps: Number of PGD steps
            - step_size: PGD step size
            - norm: Adversarial norm ("l2" or "l-inf")
            - method: Defense method ("PGD" or "FFGSM")
        train_batch_size: Default training batch size
        device: Computation device
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
        # Initialize base model
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
        attack_cls = {
            'PGD': torchattacks.PGDL2 if params['norm'] == 'l2' else torchattacks.PGD,
            'FFGSM': torchattacks.FFGSM
        }.get(params['method'], torchattacks.PGD)

        self.attack = attack_cls(
            self.model,
            eps=self.epsilon,
            alpha=params['step_size'],
            steps=params['steps']
        )

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

    def criterion(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute robust loss with dynamic batch handling"""
        # Generate adversarial examples
        if self.epsilon > 0:
            inputs = self.current_inputs  # Saved during forward pass
            adv_inputs = self.attack(inputs, labels)
            outputs = self.model(adv_inputs)

        if self.task_type == "classification":
            losses = torch.nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        else:
            losses = torch.nn.MSELoss(reduction="none")(outputs, labels)
        
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
            self.problem.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError:
            try:
                self.loss_param.value += self.numerical_eps
                self.problem.solve(solver=cp.MOSEK, verbose=False)
            except cp.SolverError:
                self.problem.solve(solver=cp.SCS, verbose=False)

        # Get optimal weights
        weights = torch.from_numpy(self.p.value).to(self.device)
        return (weights[:batch_size] * losses).sum() + weights[-1] * losses.max()

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        train_ratio: float = 0.8,
        lr: float = 1e-3,
        batch_size: int = None,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Enhanced training loop with adversarial training"""
        # Use specified batch size or default
        batch_size = batch_size or self.train_batch_size
        
        # Convert inputs to tensor and validate
        X_tensor = self._convert_to_tensor(X)
        y_tensor = self._convert_to_tensor(y, 
            dtype=torch.long if self.task_type == "classification" else None)
        self._validate_input_shape(X_tensor)

        # Call base class fit with modified parameters
        return super().fit(
            X=X_tensor,
            y=y_tensor,
            train_ratio=train_ratio,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Adversarial-robust prediction"""
        X_tensor = self._convert_to_tensor(X)
        self.current_inputs = X_tensor  # Store for adversarial generation
        
        # Generate adversarial examples if needed
        if self.epsilon > 0:
            dummy_targets = torch.zeros(X_tensor.size(0), dtype=torch.long).to(self.device)
            X_tensor = self.attack(X_tensor, dummy_targets)
        
        return super().predict(X_tensor)
    

if __name__ == "__main__":
    # Example usage
    X = np.random.randn(1000, 3, 64, 64)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    model = HRNNDRO(input_dim=3*64*64, num_classes=2, model_type='alexnet', task_type="classification")
    
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