"""Group distributionally robust neural networks.

The adversarial group weights use the exponentiated-gradient update introduced
in the reference Group DRO implementation:
https://github.com/kohpangwei/group_DRO/blob/master/loss.py
"""

from numbers import Integral
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn

from .base_nn import BaseNNDRO, DataValidationError, ParameterError, device


class GroupNNDRO(BaseNNDRO):
    r"""Neural Group Distributionally Robust Optimization model.

    The model minimizes a weighted combination of empirical group losses while
    an adversary updates its group probabilities by exponentiated gradient:

    .. math::
        q_g \leftarrow
        \frac{q_g\exp(\eta_q\,\ell_g)}
        {\sum_j q_j\exp(\eta_q\,\ell_j)}.

    Repeated updates concentrate weight on high-loss groups and approximate the
    worst-group objective.  This is the core update used by Koh et al.'s Group
    DRO code, adapted here to infer group membership from a feature column and
    retain the package-wide ``fit(X, y, ...)`` API.

    ``group_idx`` refers to a column in a two-dimensional, tabular ``X``.  Its
    values must be finite categorical numbers.  The column remains available
    to the neural network as an input feature.

    :param input_dim: Number of input features.
    :type input_dim: int
    :param num_classes: Number of classes, or 1 for regression.
    :type num_classes: int
    :param group_idx: Zero-based index of the categorical group feature.
    :type group_idx: int
    :param task_type: ``'classification'`` or ``'regression'``.
    :type task_type: str
    :param model_type: ``'mlp'`` or ``'linear'``.  Image architectures are not
        supported because a scalar feature index is required for grouping.
    :type model_type: str
    :param device: Torch device used for training.
    :type device: torch.device
    :param step_size: Positive exponentiated-gradient step size for the
        adversarial group probabilities.
    :type step_size: float

    :ivar group_values_: Sorted category values observed by :meth:`fit`.
    :ivar adv_probs: Current adversarial probability for each category in
        ``group_values_``.
    :ivar group_weights_: NumPy copy of the final adversarial probabilities.

    Reference: Sagawa, Koh, Hashimoto, and Liang, "Distributionally Robust
    Neural Networks for Group Shifts: On the Importance of Regularization for
    Worst-Case Generalization" (2020).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        group_idx: int,
        task_type: str = "classification",
        model_type: str = "mlp",
        device: torch.device = device,
        step_size: float = 0.01,
    ):
        self._check_group_idx(group_idx, input_dim)
        self._check_step_size(step_size)
        if model_type not in {"mlp", "linear"}:
            raise ParameterError(
                "GroupNNDRO supports 'mlp' and 'linear' models because groups "
                "must be read from a tabular feature column."
            )

        super().__init__(input_dim, num_classes, task_type, model_type, device)
        self.group_idx = int(group_idx)
        self.step_size = float(step_size)
        self.group_values_ = None
        self.adv_probs = None
        self.group_weights_ = None
        self.batch_group_losses_ = None
        self._group_values_tensor = None

    @staticmethod
    def _check_group_idx(group_idx: int, input_dim: int) -> None:
        """Validate a zero-based feature index."""
        if isinstance(group_idx, bool) or not isinstance(group_idx, Integral):
            raise ParameterError("group_idx must be an integer feature index.")
        if not 0 <= group_idx < input_dim:
            raise ParameterError(
                f"group_idx must be in [0, {input_dim - 1}], got {group_idx}."
            )

    @staticmethod
    def _check_step_size(step_size: float) -> None:
        """The adversarial ascent step must move toward high-loss groups."""
        if isinstance(step_size, bool) or not isinstance(
            step_size, (int, float, np.number)
        ):
            raise ParameterError("step_size must be a positive number.")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ParameterError("step_size must be a finite positive number.")

    def update(self, config: Dict[str, Any]) -> None:
        """Update Group DRO-specific configuration.

        Supported keys are ``group_idx`` and ``step_size``.  A subsequent call
        to :meth:`fit` rediscovers categories and resets adversarial weights.
        """
        if "group_idx" in config:
            self._check_group_idx(config["group_idx"], self.input_dim)
            self.group_idx = int(config["group_idx"])
        if "step_size" in config:
            self._check_step_size(config["step_size"])
            self.step_size = float(config["step_size"])

    def _criterion(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-group losses and apply an adversarial group reweighting."""
        if self._group_values_tensor is None or self.adv_probs is None:
            raise DataValidationError("Group categories are initialized by fit(X, y).")

        if self.task_type == "classification":
            per_sample_loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        else:
            # Flattening prevents accidental (batch, batch) broadcasting when
            # the network emits shape (batch, 1) and labels have shape (batch,).
            per_sample_loss = nn.MSELoss(reduction="none")(
                outputs.reshape(-1), labels.reshape(-1)
            )

        batch_groups = self.current_inputs[:, self.group_idx]
        group_map = batch_groups.unsqueeze(0).eq(
            self._group_values_tensor.unsqueeze(1)
        )
        if not torch.all(group_map.any(dim=0)):
            raise DataValidationError(
                "A training batch contains a group category not initialized by fit."
            )

        group_count = group_map.sum(dim=1)
        present = group_count > 0
        group_loss = (
            group_map.to(per_sample_loss.dtype) @ per_sample_loss
        ) / group_count.clamp_min(1).to(per_sample_loss.dtype)

        # This numerically stable form is equivalent to the exponentiated
        # adversarial update in kohpangwei/group_DRO's LossComputer.  The
        # adversarial probabilities are state, not model parameters, so the
        # update intentionally does not participate in autograd.
        with torch.no_grad():
            log_weights = torch.log(
                self.adv_probs.clamp_min(torch.finfo(self.adv_probs.dtype).tiny)
            )
            log_weights[present] += self.step_size * group_loss[present].detach()
            self.adv_probs = torch.softmax(log_weights, dim=0)

        # Arbitrary mini-batches need not contain every group. Renormalizing the
        # current adversarial weights over observed groups keeps the batch loss
        # on the same scale without inventing gradients for absent groups.
        batch_weights = self.adv_probs * present.to(self.adv_probs.dtype)
        batch_weights = batch_weights / batch_weights.sum()
        self.batch_group_losses_ = group_loss.detach().cpu().numpy()
        return group_loss @ batch_weights

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        train_ratio: float = 0.8,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Fit with the same call signature as the other neural DRO models.

        Group categories are inferred from ``X[:, group_idx]`` before the base
        training loop runs.  Adversarial probabilities start uniformly on each
        call, making repeated fits independent.
        """
        if isinstance(X, torch.Tensor):
            if X.ndim != 2:
                raise DataValidationError(
                    "GroupNNDRO requires a two-dimensional tabular feature matrix."
                )
            if X.shape[1] != self.input_dim:
                raise DataValidationError(
                    f"Expected input with {self.input_dim} features, got {X.shape[1]}."
                )
            if not torch.all(torch.isfinite(X)):
                raise DataValidationError("X must contain only finite values.")
            training_X = X
            group_column = X[:, self.group_idx].detach().cpu().to(torch.float32)
        else:
            try:
                X_array = np.asarray(X, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DataValidationError("X must contain numeric values.") from exc
            if X_array.ndim != 2:
                raise DataValidationError(
                    "GroupNNDRO requires a two-dimensional tabular feature matrix."
                )
            if X_array.shape[1] != self.input_dim:
                raise DataValidationError(
                    f"Expected input with {self.input_dim} features, got {X_array.shape[1]}."
                )
            if not np.all(np.isfinite(X_array)):
                raise DataValidationError("X must contain only finite values.")
            training_X = X_array
            group_column = torch.as_tensor(
                X_array[:, self.group_idx], dtype=torch.float32
            )

        if group_column.numel() == 0:
            raise DataValidationError("X and y must contain at least one sample.")
        if not torch.all(torch.isfinite(group_column)):
            raise DataValidationError(
                "The group feature must contain only finite category values."
            )

        if isinstance(y, torch.Tensor):
            target_values = y.detach().cpu().numpy()
        else:
            try:
                target_values = np.asarray(y, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DataValidationError("y must contain numeric values.") from exc
        if target_values.ndim == 2 and target_values.shape[1] == 1:
            target_values = target_values.reshape(-1)
        if target_values.ndim != 1:
            raise DataValidationError("y must be one-dimensional.")
        if len(target_values) != len(group_column):
            raise DataValidationError(
                "Input X and target y must have the same number of samples."
            )
        if not np.all(np.isfinite(target_values)):
            raise DataValidationError("y must contain only finite values.")
        if self.task_type == "classification" and (
            not np.all(target_values == np.floor(target_values))
            or np.min(target_values) < 0
            or np.max(target_values) >= self.num_classes
        ):
            raise DataValidationError(
                f"Classification labels must be integers in [0, {self.num_classes - 1}]."
            )
        training_y = target_values

        group_values = torch.unique(group_column, sorted=True)
        self.group_values_ = group_values.numpy()
        self._group_values_tensor = group_values.to(self.device)
        self.adv_probs = torch.full(
            (len(group_values),),
            1.0 / len(group_values),
            dtype=torch.float32,
            device=self.device,
        )

        metrics = super().fit(
            X=training_X,
            y=training_y,
            train_ratio=train_ratio,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )
        self.group_weights_ = self.adv_probs.detach().cpu().numpy().copy()
        return metrics

    def _evaluate(self, loader) -> Dict[str, float]:
        """Evaluate classification normally and regression with raw outputs."""
        if self.task_type == "classification":
            return super()._evaluate(loader)

        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.model(inputs.to(self.device)).reshape(-1)
                predictions.append(outputs.cpu())
                targets.append(labels.reshape(-1).cpu())

        predicted = torch.cat(predictions).numpy()
        observed = torch.cat(targets).numpy()
        return {"mse": float(np.mean((predicted - observed) ** 2))}

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return class indices for classification or raw regression values."""
        if self.task_type == "classification":
            return super().predict(X)

        inputs = self._convert_to_tensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs).reshape(-1).cpu().numpy()

    def score(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ):
        """Use the common classification scores or MSE for regression."""
        if self.task_type == "classification":
            return super().score(X, y)

        observed = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        return float(np.mean((self.predict(X) - np.asarray(observed).reshape(-1)) ** 2))
