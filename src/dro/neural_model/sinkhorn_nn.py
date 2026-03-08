from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, List
from .base_nn import BaseNNDRO, MLP, DROError


class SinkhornNNDROError(DROError):
    """Exception class for errors in Sinkhorn NN DRO model."""
    pass


class SinkhornNNDRO(BaseNNDRO):
    r"""Sinkhorn Distributionally Robust Optimization with Neural Networks.

    Implements the Sinkhorn DRO objective for deep learning models:

    .. math::
        \min_{\theta} \sup_{Q \in \mathcal{B}_{\epsilon,\varepsilon}(P)}
        \mathbb{E}_Q[\ell(f_\theta(X), y)]

    where the ambiguity set :math:`\mathcal{B}_{\epsilon,\varepsilon}(P)` is
    defined using the entropic-regularized (Sinkhorn) Wasserstein distance.

    The Sinkhorn DRO loss for a mini-batch is computed as:

    .. math::
        \hat{R}(\theta) = \lambda \varepsilon \cdot
        \frac{1}{N} \sum_{i=1}^{N} \log \left(
        \frac{1}{m} \sum_{j=1}^{m} \exp\left(
        \frac{\ell(f_\theta(x_i + \sigma_j), y_i)}{\lambda \varepsilon}
        \right)\right)

    where :math:`\sigma_j \sim \mathcal{N}(0, \varepsilon I)` are Gaussian
    perturbations.

    Three stochastic optimization methods are supported:

    - **SG** (Stochastic Gradient): Uses a fixed number of Monte Carlo samples
      :math:`m = 2^{K_{max}}`
    - **MLMC** (Multilevel Monte Carlo): Uses a hierarchy of sample levels for
      variance reduction
    - **RTMLMC** (Randomized Truncated MLMC): Randomly selects a single level
      per iteration for further variance reduction

    Reference: `Sinkhorn Distributionally Robust Optimization
    <https://arxiv.org/abs/2109.11926>`_
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: str = "classification",
        model_type: str = "mlp",
        reg_param: float = 1e-3,
        lambda_param: float = 1e2,
        k_sample_max: int = 5,
        optimization_type: str = "SG",
        device: torch.device = torch.device("cpu"),
    ):
        r"""Initialize Sinkhorn DRO neural model.

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

            - ``'linear'``

            - ``'resnet'``

            - ``'alexnet'``

        :type model_type: str

        :param reg_param: Entropic regularization strength
            :math:`\varepsilon > 0` controlling transport smoothness.
            Must be > 0. Defaults to 1e-3.
        :type reg_param: float

        :param lambda_param: Loss scaling factor :math:`\lambda > 0`
            balancing Wasserstein distance and loss.
            Must be > 0. Defaults to 1e2.
        :type lambda_param: float

        :param k_sample_max: Maximum level for Monte Carlo / MLMC sampling.
            The number of noise samples is :math:`2^{k\_sample\_max}`.
            Higher values improve accuracy but increase computation.
            Defaults to 5.
        :type k_sample_max: int

        :param optimization_type: Stochastic optimization algorithm. Supported:

            - ``'SG'``: Standard Stochastic Gradient (baseline)

            - ``'MLMC'``: Multilevel Monte Carlo acceleration

            - ``'RTMLMC'``: Randomized Truncated MLMC

        :type optimization_type: str

        :param device: Target computation device, defaults to CPU
        :type device: torch.device

        :raises ValueError:

            - If reg_param ≤ 0

            - If lambda_param ≤ 0

            - If k_sample_max < 1

            - If optimization_type not in {'SG', 'MLMC', 'RTMLMC'}

        Example::

            >>> model = SinkhornNNDRO(
            ...     input_dim=784,
            ...     num_classes=10,
            ...     reg_param=0.01,
            ...     lambda_param=50.0,
            ...     optimization_type='SG'
            ... )

        """
        # Parameter validation
        if reg_param <= 0:
            raise ValueError(f"reg_param must be > 0, got {reg_param}")
        if lambda_param <= 0:
            raise ValueError(f"lambda_param must be > 0, got {lambda_param}")
        if k_sample_max < 1:
            raise ValueError(
                f"k_sample_max must be >= 1, got {k_sample_max}"
            )
        if optimization_type not in {"SG", "MLMC", "RTMLMC"}:
            raise ValueError(
                f"optimization_type must be one of 'SG', 'MLMC', 'RTMLMC', "
                f"got '{optimization_type}'"
            )

        super().__init__(input_dim, num_classes, task_type, model_type, device)

        self.reg_param = reg_param
        self.lambda_param = lambda_param
        self.k_sample_max = k_sample_max
        self.optimization_type = optimization_type

    def update(self, config: Dict) -> None:
        """Update hyperparameters for Sinkhorn NN DRO.

        :param config: Dictionary containing parameter updates. Valid keys:

            - ``'reg'``: Entropic regularization strength (ε > 0)

            - ``'lambda'``: Loss scaling factor (λ > 0)

            - ``'k_sample_max'``: Maximum MLMC sampling level (int ≥ 1)

            - ``'optimization_type'``: Optimization algorithm ('SG', 'MLMC', 'RTMLMC')

            - ``'lr'``: Learning rate

            - ``'batch_size'``: Training batch size

            - ``'train_epochs'``: Number of training epochs

            - ``'layer_num'``: Number of MLP layers

            - ``'hidden_size'``: Hidden layer size for MLP

            - ``'dropout_ratio'``: Dropout rate for MLP

        :type config: dict

        :raises ValueError: If any parameter value violates constraints.

        Example::

            >>> model.update({
            ...     'reg': 0.01,
            ...     'lambda': 50.0,
            ...     'k_sample_max': 3,
            ...     'optimization_type': 'MLMC'
            ... })

        """
        if "reg" in config:
            if not isinstance(config["reg"], (float, int)) or config["reg"] <= 0:
                raise ValueError("reg must be a positive number")
            self.reg_param = float(config["reg"])

        if "lambda" in config:
            if (
                not isinstance(config["lambda"], (float, int))
                or config["lambda"] <= 0
            ):
                raise ValueError("lambda must be a positive number")
            self.lambda_param = float(config["lambda"])

        if "k_sample_max" in config:
            if (
                not isinstance(config["k_sample_max"], int)
                or config["k_sample_max"] < 1
            ):
                raise ValueError("k_sample_max must be a positive integer")
            self.k_sample_max = int(config["k_sample_max"])

        if "optimization_type" in config:
            if config["optimization_type"] not in {"SG", "MLMC", "RTMLMC"}:
                raise ValueError(
                    "optimization_type must be one of 'SG', 'MLMC', 'RTMLMC'"
                )
            self.optimization_type = config["optimization_type"]

        # MLP architecture updates
        if "layer_num" in config:
            self.model = MLP(
                self.input_dim,
                self.num_classes,
                hidden_units=config.get("hidden_size", 16),
                dropout_rate=config.get("dropout_ratio", 0.1),
                num_layers=config["layer_num"],
            ).to(self.device)

    # ------------------------------------------------------------------ #
    #  Core Sinkhorn DRO loss computation
    # ------------------------------------------------------------------ #

    def _compute_per_sample_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample loss without reduction.

        :param outputs: Model predictions of shape :math:`(N, K)`
        :param targets: Ground truth of shape :math:`(N,)` or :math:`(N, 1)`
        :return: Per-sample losses of shape :math:`(N,)`
        """
        if self.task_type == "classification":
            return nn.CrossEntropyLoss(reduction="none")(
                outputs, targets.long()
            )
        else:
            return nn.MSELoss(reduction="none")(
                outputs.squeeze(), targets.float().squeeze()
            )

    def _sinkhorn_aggregate(
        self, loss_matrix: torch.Tensor, m: int
    ) -> torch.Tensor:
        r"""Sinkhorn logsumexp aggregation.

        Computes:

        .. math::
            \lambda \varepsilon \cdot \frac{1}{N} \sum_{i=1}^{N}
            \left(\log \frac{1}{m} \sum_{j=1}^{m}
            \exp\left(\frac{\ell_{ji}}{\lambda\varepsilon}\right)\right)

        :param loss_matrix: Loss values of shape :math:`(m, N)`
        :param m: Number of Monte Carlo samples
        :return: Scalar aggregated loss
        """
        lambda_reg = self.lambda_param * self.reg_param
        residual = loss_matrix / lambda_reg
        return (
            torch.mean(torch.logsumexp(residual, dim=0) - math.log(m))
            * lambda_reg
        )

    def _forward_noisy(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        m: int,
    ) -> torch.Tensor:
        """Forward pass with Gaussian-perturbed inputs.

        Generates ``m`` noisy copies of each input sample, runs the forward
        pass, computes per-sample loss, and returns the loss matrix.

        :param inputs: Original input tensor of shape :math:`(N, ...)`
        :param targets: Target tensor of shape :math:`(N,)` or :math:`(N, 1)`
        :param m: Number of noise copies per sample
        :return: Loss matrix of shape :math:`(m, N)`
        """
        N = inputs.shape[0]

        # Generate Gaussian noise: shape [m, N, ...]
        noise_shape = [m] + list(inputs.shape)
        noise = (
            torch.randn(noise_shape, device=self.device)
            * math.sqrt(self.reg_param)
        )
        noisy_inputs = inputs.unsqueeze(0) + noise  # [m, N, ...]

        # Flatten batch dimension for forward pass: [m*N, ...]
        noisy_flat = noisy_inputs.reshape(-1, *inputs.shape[1:])

        # Repeat targets m times
        if targets.dim() == 1:
            targets_repeated = targets.repeat(m)
        else:
            targets_repeated = targets.repeat(m, 1)

        # Forward pass through the model
        noisy_outputs = self.model(noisy_flat)

        # Compute per-sample loss and reshape to [m, N]
        losses = self._compute_per_sample_loss(noisy_outputs, targets_repeated)
        return losses.reshape(m, N)

    # ------------------------------------------------------------------ #
    #  Criterion dispatch (called by BaseNNDRO.fit)
    # ------------------------------------------------------------------ #

    def _criterion(
        self, outputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute Sinkhorn DRO robust loss.

        Dispatches to the appropriate optimization method based on
        ``self.optimization_type``.

        :param outputs: Model predictions (unused; recomputed with noisy inputs)
        :param labels: Ground truth labels
        :return: Scalar robust loss
        """
        inputs = self.current_inputs  # saved in BaseNNDRO.fit()

        if self.optimization_type == "SG":
            return self._sg_criterion(inputs, labels)
        elif self.optimization_type == "MLMC":
            return self._mlmc_criterion(inputs, labels)
        elif self.optimization_type == "RTMLMC":
            return self._rtmlmc_criterion(inputs, labels)
        else:
            raise SinkhornNNDROError(
                f"Invalid optimization type: {self.optimization_type}"
            )

    # ---- SG (Stochastic Gradient) ------------------------------------ #

    def _sg_criterion(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Stochastic Gradient criterion with fixed sample count.

        Uses :math:`m = 2^{K_{max}}` Monte Carlo samples.
        """
        m = 2 ** self.k_sample_max
        loss_matrix = self._forward_noisy(inputs, labels, m)
        return self._sinkhorn_aggregate(loss_matrix, m)

    # ---- MLMC (Multilevel Monte Carlo) ------------------------------- #

    def _mlmc_criterion(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Multilevel Monte Carlo criterion with variance reduction.

        Loops over levels :math:`k = 0, \\ldots, K_{max}-1`. At each level
        :math:`k`, draws :math:`m = 2^k` noise samples on a subset of
        :math:`N_{\\ell} = 2^{K_{max}-k}` data points, then applies the
        MLMC correction (difference of logsumexp at full and half resolution).
        """
        lambda_reg = self.lambda_param * self.reg_param
        N = inputs.shape[0]
        n_levels = [2 ** (k + 1) for k in range(self.k_sample_max)]
        loss_total = torch.tensor(0.0, device=self.device)

        for k in range(self.k_sample_max):
            m = 2 ** k
            subset_size = min(n_levels[-k - 1], N)
            sub_inputs = inputs[:subset_size]
            sub_targets = labels[:subset_size]

            loss_matrix = self._forward_noisy(sub_inputs, sub_targets, m)
            residual = loss_matrix / lambda_reg

            if k == 0:
                # Base level: standard logsumexp estimator
                loss_k = (
                    torch.mean(
                        torch.logsumexp(residual, dim=0) - math.log(max(m, 1))
                    )
                    * lambda_reg
                )
            else:
                # Multilevel correction: full - 0.5 * (half1 + half2)
                half_m = m // 2
                lse_full = torch.logsumexp(residual, dim=0)
                lse_half1 = torch.logsumexp(residual[:half_m], dim=0)
                lse_half2 = torch.logsumexp(residual[half_m:], dim=0)
                loss_k = (
                    torch.mean(lse_full)
                    - 0.5 * torch.mean(lse_half1 + lse_half2)
                ) * lambda_reg

            loss_total = loss_total + loss_k

        return loss_total

    # ---- RTMLMC (Randomized Truncated MLMC) -------------------------- #

    def _rtmlmc_criterion(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Randomized Truncated MLMC criterion.

        Randomly selects a single level :math:`k` from a truncated geometric
        distribution and scales the loss by the inverse selection probability.
        """
        lambda_reg = self.lambda_param * self.reg_param
        N = inputs.shape[0]

        # Truncated geometric distribution over levels
        level_probs = np.array(
            [0.5 ** k for k in range(self.k_sample_max)]
        )
        level_probs /= level_probs.sum()

        k = int(np.random.choice(self.k_sample_max, p=level_probs))
        m = 2 ** k

        loss_matrix = self._forward_noisy(inputs, labels, m)
        residual = loss_matrix / lambda_reg

        if m == 1:
            # Single sample: standard logsumexp
            loss = (
                torch.mean(
                    torch.logsumexp(residual, dim=0) - math.log(max(m, 1))
                )
                * lambda_reg
            )
        else:
            # MLMC correction at randomly selected level
            half_m = m // 2
            lse_full = torch.logsumexp(residual, dim=0)
            lse_half1 = torch.logsumexp(residual[:half_m], dim=0)
            lse_half2 = torch.logsumexp(residual[half_m:], dim=0)
            loss = (
                torch.mean(lse_full)
                - 0.5 * torch.mean(lse_half1 + lse_half2)
            ) * lambda_reg

        # Scale by inverse selection probability
        return loss / level_probs[k]
