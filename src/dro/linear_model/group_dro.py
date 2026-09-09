"""Group distributionally robust linear models.

Group DRO minimizes the largest empirical loss among groups defined by a
categorical feature.  The group feature stays in the design matrix, so it can
also be used by the fitted predictor.
"""

from numbers import Integral
from typing import Any, Dict

import cvxpy as cp
import numpy as np

from .base import BaseLinearDRO, DataValidationError, DROError, ParameterError


class GroupDROError(DROError):
    """Raised when a Group DRO optimization problem cannot be solved."""


class GroupDRO(BaseLinearDRO):
    r"""Linear Group Distributionally Robust Optimization model.

    For observed groups :math:`\mathcal{G}`, this model solves

    .. math::
        \min_{\theta, b}\; \max_{g \in \mathcal{G}}
        \frac{1}{n_g}\sum_{i:g_i=g}\ell(\theta, b; x_i, y_i).

    A group is the value in column ``group_idx`` of ``X``.  The column must
    contain finite, discrete numeric values.  It remains part of ``X`` during
    fitting and prediction, which keeps the usual ``fit(X, y)`` API used by
    the other linear models in this package.

    :param input_dim: Number of columns in the input feature matrix.
    :type input_dim: int
    :param group_idx: Zero-based index of the categorical group feature.
    :type group_idx: int
    :param model_type: Loss/model type: ``'svm'``, ``'logistic'``, ``'ols'``,
        or ``'lad'``.
    :type model_type: str
    :param fit_intercept: Whether to fit an intercept.
    :type fit_intercept: bool
    :param solver: Installed CVXPY solver used for optimization.
    :type solver: str
    :param kernel: Kernel accepted by :class:`BaseLinearDRO`.
    :type kernel: str

    :ivar group_values_: Sorted group categories observed by the latest call
        to :meth:`fit`.
    :ivar group_losses_: Empirical loss for each value in ``group_values_`` at
        the fitted solution.
    :ivar robust_loss_: Largest fitted group loss.
    """

    def __init__(
        self,
        input_dim: int,
        group_idx: int,
        model_type: str = "svm",
        fit_intercept: bool = True,
        solver: str = "MOSEK",
        kernel: str = "linear",
    ):
        self._check_group_idx(group_idx, input_dim)
        super().__init__(input_dim, model_type, fit_intercept, solver, kernel)
        self.group_idx = int(group_idx)
        self.group_values_ = None
        self.group_losses_ = None
        self.robust_loss_ = None

    @staticmethod
    def _check_group_idx(group_idx: int, input_dim: int) -> None:
        """Validate the index separately so ``update`` can reuse the rule."""
        if isinstance(group_idx, bool) or not isinstance(group_idx, Integral):
            raise ParameterError("group_idx must be an integer feature index.")
        if not 0 <= group_idx < input_dim:
            raise ParameterError(
                f"group_idx must be in [0, {input_dim - 1}], got {group_idx}."
            )

    def update(self, config: Dict[str, Any]) -> None:
        """Update the feature used to define groups.

        :param config: Configuration dictionary.  ``group_idx`` is the only
            Group DRO-specific key; unrelated keys are ignored consistently
            with the other linear model implementations.
        :type config: Dict[str, Any]
        """
        if "group_idx" in config:
            self._check_group_idx(config["group_idx"], self.input_dim)
            self.group_idx = int(config["group_idx"])

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray):
        """Return numeric inputs and the finite group categories they contain."""
        try:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("X and y must contain numeric values.") from exc

        if X.ndim != 2:
            raise DataValidationError("X must be a two-dimensional feature matrix.")
        if X.shape[1] != self.input_dim:
            raise DataValidationError(
                f"Expected input with {self.input_dim} features, got {X.shape[1]}."
            )
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise DataValidationError("y must be one-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise DataValidationError(
                "Input X and target y must have the same number of samples."
            )
        if X.shape[0] == 0:
            raise DataValidationError("X and y must contain at least one sample.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise DataValidationError("X and y must contain only finite values.")
        if self.model_type in {"svm", "logistic"} and not np.all(
            (y == -1) | (y == 1)
        ):
            raise DataValidationError("Classification labels must be in {-1, +1}.")

        groups = X[:, self.group_idx]
        group_values = np.unique(groups)
        return X, y, groups, group_values

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit by minimizing the maximum empirical group loss.

        :param X: Numeric feature matrix of shape ``(n_samples, input_dim)``.
            Column ``group_idx`` supplies the finite group categories.
        :type X: numpy.ndarray
        :param y: Binary labels in ``{-1, +1}`` for classification, or numeric
            targets for regression.
        :type y: numpy.ndarray
        :returns: Fitted parameters and group-loss diagnostics.  The entries in
            ``group_losses`` align with ``group_values``.
        :rtype: Dict[str, Any]
        """
        X, y, groups, group_values = self._validate_training_data(X, y)
        sample_size = X.shape[0]

        # Match BaseLinearDRO's kernel representation so predict/load retain the
        # same behavior as every other linear DRO estimator.
        if self.kernel != "linear":
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, (float, int)):
                variance = np.var(X)
                if variance <= 0:
                    raise DataValidationError(
                        "A non-linear kernel requires features with positive variance."
                    )
                self.kernel_gamma = 1 / (self.input_dim * variance)
            theta_size = sample_size if self.n_components is None else self.n_components
            theta = cp.Variable(theta_size)
        else:
            theta = cp.Variable(self.input_dim)

        b = cp.Variable() if self.fit_intercept else 0
        per_sample_loss = self._cvx_loss(X, y, theta, b)

        # An epigraph variable expresses max_g mean(loss_i | group g) as a
        # convex program and works for each loss supported by BaseLinearDRO.
        robust_loss = cp.Variable()
        constraints = []
        for group_value in group_values:
            group_mask = groups == group_value
            constraints.append(
                cp.sum(per_sample_loss[group_mask]) / int(group_mask.sum())
                <= robust_loss
            )

        problem = cp.Problem(cp.Minimize(robust_loss), constraints)
        try:
            problem.solve(solver=self.solver)
        except cp.SolverError as exc:
            raise GroupDROError(
                f"Group DRO optimization failed using solver {self.solver}."
            ) from exc

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise GroupDROError(
                f"Group DRO optimization did not converge (status={problem.status})."
            )
        if theta.value is None or robust_loss.value is None:
            raise GroupDROError("Group DRO optimization returned no solution.")

        self.theta = np.asarray(theta.value).reshape(-1)
        if self.fit_intercept:
            if b.value is None:
                raise GroupDROError("Group DRO optimization returned no intercept.")
            self.b = float(b.value)

        # CVXPY evaluates this expression at the returned parameters. Using the
        # expression value also keeps logistic diagnostics numerically stable.
        fitted_losses = np.asarray(per_sample_loss.value).reshape(-1)
        self.group_values_ = group_values
        self.group_losses_ = np.asarray(
            [fitted_losses[groups == value].mean() for value in group_values]
        )
        self.robust_loss_ = float(self.group_losses_.max())
        self.robust_obj = self.robust_loss_

        return {
            "theta": self.theta.tolist(),
            "b": self.b,
            "group_values": self.group_values_.tolist(),
            "group_losses": self.group_losses_.tolist(),
            "robust_loss": self.robust_loss_,
        }
