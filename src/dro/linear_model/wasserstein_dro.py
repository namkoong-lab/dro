from .base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import Nystroem



class WassersteinDROError(Exception):
    """Base exception class for errors in Wasserstein DRO model."""
    pass


class WassersteinDRO(BaseLinearDRO):
    r"""Wasserstein Distributionally Robust Optimization (WDRO) model
    
    This model minimizes a Wasserstein-robust loss function for both regression and classification.

    The Wasserstein distance is defined as the minimum probability coupling of
    two distributions. For LAD regression, the ground cost is

    .. math::
        d((X_1, Y_1), (X_2, Y_2))
        = \|\Sigma^{1/2} (X_1 - X_2)\|_p + \kappa |Y_1 - Y_2|.

    For OLS regression, target changes are prohibited and the feature cost is
    quadratic:

    .. math::
        d((X_1, Y_1), (X_2, Y_2))
        = \begin{cases}
        \|\Sigma^{1/2} (X_1 - X_2)\|_p^2, & Y_1=Y_2,\\
        +\infty, & Y_1\ne Y_2.
        \end{cases}

    For binary classification, the label term is instead
    :math:`\kappa\mathbf{1}_{\{Y_1\ne Y_2\}}`, so a label flip costs
    exactly :math:`\kappa`.

    where parameters are:

        - :math:`\Sigma`: symmetric positive-definite cost matrix;

        - :math:`\kappa`;

        - :math:`p`;

    Reference:

    [1] OLS: <https://www.cambridge.org/core/journals/journal-of-applied-probability/article/robust-wasserstein-profile-inference-and-applications-to-machine-learning/4024D05DE4681E67334E45D039295527>

    [2] LAD / SVM / Logistic: <https://jmlr.org/papers/volume20/17-633/17-633.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', 
                 fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """Initialize Mahalanobis-Wasserstein DRO model.

        :param input_dim: Dimension of feature space. Must satisfy :math:`\text{input\_dim} \geq 1`
        :type input_dim: int

        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)

        :type model_type: str

        :param fit_intercept: Whether to learn intercept term :math:`b`.
            Set to ``False`` for pre-centered data. Defaults to True.
        :type fit_intercept: bool

        :param solver: Convex optimization solver. Valid options:
        
                - ``'MOSEK'`` (commercial, recommended)
        
        :type solver: str

        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :raises ValueError:

            - If input_dim < 1

            - If unsupported solver is selected

        Example:
            >>> model = WassersteinDRO(
            ...     input_dim=5,
            ...     model_type='svm',
            ...     solver='MOSEK'
            ... )
            >>> model.cost_matrix.shape  # (5, 5)

        .. note::
            - Changing ``cost_matrix`` after initialization requires calling ``update()``
        """
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        
        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.eps = 0 
        self.p = 1  
        self.kappa = 'inf' 

    def update(self, config: Dict[str, Any]) -> None:
        """Update Wasserstein-DRO model parameters dynamically.
        
        :param config: Configuration dictionary with keys:

            - ``'cost_matrix'``: Mahalanobis metric matrix :math:`\Sigma \succ 0`

                - Shape: (input_dim, input_dim)

                - Type: numpy.ndarray

            - ``'eps'``: Wasserstein radius :math:`\epsilon \geq 0`

            - ``'p'``: Wasserstein order :math:`p \geq 1` or ``'inf'``

            - ``'kappa'``: Y-ambiguity radius :math:`\kappa \geq 0` or ``'inf'``.
              LAD requires :math:`\kappa > 0` because free target transport
              makes the robust absolute loss unbounded. OLS requires
              ``'inf'`` because its reformulation keeps targets fixed.

        :type config: dict[str, Any]

        :raises ValueError:

            - If cost_matrix is not positive definite

            - If eps < 0

            - If p < 1 and p ≠ 'inf'

            - If kappa < 0 and kappa ≠ 'inf'

        :raises TypeError:

            - If cost_matrix is not numpy array

            - If numeric parameters are not float/int
        
    
        Example:
            >>> model = WassersteinDRO(input_dim=3)
            >>> new_config = {
            ...     'eps': 0.5,
            ...     'p': 2,
            ...     'cost_matrix': np.diag([1, 2, 3])
            ... }
            >>> model.update(new_config)
            >>> model.p  # 2.0
        """

        if 'cost_matrix' in config:
            cost_matrix = config['cost_matrix']
            if not isinstance(cost_matrix, np.ndarray):
                raise TypeError("cost_matrix must be numpy.ndarray")
            if self.kernel == 'linear' and cost_matrix.shape != (self.input_dim, self.input_dim):
                raise ValueError(f"cost_matrix must have shape ({self.input_dim}, {self.input_dim})")
            if not np.all(np.isfinite(cost_matrix)):
                raise ValueError("cost_matrix must contain only finite values")
            if not np.allclose(cost_matrix, cost_matrix.T, rtol=1e-10, atol=1e-12):
                raise ValueError("cost_matrix must be symmetric")
            cost_matrix = 0.5 * (cost_matrix + cost_matrix.T)
            if not np.all(np.linalg.eigvalsh(cost_matrix) > 0):
                raise ValueError("cost_matrix must be positive definite")
            
            self.cost_matrix = cost_matrix
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))

        if 'eps' in config:
            eps = config['eps']
            if isinstance(eps, bool) or not isinstance(eps, (float, int)):
                raise TypeError("eps must be numeric")
            if not np.isfinite(eps) or eps < 0:
                raise ValueError(f"eps must be finite and ≥ 0, got {eps}")
            self.eps = float(eps)

        if 'p' in config:
            p = config['p']
            if isinstance(p, (float, int)) and not isinstance(p, bool) and np.isinf(p):
                p = 'inf'
            if p != 'inf' and (
                isinstance(p, bool)
                or not isinstance(p, (float, int))
                or not np.isfinite(p)
                or p < 1
            ):
                raise ValueError(f"p must be ≥1 or 'inf', got {p}")
            self.p = float(p) if p != 'inf' else 'inf'

        if 'kappa' in config:
            kappa = config['kappa']
            if isinstance(kappa, (float, int)) and not isinstance(kappa, bool) and np.isinf(kappa):
                kappa = 'inf'
            if kappa != 'inf' and (
                isinstance(kappa, bool)
                or not isinstance(kappa, (float, int))
                or not np.isfinite(kappa)
                or kappa < 0
            ):
                raise ValueError(f"kappa must be ≥0 or 'inf', got {kappa}")
            if self.model_type == 'lad' and kappa != 'inf' and kappa == 0:
                raise ValueError("kappa must be strictly positive for LAD models")
            if kappa != 'inf' and self.model_type == 'ols':
                raise ValueError("kappa must be 'inf' for OLS models")
            self.kappa = float(kappa) if kappa != 'inf' else 'inf'
     
    def _penalization(self, theta: cp.Expression) -> float:
        """
        Module for computing the regularization part in the standard Wasserstein DRO problem.

        Args:
            theta (:py:class:`cvxpy.Expression`): Feature vector with shape (n_feature,).
        
        Returns:
            Float: Regularization term part.

        """
        if self.kernel != 'linear':
            if self.n_components is not None:
                if not hasattr(self, 'nystroem_transformer'):
                    raise WassersteinDROError(
                        "The Nyström feature map must be fitted before computing "
                        "the kernel penalty."
                    )
                # Use exactly the feature map fitted in ``fit``. Refitting a
                # separate randomized Nyström map here would regularize a
                # different set of coordinates from those used by the loss.
                Phi_X = self.nystroem_transformer.transform(self.support_vectors_)
                theta_K = sqrtm(Phi_X.T @ Phi_X) @ theta
            else:
                theta_K = sqrtm(pairwise_kernels(self.support_vectors_, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)) @ theta

        else:
            theta_K = theta

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type == 'lad':
            if self.kappa == 'inf':
                return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
            else:
                return cp.maximum(cp.norm(self.cost_inv_transform @ theta_K, dual_norm), 1 / self.kappa)


        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the WDRO problem.

        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param Y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type Y: numpy.ndarray

        :returns: Dictionary containing trained parameters:
        
            - ``theta``: Weight vector. Its length is ``n_features`` for a
              linear kernel, ``n_samples`` for a full nonlinear kernel, and
              ``n_components`` for a Nyström approximation.
            
            - ``b``
            
        :rtype: Dict[str, Any]
        
        .raises: WassersteinDROError: If the optimization problem fails to solve.
        """
        if self.model_type in {'logistic', 'svm'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise WassersteinDROError("classification labels not in {-1, +1}")
        
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise WassersteinDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise WassersteinDROError("Input X and target y must have the same number of samples.")

        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, (float, int)):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
                design_matrix = pairwise_kernels(
                    X,
                    self.support_vectors_,
                    metric=self.kernel,
                    gamma=self.kernel_gamma,
                )
                self.cost_matrix = np.eye(sample_size)
                self.cost_inv_transform = np.eye(sample_size)
            else:
                theta = cp.Variable(self.n_components)
                # Fit the approximation once and retain it for training,
                # regularization, and all subsequent predictions.
                self.nystroem_transformer = Nystroem(
                    kernel=self.kernel,
                    gamma=self.kernel_gamma,
                    n_components=self.n_components,
                )
                design_matrix = self.nystroem_transformer.fit_transform(X)
                self.cost_matrix = np.eye(self.n_components)
                self.cost_inv_transform = np.eye(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)
            design_matrix = X
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0


        lamb_da = cp.Variable()
        cons = [lamb_da >= self._penalization(theta)]
        if self.model_type == 'ols':
            # Kernel OLS is linear in the kernel feature coordinates.  Using
            # raw X here is dimensionally wrong when theta is indexed by the
            # training samples (full kernel) or Nyström components.
            residual = design_matrix @ theta + b - y
            final_loss = (
                cp.norm(residual) / math.sqrt(sample_size)
                + math.sqrt(self.eps) * lamb_da
            )

        else:
            if self.model_type in ['svm', 'logistic']:
                s = cp.Variable(sample_size)
                cons += [
                    s >= self._cvx_loss(
                        X, y, theta, b, design_matrix=design_matrix
                    )
                ]
                if self.kappa != 'inf':
                    cons += [
                        s >= self._cvx_loss(
                            X, -y, theta, b, design_matrix=design_matrix
                        ) - lamb_da * self.kappa
                    ]
                final_loss = cp.sum(s) / sample_size + self.eps * lamb_da
            else:
                final_loss = (
                    cp.sum(
                        self._cvx_loss(
                            X, y, theta, b, design_matrix=design_matrix
                        )
                    ) / sample_size
                    + self.eps * lamb_da
                )

        problem = cp.Problem(cp.Minimize(final_loss), cons)
        try:
            problem.solve(solver = self.solver)
        except cp.error.SolverError as e:
            raise WassersteinDROError(f"Optimization failed to solve using {self.solver}.") from e
        
        if theta.value is None:
            raise WassersteinDROError("Optimization did not converge to a solution.")

        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value
        self.robust_obj = float(problem.value)

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
        return model_params
    
    def _distance_compute(self, X_1: cp.Expression, X_2: np.ndarray, Y_1: cp.Expression, Y_2: float) -> cp.Expression:
        """
        Computing the distance between two points (X_1, Y_1), (X_2, Y_2) under our defined metric in cvxpy problem

        Args:
            X_1 (:py:class:`cvxpy.expressions.expression.Expression`): Input feature-1 (n_feature,);
            X_2 (np.ndarray): Input feature-2 (n_feature,);
            Y_1 (:py:class:`cvxpy.expressions.expression.Expression`): Input label-1;
            Y_2 (float): Input label-2;

        Returns:
            :py:class:`cvxpy.expressions.expression.Expression`: Distance value
            
        Raises:
            WassersteinDROError: If the dimensions of two input feature are different.
        """
        if X_1.shape[-1] != X_2.shape[-1]:
            raise WassersteinDROError(f"two input feature dimensions are different.")
        component_X = cp.norm(sqrtm(self.cost_matrix) @ (X_1 - X_2), self.p)
        if self.model_type == 'ols':
            component_X = component_X ** 2

        if self.model_type in {'svm', 'logistic'}:
            # Binary classification uses the indicator flip cost from the
            # model reformulation, not kappa * |1 - (-1)| = 2 * kappa.
            if isinstance(Y_1, cp.Expression):
                if not Y_1.is_constant() or Y_1.value is None:
                    raise WassersteinDROError(
                        "Classification label costs require fixed labels."
                    )
                y_1_value = float(np.asarray(Y_1.value).item())
            else:
                y_1_value = float(Y_1)
            label_changed = not np.isclose(y_1_value, float(Y_2))
            if self._is_infinite_kappa() and label_changed:
                raise WassersteinDROError(
                    "Label changes are prohibited when kappa is infinite."
                )
            component_Y = 0 if self._is_infinite_kappa() else self.kappa * float(label_changed)
        else:
            if self._is_infinite_kappa():
                if isinstance(Y_1, cp.Expression) and not Y_1.is_constant():
                    raise WassersteinDROError(
                        "Target changes are prohibited when kappa is infinite."
                    )
                y_1_value = float(np.asarray(Y_1.value).item()) if isinstance(Y_1, cp.Expression) else float(Y_1)
                if not np.isclose(y_1_value, float(Y_2)):
                    raise WassersteinDROError(
                        "Target changes are prohibited when kappa is infinite."
                    )
                component_Y = 0
            else:
                component_Y = self.kappa * cp.abs(Y_1 - Y_2)
        return component_X + component_Y
        

    def _lipschitz_norm(self):
        """
        Computing the Lipschitz norm of the loss function

        Returns:
            Float: the size of the Lipschitz norm of the loss function

        """
        if self.model_type in ['svm', 'logistic', 'lad']:
            return 1
        else:
            return np.inf

    def _is_infinite_kappa(self) -> bool:
        """Return whether output/label transportation is prohibited."""
        return self.kappa == 'inf' or (
            isinstance(self.kappa, (float, int)) and np.isinf(self.kappa)
        )

    def _feature_transport_cost(self, displacement: np.ndarray) -> float:
        """Evaluate the feature part of the configured ground cost."""
        transformed = sqrtm(self.cost_matrix) @ np.asarray(displacement, dtype=float)
        transformed = np.asarray(np.real_if_close(transformed), dtype=float)
        order = np.inf if self.p == 'inf' else self.p
        value = float(np.linalg.norm(transformed, ord=order))
        return value ** 2 if self.model_type == 'ols' else value

    def _feature_dual_direction(self) -> Tuple[np.ndarray, float]:
        r"""Return a unit-cost feature direction attaining the dual norm.

        If :math:`A=\Sigma^{1/2}`, this returns ``direction`` and ``slope``
        satisfying

        .. math::
            \|A\,\mathrm{direction}\|_p=1,\qquad
            \theta^\top\mathrm{direction}
            =\|A^{-\top}\theta\|_q=\mathrm{slope}.
        """
        theta = np.asarray(self.theta, dtype=float).reshape(-1)
        if not np.any(theta):
            return np.zeros(self.input_dim), 0.0

        if not np.allclose(
            self.cost_matrix, self.cost_matrix.T, rtol=1e-10, atol=1e-12
        ):
            raise WassersteinDROError(
                "Worst-distribution recovery requires a symmetric cost_matrix."
            )
        transform = np.asarray(
            np.real_if_close(sqrtm(self.cost_matrix)), dtype=float
        )
        dual_coordinates = np.linalg.solve(transform.T, theta)
        max_coordinate = float(np.max(np.abs(dual_coordinates)))
        if max_coordinate == 0:
            return np.zeros(self.input_dim), 0.0

        if self.p == 1:
            transformed_direction = np.zeros_like(dual_coordinates)
            index = int(np.argmax(np.abs(dual_coordinates)))
            transformed_direction[index] = np.sign(dual_coordinates[index])
        elif self.p == 'inf':
            transformed_direction = np.sign(dual_coordinates)
        else:
            q = float(self.p) / (float(self.p) - 1.0)
            scaled = np.abs(dual_coordinates) / max_coordinate
            power_sum = float(np.sum(scaled ** q))
            transformed_direction = (
                np.sign(dual_coordinates)
                * scaled ** (q - 1.0)
                / power_sum ** ((q - 1.0) / q)
            )

        direction = np.linalg.solve(transform, transformed_direction)
        order = np.inf if self.p == 'inf' else self.p
        direction_norm = float(np.linalg.norm(transform @ direction, ord=order))
        if direction_norm == 0:
            return np.zeros(self.input_dim), 0.0
        direction = np.asarray(direction / direction_norm, dtype=float)
        slope = float(theta @ direction)
        if slope < 0 and abs(slope) <= 1e-12:
            slope = 0.0
        if slope < 0:
            direction = -direction
            slope = -slope
        return direction, slope

    def _ground_transport_cost(
        self,
        target_x: np.ndarray,
        target_y: float,
        source_x: np.ndarray,
        source_y: float,
    ) -> float:
        """Evaluate the ground cost used by fitting and certification."""
        feature_cost = self._feature_transport_cost(target_x - source_x)
        if self.model_type in {'svm', 'logistic'}:
            label_changed = not np.isclose(target_y, source_y)
            if label_changed and self._is_infinite_kappa():
                return np.inf
            label_cost = 0.0 if not label_changed else float(self.kappa)
        else:
            target_change = abs(float(target_y) - float(source_y))
            if target_change > 0 and self._is_infinite_kappa():
                return np.inf
            label_cost = 0.0 if self._is_infinite_kappa() else float(self.kappa) * target_change
        return feature_cost + label_cost

    def _solve_problem(self, problem: cp.Problem, purpose: str) -> None:
        """Solve an auxiliary recovery problem and validate its status."""
        try:
            problem.solve(solver=self.solver)
        except cp.error.SolverError as exc:
            raise WassersteinDROError(
                f"Optimization failed while {purpose} using {self.solver}."
            ) from exc
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise WassersteinDROError(
                f"Optimization failed while {purpose}; solver status was {problem.status}."
            )

    def _classification_recession_direction(self, anchor_label: float) -> np.ndarray:
        """Return a unit-cost direction that increases hinge/logistic loss."""
        if np.linalg.norm(self.theta) == 0:
            return np.zeros(self.input_dim)
        direction_var = cp.Variable(self.input_dim)
        constraints = [
            cp.norm(sqrtm(self.cost_matrix) @ direction_var, self.p) <= 1
        ]
        problem = cp.Problem(cp.Maximize(direction_var @ self.theta), constraints)
        self._solve_problem(problem, "computing the recession direction")
        if direction_var.value is None:
            raise WassersteinDROError("The recession-direction problem returned no solution.")

        # Hinge and logistic losses grow when the signed margin tends to
        # -infinity.  The anchor label is therefore essential here.
        direction = -float(anchor_label) * np.asarray(direction_var.value, dtype=float)
        direction_cost = self._feature_transport_cost(direction)
        if direction_cost <= np.finfo(float).eps:
            return np.zeros(self.input_dim)
        return direction / direction_cost

    def _lad_recession_direction(
        self, tail_sign: float
    ) -> Tuple[np.ndarray, float]:
        """Return a unit-cost joint direction that increases absolute loss."""
        direction_x = cp.Variable(self.input_dim)
        if self._is_infinite_kappa():
            direction_y = 0.0
            constraints = [
                cp.norm(sqrtm(self.cost_matrix) @ direction_x, self.p) <= 1
            ]
        else:
            direction_y = cp.Variable()
            constraints = [
                cp.norm(sqrtm(self.cost_matrix) @ direction_x, self.p)
                + self.kappa * cp.abs(direction_y)
                <= 1
            ]
        problem = cp.Problem(
            cp.Maximize(self.theta @ direction_x - direction_y), constraints
        )
        self._solve_problem(problem, "computing the LAD recession direction")
        if direction_x.value is None:
            raise WassersteinDROError("The LAD recession-direction problem returned no solution.")
        x_value = float(tail_sign) * np.asarray(direction_x.value, dtype=float)
        raw_y = 0.0 if self._is_infinite_kappa() else float(direction_y.value)
        y_value = float(tail_sign) * raw_y
        direction_cost = self._feature_transport_cost(x_value)
        if not self._is_infinite_kappa():
            direction_cost += float(self.kappa) * abs(y_value)
        if direction_cost <= np.finfo(float).eps:
            return np.zeros(self.input_dim), 0.0
        return x_value / direction_cost, y_value / direction_cost

    def _certify_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidate_X: np.ndarray,
        candidate_y: np.ndarray,
        weight: np.ndarray,
        source_index: np.ndarray,
        gamma_used: Optional[float],
        objective_atol: float,
        objective_rtol: float,
        feasibility_tol: float,
        asymptotic: bool,
    ) -> Dict[str, Any]:
        """Compute feasibility and objective certificates for a candidate."""
        candidate_X = np.asarray(candidate_X, dtype=float)
        candidate_y = np.asarray(candidate_y, dtype=float)
        weight = np.asarray(weight, dtype=float)
        source_index = np.asarray(source_index, dtype=int)

        atom_count = weight.shape[0]
        if candidate_X.shape[0] != atom_count or candidate_y.shape[0] != atom_count:
            raise WassersteinDROError("Candidate atoms and weights have inconsistent sizes.")
        if source_index.shape != (atom_count,):
            raise WassersteinDROError("Each candidate atom must retain one source index.")
        if np.any(source_index < 0) or np.any(source_index >= X.shape[0]):
            raise WassersteinDROError("Candidate distribution contains an invalid source index.")
        if not (
            np.all(np.isfinite(candidate_X))
            and np.all(np.isfinite(candidate_y))
            and np.all(np.isfinite(weight))
        ):
            raise WassersteinDROError("Candidate distribution contains non-finite values.")
        if np.min(weight) < -feasibility_tol:
            raise WassersteinDROError("Candidate distribution contains a negative weight.")
        weight = np.maximum(weight, 0.0)

        weight_sum_error = abs(float(np.sum(weight)) - 1.0)
        source_mass = np.bincount(
            source_index, weights=weight, minlength=X.shape[0]
        )
        source_marginal_error = float(
            np.max(np.abs(source_mass - 1.0 / X.shape[0]))
        )
        transport_cost = 0.0
        for atom_x, atom_y, atom_weight, source in zip(
            candidate_X, candidate_y, weight, source_index
        ):
            if atom_weight == 0:
                continue
            atom_cost = self._ground_transport_cost(
                atom_x, atom_y, X[source], y[source]
            )
            transport_cost += float(atom_weight) * atom_cost

        losses = np.asarray(self._loss(candidate_X, candidate_y), dtype=float)
        if not np.all(np.isfinite(losses)):
            raise WassersteinDROError(
                "Candidate expected loss is non-finite."
            )
        expected_loss = float(np.dot(weight, losses))
        # ``fit`` minimizes the square root of the robust MSE for OLS because
        # it has the same minimizer and a simpler conic representation.  The
        # atom losses evaluated above are squared residuals, so the comparison
        # must be made in squared-loss units.
        target_objective = float(self.robust_obj)
        if self.model_type == 'ols':
            target_objective = target_objective ** 2
        optimality_gap = target_objective - expected_loss
        objective_tol = objective_atol + objective_rtol * max(
            1.0, abs(target_objective)
        )
        certified = bool(
            weight_sum_error <= feasibility_tol
            and source_marginal_error <= feasibility_tol
            and transport_cost <= self.eps + feasibility_tol
            and abs(optimality_gap) <= objective_tol
        )
        return {
            'sample_pts': [candidate_X, candidate_y],
            'weight': weight,
            'source_index': source_index,
            'gamma_used': gamma_used,
            'expected_loss': expected_loss,
            'target_objective': target_objective,
            'optimality_gap': optimality_gap,
            'transport_cost': float(transport_cost),
            'source_marginal_error': source_marginal_error,
            'certified': certified,
            'asymptotic': bool(asymptotic),
            'kappa_used': self.kappa,
        }

    def _ols_worst_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Construct and certify the exact OLS worst-case distribution."""
        if not self._is_infinite_kappa():
            raise WassersteinDROError(
                "OLS worst-distribution recovery requires kappa='inf'."
            )

        sample_size = X.shape[0]
        residual = X @ self.theta + self.b - y
        residual_norm = float(np.linalg.norm(residual) / math.sqrt(sample_size))
        direction, slope = self._feature_dual_direction()

        if self.eps == 0 or slope == 0:
            candidate_X = X.copy()
        elif residual_norm == 0:
            candidate_X = X + math.sqrt(self.eps) * direction
        else:
            displacement_scale = math.sqrt(self.eps) * residual / residual_norm
            candidate_X = X + displacement_scale[:, None] * direction[None, :]

        result = self._certify_distribution(
            X=X,
            y=y,
            candidate_X=candidate_X,
            candidate_y=y.copy(),
            weight=np.full(sample_size, 1.0 / sample_size),
            source_index=np.arange(sample_size),
            gamma_used=None,
            objective_atol=1e-5,
            objective_rtol=1e-5,
            feasibility_tol=1e-7,
            asymptotic=False,
        )
        if not result['certified']:
            raise WassersteinDROError(
                "Could not certify the exact OLS worst-case distribution: "
                f"objective_gap={result['optimality_gap']:.6g}, "
                f"transport_cost={result['transport_cost']:.6g}, "
                f"eps={self.eps:.6g}."
            )
        return result

    def worst_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        asymptotic_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r"""Construct and certify a worst-case distribution.

        OLS uses an exact, finite, label-preserving construction. Linear SVM,
        logistic, and LAD use an asymptotic construction whose numerical
        controls are grouped in ``asymptotic_options``.

        :param X: Training features with shape ``(n_samples, input_dim)``.
        :param y: Binary ``-1/+1`` labels or continuous regression targets.
        :param asymptotic_options: Options used only by SVM, logistic, and LAD:
            ``gamma``, ``objective_atol``, ``objective_rtol``,
            ``feasibility_tol``, ``max_iter``, and ``gamma_decay``. Omit this
            dictionary for OLS.

        :returns: The atoms and weights together with ``source_index``,
            ``expected_loss``, ``target_objective``, ``optimality_gap``,
            ``transport_cost``, ``gamma_used``, and ``certified`` metadata.

        .. note::
            For OLS, ``target_objective`` is the robust expected squared loss
            and therefore equals ``robust_obj**2``. For the other models, a
            positive-radius result is a certified finite member of an
            asymptotically optimal sequence.

        References:
            Blanchet, Kang, and Murthy (2019),
            Shafieezadeh-Abadeh et al. (2019),
            Shafiee et al. (2026).
        """
        if self.kernel != 'linear':
            raise WassersteinDROError(
                "Worst-distribution recovery currently requires kernel='linear'."
            )
        if asymptotic_options is not None and not isinstance(
            asymptotic_options, dict
        ):
            raise WassersteinDROError("asymptotic_options must be a dictionary or None.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.model_type == 'ols':
            if asymptotic_options:
                raise WassersteinDROError(
                    "asymptotic_options do not apply to exact OLS recovery."
                )
            if not self._is_infinite_kappa():
                raise WassersteinDROError(
                    "OLS worst-distribution recovery requires kappa='inf'."
                )
            self.fit(X, y)
            return self._ols_worst_distribution(X, y)

        asymptotic_defaults = {
            'gamma': None,
            'objective_atol': 1e-5,
            'objective_rtol': 1e-5,
            'feasibility_tol': 1e-7,
            'max_iter': 20,
            'gamma_decay': 0.5,
        }
        supplied_options = {} if asymptotic_options is None else asymptotic_options
        unknown_options = set(supplied_options) - set(asymptotic_defaults)
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise WassersteinDROError(f"Unknown asymptotic option(s): {names}.")
        options = {**asymptotic_defaults, **supplied_options}
        gamma = options['gamma']
        objective_atol = options['objective_atol']
        objective_rtol = options['objective_rtol']
        feasibility_tol = options['feasibility_tol']
        max_iter = options['max_iter']
        gamma_decay = options['gamma_decay']

        for name, value in {
            'objective_atol': objective_atol,
            'objective_rtol': objective_rtol,
            'feasibility_tol': feasibility_tol,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
            ):
                raise WassersteinDROError(
                    f"{name} must be a finite non-negative number."
                )
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise WassersteinDROError("max_iter must be a positive integer.")
        if not isinstance(gamma_decay, (float, int)) or not 0 < gamma_decay < 1:
            raise WassersteinDROError("gamma_decay must lie strictly between 0 and 1.")
        if gamma is not None and (
            isinstance(gamma, bool)
            or not isinstance(gamma, (float, int))
            or not np.isfinite(gamma)
        ):
            raise WassersteinDROError("gamma must be a finite number or None.")

        gamma_upper: Optional[float] = None
        if self.eps > 0:
            if gamma is not None and gamma <= 0:
                raise WassersteinDROError(
                    "gamma must be strictly positive when eps is positive."
                )
            if self.model_type in {'svm', 'logistic'} and not self._is_infinite_kappa():
                gamma_upper = min(self.eps, 1.0)
            else:
                gamma_upper = 1.0
            if gamma is not None and gamma > gamma_upper:
                raise WassersteinDROError(
                    f"gamma must not exceed {gamma_upper:g} for this configuration."
                )

        self.fit(X, y)
        sample_size = X.shape[0]

        if self.eps == 0:
            empirical = self._certify_distribution(
                X=X,
                y=y,
                candidate_X=X.copy(),
                candidate_y=y.copy(),
                weight=np.full(sample_size, 1.0 / sample_size),
                source_index=np.arange(sample_size),
                gamma_used=None,
                objective_atol=objective_atol,
                objective_rtol=objective_rtol,
                feasibility_tol=feasibility_tol,
                asymptotic=False,
            )
            if not empirical['certified']:
                raise WassersteinDROError(
                    "The empirical distribution does not agree with the optimized "
                    "zero-radius objective. Check kappa and solver tolerances."
                )
            return empirical

        assert gamma_upper is not None
        gamma_value = float(gamma) if gamma is not None else 0.1 * gamma_upper

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        norm_theta = float(
            np.linalg.norm(self.cost_inv_transform @ self.theta, ord=dual_norm)
        )

        # With fixed labels and a feature-insensitive classifier, the empirical
        # distribution already attains the robust value.
        if (
            self.model_type in {'svm', 'logistic'}
            and self._is_infinite_kappa()
            and norm_theta <= np.finfo(float).eps
        ):
            empirical = self._certify_distribution(
                X, y, X.copy(), y.copy(),
                np.full(sample_size, 1.0 / sample_size),
                np.arange(sample_size), None,
                objective_atol, objective_rtol, feasibility_tol, False,
            )
            if empirical['certified']:
                return empirical

        last_result: Optional[Dict[str, Any]] = None
        for _ in range(max_iter):
            if self.model_type in {'svm', 'logistic'}:
                losses = self._loss(X, y)
                anchor = int(np.argmax(losses))
                direction = self._classification_recession_direction(y[anchor])

                if self._is_infinite_kappa():
                    moved_mass = gamma_value / sample_size
                    weight = np.full(sample_size + 1, 1.0 / sample_size)
                    weight[anchor] -= moved_mass
                    weight[-1] = moved_mass
                    far_x = X[anchor] + (self.eps / moved_mass) * direction
                    candidate_X = np.vstack((X, far_x))
                    candidate_y = np.hstack((y, y[anchor]))
                    source_index = np.hstack((np.arange(sample_size), anchor))
                else:
                    eta = cp.Variable(nonneg=True)
                    alpha = cp.Variable(sample_size, nonneg=True)
                    same_loss = self._loss(X, y)
                    flipped_loss = self._loss(X, -y)
                    objective = (
                        self._lipschitz_norm() * eta * norm_theta
                        + cp.sum(cp.multiply(1 - alpha, same_loss)) / sample_size
                        + cp.sum(cp.multiply(alpha, flipped_loss)) / sample_size
                    )
                    constraints = [
                        alpha <= 1,
                        eta + self.kappa * cp.sum(alpha) / sample_size
                        == self.eps - gamma_value,
                    ]
                    problem = cp.Problem(cp.Maximize(objective), constraints)
                    self._solve_problem(
                        problem, "constructing the finite-kappa adversary"
                    )
                    if eta.value is None or alpha.value is None:
                        raise WassersteinDROError(
                            "The finite-kappa adversary problem returned no solution."
                        )
                    eta_value = max(0.0, float(eta.value))
                    alpha_value = np.clip(
                        np.asarray(alpha.value, dtype=float), 0.0, 1.0
                    )
                    denominator = (
                        eta_value + float(self.kappa) - self.eps
                        + gamma_value + 1.0
                    )
                    if denominator <= 0:
                        raise WassersteinDROError(
                            "The asymptotic mass formula has a non-positive denominator."
                        )
                    eta_gamma = gamma_value / denominator
                    if eta_gamma < -feasibility_tol or eta_gamma > 1 + feasibility_tol:
                        raise WassersteinDROError(
                            "The asymptotic construction produced an invalid mass."
                        )
                    eta_gamma = float(np.clip(eta_gamma, 0.0, 1.0))

                    unchanged_weight = (1 - alpha_value) / sample_size
                    flipped_weight = alpha_value / sample_size
                    unchanged_weight[anchor] *= 1 - eta_gamma
                    flipped_weight[anchor] *= 1 - eta_gamma
                    far_weight = eta_gamma / sample_size
                    if eta_value > 0 and far_weight > 0:
                        far_x = X[anchor] + (
                            eta_value / far_weight
                        ) * direction
                    else:
                        far_x = X[anchor].copy()
                    candidate_X = np.vstack((X, X, far_x))
                    candidate_y = np.hstack((y, -y, y[anchor]))
                    weight = np.hstack(
                        (unchanged_weight, flipped_weight, far_weight)
                    )
                    source_index = np.hstack(
                        (np.arange(sample_size), np.arange(sample_size), anchor)
                    )
            else:
                residual = X @ self.theta + self.b - y
                anchor = int(np.argmax(np.abs(residual)))
                tail_sign = 1.0 if residual[anchor] >= 0 else -1.0
                direction_x, direction_y = self._lad_recession_direction(tail_sign)
                moved_mass = gamma_value / sample_size
                weight = np.full(sample_size + 1, 1.0 / sample_size)
                weight[anchor] -= moved_mass
                weight[-1] = moved_mass
                scale = self.eps / moved_mass
                far_x = X[anchor] + scale * direction_x
                far_y = y[anchor] + scale * direction_y
                candidate_X = np.vstack((X, far_x))
                candidate_y = np.hstack((y, far_y))
                source_index = np.hstack((np.arange(sample_size), anchor))

            last_result = self._certify_distribution(
                X=X,
                y=y,
                candidate_X=candidate_X,
                candidate_y=candidate_y,
                weight=weight,
                source_index=source_index,
                gamma_used=gamma_value,
                objective_atol=objective_atol,
                objective_rtol=objective_rtol,
                feasibility_tol=feasibility_tol,
                asymptotic=True,
            )
            if last_result['certified']:
                return last_result
            gamma_value *= float(gamma_decay)
            if gamma_value <= np.finfo(float).tiny:
                break

        if last_result is None:
            raise WassersteinDROError("No adversarial distribution candidate was constructed.")
        raise WassersteinDROError(
            "Could not certify an asymptotically worst-case distribution after "
            f"{max_iter} attempts: objective_gap={last_result['optimality_gap']:.6g}, "
            f"transport_cost={last_result['transport_cost']:.6g}, eps={self.eps:.6g}. "
            "Increase the objective tolerances or max_iter, or use a larger "
            "starting gamma if the escaping atom became numerically unstable."
        )


class WassersteinDROSatisificingError(Exception):
    """Base exception class for errors in Wasserstein DRO (Robust Satisficing) model."""
    pass


class WassersteinDROsatisficing(BaseLinearDRO):
    """
    Robust satisficing version of Wasserstein DRO

    This model minimizes the subject to (approximated version) of the robust satisficing constraint of Wasserstein DRO. The Wasserstein Distance is defined as the minimum probability coupling of two distributions for the distance metric: 

    .. math::
        d((X_1, Y_1), (X_2, Y_2)) = (\|\Sigma^{1/2} (X_1 - X_2)\|_p)^{square} + \kappa |Y_1 - Y_2|,

    Reference: <https://pubsonline.informs.org/doi/10.1287/opre.2021.2238>

    """
    def __init__(self, input_dim: int, model_type: str = 'svm', 
                fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """Initialize Robust satisficing version of Wasserstein DRO.

        :param input_dim: Feature space dimension. Must satisfy :math:`d \geq 1`

        :type input_dim: int

        :param model_type: Base model architecture. Supported:

            - ``'svm'``

            - ``'logistic'``

            - ``'ols'``

            - ``'lad'``

        :type model_type: str
        :param fit_intercept: Whether to learn intercept :math:`b`.
            Disable for standardized data. Defaults to True.
        :type fit_intercept: bool

        :param solver: Convex optimization solver. Options:

            - ``'MOSEK'`` (commercial, recommended)

        :type solver: str

        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :raises ValueError:

            - If input_dim < 1

            - If invalid solver selected

        Initialization Defaults:
            1. Cost matrix initialized as identity :math:`I_d`
            2. Target ratio :math:`\tau = 1/0.8` (20% performance margin)
            3. Wasserstein order :math:`p=1` (earth mover's distance)

        Example:
            >>> model = WassersteinDROsatisficing(
            ...     input_dim=5,
            ...     model_type='svm',
            ...     solver='ECOS'
            ... )
            >>> model.cost_matrix.shape  # (5, 5)

        """
        # Parameter validation
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        
        # Initialize metric components
        self.cost_matrix = np.eye(input_dim)  
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.target_ratio = 1 / 0.8  
        self.eps = 0  
        self.p = 1 
        self.kappa = 1  

    def update(self, config: Dict[str, Any]) -> None:
        if 'cost_matrix' in config.keys():
            self.cost_matrix = config['cost_matrix']
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'target_ratio' in config.keys():
            assert (config['target_ratio'] >= 1)
            self.target_ratio = config['target_ratio']
        # the following two are only used in SVM-wasserstein
        if 'p' in config.keys():
            self.p = config['p']
        if 'kappa' in config.keys():
            self.kappa = config['kappa']
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        sample_size = len(X)
        if self.model_type in {'logistic', 'svm'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise WassersteinDROSatisificingError("classification labels not in {-1, +1}")
    
        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
                self.cost_matrix = np.eye(sample_size)
                self.cost_inv_transform = np.eye(sample_size)
            else:
                theta = cp.Variable(self.n_components)
                self.cost_matrix = np.eye(self.n_components)
                self.cost_inv_transform = np.eye(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)
        
    
        if self.kernel != 'linear':
            if self.n_components is not None:
                nystrom = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                Phi_X = nystrom.fit_transform(self.support_vectors_)
                theta_K = sqrtm(Phi_X.T @ Phi_X) @ theta
            else:
                theta_K = sqrtm(pairwise_kernels(self.support_vectors_, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)) @ theta

        else:
            theta_K = theta

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1

        sample_size, __ = X.shape
        empirical_rmse = self.fit_oracle(X, y)
        TGT = self.target_ratio * empirical_rmse
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        cons = [TGT >= cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size]
        if self.model_type == 'lad':
            if self.kappa == 'inf':
                obj = cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
            else:
                obj = cp.maximum(cp.norm(self.cost_inv_transform @ theta_K, dual_norm), 1 / self.kappa)
        elif self.model_type in ['ols', 'svm', 'logistic']:
            obj = cp.norm(self.cost_inv_transform @ theta_K, dual_norm)

        problem = cp.Problem(cp.Minimize(obj), cons)
        problem.solve(solver = self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value
        self.robust_obj = float(problem.value)

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
    
        return model_params



    # def fit_depreciate(self, X, y):
    #     """
    #     Find the best epsilon that matches the desired robust objective via bisection (depreciated)

    #     Args:
    #         X (np.ndarray): Input feature matrix with shape (n_samples, n_features).

    #         y (np.ndarray): Target vector with shape (n_samples,).

    #     Returns:
    #         Dict[str, Any]: Model parameters dictionary with 'theta' key.

    #     """

    #     warnings.warn("The bisection search is depreciated for Robust Satisficing Wasserstein DRO.")
    #     iter_num = 1
    #     # determine the empirical obj
    #     self.eps = 0
    #     empirical_rmse = self.fit_oracle(X, y)
    #     TGT = self.target_ratio * empirical_rmse
    #     # print('tgt', TGT)
    #     self.eps = 100
    #     assert (self.fit_oracle(X, y) > TGT)
    #     eps_lower, eps_upper = 0, self.eps      
    #     # binary search and find the maximum eps, such that RMSE + eps theta <= tau  
    #     for i in range(iter_num):
    #         self.eps = (eps_lower + eps_upper)/2
    #         if self.fit_oracle(X, y) > TGT:
    #             eps_upper = self.eps
    #         else:
    #             eps_lower = self.eps
        
    #     model_params = {}
    #     model_params["theta"] = self.theta.reshape(-1).tolist()
    #     return model_params
    
    def _penalization(self, theta: cp.Expression) -> float:
        """
        Module for computing the regularization part in the standard Wasserstein DRO problem.

        Args:
            theta (:py:class:`cvxpy.Expression`): Feature vector with shape (n_feature,).
        
        Returns:
            Float: Regularization term part.

        """
        if self.kernel != 'linear':
            if self.n_components is not None:
                nystrom = Nystroem(kernel = self.kernel, gamma = self.kernel_gamma, n_components = self.n_components)
                Phi_X = nystrom.fit_transform(self.support_vectors_)
                theta_K = sqrtm(Phi_X.T @ Phi_X) @ theta
            else:
                theta_K = sqrtm(pairwise_kernels(self.support_vectors_, self.support_vectors_, metric = self.kernel, gamma = self.kernel_gamma)) @ theta

        else:
            theta_K = theta

        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
        elif self.model_type == 'lad':
            if self.kappa == 'inf':
                return cp.norm(self.cost_inv_transform @ theta_K, dual_norm)
            else:
                return cp.maximum(cp.norm(self.cost_inv_transform @ theta_K, dual_norm), 1 / self.kappa)
            

    def fit_oracle(self, X, y):
        """
        Depreciated, find the optimal that given the ambiguity constraint.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).

            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            float: robust objective value

        """

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise WassersteinDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise WassersteinDROError("Input X and target y must have the same number of samples.")


        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
                self.cost_matrix = np.eye(sample_size)
                self.cost_inv_transform = np.eye(sample_size)
            else:
                theta = cp.Variable(self.n_components)
                self.cost_matrix = np.eye(self.n_components)
                self.cost_inv_transform = np.eye(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)
            
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0


        lamb_da = cp.Variable()
        cons = [lamb_da >= self._penalization(theta)]
        if self.model_type == 'ols':
            final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + math.sqrt(self.eps) * lamb_da

        else:
            if self.model_type in ['svm', 'logistic']:
                s = cp.Variable(sample_size)
                cons += [s >= self._cvx_loss(X, y, theta, b)]
                if self.kappa != 'inf':
                    cons += [s >= self._cvx_loss(X, -y, theta, b) - lamb_da * self.kappa]
                final_loss = cp.sum(s) / sample_size + self.eps * lamb_da
            else:
                # model type == 'lad' for general p.
                final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + self.eps * lamb_da

        problem = cp.Problem(cp.Minimize(final_loss), cons)

        problem.solve(solver = self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b

        self.robust_obj = float(problem.value)
        return self.robust_obj
        
    
    def worst_distribution(self, X, y):
        raise Warning("We do not compute worst case distribution for robust satisficing model since the distribution constraint is set to be held for any distribution.")

        # REQUIRED TO BE CALLED after solving the DRO problem
        # return a dict {"sample_pts": [np.array([pts_num, input_dim]), np.array(pts_num)], 'weight': np.array(pts_num)}

        # if self.is_regression == 1 or self.is_regression == 2:
        #     return NotImplementedError
        # else:
        #     sample_size, __ = X.shape
        #     if self.p == 1:
        #         dual_norm = np.inf
        #     else:
        #         dual_norm = 1 / (1 - 1 / self.p)
        #     norm_theta = np.linalg.norm(self.theta, ord = dual_norm)
        #     if self.kappa == 1000000:
        #     # not change y, we directly consider RMK 5.2 in https://arxiv.org/pdf/2308.05414.pdf, here norm_theta is lambda* there.
        #         new_X = np.zeros((sample_size, self.input_dim))
        #         for i in range(sample_size):
        #             var_x = cp.Variable(self.input_dim)
        #             obj = 1 - y[i] * var_x @ self.theta - norm_theta * cp.sum_squares(var_x - X[i])
        #             problem = cp.Problem(cp.Maximize(obj))
        #             problem.solve(solver = self.solver)
                    
        #             if 1 - y[i] * var_x.value @ self.theta < 0:
        #                 new_X[i] = X[i]
        #             else:
        #                 new_X[i] = var_x.value
        #         return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}
            
        #     else:
        #         # for general situations if we can change y, we apply Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf (SVM / logistic loss)
        #         #eta is the theta in eq(27)
        #         y_flip = -y
        #         eta = cp.Variable(nonneg = True)
        #         alpha = cp.Variable(sample_size, nonneg = True)

        #         # svm / logistic L = 1
        #         L = 1
        #         dual_loss = L * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self.loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self.loss(X, y_flip))) / sample_size
        #         cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps]
        #         problem = cp.Problem(cp.Maximize(dual_loss), cons)
        #         problem.solve(solver = self.solver)
        #         weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
        #         X = np.concatenate((X, X))
        #         y = np.concatenate((y, y_flip))
        #         return {'sample_pts': [X, y], 'weight': weight}

        






        

        
