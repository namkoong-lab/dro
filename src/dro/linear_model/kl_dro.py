from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any
import warnings


class KLDROError(Exception):
    """Base exception class for errors in KL-DRO model."""
    pass

class KLDRO(BaseLinearDRO):
    """Kullback-Leibler divergence-based Distributionally Robust Optimization (KL-DRO) model.

    This model minimizes a KL-robust loss function for both regression and classification.

    Reference: <https://optimization-online.org/wp-content/uploads/2012/11/3677.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear', eps: float = 0.0):
        """Initialize KL-divergence Distributionally Robust Optimization model.

        Inherits from BaseLinearDRO and configures KL ambiguity set parameters. 
        The ambiguity set is defined by:
        
        .. math:: 
            \\mathcal{Q} = \{ Q \\ll P \, | \, D_{KL}(Q\|P) \\leq \\epsilon \}
        
        where :math:`D_{KL}` is Kullback–Leibler divergence.

        :param input_dim: Dimension of input features. Must match training data features.
        :type input_dim: int
        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)

        :type model_type: str
        :param fit_intercept: If True, adds intercept term :math:`b` to linear model:
            :math:`\\theta^T X + b`
            Disable when data is pre-centered.
        :type fit_intercept: bool
        :param solver: Convex optimization solver, supported values:

            - ``'MOSEK'`` (recommended): Requires academic/commercial license
            
        :type solver: str
    
        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :param eps: KL divergence bound (ε ≥ 0). Special cases:

            - ε = 0: Reduces to standard empirical risk minimization (no distributional robustness)

            - ε → ∞: Approaches worst-case distribution (maximally conservative). Typical practical range: 0.01 ≤ ε ≤ 5.0

        :type eps: float

        :raises ValueError: 

            - If input_dim ≤ 0

            - If eps < 0

            - If unsupported solver specified

        Attribute Initialization:

            - ``self.dual_variable``: Stores optimal dual variable λ* after calling ``fit()``

            - ``self._p``: Internal probability vector of shape (n_samples,)

            - ``self._solver_opts``: Solver-specific options parsed from global config

        Example:
            >>> model = KLDRO(input_dim=5, model_type='logistic', eps=0.1)
            >>> model.input_dim  # 5
            >>> model.eps  # 0.1
            >>> model.dual_variable  # None (until fit is called)

        """
        
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)     

        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        elif eps > 5.0 and eps <= 100:
            warnings.warn(f"eps={eps} >5.0 may cause numerical instability", RuntimeWarning)
        elif eps > 100.0:
            raise KLDROError(f"eps >100.0 may cause system instability (got {eps})")

        self.eps = eps
        self.dual_variable = None  # To store dual variable value after fit

    def update(self, config: Dict[str, Any]) -> None:
        """Update KL-DRO model configuration parameters dynamically.

        Primarily handles robustness parameter (eps) updates while maintaining 
        optimization problem structure. Preserves existing dual variables until
        next ``fit()`` call.

        :param config: Configuration dictionary containing parameters to update.
            Recognized keys:

            - ``eps``: (float) New KL divergence bound (ε ≥ 0). Other keys are silently ignored.

        :type config: Dict[str, Any]
        :raises KLDROError: 

            - If ``eps`` value is invalid (not float/int or negative)

            - If provided ``eps`` > 100.0 (empirical stability threshold)

        :raises TypeError: If config is not a dictionary

        
        Example:
            >>> model = KLDRO(eps=0.5)
            >>> model.update({"eps": 0.8})
            >>> model.eps  # 0.8
            >>> model.update({"invalid_key": 1.0})  # No-op
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")

        if 'eps' in config:
            eps = config['eps']
            # Enhanced validation with stability check
            if not isinstance(eps, (float, int)):
                raise KLDROError(f"eps must be numeric, got {type(eps)}")
            if eps < 0:
                raise KLDROError(f"eps cannot be negative, got {eps}")
            if eps > 100.0:  # Empirical stability threshold
                raise KLDROError(f"eps >100.0 may cause system instability (got {eps})")
            
            self.eps = float(eps)


    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Solve KL-constrained distributionally robust optimization problem.

        Constructs and solves the convex optimization problem:
        
        .. math::
            \\min_{\\theta,b}  \quad \\sup_{Q \\in \\mathcal{Q}} \\mathbb{E}_Q[\\ell(\\theta,b;X,y)],\quad 
            \\text{s.t.} \quad D_{KL}(Q\|P) \\leq \\epsilon
            
        where :math:`\\mathcal{Q}` is the ambiguity set defined by KL divergence constraint.

        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray
        :param y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: Binary labels in {-1, +1}

            - Regression: Continuous real values

        :type y: numpy.ndarray
        :returns: Solution dictionary containing:

            - ``theta``: Weight vector of shape `(n_features,)`

            - ``b``: Intercept term (present if `fit_intercept=True`)

            - ``dual``: Optimal dual variable for KL constraint (λ*)

        :rtype: Dict[str, Any]
        :raises KLDROError: 

            - If problem is infeasible with current parameters

            - If solver fails to converge

        :raises ValueError:

            - If `X.shape[1] != self.input_dim`

            - If `X.shape[0] != y.shape[0]`

            - If classification labels not in {-1, +1}

        
        Example:
            >>> model = KLDRO(input_dim=3, eps=0.1)
            >>> X = np.random.randn(100, 3)
            >>> y = np.sign(np.random.randn(100))  # Binary classification
            >>> solution = model.fit(X, y)
            >>> print(solution["theta"].shape)  # (3,)
            >>> print(f"Dual variable: {solution['dual']:.4f}")
        """
        if self.model_type in {'svm', 'logistic'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise KLDROError("classification labels not in {-1, +1}")

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise KLDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise KLDROError("Input X and target y must have the same number of samples.")

        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            if self.n_components is None:
                theta = cp.Variable(sample_size)
            else:
                theta = cp.Variable(self.n_components)
        else:
            theta = cp.Variable(self.input_dim)

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable(nonneg=True)
        t = cp.Variable()
        per_loss = cp.Variable(sample_size)
        epi_g = cp.Variable(sample_size)

        # Constraints for KL-DRO
        constraints = [cp.sum(epi_g) <= sample_size * eta]
        # for i in range(sample_size):
        #     constraints.append(cp.constraints.exponential.ExpCone(per_loss[i] - t, eta, epi_g[i]))
        constraints.append(
            cp.constraints.exponential.ExpCone(
                per_loss - t, 
                eta * np.ones(sample_size),  
                epi_g 
            )
        )
        
        constraints.append(per_loss >= self._cvx_loss(X, y, theta, b))

        loss = t + eta * self.eps
        problem = cp.Problem(cp.Minimize(loss), constraints)

        try:
            problem.solve(solver=self.solver)
            self.theta = theta.value
            self.dual_variable = eta.value
        except cp.SolverError as e:
            raise KLDROError(f"Optimization failed to solve using {self.solver}.") from e

        if self.theta is None or self.dual_variable is None:
            raise KLDROError("Optimization did not converge to a solution.")

        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.tolist(), "dual": self.dual_variable, "b": self.b}

    def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute the worst-case distribution under KL divergence constraint.

        The worst-case distribution weights are computed via exponential tilting:
        
        .. math::
            w_i = \\frac{\\exp(\\ell(\\theta^*;x_i,y_i)/\\lambda^*)}{\\sum_j \\exp(\\ell(\\theta^*;x_j,y_j)/\\lambda^*)}
        
        where :math:`\\theta^*` is the optimal model parameter and :math:`\\lambda^*` is the optimal dual variable.

        :param X: Feature matrix of shape `(n_samples, n_features)`.
            Must match ``self.input_dim`` and training data dimension.
        :type X: numpy.ndarray
        :param y: Target vector of shape `(n_samples,)`. Format constraints:

            - Classification: Labels in {-1, +1}

            - Regression: Continuous values

        :type y: numpy.ndarray
        :returns: Worst-case distribution specification containing:

            - ``sample_pts``: Original samples [X, y] (reference to inputs)

            - ``weight``: Probability vector of shape `(n_samples,)`

            - ``entropy``: KL divergence :math:`D_{KL}(Q^*\|P)`

        :rtype: Dict[str, Any]
        :raises KLDROError: 

            - If inner optimization via ``fit()`` fails

            - If :math:`\\lambda^* \\leq 0` (invalid dual variable)

            - If weight normalization fails (sum → 0)

        :raises ValueError: 

            - If input dimensions mismatch

            - If classification labels violate binary constraints

        
        Example:
            >>> model = KLDRO(input_dim=3).fit(X_train, y_train)
            >>> dist = model.worst_distribution(X_test, y_test)
            >>> dist["sample_pts]
            >>> dist["weight]
        """
        self.fit(X, y)  # Fit model to obtain theta and dual variable
        # Calculate the loss with current theta
        per_loss = self._loss(X, y)
        
        if self.dual_variable is None:
            raise KLDROError("Dual variable is not set. Ensure 'fit' method has succeeded.")

        # Calculate weights for the worst-case distribution
        max_value = max(per_loss)
        weight = np.exp((per_loss-max_value) / self.dual_variable)
        weight /= np.sum(weight)  # Normalize weights

        return {'sample_pts': [X, y], 'weight': weight}