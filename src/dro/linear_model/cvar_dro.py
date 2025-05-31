from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class CVaRDROError(Exception):
    """Base exception class for errors in CVaR DRO model."""
    pass

class CVaRDRO(BaseLinearDRO):
    """Conditional Value-at-Risk (CVaR) Distributionally Robust Optimization (DRO) model.

    This model minimizes a robust loss function for both regression and classification tasks
    under a CVaR constraint.

    Reference: <https://www.risk.net/journal-risk/2161159/optimization-conditional-value-risk>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, 
                 solver: str = 'MOSEK', kernel: str = 'linear', alpha: float = 1.0):
        """Initialize a CVaR-constrained DRO model.

        :param input_dim: Number of input features. Must match training data dimension.
        :type input_dim: int
        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)
            
        :type model_type: str
        :param fit_intercept: Whether to learn an intercept term. Disable if data is pre-centered.
            Defaults to True.
        :type fit_intercept: bool
        :param solver: Convex optimization solver. Recommended: 'MOSEK' (commercial).
            Defaults to 'MOSEK'.
        :type solver: str
        :param alpha: Risk level controlling the CVaR conservativeness. Must satisfy 0 < alpha ≤ 1.
            Defaults to 1.0 (equivalent to worst-case optimization).
        :type alpha: float

        :raises ValueError: 

            - If ``model_type`` is not in ['svm', 'logistic', 'ols']

            - If ``alpha`` is outside (0, 1]

        """
        super().__init__(input_dim, model_type, fit_intercept, solver,kernel)
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.threshold_val = None

    def update(self, config: Dict[str, Any]) -> None:
        """Dynamically update CVaR-DRO model configuration parameters.
        
        Modifies risk-sensitive hyperparameters and optimization settings. Changes take effect immediately but require re-fitting the model to update solutions.

        :param config: Dictionary of configuration updates. Supported keys:

            - ``alpha``: Risk level parameter controlling CVaR conservativeness (must satisfy 0 < alpha ≤ 1)

        :type config: Dict[str, Any]

        :raises CVaRDROError: 

            - If ``alpha`` is not in (0, 1]

            - If unrecognized configuration keys are provided (future-proofing)

        Example:
            >>> model = CVaRDRO(input_dim=5, alpha=0.95)
            >>> model.update({"alpha": 0.9})  # Valid adjustment
            >>> model.alpha
            0.9
            >>> model.update({"alpha": 1.5})  # Invalid value
            Traceback (most recent call last):
                ...
            CVaRDROError: Risk parameter 'alpha' must be in the range (0, 1].

        .. note::

            - Decreasing ``alpha`` makes the model less conservative (focuses on average risk)

            - Requires manual re-fitting via :meth:`fit` after configuration changes

            - Configuration keys other than ``alpha`` will be silently ignored

        """

        if 'alpha' in config:
            alpha = config['alpha']
            if not (0 < alpha <= 1):
                raise CVaRDROError("Risk parameter 'alpha' must be in the range (0, 1].")
            self.alpha = float(alpha)


    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Solve the CVaR-constrained distributionally robust optimization problem.
        
        Constructs and solves the convex optimization problem to find model parameters 
        that minimize the worst-case CVaR of the loss distribution. The solution defines 
        a robust decision boundary/regression plane under adversarial distribution shifts.

        :param X: Training feature matrix of shape `(n_samples, n_features)`. 
            Must match the `input_dim` specified during initialization.
        :type X: numpy.ndarray
        :param y: Target values of shape `(n_samples,)`. For classification, use ±1 labels; 
            for regression, use continuous values.
        :type y: numpy.ndarray

        :returns: Dictionary containing trained parameters:

            - ``theta``: Weight vector of shape `(n_features,)`

            - ``threshold``: Optimal CVaR threshold value (stored as ``self.threshold_val``)

            - ``b``: Intercept term (only present if `fit_intercept=True`)

        :rtype: Dict[str, Any]

        :raises CVaRDROError: 

            - If the optimization solver fails to converge

            - If the problem is infeasible with current `alpha`

        :raises ValueError: 

            - If `X.shape[1] != self.input_dim`

            - If `X.shape[0] != y.shape[0]`

        Optimization Formulation:
            .. math::
                \\min_{\\theta,b} \ CVaR_\\alpha(\\ell(\\theta, b; X, y)) 
                
            where:

                - :math:`CVaR_\\alpha` = Conditional Value-at-Risk at level :math:`\\alpha`

                - :math:`\\ell` = model-specific loss (hinge loss for SVM, squared loss for OLS, etc.)

        Example:
            >>> model = CVaRDRO(input_dim=3, alpha=0.9, model_type='svm')
            >>> X_train = np.random.randn(100, 3)
            >>> y_train = np.sign(np.random.randn(100))
            >>> params = model.fit(X_train, y_train)
            >>> print(params["theta"].shape)  # (3,)
            >>> print("threshold" in params)  # True
            >>> print(model.threshold_val == params["threshold"])  # True

        .. note::

            - Higher :math:`\\alpha` values require solving more conservative (pessimistic) scenarios

            - The ``threshold`` value represents the :math:`\\alpha`-quantile of the loss distribution

            - Warm-starting not supported due to CVaR's non-smooth nature

        .. _Rockafellar2000: https://www.risk.net/journal-risk/2161159/optimization-conditional-value-risk
        """
        if self.model_type in {'svm', 'logistic'}:    
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise CVaRDROError("classification labels not in {-1, +1}")
        
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise CVaRDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise CVaRDROError("Input X and target y must have the same number of samples.")

        # Define optimization variables
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
            
        eta = cp.Variable()

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        # Define the CVaR loss function
        loss = cp.sum(cp.pos(self._cvx_loss(X, y, theta, b) - eta)) / (sample_size * self.alpha) + eta
        problem = cp.Problem(cp.Minimize(loss))

        try:
            problem.solve(solver=self.solver)
            self.theta = theta.value
            self.threshold_val = eta.value

        except cp.SolverError as e:
            raise CVaRDROError("Optimization failed to solve using MOSEK.") from e

        if self.theta is None or self.threshold_val is None:
            raise CVaRDROError("Optimization did not converge to a solution.")
        
        

        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.tolist(), "threshold": self.threshold_val, "b":self.b}
    
    def worst_distribution(self, X: np.ndarray, y: np.ndarray, precision: float = 1e-5) -> Dict[str, Any]:
        """Compute the worst-case distribution under CVaR constraint.
        
        Identifies samples contributing to the alpha-tail risk distribution and assigns uniform weights to these 
        high-loss scenarios. This represents the adversarial distribution that maximizes the conditional expected loss.

        :param X: Feature matrix of shape `(n_samples, n_features)`.
            Must match the model's `input_dim` (n_features).
        :type X: numpy.ndarray

        :param y: Target vector of shape `(n_samples,)`. Requires pre-processed labels:

            - Binary classification: ±1 labels

            - Regression: Continuous values

        :type y: numpy.ndarray
        :param precision: Perturbation tolerance for loss threshold comparison. 
            Compensates for numerical instability in loss computations. Defaults to 1e-5.
        :type precision: float

        :returns: Dictionary containing:

            -``sample_pts``: Tuple of filtered feature matrix and targets ``(X_high, y_high)``, where `X_high` shape = `(n_high_risk, n_features)`

            -``weight``: Uniform probability weights of shape `(n_high_risk,)` summing to 1

        :rtype: Dict[str, Any]

        :raises CVaRDROError: 

            - If model hasn't been fitted (threshold_val is None)

            - If all sample losses fall below CVaR threshold (empty distribution)

        Optimization Formulation:
            .. math::
                \mathcal{P}_{\\text{worst}} = \\{ (X_i,y_i) \mid \\ell(\\theta;X_i,y_i) > \\tau_\\alpha + \\epsilon \\}

            where:

                - :math:`\\tau_\\alpha` = CVaR threshold from :meth:`fit`

                - :math:`\\epsilon` = precision parameter

                - Weights are assigned uniformly: :math:`p_i = 1 / |\mathcal{P}_{\\text{worst}}|`

        Example:
            >>> model = CVaRDRO(input_dim=3, alpha=0.95)
            >>> X = np.random.randn(100, 3)
            >>> y = np.sign(np.random.randn(100))
            >>> model.fit(X, y)
            >>> dist = model.worst_distribution(X, y)
            >>> high_risk_X, high_risk_y = dist["sample_pts"]
            >>> print(high_risk_X.shape[0] == len(dist["weight"]))  # True
            >>> np.testing.assert_allclose(dist["weight"].sum(), 1.0, atol=1e-6)

        .. note::

            - The returned distribution always includes samples with loss values exceeding :math:`\\tau_\\alpha + \\epsilon`, where:

            - :math:`\\tau_\\alpha` is the CVaR threshold from the fitted model

            - :math:`\\epsilon` (epsilon) is the ``precision`` parameter

            - If all sample losses are below :math:`\tau_\alpha + \epsilon`, returns an empty weight array and raises a ``UserWarning``

            - The ``precision`` parameter mitigates false positives/negates from floating-point errors in loss comparisons (default=1e-5)
            
        """

        self.fit(X,y)
        
        # Calculate per-sample loss with the current theta
        per_loss = self._loss(X, y)

        if self.threshold_val is None:
            raise CVaRDROError("Threshold value is not set. Ensure 'fit' method has been called.")

        # Identify samples with loss above the threshold
        indices = np.where(per_loss + precision > self.threshold_val)[0]
        # Compute weights for the worst-case distribution
        weight = np.ones(len(indices)) / len(indices) if len(indices) > 0 else np.array([])

        return {'sample_pts': [X[indices], y[indices]], 'weight': weight}
