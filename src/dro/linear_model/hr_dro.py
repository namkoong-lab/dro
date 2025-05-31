import cvxpy as cp
import numpy as np
from typing import Dict, Any
from .base import BaseLinearDRO, DataValidationError

class HRDROError(Exception):
    """Base exception class for errors in HR DRO model."""
    pass

class HR_DRO_LR(BaseLinearDRO):
    """Holistic Robust DRO Linear Regression (HR_DRO_LR) model.
    
    This model supports HR DRO with additional robustness constraints for linear regression and binary classification. (Theorem 7)

    Reference: <https://arxiv.org/abs/2207.09560>
    """
    
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, 
                 solver: str = 'MOSEK', r: float = 1.0, alpha: float = 0.0, 
                 epsilon: float = 0.5, epsilon_prime: float = 1.0):
        """Initialize advanced DRO model with dual robustness parameters.

        :param input_dim: Number of input features. Must be ≥ 1.
        :type input_dim: int
        :param model_type: Base model type. Defaults to 'svm'.
        :type model_type: str
        :param fit_intercept: Whether to fit intercept term. Defaults to True.
        :type fit_intercept: bool
        :param solver: Optimization solver. Defaults to 'MOSEK'.
        :type solver: str
        :param r: Ambiguity set curvature parameter. Must satisfy r ≥ 0. Defaults to 1.0.
        :type r: float
        :param alpha: Risk level parameter. Must satisfy 0 ≤ alpha < 1. Defaults to 0.0.
        :type alpha: float
        :param epsilon: Wasserstein radius for distribution shifts. Must be ≥ 0. Defaults to 0.5.
        :type epsilon: float
        :param epsilon_prime: Robustness budget for moment constraints. Must be ≥ 0. Defaults to 1.0.
        :type epsilon_prime: float

        :raises ValueError: 

            - If any parameter violates numerical constraints (r < 0, alpha ∉ (0,1], etc.)

            - If model_type is not in allowed set

            - If input_dim < 1

        Example:
            >>> model = HR_DRO_LR(
            ...     input_dim=5, 
            ...     model_type='logistic',
            ...     r=0.5, 
            ...     epsilon=0.3,
            ...     epsilon_prime=0.8
            ... )
            >>> print(model.r, model.epsilon_prime)
            0.5 0.8
        """
        # Parameter validation
        if input_dim < 1:
            raise HRDROError("input_dim must be ≥ 1")
        if model_type in ['ols', 'logistic']:
            raise HRDROError("HR DRO does not support OLS, logistic")
        if r < 0 or alpha < 0 or alpha >= 1 or epsilon < 0 or epsilon_prime < 0:
            raise HRDROError("Parameters must satisfy: r ≥ 0, 0 ≤ alpha < 1, epsilon ≥ 0, epsilon_prime ≥ 0")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.r = r
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
    
    def update(self, config: Dict[str, Any] = {}) -> None:
        """Dynamically update robustness parameters of the Advanced DRO model.
        
        Modifies multi-level ambiguity set parameters while preserving existing model state.
        Changes require manual re-fitting to take effect.

        :param config: Dictionary containing parameter updates. Supported keys:

            - ``r``: Curvature parameter for ambiguity set (must be ≥ 0)

            - ``alpha``: Risk level for CVaR-like constraints (must satisfy 0 < alpha ≤ 1)

            - ``epsilon``: Wasserstein radius for distribution shifts (must be ≥ 0)

            - ``epsilon_prime``: Moment constraint robustness budget (must be ≥ 0)

        :type config: Dict[str, Any]

        :raises ValueError: 

            - If any parameter violates numerical constraints

            - If invalid parameter types are provided

        Parameter Relationships:

            - Larger ``epsilon`` allows more distribution shift

            - Higher ``r`` creates more conservative ambiguity set geometries

            - ``epsilon_prime`` controls second-order moment robustness

        Example:
            >>> model = HR_DRO_LR(input_dim=5, r=1.0, epsilon=0.5)
            >>> model.update({"r": 0.8, "epsilon": 0.6})  # Valid update
            >>> model.update({"alpha": 1.5})  # Invalid alpha
            Traceback (most recent call last):
                ...
            ValueError: alpha must be in (0, 1], got 1.5

        .. note::

            - Empty config dictionary will preserve current parameters

            - Partial updates are allowed (only modify specified parameters)

            - Always validate parameters before optimization phase

        """
        # Parameter validation
        new_r = config.get("r", self.r)
        new_alpha = config.get("alpha", self.alpha)
        new_epsilon = config.get("epsilon", self.epsilon)
        new_epsilon_prime = config.get("epsilon_prime", self.epsilon_prime)

        if new_r < 0:
            raise ValueError(f"r must be ≥ 0, got {new_r}")
        if not (0 < new_alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {new_alpha}")
        if new_epsilon < 0 or new_epsilon_prime < 0:
            raise ValueError(f"Epsilon parameters must be ≥ 0, got ε={new_epsilon}, ε'={new_epsilon_prime}")

        # Apply validated parameters
        self.r = new_r
        self.alpha = new_alpha
        self.epsilon = new_epsilon
        self.epsilon_prime = new_epsilon_prime
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Solve the dual-ambiguity DRO problem via convex optimization.

        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param Y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type Y: numpy.ndarray

        :returns: Dictionary containing trained parameters:
        
            - ``theta``: Weight vector of shape `(n_features,)`
        
            - ``b``: Intercept term (if `fit_intercept=True`)

        :rtype: Dict[str, Any]

        :raises HRDROError: 

            - If optimization fails due to solver errors

            - If problem becomes infeasible with current parameters

        :raises ValueError:

            - If `X.shape[1] != self.input_dim`

            - If `X.shape[0] != Y.shape[0]`

            - If parameters violate `r ≥ 0`, `epsilon ≥ 0`, etc.

        Optimization Formulation:
            .. math::
                \\min_{\\theta,b} \\sup_{Q \\in \\mathcal{Q}} \\mathbb{E}_Q[\\ell(\\theta,b;X,Y)]
                
            where the ambiguity set :math:`\\mathcal{Q}` satisfies:

            .. math::
                W(P,Q) \\leq \\epsilon 

            .. math::    
                \\text{CVaR}_\\alpha(\\ell) \\leq \\tau + r\\epsilon' 

            .. math::
                \\mathbb{E}_Q[\\phi(X)] \\leq \\mathbb{E}_P[\\phi(X)] + \\epsilon'
                
            - :math:`W(P,Q)` = Wasserstein distance between distributions

            - :math:`r` = curvature parameter from `self.r`

            - :math:`\\phi(\\cdot)` = moment constraint function

        Example:
            >>> model = HR_DRO_LR(input_dim=3, r=0.5, epsilon=0.4)
            >>> X_train = np.random.randn(100, 3)
            >>> y_train = np.sign(np.random.randn(100))  # Binary labels
            >>> params = model.fit(X_train, y_train)
            >>> print(params["theta"].shape)  # (3,)
            >>> print("dual_vars" in params)  # True

        .. note::

            - Solution time grows cubically with `n_samples`

            - Set `epsilon=0` to disable Wasserstein constraints

        """
        if self.model_type in {'svm', 'logistic'}:    
            is_valid = np.all((Y == -1) | (Y == 1))
            if not is_valid:
                raise HRDROError("classification labels not in {-1, +1}")
        
        T, feature_dim = X.shape
        if feature_dim != self.input_dim:
            raise DataValidationError(f"Expected input with {self.input_dim} features, got {feature_dim}.")

        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        w = cp.Variable(T)
        lambda_ = cp.Variable(nonneg=True)
        beta = cp.Variable(nonneg=True)
        eta = cp.Variable()
        temp = cp.Variable()

        # Define the objective function
        objective = cp.Minimize((1 / T) * cp.sum(w) + lambda_ * (self.r - 1) + beta * self.alpha + eta)

        # Constraints setup based on the model type
        constraints = []
        ## Appendix D.1 in https://arxiv.org/pdf/2207.09560v4
        if self.model_type in {'lad'}:
            for t in range(T):
                constraints.extend([
                    temp >= cp.abs(theta.T @ X[t] + b - Y[t]),
                    w[t] >= cp.rel_entr(lambda_, (eta - cp.abs(theta.T @ X[t] + b - Y[t]) - self.epsilon * cp.norm(theta, 2))),
                    w[t] >= cp.rel_entr(lambda_, (eta - temp - self.epsilon_prime * cp.norm(theta, 2))) - beta,
                    eta >= cp.abs(theta.T @ X[t]  + b - Y[t]) + self.epsilon_prime * cp.norm(theta, 2)
                ])
            
        ## Appendix D.2 in https://arxiv.org/pdf/2207.09560v4
        elif self.model_type == 'svm':
            constraints.append(eta >= 1e-6)  # Stability constraint for eta
            margin = cp.multiply(Y, X @ theta + b)
            theta_norm = cp.norm(theta, 2)
            
            constraints += [
                temp <= cp.min(margin),
                w >= cp.rel_entr(lambda_, eta),
                w >= cp.rel_entr(lambda_, (eta - 1) + margin - self.epsilon * theta_norm),
                w >= cp.rel_entr(lambda_, (eta - 1) + temp - self.epsilon_prime * theta_norm) - beta,
                eta >= cp.max(1 - margin + self.epsilon * theta_norm)
            ]

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.MOSEK,
                mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8},
                verbose=True)
        except cp.error.SolverError as e:
            raise HRDROError(f"Optimization failed to solve using {self.solver}.") from e

        if theta.value is None:
            raise HRDROError("Optimization did not converge to a solution.")

        # Store fitted parameters
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        self.w = w.value
        self.lambda_ = lambda_.value
        self.beta = beta.value
        self.eta = eta.value

        # Return model parameters in dictionary format
        return {"theta": self.theta.tolist(), "b": self.b}
