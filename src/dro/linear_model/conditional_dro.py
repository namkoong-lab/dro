from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class ConditionalCVaRDROError(Exception):
    """Exception class for errors in Marginal CVaR DRO model."""
    pass


class ConditionalCVaRDRO(BaseLinearDRO):
    """Y|X (ConditionalShiftBased) Conditional Value-at-Risk (Conditional-CVaR) Distributionally Robust Optimization (DRO) model that only allow likelihood ratio changes in Y|X.

    This model minimizes a robust loss function for both regression and classification tasks
    under a CVaR constraint only for the distribution of Y|X.    

    Conditional CVaR DRO model following Theorem 2 in:
    with alpha(x) to be beta^T x for simplicity
    alpha corresponds to Gamma in the paper.

    Reference: <https://arxiv.org/pdf/2209.01754.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', 
                 fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """Initialize the linear DRO model.

        :param input_dim: Number of input features. Must be ≥ 1.
        :type input_dim: int
        :param model_type: Type of base model. Valid options: 
            'svm', 'logistic', 'ols', 'lad'. Defaults to 'svm'.
        :type model_type: str
        :param fit_intercept: Whether to learn an intercept term. 
            Set False for pre-centered data. Defaults to True.
        :type fit_intercept: bool
        :param solver: Optimization solver. 
            See class-level documentation for recommended options. Defaults to 'MOSEK'.
        :type solver: str
        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str
        :raises ValueError: 
            - If ``input_dim`` < 1
            - If ``model_type`` is not in ['svm', 'logistic', 'ols', 'lad']
        
        Example:
            >>> model = ConditionalCVaRDRO(input_dim=5, model_type='logistic')
            >>> print(model.model_type)
            'logistic'
            >>> print(model.alpha)
            1.0
        """

        if input_dim < 1:
            raise ValueError("input_dim must be ≥ 1")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        self.alpha = 1.0      
        self.control_name = None  

    def update(self, config: Dict[str, Any]) -> None:
        """Update Conditional CVaR-DRO model configuration.
        
        Modifies control features and risk sensitivity parameters dynamically. 
        Changes affect subsequent optimization but require manual re-fitting.

        :param config: Dictionary of configuration updates. Valid keys:

            - ``control_name``: Indices of controlled features (0 ≤ index < input_dim)

            - ``alpha``: Risk level for CVaR constraint (0 < alpha ≤ 1)

        :type config: Dict[str, Any]
        :raises ConditionalCVaRDROError: 

            - If ``control_name`` contains invalid indices

            - If ``alpha`` is outside (0, 1]

            - If unrecognized configuration keys are provided

        Control Features:

            - Controlled features (``control_name``) are protected from distribution shifts

            - Indices must satisfy: :math:`0 \\leq \\text{index} < \\text{input_dim}`

        Example:
            >>> model = ConditionalCVaRDRO(input_dim=5)
            >>> model.update({
            ...     'control_name': [0, 2],  # Protect 1st and 3rd features
            ...     'alpha': 0.95
            ... })
            >>> model.update({'control_name': [5]})  # Invalid index for input_dim=5
            Traceback (most recent call last):
                ...
            ConditionalCVaRDROError: All indices in 'control_name' must be in [0, input_dim - 1]

        .. note::

            - Setting ``control_name=None`` disables feature protection

            - Lower ``alpha`` values reduce conservatism (focus on average risk)

            - Configuration changes invalidate previous solutions (requires re-fitting)

        """
        if 'control_name' in config:
            control_name = config['control_name']
            if control_name is not None:
                if not all(0 <= x < self.input_dim for x in control_name):
                    raise ConditionalCVaRDROError(
                        f"All indices in 'control_name' must be in [0, {self.input_dim-1}]"
                    )
            self.control_name = control_name

        if 'alpha' in config:
            alpha = config['alpha']
            if not (0 < alpha <= 1):
                raise ConditionalCVaRDROError("Parameter 'alpha' must be in (0, 1]")
            self.alpha = float(alpha)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Solve the Conditional CVaR-constrained DRO problem via convex optimization.
        
        Constructs and optimizes a distributionally robust model that minimizes the 
        worst-case conditional expected loss, where the ambiguity set is constrained 
        by both CVaR and feature control parameters.

        :param X: Training feature matrix of shape `(n_samples, n_features)`. 
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type y: numpy.ndarray

        :returns: Dictionary containing trained parameters:

            - ``theta``: Weight vector of shape `(n_features,)`

            - ``b``: Intercept term (exists if `fit_intercept=True`)

            - ``cvar_threshold``: Optimal CVaR threshold value

        :rtype: Dict[str, Any]
        :raises ConditionalCVaRDROError: 
            - If optimization fails (solver error/infeasible)
            - If `X` and control features have dimension mismatch
        :raises ValueError:
            - If `X.shape[1] != self.input_dim`
            - If `X.shape[0] != y.shape[0]`

        Optimization Formulation:
            .. math::
                \\min_{\\theta,b} \ \\sup_{Q \in \\mathcal{Q}} \\mathbb{E}_Q[\\ell(\\theta,b;X,y)]
                
            where the ambiguity set :math:`\mathcal{Q}` satisfies:

            .. math::
                \\text{CVaR}_\\alpha(\\ell) \\leq \\tau \quad \\text{and} \quad X_{\\text{control}} = \\mathbb{E}_P[X_{\\text{control}}]

            - :math:`X_{\\text{control}}` = features specified by `control_name`

            - :math:`\\tau` = CVaR threshold (optimization variable)

        Example:
            >>> model = ConditionalCVaRDRO(input_dim=5, control_name=[0,2], alpha=0.9)
            >>> X_train = np.random.randn(100, 5)
            >>> y_train = np.sign(np.random.randn(100)) 
            >>> params = model.fit(X_train, y_train)
            >>> assert params["theta"].shape == (5,)
            >>> assert "b" in params
            >>> print(f"CVaR threshold: {params['cvar_threshold']:.4f}")

        .. note::

            - Controlled features (``control_name``) are assumed fixed under distribution shifts

            - Solution cache is invalidated after calling :meth:`update`
            
        """
        if self.model_type in {'svm', 'logistic'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise ConditionalCVaRDROError("classification labels not in {-1, +1}")
            
        sample_size, feature_size = X.shape

        if feature_size != self.input_dim:
            raise ConditionalCVaRDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise ConditionalCVaRDROError("Input X and target y must have the same number of samples.")


        if self.control_name is not None:
            control_X = X[:,self.control_name]
        else:
            control_X = X
        
        if self.kernel != 'linear':
            self.support_vectors_ = X
            if not isinstance(self.kernel_gamma, float):
                self.kernel_gamma = 1 / (self.input_dim * np.var(X))
            theta = cp.Variable(sample_size)
        else:
            theta = cp.Variable(self.input_dim)

        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0

        beta = cp.Variable(len(control_X[0]))
        cost = cp.sum(self._cvx_loss(X, y, theta, b)) / (1 / self.alpha) + (1 / self.alpha - self.alpha) * cp.sum(cp.pos(self._cvx_loss(X, y, theta, b) - control_X @ beta)) + (1 - self.alpha) * cp.sum(control_X @ beta) 

        prob = cp.Problem(cp.Minimize(cost / sample_size))
        
        try:
            prob.solve(solver = self.solver)
            self.theta = theta.value
        except cp.error.SolverError as e:
            raise ConditionalCVaRDROError(f"Optimization failed to solve using {self.solver}.") from e

        if self.theta is None:
            raise ConditionalCVaRDROError("Optimization did not converge to a solution.")

        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params['theta'] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b


        return model_params
