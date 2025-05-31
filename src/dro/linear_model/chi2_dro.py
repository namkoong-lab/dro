from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any

class Chi2DROError(Exception):
    """Base exception class for errors in Chi-squared DRO model.
    :meta private:
    """
    pass

class Chi2DRO(BaseLinearDRO):
    """Chi-squared Distributionally Robust Optimization (chi2-DRO) model.

    This model minimizes a chi-squared robust loss function for both regression and classification.

    Reference: <https://www.jmlr.org/papers/volume20/17-750/17-750.pdf>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear'):
        """Initialize the Chi-squared Distributionally Robust Optimization (Chi2-DRO) model.

        :param input_dim: Dimensionality of the input feature space. 
            Must match the number of columns in the training data.
        :type input_dim: int
        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)

            
        :type model_type: str
        :param fit_intercept: Whether to learn an intercept/bias term. 
            If False, assumes data is already centered. Defaults to True.
        :type fit_intercept: bool
        :param solver: Convex optimization solver. 
            Recommended solvers: 
            - ``'MOSEK'`` (requires license)
            Defaults to 'MOSEK'.
        :type solver: str

        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str

        :raises ValueError: 
            - If `model_type` is not in ['svm', 'logistic', 'ols', 'lad']
            - If `input_dim` ≤ 0

        .. note::
            - 'lad' (L1 loss) produces sparse solutions but requires longer solve times
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        self.eps = 0.0

    def update(self, config: Dict[str, Any] = {}):
        """Update the Chi-squared DRO model configuration parameters.
        
        Dynamically modify robustness settings and optimization parameters after model initialization.
        Changes will affect subsequent operations (e.g., re-fitting the model).

        :param config: Dictionary containing configuration updates. Supported keys:

            - ``eps``: Robustness parameter controlling the size of the chi-squared ambiguity set (must be ≥ 0)

            - ``solver``: Optimization solver to use (must be installed)

        :type config: Dict[str, Any]
        :raises Chi2DROError: 
            - If ``eps`` is not a non-negative numeric value
            - If unrecognized configuration keys are provided

        Example:
            >>> model = Chi2DRO(input_dim=5)
            >>> model.update({"eps": 0.5})  # Valid update
            >>> model.eps  # Verify new value
            0.5
            >>> model.update({"eps": "invalid"})  # Will raise error
            Traceback (most recent call last):
                ...
            Chi2DROError: Robustness parameter 'eps' must be a non-negative float.

        .. note::
            - Configuration changes don't trigger automatic re-optimization
            - Larger ``eps`` values make solutions more conservative
        """


        if 'eps' in config:
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise Chi2DROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the Chi-squared DRO model by solving the convex optimization problem.

        Constructs and solves the distributionally robust optimization problem using CVXPY,
        where the ambiguity set is defined by the chi-squared divergence. The optimization
        objective and constraints are built dynamically based on input data.

        :param X: Training feature matrix. Must have shape `(n_samples, n_features)`,
            where `n_features` should match the `input_dim` specified during initialization.
        :type X: numpy.ndarray
        :param y: Target values. For classification tasks, expected to be binary (±1 labels).
            Shape must be `(n_samples,)`.
        :type y: numpy.ndarray
        :returns: Dictionary containing the trained model parameters:

            - ``theta``: Weight vector of shape `(n_features,)`

            - ``b``: Intercept term (only present if `fit_intercept=True`)

        :rtype: Dict[str, Any]
        :raises Chi2DROError: 
            - If the optimization solver fails to converge
            - If the problem is infeasible due to invalid hyperparameters
        :raises ValueError:
            - If `X` and `y` have inconsistent sample sizes (`X.shape[0] != y.shape[0]`)
            - If `X` has incorrect feature dimension (`X.shape[1] != input_dim`)

        Optimization Problem:
            .. math::
                \\min_{\\theta,b} \\max_{P \in \\mathcal{P}} \\mathbb{E}_P[\\ell(\\theta, b; X, y)]
            
            where :math:`\\mathcal{P}` is the ambiguity set defined by chi-squared divergence:

            .. math::
                \\mathcal{P} = \{ P: D_{\\chi^2}(P \| P_0) \\leq \\epsilon \}

        Example:
            >>> model = Chi2DRO(input_dim=5, eps=0.5, fit_intercept=True)
            >>> X_train = np.random.randn(100, 5)
            >>> y_train = np.sign(np.random.randn(100))  # Binary labels
            >>> params = model.fit(X_train, y_train)
            >>> print(params["theta"].shape)  # (5,)
            >>> print("b" in params)  # True

        .. note::
            - Large values of `eps` increase robustness but may lead to conservative solutions
            - Warm-starting is not supported due to DRO problem structure
        """
        if self.model_type in {'svm', 'logistic'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise Chi2DROError("classification labels not in {-1, +1}")
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise Chi2DROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise Chi2DROError("Input X and target y must have the same number of samples.")

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
        eta = cp.Variable()

        loss = (np.sqrt(1 + self.eps) / np.sqrt(sample_size) * 
                cp.norm(cp.pos(self._cvx_loss(X, y, theta, b) - eta), 2) + eta)
        
        problem = cp.Problem(cp.Minimize(loss))
        try:
            problem.solve(solver=self.solver)
        except cp.error.SolverError as e:
            raise Chi2DROError(f"Optimization failed to solve using {self.solver}.") from e

        if theta.value is None:
            raise Chi2DROError("Optimization did not converge to a solution.")

        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        return {"theta": self.theta.reshape(-1).tolist(), "b": self.b}


    def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute the worst-case distribution within the chi-squared ambiguity set.

        This method solves a convex optimization problem to find the probability distribution 
        that maximizes the expected loss under the chi-squared divergence constraint. The result
        characterizes the adversarial data distribution the model is robust against.

        :param X: Feature matrix of shape `(n_samples, n_features)`. 
            Must match the model's `input_dim` (n_features).
        :type X: numpy.ndarray
        :param y: Target vector of shape `(n_samples,)`. For regression tasks, continuous values 
            are expected; for classification, ±1 labels.
        :type y: numpy.ndarray

        :returns: Dictionary containing:

            - ``sample_pts``: Original data points as a tuple ``(X, y)``

            - ``weight``: Worst-case probability weights of shape `(n_samples,)`

        :rtype: Dict[str, Any]

        :raises Chi2DROError: 

            - If the optimization solver fails to converge

            - If the solution is infeasible or returns null weights

        :raises ValueError: 

            - If `X` and `y` have inconsistent sample sizes

            - If `X` feature dimension ≠ `input_dim`

        Optimization Formulation:
            .. math::
                \\max_{p \\in \\Delta} \ \\sum_{i=1}^n p_i \\cdot \\ell_i 
                \ \ \ s.t. \ \\sum_{i=1}^n n(p_i - 1/n)^2 \\leq \\epsilon
            
            where:

                - :math:`\\ell_i` is the loss for the i-th sample

                - :math:`\\Delta` is the probability simplex

                - :math:`\\epsilon` is the robustness parameter ``self.eps``

        Example:
            >>> model = Chi2DRO(input_dim=5, eps=0.5)
            >>> X = np.random.randn(100, 5)
            >>> y = np.sign(np.random.randn(100))  # Binary labels
            >>> dist = model.worst_distribution(X, y)
            >>> print(dist["weight"].shape)  # (100,)
            >>> np.testing.assert_allclose(dist["weight"].sum(), 1.0, rtol=1e-3)  # Sum to 1

        .. note::

            - The weights are guaranteed to be non-negative and sum to 1

            - Larger ``eps`` allows more deviation from the empirical distribution
            
            - Requires prior model fitting via :meth:`fit`

        .. _reference: https://jmlr.org/papers/volume20/17-750/17-750.pdf
        """
        
        self.fit(X, y)

        sample_size, _ = X.shape
        
        per_loss = self._loss(X, y)
        prob = cp.Variable(sample_size, nonneg=True)
        
        constraints = [
            cp.sum(prob) == 1,
            cp.sum_squares(sample_size * prob - 1) <= sample_size * self.eps
        ]
        
        problem = cp.Problem(cp.Maximize(prob @ per_loss), constraints)
        try:
            problem.solve(solver=self.solver)
        except cp.error.SolverError as e:
            raise Chi2DROError("Optimization failed to solve for worst-case distribution.") from e

        if prob.value is None:
            raise Chi2DROError("Worst-case distribution optimization did not converge to a solution.")

        return {'sample_pts': [X, y], 'weight': prob.value}

