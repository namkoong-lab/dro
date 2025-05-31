from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors  
from scipy.sparse import triu

class MarginalCVaRDROError(Exception):
    """Exception class for errors in Marginal CVaR DRO model."""
    pass

class MarginalCVaRDRO(BaseLinearDRO):
    """Marginal-X Conditional Value-at-Risk (Conditional-CVaR) Distributionally Robust Optimization (DRO) model that only allow likelihood ratio changes in X.

    This model minimizes a robust loss function for both regression and classification tasks
    under a CVaR constraint only for the marginal distribution of X.
    
    Reference:
    [1] <https://pubsonline.informs.org/doi/10.1287/opre.2022.2363>
    [2] The specific model follows Equation (27) in:
    https://arxiv.org/pdf/2007.13982.pdf with parameters L and p.
    """
    
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK', kernel: str = 'linear', alpha: float = 1.0, L: float = 10.0, p: int = 2):
        """
        :param input_dim: Dimension of input features. Must be ≥ 1.
        :type input_dim: int
        :param model_type: Base model architecture. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'logistic'``: Logistic loss (classification)

            - ``'ols'``: Least squares (regression)

            - ``'lad'``: Least absolute deviation (regression)

        :type model_type: str
        :param fit_intercept: Whether to learn intercept term :math:`b`. 
            Set False for pre-centered data. Defaults to True.
        :type fit_intercept: bool
        :param solver: Convex optimization solver. Defaults to 'MOSEK'.
        :type solver: str
        :param kernel: the kernel type to be used in the optimization model, default = 'linear'
        :type kernel: str
        :param alpha: CVaR risk level controlling tail expectation. 
            Must satisfy 0 < α ≤ 1. 
            Defaults to 1.0.
        :type alpha: float
        :param L: Wasserstein radius scaling factor. 
            Larger values increase distributional robustness. Must satisfy L ≥ 0.
            Defaults to 10.0.
        :type L: float
        :param p: Order of Wasserstein distance. 
            Supported values: 1 (Earth Mover's Distance) or 2 (Quadratic Wasserstein).
            Defaults to 2.
        :type p: int
        :raises ValueError: 

            - If input_dim < 1

            - If alpha ∉ (0, 1]

            - If L < 0

            - If p < 1

        Example:
            >>> model = MarginalCVaRDRO(
            ...     input_dim=5, 
            ...     model_type='lad',
            ...     alpha=0.95,
            ...     L=5.0,
            ...     p=1
            ... )
            >>> model.L  # 5.0
            >>> model.p  # 1
        """

        # Parameter validation
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if L < 0:
            raise ValueError(f"L must be ≥ 0, got {L}")
        if p < 1:
            raise ValueError(f"p must be 1 or 2, got {p}")

        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver, kernel)
        self.alpha = alpha
        self.L = L
        self.p = p
        self.control_name = None  
        self.threshold_val = None  
        self.B_val = None  
        self.n_components = 100

    def update(self, config: Dict[str, Any]) -> None:
        """Update Marginal CVaR-DRO model configuration parameters.

        Dynamically adjusts the robustness parameters for marginal distribution shifts. 
        Preserves existing solutions until next ``fit()`` call.

        :param config: Configuration dictionary with optional keys:

            - ``control_name`` (List[int]): 
                Indices of features to protect against marginal shifts. Constraints:

                - All indices must satisfy :math:`0 \\leq \\text{index} < \\text{input_dim}`

                - Empty list disables marginal robustness

            - ``L`` (float): 
                Wasserstein radius scaling factor. Must satisfy :math:`L > 0`
                Larger values increase conservativeness

            - ``p`` (int): 
                Wasserstein metric order. Must satisfy :math:`p \\geq 1`
                Supported values: 1 (EMD), 2 (Quadratic)

            - ``alpha`` (float): 
                CVaR risk level. Must satisfy :math:`0 < \\alpha \leq 1`
                :math:`\\alpha \\to 0` focuses on average loss, :math:`\\alpha = 1` is worst-case

        :type config: Dict[str, Any]

        :raises MarginalCVaRDROError: 

            - If ``control_name`` contains invalid indices

            - If ``L`` is non-positive

            - If ``p`` < 1

            - If ``alpha`` ∉ (0, 1]

            - If config contains unsupported keys

        Example:
            >>> model = MarginalCVaRDRO(input_dim=5, control_name=[1,3])
            >>> model.update({
            ...     "L": 15.0,      
            ...     "alpha": 0.95  
            ... })
            >>> model.L  # 15.0

        """

        if 'control_name' in config:
            control_name = config['control_name']
            if not all(0 <= x < self.input_dim for x in control_name):
                raise MarginalCVaRDROError("All indices in 'control_name' must be in the range [0, input_dim - 1].")
            self.control_name = control_name
        
        if 'L' in config:
            L = config['L']
            if L <= 0:
                raise MarginalCVaRDROError("Parameter 'L' must be positive.")
            self.L = float(L)
        
        if 'p' in config:
            p = config['p']
            if p < 1:
                raise MarginalCVaRDROError("Parameter 'p' must be >= 1.")
            self.p = int(p)
        
        if 'alpha' in config:
            alpha = config['alpha']
            if not (0 < alpha <= 1):
                raise MarginalCVaRDROError("Parameter 'alpha' must be in the range (0, 1].")
            self.alpha = float(alpha)

        
    def fit(self, X: np.ndarray, y: np.ndarray, accelerate: bool = True) -> Dict[str, Any]:
        """Solve the Marginal CVaR-DRO problem via convex optimization with approximation.

        Constructs and solves the following distributionally robust optimization problem:

        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray
        
        :param y: Target vector of shape `(n_samples,)`. Format requirements depend on model_type:

            - Classification (`svm`/`logistic`):
                
                - Binary labels in {-1, +1}
                
                - No missing values allowed

            - Regression** (`ols`/`lad`):
                
                - Continuous real values

                - May contain NaN

        :type y: numpy.ndarray

        :param accelerate: Whether to use acceleration for quadratic approximation.

        :type accelerate: bool

        :returns: Solution dictionary containing:
            
            - ``theta``: Weight vector of shape `(n_features,)`

            - ``b``: Intercept term (exists if `fit_intercept=True`)

            - ``B``: Marginal robustness dual matrix of shape `(n_samples, n_samples)`

            - ``threshold``: CVaR threshold value

            
        :rtype: Dict[str, Any]

        :raises MarginalCVaRDROError: 

            - If `X.shape[1] != self.input_dim`

            - If `X.shape[0] != y.shape[0]`

            - If optimization fails (problem infeasible/solver error)

            - If `control_name` indices exceed feature dimensions

        Example:
            >>> model = MarginalCVaRDRO(input_dim=3, control_name=[0,2], L=5.0)
            >>> X = np.random.randn(100, 3)
            >>> y = np.sign(np.random.randn(100)) 
            >>> sol = model.fit(X, y)
            >>> print(sol["theta"].shape)  # (3,)
            >>> print(sol["B"].shape)      # (2, 2)
        """
        
        if self.model_type in {'svm', 'logistic'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise MarginalCVaRDROError("classification labels not in {-1, +1}")

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise MarginalCVaRDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise MarginalCVaRDROError("Input X and target y must have the same number of samples.")

        
        control_X = X[:, self.control_name] if self.control_name else X
        nbrs = NearestNeighbors(n_neighbors=50).fit(control_X)
        dist_sparse = nbrs.kneighbors_graph(mode='distance')
        dist = triu(dist_sparse, format='coo')
        dist.data = (dist.data ** (self.p-1)).astype(np.float32)


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
        

        if accelerate == True:
            B_rowsum = cp.Variable(sample_size)  
            B_colsum = cp.Variable(sample_size) 

            constraints = []

            b_var = cp.Variable(name='b') if self.fit_intercept else 0.0
            eta = cp.Variable(nonneg=True, name='eta')
            s = cp.Variable(sample_size, nonneg=True, name='s')
            
            loss_vector = self._cvx_loss(X, y, theta, b_var)
            constraints += [
                s >= loss_vector - (B_rowsum - B_colsum)/sample_size - eta,
                B_rowsum >= 0,
                B_colsum >= 0
            ]

            trace_term = cp.sum(dist.data * (B_rowsum[dist.row] + B_colsum[dist.col])/2)
            cost = (
                cp.sum(s)/(self.alpha * sample_size) +
                self.L ** (self.p-1) * trace_term/(sample_size ** 2) +
                eta
            )

            problem = cp.Problem(cp.Minimize(cost), constraints)
            try:
                problem.solve(
                    solver = self.solver
                )
            except cp.SolverError as e:
                raise RuntimeError(f"Solver Failed: {str(e)}")

            if theta.value is None or not np.isfinite(eta.value):
                raise RuntimeError("Not Converged!")

            self.theta = theta.value
            self.b = b_var.value if self.fit_intercept else 0.0
            self.threshold = eta.value

            return {
                "theta": self.theta.tolist(),
                "b": self.b,
                "threshold": self.threshold
            }

        else:
            if self.fit_intercept == True:
                b = cp.Variable()
            else:
                b = 0
            eta = cp.Variable()
            B_var = cp.Variable((sample_size, sample_size), nonneg=True)
            s = cp.Variable(sample_size, nonneg=True)

            # Define constraints for optimization problem
            cons = [
                s >= self._cvx_loss(X, y, theta, b) - (cp.sum(B_var, axis=1) - cp.sum(B_var, axis=0)) / sample_size - eta,
                B_var >= 0
            ]

            # Define the cost function
            cost = (
                cp.sum(s) / (self.alpha * sample_size) +
                self.L ** (self.p - 1) * cp.sum(cp.multiply(dist, B_var)) / (sample_size ** 2)
            )

            # Set up and solve the optimization problem
            problem = cp.Problem(cp.Minimize(cost + eta), cons)
            
            try:
                
                problem.solve(solver=self.solver)
                
                self.theta = theta.value
                self.B_val = B_var.value
                self.threshold_val = eta.value
            except cp.error.SolverError as e:
                raise MarginalCVaRDROError(f"Optimization failed to solve using {self.solver}.") from e

            if self.theta is None or self.threshold_val is None or self.B_val is None:
                raise MarginalCVaRDROError("Optimization did not converge to a solution.")

            if self.fit_intercept == True:
                self.b = b.value

            return {
                "theta": self.theta.tolist(),
                "B": self.B_val.tolist(),
                "b": self.b,
                "threshold": self.threshold_val
            }
        
