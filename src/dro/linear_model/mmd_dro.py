from .base import BaseLinearDRO, DataValidationError, ParameterError
import numpy as np
import cvxpy as cp
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import Nystroem
from joblib import Parallel, delayed

class MMDDROError(Exception):
    """Exception class for errors in Marginal CVaR DRO model."""
    pass

class MMD_DRO(BaseLinearDRO):
    """
    MMD-DRO (Maximum Mean Discrepancy - Distributionally Robust Optimization)
    Implementation with flexible sampling methods and model types.

    Reference: <https://arxiv.org/abs/2006.06981>
    """

    _SUPPORTED_KERNELS = {'rbf', 'laplacian', 'polynomial', 'linear'}
    _KERNEL_ALIASES = {'poly': 'polynomial', 'gaussian': 'rbf'}

    def __init__(
        self,
        input_dim: int,
        model_type: str = 'svm',
        fit_intercept: bool = True,
        solver: str = 'MOSEK',
        sampling_method: str = 'bound',
        kernel: str = 'rbf',
        kernel_gamma: Optional[float] = None,
        kernel_degree: int = 2,
        kernel_coef0: float = 1.0,
    ):
        """Initialize MMD-DRO with kernel-based ambiguity set.

        :param input_dim: Dimension of input features. Must match training data.
        :type input_dim: int
        :param model_type: Base model type. Supported:
            
            - ``'svm'``: Support Vector Machine (hinge loss)

            - ``'logistic'``: Logistic Regression (log loss)

            - ``'ols'``: Ordinary Least Squares (L2 loss)

            - ``'lad'``: Least Absolute Deviation (L1 loss)

        :type model_type: str

        :param sampling_method: Supported:
            
            - ``'bound'``

            - ``'hull'``

        :type sampling_method: str

        :param kernel: Kernel defining the MMD ambiguity set. Supported values are
            ``'rbf'`` (default), ``'laplacian'``, ``'polynomial'`` (or
            ``'poly'``), and ``'linear'``.
        :type kernel: str
        :param kernel_gamma: Positive kernel scale for RBF, Laplacian, and
            polynomial kernels. If ``None``, RBF and Laplacian use a robust
            median-distance heuristic, while polynomial uses
            ``1 / (input_dim + 1)``.
        :type kernel_gamma: float or None
        :param kernel_degree: Positive integer degree of the polynomial kernel.
            Defaults to 2.
        :type kernel_degree: int
        :param kernel_coef0: Non-negative independent term in the polynomial
            kernel. Defaults to 1.0.
        :type kernel_coef0: float

        :raises ValueError: 

            - If `model_type` not in supported list

            - If `input_dim` ≤ 0

            - If `sampling_method` is invalid

            - If `kernel` or its hyperparameters are invalid
        
        Example:
            >>> model = MMD_DRO(input_dim=128, model_type='svm', kernel='rbf')
            >>> model.sampling_method = 'hull' 
            >>> model.eta = 0.5  

        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)

        if not sampling_method in {'bound', 'hull'}:
            raise MMDDROError(f"Invalid sampling method: {sampling_method}")

        # This kernel defines the MMD ambiguity set; MMD_DRO still learns a
        # linear prediction model (see predict and _loss below).
        self.kernel = self._validate_kernel(kernel)
        self._validate_kernel_hyperparameters(
            kernel_gamma, kernel_degree, kernel_coef0
        )
        self.kernel_gamma = kernel_gamma
        self.kernel_degree = kernel_degree
        self.kernel_coef0 = float(kernel_coef0)

        self.eta = 0.1
        self.sampling_method = sampling_method  
        self.n_certify_ratio = 1  
        self.n_components = None      

    @classmethod
    def _validate_kernel(cls, kernel: str) -> str:
        """Normalize and validate an MMD ambiguity-set kernel name."""
        if not isinstance(kernel, str):
            raise TypeError("Parameter 'kernel' must be a string.")
        kernel = cls._KERNEL_ALIASES.get(kernel.lower(), kernel.lower())
        if kernel not in cls._SUPPORTED_KERNELS:
            supported = ", ".join(sorted(cls._SUPPORTED_KERNELS))
            raise MMDDROError(
                f"Invalid kernel: {kernel}. Supported kernels are: {supported}."
            )
        return kernel

    @staticmethod
    def _validate_kernel_hyperparameters(
        gamma: Optional[float], degree: int, coef0: float
    ) -> None:
        """Validate parameters while preserving positive semidefiniteness."""
        if gamma is not None and (
            isinstance(gamma, bool)
            or not isinstance(gamma, (float, int))
            or gamma <= 0
        ):
            raise ValueError("Parameter 'kernel_gamma' must be None or positive.")
        if isinstance(degree, bool) or not isinstance(degree, int) or degree <= 0:
            raise ValueError("Parameter 'kernel_degree' must be a positive integer.")
        if (
            isinstance(coef0, bool)
            or not isinstance(coef0, (float, int))
            or coef0 < 0
        ):
            raise ValueError("Parameter 'kernel_coef0' must be non-negative.")

    def update_kernel(self, config: Dict[str, Any]) -> None:
        """Update the MMD kernel using the package's kernel-update convention.

        ``metric`` is accepted as an alias for ``kernel``; ``degree`` and
        ``coef0`` are accepted as aliases for their ``kernel_`` counterparts.
        """
        allowed_keys = {
            'metric',
            'kernel',
            'kernel_gamma',
            'degree',
            'kernel_degree',
            'coef0',
            'kernel_coef0',
            'n_components',
        }
        unknown_keys = set(config) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unrecognized kernel parameter(s): {unknown}.")

        kernel_config: Dict[str, Any] = {}
        if 'metric' in config and 'kernel' in config:
            if self._validate_kernel(config['metric']) != self._validate_kernel(
                config['kernel']
            ):
                raise ValueError("'metric' and 'kernel' must select the same kernel.")
        if 'metric' in config or 'kernel' in config:
            kernel_config['kernel'] = config.get('kernel', config.get('metric'))
        if 'kernel_gamma' in config:
            kernel_config['kernel_gamma'] = config['kernel_gamma']
        if 'degree' in config and 'kernel_degree' in config:
            if config['degree'] != config['kernel_degree']:
                raise ValueError(
                    "'degree' and 'kernel_degree' must have the same value."
                )
        if 'degree' in config or 'kernel_degree' in config:
            kernel_config['kernel_degree'] = config.get(
                'kernel_degree', config.get('degree')
            )
        if 'coef0' in config and 'kernel_coef0' in config:
            if config['coef0'] != config['kernel_coef0']:
                raise ValueError(
                    "'coef0' and 'kernel_coef0' must have the same value."
                )
        if 'coef0' in config or 'kernel_coef0' in config:
            kernel_config['kernel_coef0'] = config.get(
                'kernel_coef0', config.get('coef0')
            )

        n_components = self.n_components
        if 'n_components' in config:
            n_components = config['n_components']
            if (
                isinstance(n_components, bool)
                or not isinstance(n_components, int)
                or n_components <= 0
            ):
                raise ValueError("Parameter 'n_components' must be a positive integer.")

        self.update(kernel_config)
        self.n_components = n_components

    def update(self, config: Dict[str, Any]) -> None:
        """Update MMD-DRO model configuration.

        :param config: Configuration dictionary containing optional keys:

            - ``eta`` (float): 
                MMD radius controlling distributional robustness. 
                Must satisfy :math:`\eta > 0`.
                Defaults to current value.

            - ``sampling_method`` (str): 
                Ambiguity set sampling strategy. Valid options:
                
                - ``'bound'``: Sample on MMD ball boundary

                - ``'hull'``: Sample within convex hull
                
                
            - ``n_certify_ratio`` (float): 
                Ratio of certification samples to training data size. 
                Must satisfy :math:`0 < \text{ratio} \leq 1`.
                Defaults to current ratio.

            - ``kernel`` (str):
                MMD kernel. One of ``'rbf'``, ``'laplacian'``,
                ``'polynomial'``/``'poly'``, or ``'linear'``.

            - ``kernel_gamma`` (float or None):
                Positive kernel scale. ``None`` selects the kernel-specific
                data-driven default.

            - ``kernel_degree`` (int):
                Positive polynomial degree. Defaults to 2.

            - ``kernel_coef0`` (float):
                Non-negative polynomial offset. Defaults to 1.0.

        :type config: Dict[str, Any]

        :raises ValueError: 

            - If ``eta`` is non-positive

            - If ``sampling_method`` not in {'bound', 'hull'}

            - If ``n_certify_ratio`` ∉ (0, 1]

            - If config contains unrecognized keys

        
        Example:
            >>> model = MMD_DRO(input_dim=10, model_type='svm')
            >>> model.update({
            ...     'eta': 0.5,              
            ...     'sampling_method': 'hull',
            ...     'kernel': 'polynomial',
            ...     'kernel_degree': 2
            ... })

        """
        eta = config.get('eta', self.eta)
        sampling_method = config.get('sampling_method', self.sampling_method)
        n_certify_ratio = config.get('n_certify_ratio', self.n_certify_ratio)
        kernel = self._validate_kernel(config.get('kernel', self.kernel))
        kernel_gamma = config.get('kernel_gamma', self.kernel_gamma)
        kernel_degree = config.get('kernel_degree', self.kernel_degree)
        kernel_coef0 = config.get('kernel_coef0', self.kernel_coef0)

        if sampling_method not in ['bound', 'hull']:
            raise MMDDROError("sampling_method must be either 'bound' or 'hull'")

        # Validate parameter types
        if not isinstance(eta, (float, int)) or eta <= 0:
            raise ValueError("Parameter 'eta' must be a positive float or int.")
        if not isinstance(n_certify_ratio, (float, int)) or n_certify_ratio <= 0 or n_certify_ratio > 1:
            raise ValueError("Parameter 'n_certify_ratio' must be a positive float or int between (0,1]).")
        self._validate_kernel_hyperparameters(
            kernel_gamma, kernel_degree, kernel_coef0
        )

        self.eta = eta
        self.sampling_method = sampling_method
        self.n_certify_ratio = n_certify_ratio
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        self.kernel_degree = kernel_degree
        self.kernel_coef0 = float(kernel_coef0)

    @staticmethod
    def _matrix_decomp(K: np.ndarray) -> np.ndarray:
        """Perform matrix decomposition for kernel matrix K."""
        try:
            return np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            eigenvalues = np.clip(eigenvalues, 0, None)  # Remove small negative eigenvalues
            return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    @staticmethod
    def _positive_median(distances: np.ndarray) -> float:
        """Return the median nonzero distance with a safe constant fallback."""
        positive_distances = distances[distances > np.finfo(float).eps]
        if positive_distances.size == 0:
            return 1.0
        return float(np.median(positive_distances))

    def _median_heuristic(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Calculate kernel width and gamma using the median heuristic."""
        distsqr = euclidean_distances(X, Y, squared=True)
        median_distsqr = self._positive_median(distsqr)
        kernel_width = np.sqrt(0.5 * median_distsqr)
        kernel_gamma = 1.0 / (2 * kernel_width ** 2)
        return kernel_width, kernel_gamma

    def _medium_heuristic(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Backward-compatible alias for the formerly misspelled method name."""
        return self._median_heuristic(X, Y)

    def _kernel_parameters(self, zeta: np.ndarray) -> Dict[str, Any]:
        """Resolve data-dependent parameters for the configured MMD kernel."""
        if self.kernel == 'linear':
            return {}

        if self.kernel_gamma is not None:
            gamma = float(self.kernel_gamma)
        elif self.kernel == 'rbf':
            _, gamma = self._median_heuristic(zeta, zeta)
        elif self.kernel == 'laplacian':
            distances = pairwise_distances(zeta, metric='manhattan')
            gamma = 1.0 / self._positive_median(distances)
        else:  # Polynomial kernel: (gamma * <x, y> + coef0) ** degree
            gamma = 1.0 / zeta.shape[1]

        parameters: Dict[str, Any] = {'gamma': gamma}
        if self.kernel == 'polynomial':
            parameters.update(
                degree=self.kernel_degree,
                coef0=self.kernel_coef0,
            )
        return parameters

    def _kernel_matrix(self, zeta: np.ndarray) -> np.ndarray:
        """Compute the exact Gram matrix for the configured MMD kernel."""
        return pairwise_kernels(
            zeta,
            metric=self.kernel,
            filter_params=True,
            **self._kernel_parameters(zeta),
        )

    def _linear_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute scores for the linear predictor learned by MMD-DRO."""
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            actual_dim = X.shape[1] if X.ndim == 2 else 'unknown'
            raise DataValidationError(
                f"Expected input with {self.input_dim} features, got {actual_dim}."
            )
        return X @ self.theta + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the linear model; ``kernel`` only defines the MMD set."""
        scores = self._linear_scores(X)
        if self.model_type in {'ols', 'lad'}:
            return scores
        threshold = 0 if self.model_type == 'svm' else 0.5
        return np.where(scores >= threshold, 1, -1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate per-sample loss for the linear prediction model."""
        scores = self._linear_scores(X)
        if self.model_type == 'svm':
            return np.maximum(1 - y * scores, 0)
        if self.model_type == 'logistic':
            return np.logaddexp(0, -y * scores)
        if self.model_type == 'ols':
            return (y - scores) ** 2
        if self.model_type == 'lad':
            return np.abs(y - scores)
        raise NotImplementedError(
            "Loss function not implemented for the specified model_type value."
        )

    def _cvx_loss(
        self,
        X: cp.Expression,
        y: cp.Expression,
        theta: cp.Expression,
        b: cp.Expression,
    ) -> cp.Expression:
        """Construct the convex loss for MMD-DRO's linear predictor."""
        assert X.shape[-1] == self.input_dim, (
            "Mismatch between feature and input dimension."
        )
        inner_product = X @ theta
        if self.model_type == 'svm':
            return cp.pos(1 - cp.multiply(y, inner_product + b))
        if self.model_type == 'logistic':
            return cp.logistic(-cp.multiply(y, inner_product + b))
        if self.model_type == 'ols':
            return cp.square(y - inner_product - b)
        if self.model_type == 'lad':
            return cp.abs(y - inner_product - b)
        raise NotImplementedError(
            "CVXPY loss not implemented for the specified model_type value."
        )

    def load(self, config: Dict[str, Any]) -> None:
        """Load linear MMD-DRO parameters independently of the MMD kernel."""
        try:
            theta = np.asarray(config['theta'])
        except KeyError as exc:
            raise ParameterError("Config must contain 'theta' key.") from exc
        if theta.shape != (self.input_dim,):
            raise DataValidationError(
                f"Theta must have shape ({self.input_dim},) in MMD-DRO."
            )
        self.theta = theta
        if 'b' in config:
            self.b = float(config['b'])

    def evaluate(self, X: np.ndarray, y: np.ndarray, fast: bool = True) -> float:
        """Evaluate the linear predictor independently of the MMD kernel."""
        del fast  # Kept for API compatibility with BaseLinearDRO.evaluate.
        sample_num = X.shape[0]
        errors = self._loss(X, y)

        if self.model_type == 'ols':
            design = X
            if self.fit_intercept:
                design = np.hstack([X, np.ones((sample_num, 1))])
            covariance = np.atleast_2d(np.cov(design.T))
            covariance_inverse = np.linalg.pinv(covariance)
            weighted_design = errors.reshape(-1, 1) * design
            gradient_square = weighted_design.T @ weighted_design
            bias = 2 * np.trace(
                covariance_inverse @ gradient_square
            ) / sample_num ** 3
            return float(np.mean(errors) + bias)

        if self.model_type == 'logistic':
            design = np.hstack([X, np.ones((sample_num, 1))])
            theta_full = np.append(self.theta, self.b)
            sigmoid = 1 / (1 + np.exp(y * (design @ theta_full)))
            information = design.T @ (
                (sigmoid * (1 - sigmoid)).reshape(-1, 1) * design
            )
            score_covariance = design.T @ (
                np.square(sigmoid).reshape(-1, 1) * design
            )
            bias = np.trace(
                np.linalg.pinv(information) @ score_covariance
            ) / sample_num
            return float(np.mean(errors) + bias)

        return float(np.mean(errors))
    


    def fit(self, X: np.ndarray, y: np.ndarray, accelerate: bool = True) -> None:
        """Vectorized implementation of MMD-DRO fit function."""
        """Fit the MMD-DRO model to the data.
        
        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type y: numpy.ndarray

        :param accelerate: Whether to use acceleration for kernel approximation.

        :type accelerate: bool

        :returns: Dictionary containing trained parameters:
        
            - ``theta``: Weight vector of shape `(n_features,)`
        
        :rtype: Dict[str, Any]

        """
        
        if self.model_type in {'svm', 'logistic'}:
            if not np.all(np.isin(y, [-1, 1])):
                raise MMDDROError("classification labels must be in {-1, +1}")

        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("X must be 2D array and y must be 1D array")

        sample_size, input_dim = X.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")

        n_certify = int(self.n_certify_ratio * sample_size)

        if accelerate == True:

            if self.fit_intercept == True:
                b = cp.Variable()
            else:
                b = 0
            
            theta = cp.Variable(self.input_dim)
            
            # Generate all certify samples at once (no loops)
            if self.sampling_method == 'bound':
                # Uniform sampling in [-1, 1]^d
                zeta_certify = np.random.uniform(-1, 1, (n_certify, self.input_dim + 1))
            elif self.sampling_method == 'hull':
                # Feature-wise bounds
                feat_mins = X.min(axis=0)
                feat_maxs = X.max(axis=0)
                # Generate features in [min, max]^d via matrix op
                zeta_feat = np.random.uniform(feat_mins, feat_maxs, (n_certify, self.input_dim))
                
                # Label generation
                if self.model_type in ["svm", "logistic"]:
                    zeta_label = np.random.choice([-1, 1], n_certify)
                else:
                    label_min, label_max = y.min(), y.max()
                    zeta_label = np.random.uniform(label_min, label_max, n_certify)
                
                zeta_certify = np.hstack([zeta_feat, zeta_label.reshape(-1, 1)])

            # Merge with original data (no loops)
            zeta = np.vstack([
                np.hstack([X, y.reshape(-1, 1)]),  # Original data
                zeta_certify                       # Certify samples
            ])

            n_components = min(self.n_components or 100, len(zeta))
            nystroem = Nystroem(
                kernel=self.kernel,
                n_components=n_components,
                random_state=0,
                **self._kernel_parameters(zeta),
            )
            nystroem.fit(zeta)
            self.mmd_nystroem_transformer = nystroem

            batches = [zeta[i:i+5000] for i in range(0, len(zeta), 5000)]
            K_approx_list = Parallel(n_jobs=4)(
                delayed(nystroem.transform)(batch) for batch in batches
            )
            K_approx = np.vstack(K_approx_list)



            a = cp.Variable(K_approx.shape[1])
            f0 = cp.Variable()


            n_total = zeta.shape[0]
            n_selected = min(5000, n_total)
            selected_indices = np.random.choice(n_total, n_selected, replace=False)

            # Batch extraction of selected samples (no loops)
            X_selected = zeta[selected_indices, :-1]  # Shape: (n_selected, input_dim)
            y_selected = zeta[selected_indices, -1]   # Shape: (n_selected,)

            # Vectorized loss computation
            if self.model_type == 'svm':
                # SVM: 1 - y*(X@theta + b) <= s --> s >= 1 - y*(X@theta + b)
                losses = 1 - cp.multiply(y_selected, (X_selected @ theta + b))
            elif self.model_type == 'logistic':
                # Logistic: log(1 + exp(-y*(X@theta + b))) <= s
                linear_term = cp.multiply(y_selected, (X_selected @ theta + b))
                losses = cp.logistic(-linear_term)
            elif self.model_type == 'ols':
                # OLS: (y - X@theta - b)^2 <= s
                residuals = y_selected - (X_selected @ theta + b)
                losses = cp.square(residuals)
            elif self.model_type == 'lad':
                # LAD: |y - X@theta - b| <= s
                residuals = y_selected - (X_selected @ theta + b)
                losses = cp.abs(residuals)

            # Vectorized RHS: f0 + K_approx_selected @ a
            rhs = f0 + K_approx[selected_indices] @ a

            # All constraints in one line (no loops)
            constraints = [losses <= rhs]

            loss_term = cp.sum(K_approx[:sample_size] @ a) / sample_size
            reg_term = self.eta * cp.norm(K_approx @ a)

            objective = cp.Minimize(f0 + loss_term + reg_term)

            problem = cp.Problem(objective, constraints)
            problem.solve(
                solver=self.solver, 
                verbose=True, 
                mosek_params={'MSK_IPAR_NUM_THREADS': 8} if self.solver == 'MOSEK' else {}
            )

            self.theta = theta.value
            if self.fit_intercept:
                self.b = b.value
            else:
                self.b = 0.0
            self.robust_obj = float(problem.value)

            return {"theta": self.theta.tolist(), "b": float(self.b)}
        else:
            # Define decision variable
            theta = cp.Variable(self.input_dim)

            # DRO variables
            a = cp.Variable(sample_size + n_certify)
            f0 = cp.Variable()

            # --------------------------------------------------------------------------------
            # Step 1: Generate the sampled support
            # --------------------------------------------------------------------------------
            if self.sampling_method == 'bound':
                zeta = np.random.uniform(-1, 1, size=(n_certify, self.input_dim + 1))
            elif self.sampling_method == 'hull':
                if self.model_type in ["ols", "lad"]:
                    zeta1 = np.random.uniform(np.min(X), np.max(X), size=(n_certify, self.input_dim))
                    zeta2 = np.random.uniform(np.min(y), np.max(y), size=(n_certify, 1))
                elif self.model_type in ["svm", "logistic"]:
                    zeta1 = np.random.uniform(-1, 1, size=(n_certify, self.input_dim))
                    zeta2 = np.random.choice([-1, 1], size=(n_certify, 1))
                else:
                    raise NotImplementedError(f"Model type {self.model_type} is not supported.")
                zeta = np.concatenate([zeta1, zeta2], axis=1)

            # Include empirical data in sampled support
            data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
            zeta = np.concatenate([data, zeta])

            # Validate zeta dimensions
            assert zeta.shape[1] == self.input_dim + 1, "Generated zeta does not match expected dimensions."

            if self.fit_intercept == True:
                b = cp.Variable()
            else:
                b = 0

            # --------------------------------------------------------------------------------
            # Step 2: Kernel matrix computation
            # --------------------------------------------------------------------------------
            K = self._kernel_matrix(zeta)

            # --------------------------------------------------------------------------------
            # Step 3: Define objective and constraints
            # --------------------------------------------------------------------------------
            f = a @ K
            constraints = [self._cvx_loss(zeta[i][:-1], zeta[i][-1], theta, b) <= f0 + f[i] for i in range(len(zeta))]
            objective = f0 + cp.sum(f[:sample_size]) / sample_size + self.eta * cp.norm(a.T @ self._matrix_decomp(K))

            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=self.solver)

            # Check optimization status
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise ValueError("Optimization problem did not converge.")

            # Store results
            self.theta = theta.value

            # Validate optimization results
            if self.theta is None or not np.all(np.isfinite(self.theta)):
                raise ValueError("Optimization resulted in invalid theta values.")

            if self.fit_intercept == True:
                self.b = b.value
            self.robust_obj = float(problem.value)

            # Return model parameters in dictionary format
            return {"theta": self.theta.tolist(), "b": self.b}
