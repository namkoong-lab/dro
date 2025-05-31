from .base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any, Tuple
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances
from sklearn.kernel_approximation import Nystroem, RBFSampler
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

    def __init__(self, input_dim: int, model_type: str = 'svm',  fit_intercept: bool = True, solver: str = 'MOSEK', sampling_method: str = 'bound'):
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

        :raises ValueError: 

            - If `model_type` not in supported list

            - If `input_dim` ≤ 0

            - If `sampling_method` is invalid
        
        Example:
            >>> model = MMD_DRO(input_dim=128, model_type='svm')
            >>> model.sampling_method = 'hull' 
            >>> model.eta = 0.5  

        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)

        if not sampling_method in {'bound', 'hull'}:
            raise MMDDROError(f"Invalid sampling method: {sampling_method}")
        
        self.eta = 0.1
        self.sampling_method = sampling_method  
        self.n_certify_ratio = 1  
        self.n_components = None      

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
            ...     'sampling_method': 'hull' 
            ... })

        """
        self.eta = config.get('eta', self.eta)
        self.sampling_method = config.get('sampling_method', self.sampling_method)
        if self.sampling_method not in ['bound', 'hull']:
            raise MMDDROError("sampling_method must be either 'bound' or 'hull'")
        self.n_certify_ratio = config.get('n_certify_ratio', self.n_certify_ratio)

        # Validate parameter types
        if not isinstance(self.eta, (float, int)) or self.eta <= 0:
            raise ValueError("Parameter 'eta' must be a positive float or int.")
        if not isinstance(self.n_certify_ratio, (float, int)) or self.n_certify_ratio <= 0 or self.n_certify_ratio > 1:
            raise ValueError("Parameter 'n_certify_ratio' must be a positive float or int between (0,1]).")

    @staticmethod
    def _matrix_decomp(K: np.ndarray) -> np.ndarray:
        """Perform matrix decomposition for kernel matrix K."""
        try:
            return np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            eigenvalues = np.clip(eigenvalues, 0, None)  # Remove small negative eigenvalues
            return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def _medium_heuristic(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Calculate kernel width and gamma using the median heuristic."""
        if self.model_type in ["ols", "lad"]:
            distsqr = euclidean_distances(X, Y, squared=True)
        elif self.model_type in ["svm", "logistic"]:
            distsqr = euclidean_distances(X[:, :-1], Y[:, :-1], squared=True)
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not supported.")

        kernel_width = np.sqrt(0.5 * np.median(distsqr))
        kernel_gamma = 1.0 / (2 * kernel_width ** 2)
        return kernel_width, kernel_gamma
    


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

        n_components = 100  # Low-rank kernel approximation
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

            if self.model_type in ["ols", "lad"]:
                distsqr = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
            else:
                distsqr = np.sum((X[:, None, :-1] - X[None, :, :-1]) ** 2, axis=-1)
            kernel_width = np.sqrt(0.5 * np.median(distsqr))
            kernel_gamma = 1.0 / (2 * kernel_width ** 2)        
            
            def transform_batch(batch):
                nystroem = Nystroem(
                    kernel='rbf', gamma=kernel_gamma, 
                    n_components=n_components, random_state=0
                )
                return nystroem.fit_transform(batch)

            batches = [zeta[i:i+5000] for i in range(0, len(zeta), 5000)]
            K_approx_list = Parallel(n_jobs=4)(delayed(transform_batch)(b) for b in batches)
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
            kernel_width, kernel_gamma = self._medium_heuristic(zeta, zeta)
            K = rbf_kernel(zeta, zeta, gamma=kernel_gamma)

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

            # Return model parameters in dictionary format
            return {"theta": self.theta.tolist(), "b": self.b}