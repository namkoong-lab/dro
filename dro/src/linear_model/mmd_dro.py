from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import cvxpy as cp
from typing import Dict, Any, Tuple
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances



class MMD_DRO(BaseLinearDRO):
    """
    MMD-DRO (Maximum Mean Discrepancy - Distributionally Robust Optimization)
    Implementation with flexible sampling methods and model types.

    Reference: <https://arxiv.org/abs/2006.06981>
    """

    def __init__(self, input_dim: int, model_type: str):
        BaseLinearDRO.__init__(input_dim, model_type)
        self.eta = 0.1
        self.sampling_method = 'bound'  # Sampling method: 'bound' or 'hull'
        self.n_certify_ratio = 1        # Ratio of additional certification samples


    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration parameters."""
        self.eta = config.get('eta', self.eta)
        self.sampling_method = config.get('sampling_method', self.sampling_method)
        if self.sampling_method not in ['bound', 'hull']:
            raise ValueError("sampling_method must be either 'bound' or 'hull'")
        self.n_certify_ratio = config.get('n_certify_ratio', self.n_certify_ratio)

        # Validate parameter types
        if not isinstance(self.eta, (float, int)) or self.eta <= 0:
            raise ValueError("Parameter 'eta' must be a positive float or int.")
        if not isinstance(self.n_certify_ratio, (float, int)) or self.n_certify_ratio <= 0 or self.n_certify_ratio > 1:
            raise ValueError("Parameter 'n_certify_ratio' must be a positive float or int between (0,1]).")

    @staticmethod
    def matrix_decomp(K: np.ndarray) -> np.ndarray:
        """Perform matrix decomposition for kernel matrix K."""
        try:
            return np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            eigenvalues = np.clip(eigenvalues, 0, None)  # Remove small negative eigenvalues
            return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def medium_heuristic(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """Calculate kernel width and gamma using the median heuristic."""
        if self.model_type in ["linear", "logistic"]:
            distsqr = euclidean_distances(X, Y, squared=True)
        elif self.model_type == "svm":
            distsqr = euclidean_distances(X[:, :-1], Y[:, :-1], squared=True)
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not supported.")

        kernel_width = np.sqrt(0.5 * np.median(distsqr))
        kernel_gamma = 1.0 / (2 * kernel_width ** 2)
        return kernel_width, kernel_gamma

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the MMD-DRO model to the data."""
        # Ensure input data is valid
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")

        sample_size, _ = X.shape
        n_certify = int(self.n_certify_ratio * sample_size)

        # Define decision variable
        theta = cp.Variable(self.input_dim)

        # KDRO variables
        a = cp.Variable(sample_size + n_certify)
        f0 = cp.Variable()

        # --------------------------------------------------------------------------------
        # Step 1: Generate the sampled support
        # --------------------------------------------------------------------------------
        if self.sampling_method == 'bound':
            zeta = np.random.uniform(-1, 1, size=(n_certify, self.input_dim + 1))
        elif self.sampling_method == 'hull':
            if self.model_type in ["linear", "logistic"]:
                zeta1 = np.random.uniform(np.min(X), np.max(X), size=(n_certify, self.input_dim))
                zeta2 = np.random.uniform(np.min(y), np.max(y), size=(n_certify, 1))
            elif self.model_type == "svm":
                zeta1 = np.random.uniform(-1, 1, size=(n_certify, self.input_dim))
                zeta2 = np.random.choice([-1, 1], size=(n_certify, 1))
            else:
                raise NotImplementedError(f"Model type {self.model_type} is not supported.")
            zeta = np.concatenate([zeta1, zeta2], axis=1)
        else:
            raise ValueError("Invalid sampling method.")

        # Include empirical data in sampled support
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        zeta = np.concatenate([data, zeta])

        # Validate zeta dimensions
        assert zeta.shape[1] == self.input_dim + 1, "Generated zeta does not match expected dimensions."

        # --------------------------------------------------------------------------------
        # Step 2: Kernel matrix computation
        # --------------------------------------------------------------------------------
        kernel_width, kernel_gamma = self.medium_heuristic(zeta, zeta)
        K = rbf_kernel(zeta, zeta, gamma=kernel_gamma)

        # --------------------------------------------------------------------------------
        # Step 3: Define objective and constraints
        # --------------------------------------------------------------------------------
        f = a @ K
        constraints = [self._cvx_loss(theta, zeta[i]) <= f0 + f[i] for i in range(len(zeta))]
        objective = f0 + cp.sum(f[:sample_size]) / sample_size + self.eta * cp.norm(a.T @ self.matrix_decomp(K))

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

        # Return model parameters in dictionary format
        return {"theta": self.theta.tolist()}