from .base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from typing import Dict, Any

class ORWDROError(Exception):
    """Base exception class for errors in OR-WDRO model."""
    pass

class ORWDRO(BaseLinearDRO):
    """Outlier-Robust Wasserstein Distributionally Robust Optimization (OR-WDRO) model.

    Implements TV-corrupted p-Wasserstein DRO with dual norm constraints:

    .. math::
        \\min_{\\theta} \\sup_{Q \\in \\mathcal{B}_\\epsilon(P)} \\mathbb{E}_Q[\\ell(\\theta;X,y)]
        + \\eta \\cdot \\text{TV}(P,Q)

    where :math:`\\mathcal{B}_\\epsilon(P)` is the Wasserstein ball and :math:`\\text{TV}` is total variation.

    ORWDRO_Paper: https://arxiv.org/pdf/2311.05573
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', solver: str = 'MOSEK', eps: float = 0.0, 
                 eta: float = 0.0, dual_norm: int = 1):
        """Initialize OR-WDRO model with anomaly-robust parameters.

        :param input_dim: Feature space dimension. Must be ≥ 1
        :type input_dim: int

        :param model_type: Base learner type. Supported:

            - ``'svm'``: Hinge loss (classification)

            - ``'lad'``: Least absolute deviation (regression)

        :type model_type: str

        :param eps: Wasserstein robustness radius. Defaults to 0.0

            - 0: Standard empirical risk minimization

            - >0: Controls distributional robustness

            
        :type eps: float
        :param eta: Expected outlier fraction. 
            Must satisfy :math:`0 \\leq \\eta \\leq 0.5`. Defaults to 0.0
        :type eta: float

        :param dual_norm: Wasserstein dual norm order. Valid values: 
        
            - 1: ℓ¹-Wasserstein (transportation cost)

            - 2: ℓ²-Wasserstein (default)

        :type dual_norm: int

        :raises ValueError:

            - If input_dim < 1

            - If model_type not in allowed set

            - If eps < 0 or eta < 0 or eta > 0.5

            - If dual_norm ∉ {1, 2}

        Example:
            >>> model = ORWDRO(
            ...     input_dim=5,
            ...     model_type='svm',
            ...     eps=0.1,
            ...     eta=0.05,
            ...     dual_norm=2
            ... )
            >>> model.sigma  # sqrt(5) ≈ 2.236

        .. note::
            - Computation complexity scales as :math:`O(\epsilon^{-2})`
            - Set ``eta=0`` to disable outlier robustness
        """
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be ≥ 1, got {input_dim}")
        if model_type not in {'svm', 'lad'}:
            raise ValueError(f"Invalid model_type: {model_type}")
        if eps < 0 or eta < 0 or eta > 0.5:
            raise ValueError(f"Invalid robustness params: eps={eps}, eta={eta}")
        if dual_norm not in {1, 2}:
            raise ValueError(f"dual_norm must be 1 or 2, got {dual_norm}")

        BaseLinearDRO.__init__(self, input_dim, model_type)
        self.eps = eps
        self.eta = eta
        self.dual_norm = dual_norm
        self.sigma = math.sqrt(input_dim)

    def update(self, config: Dict[str, Any]) -> None:
        """Update robustness parameters for OR-WDRO optimization.
        
        :param config: Dictionary containing parameters to update. Valid keys:

            - ``'eps'``: New Wasserstein radius (ε ≥ 0)

            - ``'eta'``: New outlier fraction (0 ≤ η ≤ 0.5)

            - ``'dual_norm'``: Norm order (1 or 2)

        :type config: dict[str, Any]

        :raises ValueError: 

            - If parameter values violate type or range constraints

            - If unknown parameters are provided
        
        Example:
            >>> model.update({
            ...     'eps': 0.2,
            ...     'eta': 0.1,
            ...     'dual_norm': 2
            ... })  # Updates multiple parameters atomically
            
        """
        if 'eps' in config:
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise ValueError("eps must be non-negative float")
            self.eps = float(eps)
            
        if 'eta' in config:
            eta = config['eta']
            if not isinstance(eta, (float, int)) or eta < 0 or eta >= 1:
                raise ValueError("eta must be in [0, 1]")
            self.eta = float(eta)
            
        if 'dual_norm' in config:
            dn = config['dual_norm']
            if dn not in {1, 2}:
                raise ValueError("dual_norm must be 1 or 2")
            self.dual_norm = int(dn)
        
    # def fit_old(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    #     """
    #     :param X: Training feature matrix of shape `(n_samples, n_features)`.
    #         Must satisfy `n_features == self.input_dim`.
    #     :type X: numpy.ndarray

    #     :param Y: Target values of shape `(n_samples,)`. Format requirements:

    #         - Classification: ±1 labels

    #         - Regression: Continuous values

    #     :type Y: numpy.ndarray

    #     :returns: Dictionary containing trained parameters:
        
    #         - ``theta``: Weight vector of shape `(n_features,)`
            
    #     :rtype: Dict[str, Any]
    #     """
    #     if self.model_type in {'svm'}:
    #         is_valid = np.all((y == -1) | (y == 1))
    #         if not is_valid:
    #             raise ORWDROError("classification labels not in {-1, +1}")
        
        
    #     sample_size, feature_size = X.shape
    #     if feature_size != self.input_dim:
    #         raise ORWDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
    #     if sample_size != y.shape[0]:
    #         raise ORWDROError("Input X and target y must have the same number of samples.")
    
    #     z_0, sgn = self._cheap_robust_mean_estimate(X, y)
    #     if self.model_type == 'lad':
    #         Z = np.hstack([X, y.reshape(-1, 1)])
    #     elif self.model_type == 'svm':
    #         Z = X
    #     J = 2
        
    #     # ----------------------------------------------------------------------
    #     # 2) Define CVXPY Variables
    #     # ----------------------------------------------------------------------
    #     lambda_1 = cp.Variable(nonneg=True)     # scalar >= 0
    #     lambda_2 = cp.Variable(nonneg=True)     # scalar >= 0
    #     alpha    = cp.Variable()                # scalar (unconstrained in sign, can be +/-, user sets constraints if needed)
    #     s        = cp.Variable(shape=(sample_size,), nonneg=True)    # s(i) >= 0
    #     zeta_G = [cp.Variable(shape=(feature_size + 1 - sgn, sample_size)) for _ in range(2)]
    #     zeta_W = [cp.Variable(shape=(feature_size + 1 - sgn, sample_size)) for _ in range(2)]

    #     aux_W = cp.Variable(shape = (2,sample_size), nonneg = True)
    #     tau = cp.Variable(shape=(sample_size, J), nonneg=True)
    #     theta = cp.Variable(shape=(feature_size,))

    #     # ----------------------------------------------------------------------
    #     # 3) Objective function
    #     #    objective = lambda_1 * sigma^q + lambda_2 * rho^p
    #     #                + (1/(n*(1 - vareps))) * sum(s) + alpha
    #     # ----------------------------------------------------------------------
    #     objective_expr = (
    #         lambda_1 * (self.sigma ** 2)
    #         + lambda_2 * (self.eps ** 1)
    #         + (1.0 / (sample_size * (1.0 - self.eta))) * cp.sum(s)
    #         + alpha
    #     )
    #     objective = cp.Minimize(objective_expr)


    #     # ----------------------------------------------------------------------
    #     # 4) Build constraints
    #     # ----------------------------------------------------------------------
    #     constraints = []

    #     # Already declared lambda_1 >= 0, lambda_2 >= 0, s >= 0, tau >= 0 in the variable definitions.

    #     # Loop over each sample i
    #     for i in range(sample_size):
    #         lhs_0 = (sgn + z_0 @ zeta_G[0][:, i]
    #                 + tau[i, 0]
    #                 + Z[i, :] @ zeta_W[0][:, i]
    #                 - alpha )
    #         constraints.append( s[i] >= lhs_0 )

    #         lhs_1 = ( z_0 @ zeta_G[1][:, i]
    #                 + tau[i, 1]
    #                 + Z[i, :] @ zeta_W[1][:, i]
    #                 - alpha )
    #         constraints.append( s[i] >= lhs_1 )

    #         if self.model_type == 'lad':
    #             minus_theta_plus1 = cp.hstack([-theta, 1.0])  # shape (d,)
    #             constraints.append( minus_theta_plus1 + zeta_G[0][:, i] + zeta_W[0][:, i] == 0 )

    #             # [theta; -1] + zeta_G(:, i, 2) + zeta_W(:, i, 2) == 0
    #             plus_theta_minus1 = cp.hstack([theta, -1.0])  # shape (d,)
    #             constraints.append( plus_theta_minus1 + zeta_G[1][:, i] + zeta_W[1][:, i] == 0 )
    #         elif self.model_type == 'svm':
    #             constraints.append( y[i] * theta + zeta_G[0][:, i] + zeta_W[0][:, i] == 0 )

    #             # zeta_G(:, i, 2) + zeta_W(:, i, 2) == 0
    #             constraints.append( zeta_G[1][:, i] + zeta_W[1][:, i] == 0 )


    #         constraints.append( cp.quad_over_lin(zeta_G[0][:, i], lambda_1) <= tau[i, 0])
    #         constraints.append( cp.quad_over_lin(zeta_G[1][:, i], lambda_1) <= tau[i, 1])
    #         constraints.append( cp.norm(zeta_W[0][:, i], self.dual_norm) <= lambda_2 )
    #         constraints.append( cp.norm(zeta_W[1][:, i], self.dual_norm) <= lambda_2 )

    #     # ----------------------------------------------------------------------
    #     # 5) Solve the problem
    #     # ----------------------------------------------------------------------
    #     problem = cp.Problem(objective, constraints)
    #     try: 
    #         problem.solve(solver = self.solver)
    #         self.theta = theta.value
    #     except cp.SolverError as e:
    #         raise ORWDROError(f"Optimization failed to solve using {self.solver}.") from e

    #     print(problem.status)
    #     if self.theta is None:
    #         raise ORWDROError("Optimization did not converge to a solution.")

    #     return {"theta": self.theta.tolist()}


    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if self.model_type in {'svm'}:
            if not np.all(np.isin(y, [-1, 1])):
                raise ORWDROError("Classification labels must be in {-1, +1}")

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise ORWDROError(f"Expected {self.input_dim} features, got {feature_size}")
        if sample_size != y.shape[0]:
            raise ORWDROError("X and y must have the same number of samples")

        z_0, sgn = self._cheap_robust_mean_estimate(X, y)
        print('check z0', z_0)
        assert not np.any(np.isnan(z_0)), "NaN detected in z_0"
        assert not np.any(np.isinf(z_0)), "Inf detected in z_0"

        
        if self.model_type == 'lad':
            Z = np.hstack([X, y.reshape(-1, 1)])  # (n, d+1)
        elif self.model_type == 'svm':
            Z = X  # (n, d)
        else:
            raise ORWDROError(f"Unsupported model type: {self.model_type}")


        zeta_G = [cp.Variable(shape=(sample_size, feature_size + 1 - sgn)) for _ in range(2)]
        zeta_W = [cp.Variable(shape=(sample_size, feature_size + 1 - sgn)) for _ in range(2)]


        lambda_1 = cp.Variable(nonneg=True, name='lambda_1')
        lambda_2 = cp.Variable(nonneg=True, name='lambda_2')
        alpha = cp.Variable(name='alpha')
        s = cp.Variable(sample_size, nonneg=True, name='s')
        tau = cp.Variable((sample_size, 2), nonneg=True, name='tau')
        theta = cp.Variable(feature_size, name='theta')

        objective = cp.Minimize(
            lambda_1 * (self.sigma ** 2) 
            + lambda_2 * (self.eps ** 1) 
            + (1.0 / (sample_size * (1.0 - self.eta))) * cp.sum(s) 
            + alpha
        )

        constraints = []
        z_0 = z_0.reshape(1, -1)  

        for j in range(2):
            term_zG = z_0 @ zeta_G[j].T  # (1, n)
            term_ZW = cp.sum(cp.multiply(Z, zeta_W[j]), axis=1)  # (n,)
            lhs = cp.reshape(term_zG, (sample_size,)) + tau[:, j] + term_ZW - alpha
            if j == 0 and self.model_type == 'svm':
                constraints.append(s >= 1 + lhs)
            else:
                constraints.append(s >= lhs)

        ## constraint: model fit
        if self.model_type == 'lad':
            theta_ext = cp.hstack([-theta, 1.0])
            constraints += [
                zeta_G[0] + zeta_W[0] == theta_ext,
                zeta_G[1] + zeta_W[1] == -theta_ext
            ]
        elif self.model_type == 'svm':
            constraints += [
                cp.multiply(y.reshape(-1,1), cp.reshape(theta, (1, feature_size))) + zeta_G[0] + zeta_W[0] == 0,
                zeta_G[1] + zeta_W[1] == 0
            ]

        # constraint: second-order cone
        for j in range(2):
            constraints.append(
                cp.quad_over_lin(zeta_G[j], lambda_1) <= tau[:, j]
            )
            constraints.append(
                cp.norm(zeta_W[j], self.dual_norm, axis=1) <= lambda_2
            )

        # solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(
                solver = self.solver
            )
            self.theta = theta.value
        except cp.SolverError as e:
            raise ORWDROError(f"Solver failed: {str(e)}")

        if self.theta is None or not np.all(np.isfinite(self.theta)):
            raise ORWDROError("Optimization did not converge to a valid solution")

        return {"theta": self.theta.tolist()}

    def _cheap_robust_mean_estimate(self, X, y):
        """
        robust mean estimation with 2 * self.eta trimming.
        """
        n, d = X.shape
        means = np.zeros(d)
        for j in range(d):
            col = X[:, j]
            # Determine the lower and upper cutoffs
            lower_cut = np.quantile(col, self.eta)
            upper_cut = np.quantile(col, 1 - self.eta)
            # Keep only the values within [lower_cut, upper_cut]
            trimmed_col = col[(col >= lower_cut) & (col <= upper_cut)]
            # Compute the mean of the remaining values
            means[j] = trimmed_col.mean()

        # lower_cuts = np.quantile(X, self.eta, axis=0)
        # upper_cuts = np.quantile(X, 1 - self.eta, axis=0)
        # mask = (X >= lower_cuts) & (X <= upper_cuts)
        # means = np.sum(X * mask, axis=0) / np.sum(mask, axis=0)

        if self.model_type == 'lad':
            means_aug = np.zeros(d + 1)
            means_aug[0:d] = means
            lower_cut = np.quantile(y, self.eta)
            upper_cut = np.quantile(y, 1 - self.eta)
            # Keep only the values within [lower_cut, upper_cut]
            trimmed_col = y[(y >= lower_cut) & (y <= upper_cut)]
            means_aug[d] = trimmed_col.mean()
            return means_aug, 0
        elif self.model_type == 'svm':
            return means, 1
        raise NotImplementedError
    
    def worst_distribution(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute the worst-case distribution.

        Reference: Theorem 3 in https://arxiv.org/pdf/2311.05573

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            ORWDROError: If the worst-case distribution optimization fails.
        """
        
        n, d = X.shape
        J = 2
        # Variables
        q = [[cp.Variable(nonneg = True) for _ in range(J)] for _ in range(n)]        # q_{ij}
        xi = cp.Variable((d+1, n*J))                 # each col is \xi_{ij}

        ################
        # 2) Constraints
        ################
        constraints = []
        for i in range(n):
            constraints.append( cp.sum(q[i]) <= 1.0/(n*(1.0 - self.eta)) )
        q_sum = 0
        for i in range(n):
            q_sum += cp.sum(q[i])
        constraints.append( q_sum == 1 )

        z0, __ = self._cheap_robust_mean_estimate(X, y)
        
        # TODO for tianyu: please check whether this is ok
        z0 = np.concatenate([z0, [1]])

        Ztilde = np.hstack([X, y.reshape(-1, 1)])

        dist_p_list = []
        dist_2_list = []
        p = 1  # or 2, if you want
        for i in range(n):
            for j in range(J):
                idx = j*n + i
                dist_p = cp.norm( xi[:, idx] - q[i][j]*Ztilde[i], p )
                dist_p_list.append(dist_p)
                ell_0 = cp.sum_squares(xi[:, idx] - q[i][j]*z0)
                dist_2 = cp.perspective(ell_0, q[i][j], ell_0)
                dist_2_list.append(dist_2)
        constraints.append( cp.sum(dist_p_list) <= self.eps)
        constraints.append( cp.sum(dist_2_list) <= self.sigma ** 2)

  


        ################
        # 3) Objective
        ################
        theta = self.theta

        obj_terms = 0
        for i in range(n):
            for j in range(J):
                idx = j*n + i
                if j==0:
                    # ell_1(a)= aY - theta^T aX
                    ell_1 = xi[d, idx] - theta @ xi[:d, idx]
                    obj_terms += cp.perspective(ell_1, q[i][j], ell_1)
                else:
                    # ell_2(a)= theta^T aX - aY
                    ell_2 = theta @ xi[:d, idx] - xi[d, idx]
                    obj_terms += cp.perspective(ell_2, q[i][j], ell_2)

        objective = cp.Minimize(obj_terms)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver = self.solver, verbose = True)

        except cp.error.SolverError as e:
            raise ORWDROError("Optimization failed to solve for worst-case distribution.") from e

        if problem.value is None:
            raise ORWDROError("Worst-case distribution optimization did not converge to a solution.")
        
        # q_values = [[q[i][j].value for j in range(J)] for i in range(n)]
        # print(xi.value, problem.status)
        # xi_values = xi.value
        # sample_pts = []
        # for i in range(n):
        #     for j in range(J):
        #         idx = j * n + i
        #         sample_pts.append(xi_values[:, idx]) 
        # sample_pts = np.array(sample_pts) 
        # weights = []
        # for i in range(n):
        #     for j in range(J):
        #         weights.append(q_values[i][j])
        # weights = np.array(weights)
        # assert np.isclose(weights.sum(), 1.0)
        # return {
        #     "sample_pts": sample_pts,  
        #     "weight": weights          
        # }
        