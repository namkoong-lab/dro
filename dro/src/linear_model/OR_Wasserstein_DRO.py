from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from typing import Dict, Any

class ORWDROError(Exception):
    """Base exception class for errors in OR-WDRO model."""
    pass

class OR_Wasserstein_DRO(BaseLinearDRO):
    """
    Outlier-Robust Wasserstein DRO.

    This model minimizes the TV corrupted p-Wasserstein loss function for both regression and classification.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression).
        eps (float): Robustness parameter for OR-WDRO. 
        eta (float): Fraction of outlier for OR-WDRO.
        dual norm (int): used in the optimization

    Reference:<https://arxiv.org/pdf/2311.05573>
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', eps: float = 0.0, eta: float = 0.0, dual_norm: int = 1):
        """
        Initialize the ORWDRO model with specified input dimension and model type.
        """
        BaseLinearDRO.__init__(self, input_dim, model_type)
        self.eps = eps
        self.eta = eta
        self.dual_norm = dual_norm
        self.sigma = math.sqrt(input_dim)
        
    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing 'eps' key for robustness parameter.
        TODO: other description
        Raises:
            ORDROError: If 'eps', 'eta' is provided but is not a non-negative float.
        """
        if 'eps' in config:
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise ORWDROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)
        if 'eta' in config:
            eta = config['eta']
            if not isinstance(eta, (float, int)) or eta < 0 or eta >= 1:
                raise ORWDROError("Fraction of outlier 'eta' must be a non-negative float between 0 and 1")
            self.eta = float(eta)
        if 'dual_norm' in config:
            dual_norm = config['dual_norm']
            if dual_norm not in [1, 2]:
                raise ORWDROError("dual_norm must be 1 or 2")
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # TODO: description
        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise ORWDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise ORWDROError("Input X and target y must have the same number of samples.")
    
        z_0, sgn = self.cheap_robust_mean_estimate(X, y)
        if self.model_type == 'lad':
            Z = np.hstack([X, y.reshape(-1, 1)])
        elif self.model_type == 'svm':
            Z = X
        # J = 2 (as in your original code)
        J = 2
        # p = 1, q = 2, from the comments in your original code

        # ----------------------------------------------------------------------
        # 2) Define CVXPY Variables
        # ----------------------------------------------------------------------
        lambda_1 = cp.Variable(nonneg=True)     # scalar >= 0
        lambda_2 = cp.Variable(nonneg=True)     # scalar >= 0
        alpha    = cp.Variable()                # scalar (unconstrained in sign, can be +/-, user sets constraints if needed)
        s        = cp.Variable(shape=(sample_size,), nonneg=True)    # s(i) >= 0
        # We'll store zeta_G and zeta_W in 3D shape (d, n, J).
        # CVXPY does allow multi-dimensional Variables, but indexing must be handled carefully.
        zeta_G = [cp.Variable(shape=(feature_size + 1 - sgn, sample_size)) for _ in range(2)]
        zeta_W = [cp.Variable(shape=(feature_size + 1 - sgn, sample_size)) for _ in range(2)]

        # for reformulating rotated SOC
        aux_W = cp.Variable(shape = (2,sample_size), nonneg = True)
        # tau: shape (n, J), both entries >= 0
        tau = cp.Variable(shape=(sample_size, J), nonneg=True)
        # theta: shape (d-1, )
        theta = cp.Variable(shape=(feature_size,))

        # ----------------------------------------------------------------------
        # 3) Objective function
        #    objective = lambda_1 * sigma^q + lambda_2 * rho^p
        #                + (1/(n*(1 - vareps))) * sum(s) + alpha
        # ----------------------------------------------------------------------
        objective_expr = (
            lambda_1 * (self.sigma ** 2)
            + lambda_2 * (self.eps ** 1)
            + (1.0 / (sample_size * (1.0 - self.eta))) * cp.sum(s)
            + alpha
        )
        objective = cp.Minimize(objective_expr)

        # ----------------------------------------------------------------------
        # 4) Build constraints
        # ----------------------------------------------------------------------
        constraints = []

        # Already declared lambda_1 >= 0, lambda_2 >= 0, s >= 0, tau >= 0 in the variable definitions.

        # Loop over each sample i
        for i in range(sample_size):
            # s(i) >= z_0' * zeta_G(:, i, 1) + tau(i, 1) + Z(i,:) * zeta_W(:, i, 1) - alpha
            # But in Python indexing, j=0 or j=1 for the 2 constraints
            # We'll build them explicitly:

            # s(i) >= <z_0, zeta_G(:,i,0)> + tau(i,0) + <Z(i,:), zeta_W(:,i,0)> - alpha
            lhs_0 = (sgn + z_0 @ zeta_G[0][:, i]
                    + tau[i, 0]
                    + Z[i, :] @ zeta_W[0][:, i]
                    - alpha )
            constraints.append( s[i] >= lhs_0 )

            # s(i) >= z_0' * zeta_G(:, i, 2) + tau(i, 2) + Z(i,:) * zeta_W(:, i, 2) - alpha
            lhs_1 = ( z_0 @ zeta_G[1][:, i]
                    + tau[i, 1]
                    + Z[i, :] @ zeta_W[1][:, i]
                    - alpha )
            constraints.append( s[i] >= lhs_1 )

            # Next the linear constraints:
            # [-theta; 1] + zeta_G(:, i, 1) + zeta_W(:, i, 1) == 0
            # We'll build the vector [-theta, 1] via cp.hstack:
            if self.model_type == 'lad':
                minus_theta_plus1 = cp.hstack([-theta, 1.0])  # shape (d,)
                constraints.append( minus_theta_plus1 + zeta_G[0][:, i] + zeta_W[0][:, i] == 0 )

                # [theta; -1] + zeta_G(:, i, 2) + zeta_W(:, i, 2) == 0
                plus_theta_minus1 = cp.hstack([theta, -1.0])  # shape (d,)
                constraints.append( plus_theta_minus1 + zeta_G[1][:, i] + zeta_W[1][:, i] == 0 )
            elif self.model_type == 'svm':
                constraints.append( y[i] * theta + zeta_G[0][:, i] + zeta_W[0][:, i] == 0 )

                # zeta_G(:, i, 2) + zeta_W(:, i, 2) == 0
                constraints.append( zeta_G[1][:, i] + zeta_W[1][:, i] == 0 )


            # Rotated cone constraints:
            # rcone(zeta_G(:, i, 1), lambda_1, 0.5 * tau(i, 1)) =>  ||zeta_G||^2 <= lambda_1 * tau(i, 1)
            # In cvxpy, we can do sum_squares(...) <= lambda_1 * tau[i, 0]
            # constraints.append( cp.sum_squares(zeta_G[0][:, i]) <= aux_W[0][i])
            # constraints.append( cp.SOC(cp.sqrt(2 * aux_W[0][i]), cp.hstack([lambda_1, tau[i, 0]])))
            # constraints.append( cp.sum_squares(zeta_G[1][:, i]) <= aux_W[1][i])
            # constraints.append( cp.SOC(cp.sqrt(2 * aux_W[1][i]), cp.hstack([lambda_1, tau[i, 1]])))
            constraints.append( cp.quad_over_lin(zeta_G[0][:, i], lambda_1) <= tau[i, 0])
            constraints.append( cp.quad_over_lin(zeta_G[1][:, i], lambda_1) <= tau[i, 1])

            # norm(zeta_W(:, i, 1), dual_norm) <= lambda_2
            # similarly for j=0,1
            constraints.append( cp.norm(zeta_W[0][:, i], self.dual_norm) <= lambda_2 )
            constraints.append( cp.norm(zeta_W[1][:, i], self.dual_norm) <= lambda_2 )

        # ----------------------------------------------------------------------
        # 5) Solve the problem
        # ----------------------------------------------------------------------
        problem = cp.Problem(objective, constraints)
        try: 
            problem.solve(solver = cp.GUROBI)
            self.theta = theta.value
        except cp.SolverError as e:
            raise ORWDROError("Optimization failed to solve using MOSEK.") from e

        if self.theta is None:
            raise ORWDROError("Optimization did not converge to a solution.")

        return {"theta": self.theta.tolist()}



    def cheap_robust_mean_estimate(self, X, y):
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
            ORWError: If the worst-case distribution optimization fails.
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
        # 2a) sum_j q_{ij} <= 1/(n*(1-vareps))
        for i in range(n):
            constraints.append( cp.sum(q[i]) <= 1.0/(n*(1.0 - self.eta)) )
        # 2b) sum_{i,j} q_{ij} = 1
        q_sum = 0
        for i in range(n):
            q_sum += cp.sum(q[i])
        constraints.append( q_sum == 1 )

        # 2c) norm constraints
        z0, __ = self.cheap_robust_mean_estimate(X, y)
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
        return q
        return {'sample_pts': [X, y], 'weight': [[q[i][j].value for j in range(J)] for i in range(n)]}