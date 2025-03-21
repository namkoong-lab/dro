from dro.src.linear_model.base import BaseLinearDRO
import cvxpy as cp
from typing import Dict, Any

class MOTDROError(Exception):
    """Base exception class for errors in Wasserstein DRO model."""
    pass


## MOT DRO
class MOTDRO(BaseLinearDRO):
    """
    OT discrepancy with conditional moment constraint, the cost is set as:
    c(((X_1, Y_1), w), ((X_2, Y_2), \hat w)) = theta1 * w * d((X_1, Y_1), (X_2, Y_2)) + theta2 * (\phi(w) - \phi(\hat w))^+, where \phi(w) = w log (w) - w + 1 is the KL divergence.

    Args:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for DRO.
        cost matrix (np.ndarray): the feature importance perturbation matrix with the dimension being (input_dim, input_dim).
        p (float or 'inf'): Norm parameter for controlling the perturbation moment of X.
        theta1 (float): theta1 \geq 0 controls the strength of Wasserstein Distance
        theta2 (float): theta2 \geq 0 controls the strength of KL Divergence

    Reference: <https://arxiv.org/abs/2308.05414>
    """
    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK'):
        """
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'lad').
        """
        if model_type in ['ols', 'logistic']:
            raise MOTDROError("MOT DRO does not support OLS, logistic")
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.theta1 = 1
        self.theta2 = 1
        self.eps = 0        
        self.p = 2
        self.square = 2

    def update(self, config = Dict[str, Any]) -> None:
        """Update the model configuration
        
        Args: 
            config (Dict[str, Any]): Configuration dictionary containing 'theta1', 'theta2', 'eps', 'p', 'kappa', 'square' keys for robustness parameter.
            
        Raises:
            MOTDROError: If any of the configs does not fall into its domain.
        """

        if "theta1" in config.keys():
            self.theta1 = config["theta1"]
            self.theta2 = 1/(1-1/self.theta1)
            assert self.theta1 > 1
        if "theta2" in config.keys():
            self.theta2 = config["theta2"]
            self.theta1 = 1/(1-1/self.theta2)
            assert self.theta2 > 1
        if "eps" in config.keys():
            eps = config["eps"]
            if not isinstance(eps, (float, int)) or eps < 0:
                raise MOTDROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)

        if 'p' in config.keys():
            p = config['p']
            if p != 'inf':
                if not isinstance(p, (float, int)) or p < 1:
                    raise MOTDROError("Norm parameter 'p' must be float and larger than 1.")
                self.p = float(p)
            else:
                self.p = p
        
        if 'square' in config.keys():
            square = config['square']
            if square not in [1, 2]:
                raise MOTDROError("Distance parameter 'square' can only take values in 1,2.")
            self.square = square


    def fit(self, X, y):
        """Fit the model using CVXPY to solve the optimal-transportdistributionally robust optimization problem with conditional moment.

        Args:
            X (np.ndarray): Input feature matrix with shape (N_train, dim).
            y (np.ndarray): Target vector with shape (N_train,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' key.

        Raises:
            WassersteinDROError: If the optimization problem fails to solve.        
        """
        N_train = X.shape[0]
        dim = X.shape[1]

        if dim != self.input_dim:
            raise MOTDROError(f"Expected input with {self.input_dim} features, got {dim}.")
        if N_train != y.shape[0]:
            raise MOTDROError("Input X and target y must have the same number of samples.")

        self.theta1_par = self.theta1
        self.theta2_par = self.theta2

        if self.p == 1:
            q = "inf"
        elif self.p == 2:
            q = 2
        elif self.p == "inf":
            q = 1

        # Decision variables
        t = cp.Variable(1)
        epig_ = cp.Variable([N_train, 1])
        self.lambda_ = cp.Variable(1, pos = True)
        theta = cp.Variable(dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable([N_train, 1], pos = True)

        # Constraints
        cons = []
        
        ## our version
        for i in range(N_train):
            cons.append(
                cp.constraints.exponential.ExpCone(
                    epig_[i] - t, self.theta2 * self.lambda_, eta[i]
                )
            )
            if self.square == 1:
                cons.append(self.lambda_ * self.theta1 >= cp.norm(self.theta, q))
                cons.append(self._cvx_loss(X[i], y[i], theta, b) <= epig_)  
            elif self.square == 2:
                cons.append(self._cvx_loss(X[i], y[i], theta, b) + cp.quad_over_lin(cp.norm(theta, q), self.lambda_*4*self.theta1) <= epig_[i]) 

        cons.append(N_train * self.lambda_ * self.theta2 >= cp.sum(eta))
        obj = self.eps * self.lambda_ + t
        problem = cp.Problem(cp.Minimize(obj), cons)

        problem.solve(solver=self.solver)
        print(problem.status)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
        return model_params