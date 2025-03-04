from .base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm
import time
from collections import namedtuple
from typing import Dict, Any
from sklearn.metrics import f1_score

class MOTDROError(Exception):
    """Base exception class for errors in Wasserstein DRO model."""
    pass


## MOT DRO
class MOTDRO(BaseLinearDRO):
    """
    OT discrepancy with conditional moment constraint, the cost is set as:
    c(((X_1, Y_1), w), ((X_2, Y_2), \hat w)) = theta1 * w * d((X_1, Y_1), (X_2, Y_2)) + theta2 * (\phi(w) - \phi(\hat w))^+, where \phi() is the KL divergence.

    Attribute:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        eps (float): Robustness parameter for DRO.
        cost matrix (np.ndarray): the feature importance perturbation matrix with the dimension being (input_dim, input_dim).
        p (float or 'inf'): Norm parameter for controlling the perturbation moment of X.
        
        theta1 (float): theta1 \geq 0 controls the strength of Wasserstein Distance
        theta2 (float): theta2 \geq 0 controls the strength of KL Divergence
    Reference: <https://arxiv.org/abs/2308.05414>
    """
    def __init__(self, input_dim: int, model_type: str):
        """
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
        """
        BaseLinearDRO.__init__(self, input_dim, model_type)

        self.theta1 = 2
        self.theta2 = 2
        self.eps = 0        
        self.p = 2

    def update(self, config = Dict[str, Any]) -> None:
        """Update the model configuration
        
        Args: 
            config (Dict[str, Any]): Configuration dictionary containing 'theta1', 'theta2', 'eps', 'p', 'kappa' keys for robustness parameter.
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
        epig_ = cp.Variable([N_train, 1], pos=True)
        self.lambda_ = cp.Variable(1)
        self.theta = cp.Variable(dim)
        eta = cp.Variable([N_train, 1])

        # Constraints
        cons = []
        cons.append(eta >= 0)
        cons.append(self.lambda_ >= 0)
        
        ## our version
        cons.append(self._cvx_loss(X, y, self.theta) <= epig_)  
        for i in range(N_train):
            cons.append(
                cp.constraints.exponential.ExpCone(
                    epig_[i] - t, self.theta2 * self.lambda_, eta[i]
                )
            )
          
        cons.append(self.lambda_ * self.theta1 >= cp.norm(self.theta, q))
        cons.append(N_train * self.lambda_ * self.theta2 >= cp.sum(eta))
        obj = self.eps * self.lambda_ + t
        problem = cp.Problem(cp.Minimize(obj), cons)

        problem.solve(solver=self.solver)
        self.theta = self.theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params


# ## L_inf cost function for Unified DRO
# class MOT_Robust_CLF_Linf:
#     def __init__(
#         self, fit_intercept=False, theta1=2, theta2=2, c_r=0, p="inf", verbose=True):
#         self.fit_intercept = fit_intercept
#         self.theta1 = theta1
#         self.theta2 = theta2
#         self.c_r = c_r
#         self.p = p
#         self.verbose = verbose
#         self.training_time = 0
#         self.model_prepared = False


#     def update(self, config = {}):
#         if "theta1" in config.keys():
#             self.theta1 = config["theta1"]
#         if "theta2" in config.keys():
#             self.theta2 = config["theta2"]
#         if "rho" in config.keys():
#             self.c_r = config["rho"]


#     def fit(self, X, y):
#         if self.model_prepared:
#             self.param_fit()
#         else:
#             self.prepare_model(X, y)
#             self.param_fit()
        
#         model_params = {}
#         model_params["theta"] = self.coef_.reshape(-1).tolist()
#         return model_params

#     def prepare_model(self, X, y):
#         start_time = time.time()
#         N_train = X.shape[0]
#         dim = X.shape[1]
#         self.r = cp.Parameter(nonneg=True)
#         self.theta1_par = cp.Parameter(nonneg=True)
#         self.theta2_par = cp.Parameter(nonneg=True)

#         if self.p == 1:
#             q = "inf"
#         elif self.p == 2:
#             q = 2
#         elif self.p == "inf":
#             q = 1

#         # Decision variables
#         t = cp.Variable(1)
#         h = cp.Variable(1, pos=True)
#         epig_ = cp.Variable([N_train, 1], pos=True)
#         self.lambda_ = cp.Variable(1)
#         self.beta = cp.Variable(dim)
#         eta = cp.Variable([N_train, 1])

#         # Constraints
#         cons = []
#         cons.append(
#             cp.norm2(self.beta) <= 1
#         )  # Bounded SVM constraint from the original problem
#         cons.append(eta >= 0)
#         cons.append(self.lambda_ >= 0)
        

#         # original version
#         for i in range(N_train):
#             cons.append(
#                 cp.constraints.exponential.ExpCone(
#                     epig_[i] - t, self.theta2 * self.lambda_, eta[i]
#                 )
#             )

#             ## original version:
#             cons.append(cp.pos(1 - y[i] * (self.beta.T @ X[i, :])) <= epig_[i])
#         cons.append(self.lambda_ * self.theta1 >= cp.norm(self.beta, q))

#         cons.append(N_train * self.lambda_ * self.theta2 >= cp.sum(eta))
#         self.obj = self.r * self.lambda_ + t
#         self.problem = cp.Problem(cp.Minimize(self.obj), cons)
#         self.model_prepared = True
#         stop_time = time.time()
#         self.model_building_time = stop_time - start_time

#     def param_fit(self):
#         """Can be only called after prepare_model"""
#         start_time = time.time()
#         self.r.value = self.c_r
#         self.theta1_par.value = self.theta1
#         self.theta2_par.value = self.theta2
#         self.problem.solve(
#             solver=self.solver,
#             verbose=self.verbose,
#         )
#         self.coef_ = self.beta.value
#         self.obj_opt = self.obj.value
#         stop_time = time.time()
#         self.param_fit_time = stop_time - start_time

#     def loss(self, X, y):
#         loss = np.mean(
#             cp.pos(
#                 1 - np.multiply(y.flatten(), self.coef_.T @ X.T)
#             ).value
#         )
#         return loss

#     def predict(self, X):
#         scores = self.coef_.T @ X.T
#         preds = scores.copy()
#         preds[scores >= 0] = 1
#         preds[scores < 0] = 0
#         return preds

#     def score(self, X, y):
#         # calculate accuracy of the given test data set
#         predictions = self.predict(X)
#         acc = np.mean([predictions.flatten() == y.flatten()])
#         f1 = f1_score(y, predictions, average='macro')
#         return acc, f1 
