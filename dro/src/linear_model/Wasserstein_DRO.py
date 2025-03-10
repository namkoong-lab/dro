from dro.src.linear_model.base import BaseLinearDRO
import numpy as np
import math
import cvxpy as cp
from scipy.linalg import sqrtm
from typing import Dict, Any
import warnings

class WassersteinDROError(Exception):
    """Base exception class for errors in Wasserstein DRO model."""
    pass


class WassersteinDRO(BaseLinearDRO):
    """Wasserstein Distributionally Robust Optimization (WDRO) model
    
    This model minimizes a Wasserstein-robust loss function for both regression and classification.

    The Wasserstein distance is defined as the minimum probability coupling of two distributions for the distance metric: 
        d((X_1, Y_1), (X_2, Y_2)) = (|cost_matrix^{1/2} @ (X_1 - X_2)|_p)^{square} + kappa |Y_1 - Y_2|, 
    where parameters are:
        - cost matrix, (a PSD Matrix); 
        - kappa;
        - p;
        - square (notation depending on the model type), where square = 2 for 'svm', 'logistic', 'lad'; square = 1 for 'ols'.

    Attribute:
        input_dim (int): Dimensionality of the input features.
        model_type (str, default = 'svm'): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression for OLS, 'lad' for Linear Regression for LAD).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'.
        eps (float): Robustness parameter for DRO.
        cost matrix (np.ndarray): the feature importance perturbation matrix with the dimension being (input_dim, input_dim).
        p (float or 'inf'): Norm parameter for controlling the perturbation moment of X.
        kappa (float or 'inf'): Robustness parameter for the perturbation of Y. Note that if we set kappa to be large enough to approximately infinite, this is equivalent to saying that we do not allow changes in Y.

    Reference:
    [1] OLS: <https://www.cambridge.org/core/journals/journal-of-applied-probability/article/robust-wasserstein-profile-inference-and-applications-to-machine-learning/4024D05DE4681E67334E45D039295527>
    [2] LAD / SVM / Logistic: <https://jmlr.org/papers/volume20/17-633/17-633.pdf>
    """


    def __init__(self, input_dim: int, model_type: str = 'svm', fit_intercept: bool = True, solver: str = 'MOSEK'):
        """
        Args:
            input_dim (int): Dimension of the input features.
            model_type (str): Type of model ('svm', 'logistic', 'ols', 'lad').
            fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'
        """
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)

        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        self.eps = 0
        self.p = 1
        self.kappa = 'inf'
        

    def update(self, config: Dict[str, Any]) -> None:
        """Update the model configuration
        
        Args: 
            config (Dict[str, Any]): Configuration dictionary containing 'cost_matrix', 'eps', 'p', 'kappa' keys for robustness parameter.
        Raises:
            WassersteinDROError: If any of the configs does not fall into its domain.
        """
        if 'cost_matrix' in config.keys():
            cost_matrix = config['cost_matrix']
            if not isinstance(cost_matrix, np.ndarray) or cost_matrix.shape != (self.input_dim, self.input_dim) or not np.all(np.linalg.eigvals(cost_matrix) > 0):
                raise WassersteinDROError("Cost Adjust Matrix 'cost matrix' must be a PD matrix")
            self.cost_matrix = cost_matrix
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'eps' in config.keys():
            eps = config['eps']
            if not isinstance(eps, (float, int)) or eps < 0:
                raise WassersteinDROError("Robustness parameter 'eps' must be a non-negative float.")
            self.eps = float(eps)
        if 'p' in config.keys():
            p = config['p']
            if p != 'inf' and ((not isinstance(p, (float, int)) or p < 1)):
                raise WassersteinDROError("Norm parameter 'p' must be float and larger than 1.")
            self.p = float(p)

        if 'kappa' in config.keys():
            kappa = config['kappa']
            if kappa != 'inf' and ((not isinstance(kappa, (float, int))) or kappa < 0):
                raise WassersteinDROError("Y-Robustness parameter 'kappa' must be a non-negative float.")
            if kappa != 'inf' and self.model_type == 'ols':
                warnings.warn("Wasserstein Distributionally Robust OLS does not support changes of Y in the ambiguity set")
            self.kappa = float(kappa)

        
    def penalization(self, theta: cp.Expression) -> float:
        """
        Module for computing the regularization part in the standard Wasserstein DRO problem.

        Args:
            theta (cp.Expression): Feature vector with shape (n_feature,).
        
        Returns:
            Float: Regularization term part.

        """
        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1

        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta, dual_norm)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta, dual_norm)
        elif self.model_type == 'lad':
            # the dual of the \|\theta, -1\|_dual_norm penalization
            if self.kappa == 'inf':
                return cp.max(cp.norm(self.cost_inv_transform @ theta, dual_norm))
            else:
                return cp.max(cp.norm(self.cost_inv_transform @ theta, dual_norm), 1 / self.kappa)




        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model using CVXPY to solve the Wasserstein distributionally robust optimization problem.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' key.

        Raises:
            WassersteinDROError: If the optimization problem fails to solve.        
        """

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise WassersteinDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise WassersteinDROError("Input X and target y must have the same number of samples.")



        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0


        lamb_da = cp.Variable()
        cons = [lamb_da >= self.penalization(theta)]
        if self.model_type == 'ols':
            final_loss = cp.norm(X @ theta + b - y) / math.sqrt(sample_size) + math.sqrt(self.eps) * lamb_da

        else:
            if self.model_type in ['svm', 'logistic']:
                s = cp.Variable(sample_size)
                cons += [s >= self._cvx_loss(X, y, theta, b)]
                if self.kappa != 'inf':
                    cons += [s >= self._cvx_loss(X, -y, theta, b) - lamb_da * self.kappa]
                final_loss = cp.sum(s) / sample_size + self.eps * lamb_da
            else:
                # model type == 'lad' for general p.
                final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + self.eps * lamb_da

        problem = cp.Problem(cp.Minimize(final_loss), cons)
        try:
            problem.solve(solver = self.solver)
        except cp.error.SolverError as e:
            raise WassersteinDROError(f"Optimization failed to solve using {self.solver}.") from e
        
        if theta.value is None:
            raise WassersteinDROError("Optimization did not converge to a solution.")

        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
        return model_params
    
    def distance_compute(self, X_1: cp.Expression, X_2: np.ndarray, Y_1: cp.Expression, Y_2: float) -> cp.Expression:
        """
        Computing the distance between two points (X_1, Y_1), (X_2, Y_2) under our defined metric in cvxpy problem

        Args:
            X_1 (cp.Expression): Input feature-1 (n_feature,);
            Y_1 (float): Input label-1;
            X_2 (cp.Expression): Input feature-2 (n_feature,);
            Y_2 (float): Input label-2;

        Returns:
            Float: d((X_1, Y_1), (X_2, Y_2))
            
        Raises:
            WassersteinDROError: If the dimensions of two input feature are different.
        """
        if X_1.shape[-1] != X_2.shape[-1]:
            raise WassersteinDROError(f"two input feature dimensions are different.")
        # if Y_1 != Y_2 and self.kappa != 'inf':
        #     warnings.warn("Despite labels are different, we do not count their difference since we do not allow change in Y.")

        component_X = cp.norm(sqrtm(self.cost_matrix) @ (X_1 - X_2), self.p)
        if self.model_type == 'ols':
            component_X = component_X ** 2

        if self.kappa != 'inf':
            component_Y = self.kappa * cp.abs(Y_1 - Y_2)

        else:
            # change of Y is not allowed
            component_Y = 0
        return component_X + component_Y
        

    def lipschitz_norm(self):
        """
        Computing the Lipschitz norm of the loss function

        Returns:
            Float: the size of the Lipschitz norm of the loss function

        """
        if self.model_type in ['svm', 'logistic', 'lad']:
            return 1
        else:
            return np.inf
        

    def worst_distribution(self, X: np.ndarray, y: np.ndarray, compute_type: int) -> Dict[str, Any]:
        """Compute the worst-case distribution based on Wasserstein Distance
        
        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).
            compute_type (int): type of computing the worst case distribution, only suitable for 'lad', 'svm', 'logistic'. 1 refers to computing via [1] (which is asymptotically, not exact and not exact and cannot do when kappa = \infty), 2 refers to computing via [2].

        Returns:
            Dict[str, Any]: Dictionary containing 'sample_pts' and 'weight' keys for worst-case distribution.

        Raises:
            Warnings: when ols is set for the compute_type == 1
            WassersteinDROError: the compute_type == 1 does not support kappa = infty.

        Reference of Worst-case Distribution:
        [1] SVM / Logistic / LAD: Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf, where eta is the theta in eq(27) and gamma = 0 in that equation.
        [2] In all cases, we use a reduced dual case (e.g., Remark 5.2 of https://arxiv.org/pdf/2308.05414) to compute their worst-case distribution.
        [3] General Worst-case Distributions can be found in: https://pubsonline.informs.org/doi/abs/10.1287/moor.2022.1275, where norm_theta is lambda* here.
        """

        if self.model_type == 'ols' and compute_type == 1:
            warnings.warn("OLS does not support the corresponding computation method.")
        elif self.kappa == 'inf' and compute_type == 1:
            raise WassersteinDROError("The corresponding computation method do not support kappa = infty!")
        
        sample_size, __ = X.shape


        self.fit(X, y)
        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1
        norm_theta = np.linalg.norm(self.cost_inv_transform @ self.theta, ord = dual_norm)

        if compute_type == 2:
            if self.model_type == 'ols':
                dual_norm_parameter = np.linalg.norm(self.cost_inv_transform @ self.theta, dual_norm) ** 2
                new_X = np.zeros((sample_size, self.input_dim))
                for i in range(sample_size):
                    var_x = cp.Variable(self.input_dim)
                    var_y = cp.Variable()
                    obj = (y[i] - self.theta @ var_x - self.b) ** 2 - dual_norm_parameter * self.distance_compute(var_x, X[i], var_y, y[i])
                    problem = cp.Problem(cp.Maximize(obj))
                    problem.solve(solver = self.solver)
                    new_X[i] = var_x.value
                return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}

            else: # linear classification or regression with Lipschitz norm
                # we denote the following case when we do not change Y.
                new_X = np.zeros((sample_size, self.input_dim))
                if self.model_type == 'svm':
                    for i in range(sample_size):
                        var_x = cp.Variable(self.input_dim)
                        var_y = cp.Variable()
                        obj = 1 - y[i] * var_x @ self.theta - self.b - norm_theta * self.distance_compute(var_x, X[i], var_y, y[i])
                        problem = cp.Problem(cp.Maximize(obj))
                        problem.solve(solver = self.solver)
                        
                        if 1 - y[i] * var_x.value @ self.theta - self.b < 0:
                            new_X[i] = X[i]
                        else:
                            new_X[i] = var_x.value
                    return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}

                elif self.model_type in ['lad', 'logistic']:
                    for i in range(sample_size):
                        var_x = cp.Variable(self.input_dim)
                        var_y = cp.Variable()
                        obj = self._cvx_loss(var_x, y[i], self.theta, self.b) - norm_theta * self.distance_compute(var_x, X[i], var_y, y[i])
                        problem = cp.Problem(cp.Maximize(obj))
                        problem.solve(solver = self.solver)
                        new_X[i] = var_x.value
                    return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}

                    
            
        else:
            # in the following cases, we take gamma = 1 / sample_size since we want the asymptotic with respect to n
            if self.model_type in ['svm', 'logistic']:
            # Theorem 20 in https://jmlr.org/papers/volume20/17-633/17-633.pdf, where eta refers to theta in their equation, and eta_gamma refers to eta(\gamma)
                gamma = 1 / sample_size
                eta = cp.Variable(nonneg = True)
                alpha = cp.Variable(sample_size, nonneg = True)

                # svm / logistic L = 1
                dual_loss = self.lipschitz_norm() * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self._loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self._loss(X, -y))) / sample_size
                cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps - gamma]
                problem = cp.Problem(cp.Maximize(dual_loss), cons)
                problem.solve(solver = self.solver)
                eta_gamma = gamma / (eta.value + self.kappa - self.eps + gamma + 1)
                weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
                weight = np.hstack((weight, eta_gamma / sample_size))
                weight[0] = weight[0] * (1 - eta_gamma)
                weight[sample_size] = weight[sample_size] * (1 - eta_gamma)
                # solve the following perturbation problem
                X_star = cp.Variable(self.input_dim)
                cons = [cp.norm(sqrtm(self.cost_matrix) @ X_star, self.p) <= 1]
                problem = cp.Problem(cp.Maximize(X_star @ self.theta), cons)
                problem.solve(solver = self.solver)

                new_X = X[0] + X_star.value * sample_size * eta.value / eta_gamma
                new_y = y[0]

                X = np.concatenate((X, X))
                X = np.vstack((X, new_X))
                y = np.concatenate((y, -y))
                y = np.hstack((y, new_y))
                return {'sample_pts': [X, y], 'weight': weight}

            elif self.model_type == 'lad':
            # Theorem 9 in https://jmlr.org/papers/volume20/17-633/17-633.pdf
                gamma = 1 / sample_size
                weight = np.zeros(sample_size + 1)
                weight[1:-1] = np.ones(sample_size - 1) / sample_size
                weight[0] = (1 - gamma) / sample_size
                weight[-1] = gamma / sample_size
                # solve the following perturbation problem
                X_star = cp.Variable(self.input_dim)
                y_star = cp.Variable()
                cons = [cp.norm(sqrtm(self.cost_matrix) @ X_star, self.p) + self.kappa * cp.abs(y_star) <= 1]
                dual_loss = cp.dot(self.theta, X_star) - y_star
                problem = cp.Problem(cp.Maximize(dual_loss), cons)
                problem.solve(solver = self.solver)
                new_X = X[0] + self.eps * sample_size / gamma * X_star.value
                new_y = y[0] + self.eps * sample_size / gamma * y_star.value
                worst_X = np.vstack((X, new_X))
                worst_y = np.hstack((y, new_y))
                return {'sample_pts': [worst_X, worst_y], 'weight': weight}


class WassersteinDROSatisificingError(Exception):
    """Base exception class for errors in Wasserstein DRO (Robust Satisficing) model."""
    pass


class Wasserstein_DRO_satisficing(BaseLinearDRO):
    """
    Robust satisficing version of Wasserstein DRO

    This model minimizes the subject to (approximated version) of the robust satisficing constraint of Wasserstein DRO. The Wasserstein Distance is defined as the minimum probability coupling of two distributions for the distance metric: 
    d((X_1, Y_1), (X_2, Y_2)) = (|cost_matrix^{1/2} @ (X_1 - X_2)|_p)^{square} + kappa |Y_1 - Y_2|, 
        where parameters are:
        - cost matrix, (a PSD Matrix); 
        - kappa;
        - p;
        - square (notation depending on the model type), where square = 2 for 'svm', 'logistic', 'lad'; square = 1 for 'ols'.


    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator (e.g., 'svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression with L2-loss, 'lad' for Linear Regression with L1-loss).
        model_type (str, default = 'svm'): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression for OLS, 'lad' for Linear Regression for LAD).
        fit_intercept (bool, default = True): Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        solver (str, default = 'MOSEK'): Optimization solver to solve the problem, default = 'MOSEK'
        target ratio (float): target ratio (required to be >=1, against the empirical objective).
 
    Reference: <https://pubsonline.informs.org/doi/10.1287/opre.2021.2238>

    """

    def __init__(self, input_dim: int, model_type: str, fit_intercept: bool = True, solver: str = 'MOSEK'):
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.cost_matrix = np.eye(input_dim)
        self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        # target that the robust error need to achieve
        self.target_ratio = 1 / 0.8
        self.eps = 100
        self.p = 1
        self.kappa = 1

    def update(self, config: Dict[str, Any]) -> None:
        if 'cost_matrix' in config.keys():
            self.cost_matrix = config['cost_matrix']
            self.cost_inv_transform = np.linalg.inv(sqrtm(self.cost_matrix))
        if 'target_ratio' in config.keys():
            assert (config['target_ratio'] >= 1)
            self.target_ratio = config['target_ratio']
        # the following two are only used in SVM-wasserstein
        if 'p' in config.keys():
            self.p = config['p']
        if 'kappa' in config.keys():
            self.kappa = config['kappa']
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if self.p == 1:
            dual_norm = np.inf
        elif self.p != 'inf':
            dual_norm = 1 / (1 - 1 / self.p)
        else:
            dual_norm = 1

        sample_size, __ = X.shape
        empirical_rmse = self.fit_oracle(X, y)
        TGT = self.target_ratio * empirical_rmse
        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        cons = [TGT >= cp.sum(self._cvx_loss(X, y, theta, b)) * sample_size]
        if self.model_type == 'lad':
            obj = cp.norm(cp.hstack([theta, -1]), dual_norm)
        elif self.model_type in ['ols', 'svm', 'logistic']:
            obj = cp.norm(theta, dual_norm)

        problem = cp.Problem(cp.Minimize(obj), cons)
        problem.solve(solver = self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
    
        return model_params



    def fit_depreciate(self, X, y):
        """
        Find the best epsilon that matches the desired robust objective via bisection (depreciated)

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            Dict[str, Any]: Model parameters dictionary with 'theta' key.

        """

        warnings.warn("The bisection search is depreciated for Robust Satisficing Wasserstein DRO.")
        iter_num = 1
        # determine the empirical obj
        self.eps = 0
        empirical_rmse = self.fit_oracle(X, y)
        TGT = self.target_ratio * empirical_rmse
        # print('tgt', TGT)
        self.eps = 100
        assert (self.fit_oracle(X, y) > TGT)
        eps_lower, eps_upper = 0, self.eps      
        # binary search and find the maximum eps, such that RMSE + eps theta <= tau  
        for i in range(iter_num):
            self.eps = (eps_lower + eps_upper)/2
            if self.fit_oracle(X, y) > TGT:
                eps_upper = self.eps
            else:
                eps_lower = self.eps
        
        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params
    
    def penalization(self, theta: np.ndarray) -> float:
        """
        Module for computing the regularization part in the standard Wasserstein DRO problem.

        Args:
            theta (np.ndarray): Feature vector with shape (n_feature,).
        
        Returns:
            Float: Regularization term part.

        """
        if self.p == 1:
            dual_norm = np.inf
        else:
            dual_norm = 1 / (1 - 1 / self.p)
        if self.model_type == 'ols':
            return cp.norm(self.cost_inv_transform @ theta)
        elif self.model_type in ['svm', 'logistic']:
            return cp.norm(self.cost_inv_transform @ theta, dual_norm)
        elif self.model_type == 'lad':
            # the dual of the \|\theta, -1\|_dual_norm penalization
            return cp.max(cp.norm(self.cost_inv_transform @ theta, dual_norm), 1 / self.kappa)
            

    def fit_oracle(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Depreciated, find the optimal thet given the ambiguity constraint.

        Args:
            X (np.ndarray): Input feature matrix with shape (n_samples, n_features).
            y (np.ndarray): Target vector with shape (n_samples,).

        Returns:
            float: robust objective value

        """

        sample_size, feature_size = X.shape
        if feature_size != self.input_dim:
            raise WassersteinDROError(f"Expected input with {self.input_dim} features, got {feature_size}.")
        if sample_size != y.shape[0]:
            raise WassersteinDROError("Input X and target y must have the same number of samples.")


        theta = cp.Variable(self.input_dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0


        lamb_da = cp.Variable()
        cons = [lamb_da >= self.penalization(theta)]
        if self.model_type == 'ols':
            final_loss = cp.norm(X @ theta + b - y) / math.sqrt(sample_size) + math.sqrt(self.eps) * lamb_da

        else:
            if self.model_type in ['svm', 'logistic']:
                s = cp.Variable(sample_size)
                cons += [s >= self._cvx_loss(X, y, theta, b)]
                if self.kappa != 'inf':
                    cons += [s >= self._cvx_loss(X, -y, theta, b) - lamb_da * self.kappa]
                final_loss = cp.sum(s) / sample_size + self.eps * lamb_da
            else:
                # model type == 'lad' for general p.
                final_loss = cp.sum(self._cvx_loss(X, y, theta, b)) / sample_size + self.eps * lamb_da

        problem = cp.Problem(cp.Minimize(final_loss), cons)

        problem.solve(solver = self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b

        return problem.value
        
    
    def worst_distribution(self, X, y):
        warnings.warn("We do not compute worst case distribution for robust satisficing model since the distribution constraint is set to be held for any distribution.")


        # REQUIRED TO BE CALLED after solving the DRO problem
        # return a dict {"sample_pts": [np.array([pts_num, input_dim]), np.array(pts_num)], 'weight': np.array(pts_num)}

        # if self.is_regression == 1 or self.is_regression == 2:
        #     return NotImplementedError
        # else:
        #     sample_size, __ = X.shape
        #     if self.p == 1:
        #         dual_norm = np.inf
        #     else:
        #         dual_norm = 1 / (1 - 1 / self.p)
        #     norm_theta = np.linalg.norm(self.theta, ord = dual_norm)
        #     if self.kappa == 1000000:
        #     # not change y, we directly consider RMK 5.2 in https://arxiv.org/pdf/2308.05414.pdf, here norm_theta is lambda* there.
        #         new_X = np.zeros((sample_size, self.input_dim))
        #         for i in range(sample_size):
        #             var_x = cp.Variable(self.input_dim)
        #             obj = 1 - y[i] * var_x @ self.theta - norm_theta * cp.sum_squares(var_x - X[i])
        #             problem = cp.Problem(cp.Maximize(obj))
        #             problem.solve(solver = self.solver)
                    
        #             if 1 - y[i] * var_x.value @ self.theta < 0:
        #                 new_X[i] = X[i]
        #             else:
        #                 new_X[i] = var_x.value
        #         return {'sample_pts': [new_X, y], 'weight': np.ones(sample_size) / sample_size}
            
        #     else:
        #         # for general situations if we can change y, we apply Theorem 20 (ii) in https://jmlr.org/papers/volume20/17-633/17-633.pdf (SVM / logistic loss)
        #         #eta is the theta in eq(27)
        #         y_flip = -y
        #         eta = cp.Variable(nonneg = True)
        #         alpha = cp.Variable(sample_size, nonneg = True)

        #         # svm / logistic L = 1
        #         L = 1
        #         dual_loss = L * eta * norm_theta + cp.sum(cp.multiply(1 - alpha, self.loss(X, y))) / sample_size + cp.sum(cp.multiply(alpha, self.loss(X, y_flip))) / sample_size
        #         cons = [alpha <= 1, eta + self.kappa * cp.sum(alpha) / sample_size == self.eps]
        #         problem = cp.Problem(cp.Maximize(dual_loss), cons)
        #         problem.solve(solver = self.solver)
        #         weight = np.concatenate(((1 - alpha.value) / sample_size, alpha.value / sample_size))
        #         X = np.concatenate((X, X))
        #         y = np.concatenate((y, y_flip))
        #         return {'sample_pts': [X, y], 'weight': weight}

        






        

        
