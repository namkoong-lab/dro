from .base import BaseLinearDRO
import cvxpy as cp
from typing import Dict, Any
import numpy as np 

class MOTDROError(Exception):
    """Base exception class for errors in Wasserstein DRO model."""
    pass


## MOT DRO
class MOTDRO(BaseLinearDRO):
    """Optimal Transport DRO with Conditional Moment Constraints.

    Implements DRO with composite transportation cost:
    
    .. math::
        c\\big(((X_1,Y_1),w), ((X_2,Y_2),\\hat{w})\\big) = \\theta_1 w \\cdot d((X_1,Y_1),(X_2,Y_2)) 
         + \\theta_2 (\\phi(w) - \\phi(\\hat{w}))^+
        
    where :math:`\\phi(\\cdot)` is the KL divergence.

    :ivar theta1: Wasserstein distance scaling factor (:math:`\\theta_1 \\geq 0`)
    :vartype theta1: float
    :ivar theta2: KL divergence penalty coefficient (:math:`\\theta_2 \\geq 0`)
    :vartype theta2: float
    :ivar eps: Robustness radius for OT ambiguity set (:math:`\\epsilon \\geq 0`)
    :vartype eps: float
    :ivar p: Norm order for feature perturbation. Valid values: 2 (L2), 'inf' (L_inf)
    :vartype p: Union[float, str]

    .. _MOTDRO_Paper: https://arxiv.org/abs/2308.05414
    """

    def __init__(self, input_dim: int, model_type: str = 'svm', 
                 fit_intercept: bool = True, solver: str = 'MOSEK'):
        """Initialize MOT-DRO with restricted model types.

        :param input_dim: Dimension of input features. Must be ≥ 1.
        :type input_dim: int
        :param model_type: Base model type. Defaults to 'svm'. Supported:
            
            - ``'svm'``: Support Vector Machine (hinge loss)

            - ``'lad'``: Least Absolute Deviation (L1 loss)

        :type model_type: str
        :param fit_intercept: Whether to learn intercept term. 
            Set False for pre-centered data. Defaults to True.
        :type fit_intercept: bool
        :param solver: Convex optimization solver. Defaults to 'MOSEK'.
        :type solver: str

        :raises MOTDROError: 

            - If ``model_type`` in ['ols', 'logistic']
            
            - If ``input_dim`` < 1

        Default Parameters:

            - ``theta1``: 1.0 (minimum Wasserstein scaling)

            - ``square``: 2.0 (quadratic regularization strength)

        Example:
            >>> model = MOTDRO(input_dim=5, model_type='svm')
            >>> model.theta1 = 1.5  
            >>> model.square = 1.0  
        """
        if model_type in ['ols', 'logistic']:
            raise MOTDROError("MOT DRO does not support OLS, logistic")
        BaseLinearDRO.__init__(self, input_dim, model_type, fit_intercept, solver)
        self.theta1 = 1
        self.theta2 = 1
        self.eps = 0        
        self.p = 2
        self.square = 2

    def update(self, config: Dict[str, Any]) -> None:
        """Update MOTDRO model configuration with coupled parameters.

        :param config: Dictionary containing parameters to update. Valid keys:
            
            - ``theta1`` (float > 1): 
                Wasserstein distance scaling factor. Updates trigger:
                :math:`\\theta_2 = \\frac{1}{1 - 1/\\theta_1}`
            
            - ``theta2`` (float > 1): 
                KL divergence penalty coefficient. Updates trigger:
                :math:`\\theta_1 = \\frac{1}{1 - 1/\\theta_2}`
            
            - ``eps`` (float ≥ 0): 
                Robustness radius for OT ambiguity set
            
            - ``p`` (float ≥1 or 'inf'): 
                Perturbation norm order (1 ≤ p ≤ ∞)

        :raises MOTDROError: 
            - If ``theta1`` ≤ 1 or ``theta2`` ≤ 1
            - If ``theta1``/``theta2`` relationship becomes invalid
            - If ``eps`` < 0 or non-numeric
            - If ``p`` < 1 (when numeric) or not 'inf'
            - If parameter types are incorrect

        Parameter Coupling:
            - :math:`\\theta_1` and :math:`\\theta_2` are related through:
                :math:`\\theta_1 = \\frac{1}{1 - 1/\\theta_2}`
            - Updating one automatically adjusts the other

        Example:
            >>> model.update({
            ...     'theta1': 2.5,       
            ...     'p': 'inf',          
            ...     'eps': 0.3           
            ... })

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
        """Fit the MMD-DRO model to the data.
        
        :param X: Training feature matrix of shape `(n_samples, n_features)`.
            Must satisfy `n_features == self.input_dim`.
        :type X: numpy.ndarray

        :param y: Target values of shape `(n_samples,)`. Format requirements:

            - Classification: ±1 labels

            - Regression: Continuous values

        :type y: numpy.ndarray

        :returns: Dictionary containing trained parameters:
        
            - ``theta``: Weight vector of shape `(n_features,)`
        
        :rtype: Dict[str, Any]

        """
        if self.model_type in {'svm', 'logistic'}:
            is_valid = np.all((y == -1) | (y == 1))
            if not is_valid:
                raise MOTDROError("classification labels not in {-1, +1}")
            
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
        epig_ = cp.Variable([N_train])
        self.lambda_ = cp.Variable(1, pos = True)
        theta = cp.Variable(dim)
        if self.fit_intercept == True:
            b = cp.Variable()
        else:
            b = 0
        eta = cp.Variable([N_train], pos = True)

        # Constraints
        cons = []
        ones_vector = np.ones(N_train)
        exp_cones = [
            cp.constraints.exponential.ExpCone(
                epig_ - t, 
                self.theta2 * self.lambda_* ones_vector, 
                eta
            )
        ]
        cons.extend(exp_cones)

        if self.square == 1:
            cons.append(self.lambda_ * self.theta1 >= cp.norm(theta, q))
            loss_constraints = self._cvx_loss(X, y, theta, b) <= epig_
            cons.append(loss_constraints)  
        elif self.square == 2:
            reg_denominator = 4 * self.lambda_ * self.theta1
            quad_terms = cp.quad_over_lin(cp.norm(theta, q), reg_denominator)
            combined_loss = self._cvx_loss(X, y, theta, b) + quad_terms
            cons.append(combined_loss <= epig_)   

        # ## our version
        # for i in range(N_train):
        #     cons.append(
        #         cp.constraints.exponential.ExpCone(
        #             epig_[i] - t, self.theta2 * self.lambda_, eta[i]
        #         )
        #     )
        #     if self.square == 1:
        #         cons.append(self.lambda_ * self.theta1 >= cp.norm(self.theta, q))
        #         cons.append(self._cvx_loss(X[i], y[i], theta, b) <= epig_)  
        #     elif self.square == 2:
        #         cons.append(self._cvx_loss(X[i], y[i], theta, b) + cp.quad_over_lin(cp.norm(theta, q), self.lambda_*4*self.theta1) <= epig_[i]) 

        cons.append(N_train * self.lambda_ * self.theta2 >= cp.sum(eta))
        obj = self.eps * self.lambda_ + t
        problem = cp.Problem(cp.Minimize(obj), cons)

        problem.solve(solver=self.solver)
        self.theta = theta.value
        if self.fit_intercept == True:
            self.b = b.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        model_params["b"] = self.b
        return model_params