from dro.src.linear_model.base import BaseLinearDRO
import cvxpy as cp
from typing import Dict, Any

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

    def __init__(self, input_dim: int, model_type: str):
        """Initialize MOTDRO with composite transportation cost.

        :param input_dim: Dimension of input features. Must be ≥ 1.
        :type input_dim: int
        :param model_type: Base model type. Supported:
            
            - ``'svm'``: Support Vector Machine (hinge loss)

            - ``'logistic'``: Logistic Regression (log loss)

            - ``'ols'``: Ordinary Least Squares (L2 loss)

            - ``'lad'``: Least Absolute Deviation (L1 loss)

        :type model_type: str
        :raises ValueError: 

            - If ``input_dim`` < 1

            - If ``model_type`` not in allowed set

        Default Parameters:
            - ``theta1``: 2.0 (Wasserstein scaling)
            - ``theta2``: 2.0 (KL penalty)
            - ``eps``: 0.0 (non-robust baseline)
            - ``p``: 2 (L2 norm perturbation)

        Example:
            >>> model = MOTDRO(input_dim=10, model_type='logistic')
            >>> model.theta1 = 1.5 
            >>> model.p = 'inf'    

        """
        super().__init__(input_dim, model_type)
        
        self.theta1 = 2.0    
        self.theta2 = 2.0    
        self.eps = 0.0       
        self.p = 2           

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
        # TODO: double check
        cons.append(self.lambda_ * self.theta1 >= cp.norm(self.theta, q))
        cons.append(N_train * self.lambda_ * self.theta2 >= cp.sum(eta))
        obj = self.eps * self.lambda_ + t
        problem = cp.Problem(cp.Minimize(obj), cons)

        problem.solve(solver=self.solver)
        self.theta = self.theta.value

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params