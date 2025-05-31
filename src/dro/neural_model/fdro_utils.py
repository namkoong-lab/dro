import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize as sopt
from typing import Callable
import warnings

# Constants
GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5
BISECTION_TOL = 1e-2
MAX_BISECTION_ITER = 500

class DROError(Exception):
    """Base class for all distributionally robust optimization errors."""
    pass

class ParameterError(DROError):
    """Raised for invalid parameter configurations."""
    pass

class NumericalError(DROError):
    """Raised for numerical computation failures."""
    pass

def chi_square_value(p: torch.Tensor, v: torch.Tensor, reg: float) -> torch.Tensor:
    """Compute χ²-regularized value: ⟨p,v⟩ - reg * χ²(p, uniform)."""
    m = p.shape[0]
    chi2 = (0.5 / m) * reg * torch.norm(m * p - torch.ones_like(p), p=2)**2
    return torch.dot(p.squeeze(), v.squeeze()) - chi2

def cvar_value(p: torch.Tensor, v: torch.Tensor, reg: float) -> torch.Tensor:
    """Compute CVaR-regularized value: ⟨p,v⟩ - reg * KL(p, uniform)."""
    m = p.shape[0]
    kl = torch.log(torch.tensor(m)) + (p * torch.log(p + 1e-10)).sum()
    return torch.dot(p, v) - reg * kl

def bisection(
    eta_min: float,
    eta_max: float,
    f: Callable[[float], float],
    tol: float = BISECTION_TOL,
    max_iter: int = MAX_BISECTION_ITER
) -> float:
    """Robust bisection method with dynamic interval expansion."""
    # Validate input parameters
    if eta_min >= eta_max:
        raise ParameterError(f"Invalid interval: [{eta_min}, {eta_max}]")

    # Dynamic interval expansion
    lower, upper = f(eta_min), f(eta_max)
    while lower > 0 or upper < 0:
        interval = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min -= 2 * interval
        else:
            eta_min = eta_max
            eta_max += 2 * interval
        lower, upper = f(eta_min), f(eta_max)

    # Bisection iterations
    for _ in range(max_iter):
        eta = (eta_min + eta_max) / 2
        val = f(eta)
        
        if abs(val) <= tol:
            return eta
            
        if val > 0:
            eta_max = eta
        else:
            eta_min = eta

    # Return best approximation if not converged
    warnings.warn(f"Bisection not converge into {tol}", RuntimeWarning)
    return (eta_min + eta_max) / 2

class RobustLoss(nn.Module):
    """Distributionally Robust Loss with Configurable Geometry.
    
    Features:
    - Supports both CVaR and χ² divergence geometries
    - Automatic handling of regularization parameters
    - Batch-wise robust optimization
    
    Args:
        size: Uncertainty set size (α for CVaR, ρ for χ²)
        reg: Regularization strength (λ ≥ 0)
        geometry: Divergence type ('cvar' or 'chi-square')
        tol: Numerical tolerance for optimization
        max_iter: Maximum iterations for bisection
        is_regression: Flag for regression tasks
    """
    
    def __init__(
        self,
        size: float,
        reg: float,
        geometry: str,
        tol: float = BISECTION_TOL,
        max_iter: int = MAX_BISECTION_ITER,
        is_regression: bool = False
    ):
        super().__init__()
        self.size = size
        self.reg = reg

        if geometry is None:
            raise ParameterError(f"Unsupported geometry: {geometry}")

        self.geometry = geometry.lower()
        self.tol = tol
        self.max_iter = max_iter
        self.is_regression = is_regression

        self._validate_params()

    def _validate_params(self):
        """Validate initialization parameters."""
        if self.geometry not in GEOMETRIES:
            raise ParameterError(f"Unsupported geometry: {self.geometry}")
            
        if self.geometry == 'cvar' and self.size > 1:
            raise ParameterError(f"CVaR α must be ≤ 1, got {self.size}")
            
        if self.reg < 0 or self.size <= 0:
            raise ParameterError(f"Regularization must be ≥ 0, got {self.reg}")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute robust loss value.
        
        Args:
            outputs: Model predictions (logits for classification)
            targets: Ground truth values
            
        Returns:
            Robust loss value (scalar tensor)
        """
        individual_loss = self._compute_individual_loss(outputs, targets)
        return self._compute_robust_loss(individual_loss)

    def _compute_individual_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-sample losses."""
        if self.is_regression:
            return (outputs.squeeze() - targets.squeeze()).pow(2)
        return F.cross_entropy(outputs, targets.long(), reduction='none')

    def _compute_robust_loss(self, v: torch.Tensor) -> torch.Tensor:
        """Core robust loss computation."""
        if self.size == 0:  # ERM fallback
            return v.mean()

        with torch.no_grad():
            p = self._compute_optimal_weights(v)

        if self.geometry == 'cvar':
            return cvar_value(p, v, self.reg)
        return chi_square_value(p, v, self.reg)

    def _compute_optimal_weights(self, v: torch.Tensor) -> torch.Tensor:
        """Compute optimal weights based on specified geometry."""
        if self.geometry == 'cvar':
            return self._cvar_weights(v)
        return self._chisquare_weights(v)

    def _cvar_weights(self, v: torch.Tensor) -> torch.Tensor:
        """Compute CVaR-optimal weights."""
        if self.reg == 0:
            return self._hard_cvar(v)
        return self._soft_cvar(v)

    def _soft_cvar(self, v: torch.Tensor) -> torch.Tensor:
        """Regularized CVaR weights computation."""
        m = v.shape[0]
        eta_min = self.reg * torch.logsumexp(v/self.reg - np.log(m), 0)
        eta_max = v.max()

        def f(eta):
            p = torch.min(torch.exp((v - eta)/self.reg), 
                         torch.tensor(1/self.size, device=v.device)) / m
            return 1.0 - p.sum()

        eta_star = bisection(eta_min.item(), eta_max.item(), f, self.tol, self.max_iter)
        return torch.min(torch.exp((v - eta_star)/self.reg), 
                        torch.tensor(1/self.size, device=v.device)) / m

    def _hard_cvar(self, v: torch.Tensor) -> torch.Tensor:
        """Hard thresholding CVaR weights."""
        m = v.shape[0]
        cutoff = int(self.size * m)
        surplus = 1.0 - cutoff / (self.size * m)

        p = torch.zeros_like(v)
        idx = torch.argsort(v, descending=True)
        p[idx[:cutoff]] = 1.0 / (self.size * m)
        
        if cutoff < m:
            p[idx[cutoff]] = surplus
            
        return p

    def _chisquare_weights(self, v: torch.Tensor) -> torch.Tensor:
        """χ²-optimal weights computation."""
        m = v.shape[0]
        
        # Handle numerical edge cases
        if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
            return torch.ones_like(v) / m

        # if self.size == float('inf'):
        #     return self._unconstrained_chisquare(v, m)
        return self._constrained_chisquare(v, m)

    # def _unconstrained_chisquare(self, v: torch.Tensor, m: int) -> torch.Tensor:
    #     """Unconstrained χ² weights."""
    #     def f(eta):
    #         p = torch.relu(v - eta) / (self.reg * m)
    #         return 1.0 - p.sum()

    #     print(v, m)
    #     eta_min = (v.sum() - self.reg * m).item()
    #     eta_max = v.max().item()
    #     eta_star = bisection(eta_min, eta_max, f, self.tol, self.max_iter)
    #     return torch.relu(v - eta_star) / (self.reg * m)

    def _constrained_chisquare(self, v: torch.Tensor, m: int) -> torch.Tensor:
        """Constrained χ² weights."""
        if m <= 1 + 2 * self.size:
            return (v == v.max()).float() / (v == v.max()).sum()

        if self.reg == 0:
            return self._unregularized_constrained_chisquare(v, m)
        return self._regularized_constrained_chisquare(v, m)

    def _unregularized_constrained_chisquare(self, v: torch.Tensor, m: int) -> torch.Tensor:
        """Unregularized constrained χ² weights."""
        def f(eta):
            p = torch.relu(v - eta)
            if p.sum() == 0:
                return float('inf')
            return 0.5 * torch.norm(m * p/p.sum() - 1)**2 / m - self.size

        eta_min = -(1.0 / (np.sqrt(2*self.size + 1) - 1)) * v.max().item()
        eta_max = v.max().item()
        eta_star = bisection(eta_min, eta_max, f, self.tol, self.max_iter)
        p = torch.relu(v - eta_star)
        return p / p.sum()

    def _regularized_constrained_chisquare(self, v: torch.Tensor, m: int) -> torch.Tensor:
        """Regularized constrained χ² weights."""
        def f(eta):
            p = torch.relu(v - eta)
            opt_lam = max(self.reg, torch.norm(p) / np.sqrt(m*(1+2*self.size)))
            return 1 - (p / (m * opt_lam)).sum()

        eta_min = v.min().item() - 1
        eta_max = v.max().item()
        eta_star = bisection(eta_min, eta_max, f, self.tol, self.max_iter)
        p = torch.relu(v - eta_star)
        opt_lam = max(self.reg, torch.norm(p) / np.sqrt(m*(1+2*self.size)))
        return p / (m * opt_lam)

def chi_square_doro(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    eps: float
) -> torch.Tensor:
    """χ²-DORO loss implementation.
    
    Args:
        outputs: Model predictions
        targets: Ground truth labels
        alpha: Confidence level parameter
        eps: Contamination ratio
        
    Returns:
        Robust loss value
    """
    batch_size = targets.shape[0]
    loss = F.cross_entropy(outputs, targets.long(), reduction='none')
    
    C = math.sqrt(1 + (1/alpha - 1)**2)
    n = int(eps * batch_size)
    sorted_idx = torch.argsort(loss, descending=True)
    filtered_loss = loss[sorted_idx[n:]]
    
    def optimization_fn(eta: float) -> float:
        return C * torch.sqrt(F.relu(filtered_loss - eta).pow(2).mean()).item() + eta
        
    opt_eta = sopt.brent(optimization_fn, brack=(0, loss.max().item()), tol=1e-2)
    return C * torch.sqrt(F.relu(filtered_loss - opt_eta).pow(2).mean()) + opt_eta

def cvar_doro(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    alpha: float
) -> torch.Tensor:
    """CVaR-DORO loss implementation.
    
    Args:
        outputs: Model predictions
        targets: Ground truth labels
        eps: Contamination ratio
        alpha: Confidence level parameter
        
    Returns:
        Robust loss value
    """
    batch_size = targets.shape[0]
    loss = F.cross_entropy(outputs, targets.long(), reduction='none')
    
    gamma = eps + alpha * (1 - eps)
    n1, n2 = int(gamma * batch_size), int(eps * batch_size)
    sorted_idx = torch.argsort(loss, descending=True)
    
    return loss[sorted_idx[n2:n1]].sum() / (alpha * (batch_size - n2))