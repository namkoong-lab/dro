"""Post-training empirical radius certificates for neural DRO models."""

import math
from typing import Dict, Optional

import numpy as np
import torch


def _vector(values, name):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def _scalar(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _weights(values, count):
    weights = _vector(values, "candidate_weights")
    if weights.size != count or np.any(weights < 0):
        raise ValueError("candidate_weights must be nonnegative and match losses")
    total = math.fsum(weights)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("candidate_weights must sum to one")
    return weights / total


def wasserstein_radius(
    transport_budget, adversarial_losses, transport_costs,
    certified_inner_errors, transport_penalty,
) -> Optional[Dict[str, float]]:
    """Certify a squared-cost transport budget at the adversarial mean loss."""
    budget = _scalar(transport_budget, "transport_budget")
    penalty = _scalar(transport_penalty, "transport_penalty")
    if budget < 0 or penalty <= 0:
        raise ValueError("transport_budget must be nonnegative and penalty positive")
    losses = _vector(adversarial_losses, "adversarial_losses")
    costs = _vector(transport_costs, "transport_costs")
    errors = _vector(certified_inner_errors, "certified_inner_errors")
    if costs.size != losses.size or errors.size != losses.size:
        raise ValueError("losses, costs, and error bounds must have the same length")
    if np.any(costs < 0) or np.any(errors < 0):
        raise ValueError("transport costs and error bounds must be nonnegative")

    loss_bound = math.fsum(losses) / losses.size
    mean_cost = math.fsum(costs) / costs.size
    mean_error = math.fsum(errors) / errors.size
    if mean_cost > budget:
        raise ValueError("the adversarial coupling exceeds transport_budget")
    gap = penalty * (budget - mean_cost) + mean_error
    upper_bound = loss_bound + gap
    if not math.isfinite(upper_bound):
        raise ValueError("certificate values must be finite")
    radius = mean_cost - mean_error / penalty
    if radius < 0:
        return None
    return {
        "epsilon_cert": min(radius, budget),
        "loss_bound": loss_bound,
        "input_upper_bound": upper_bound,
        "gap": gap,
    }


def chi2_radius(epsilon, losses, candidate_weights, eta) -> Optional[Dict[str, float]]:
    """Invert the empirical chi-square upper bound at a feasible lower loss."""
    epsilon = _scalar(epsilon, "epsilon")
    eta = _scalar(eta, "eta")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    losses = _vector(losses, "losses")
    weights = _weights(candidate_weights, losses.size)
    divergence = math.fsum((losses.size * weights - 1.0) ** 2) / (2 * losses.size)
    if divergence > epsilon:
        raise ValueError("candidate_weights exceed the chi-square radius")

    loss_bound = math.fsum(weights * losses)
    scale = math.sqrt(math.fsum(np.maximum(losses - eta, 0.0) ** 2) / losses.size)
    root = math.sqrt(1 + 2 * epsilon)
    upper_bound = eta + root * scale
    gap = upper_bound - loss_bound
    if not all(math.isfinite(x) for x in (scale, upper_bound, gap)) or gap < 0:
        raise ValueError("candidate loss and upper bound are inconsistent")
    if scale == 0:
        radius = epsilon if gap == 0 else None
    elif gap > scale * (2 * epsilon / (root + 1)):
        radius = None
    else:
        adjusted_root = root - gap / scale
        radius = (adjusted_root - 1) * (adjusted_root + 1) / 2
        if radius < 0:
            radius = None
    if radius is None:
        return None
    return {
        "epsilon_cert": min(radius, epsilon),
        "loss_bound": loss_bound,
        "input_upper_bound": upper_bound,
        "gap": gap,
    }


def cvar_alpha(alpha, losses, candidate_weights, eta) -> Optional[Dict[str, float]]:
    """Invert the empirical CVaR upper bound at a feasible lower loss."""
    alpha = _scalar(alpha, "alpha")
    eta = _scalar(eta, "eta")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    losses = _vector(losses, "losses")
    weights = _weights(candidate_weights, losses.size)
    if np.any(weights > 1 / (alpha * losses.size)):
        raise ValueError("candidate_weights exceed the CVaR weight cap")

    loss_bound = math.fsum(weights * losses)
    scale = math.fsum(np.maximum(losses - eta, 0.0)) / losses.size
    upper_bound = eta + scale / alpha
    gap = upper_bound - loss_bound
    if not all(math.isfinite(x) for x in (scale, upper_bound, gap)) or gap < 0:
        raise ValueError("candidate loss and upper bound are inconsistent")
    if scale == 0:
        certified_alpha = alpha if gap == 0 else None
    elif gap > scale * (1 / alpha - 1):
        certified_alpha = None
    else:
        certified_alpha = 1 / (1 / alpha - gap / scale)
        if certified_alpha > 1:
            certified_alpha = None
    if certified_alpha is None:
        return None
    return {
        "alpha_cert": max(alpha, certified_alpha),
        "loss_bound": loss_bound,
        "input_upper_bound": upper_bound,
        "gap": gap,
    }
