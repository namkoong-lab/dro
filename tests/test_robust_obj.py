"""Tests for the fitted robust empirical objective exposed by DRO models."""

from typing import Callable

import cvxpy as cp
import numpy as np
import pytest

import src.dro.tree_model.lgbm as lgbm_module
import src.dro.tree_model.xgb as xgb_module
from src.dro.linear_model.cvar_dro import CVaRDRO
from src.dro.neural_model.base_nn import BaseNNDRO


RAW_TREE_SCORES = np.array([-1.5, -0.5, 0.5, 1.5])
TREE_LABELS = np.array([0.0, 0.0, 1.0, 1.0])
TREE_X = np.arange(8, dtype=np.float32).reshape(4, 2)


def _tree_losses(raw_scores: np.ndarray, kind: str) -> np.ndarray:
    if kind == "regression":
        return (raw_scores - TREE_LABELS) ** 2

    probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
    return (
        -TREE_LABELS * np.log(probabilities + 1e-8)
        - (1 - TREE_LABELS) * np.log(1 - probabilities + 1e-8)
    )


def _expected_kl(losses: np.ndarray, epsilon: float) -> float:
    lambda_param = 1.0 / epsilon
    scaled_losses = losses / lambda_param
    maximum = np.max(scaled_losses)
    return float(
        lambda_param
        * (
            maximum
            + np.log(np.mean(np.exp(scaled_losses - maximum)))
        )
    )


def _expected_chi2(losses: np.ndarray, epsilon: float) -> float:
    centered_losses = losses - np.mean(losses)
    return float(
        np.mean(losses)
        + np.sqrt(epsilon / len(losses)) * np.linalg.norm(centered_losses)
    )


def _expected_cvar(losses: np.ndarray, epsilon: float) -> float:
    threshold = np.percentile(losses, epsilon * 100)
    return float(
        threshold
        + np.mean(np.maximum(losses - threshold, 0)) / (1 - epsilon)
    )


TREE_MODEL_CASES = [
    (lgbm_module, lgbm_module.KLDRO_LGBM, _expected_kl),
    (lgbm_module, lgbm_module.Chi2DRO_LGBM, _expected_chi2),
    (lgbm_module, lgbm_module.CVaRDRO_LGBM, _expected_cvar),
    (xgb_module, xgb_module.KLDRO_XGB, _expected_kl),
    (xgb_module, xgb_module.Chi2DRO_XGB, _expected_chi2),
    (xgb_module, xgb_module.CVaRDRO_XGB, _expected_cvar),
]


@pytest.mark.parametrize(
    "module,model_class,expected_objective",
    TREE_MODEL_CASES,
    ids=[
        "lgbm-kl",
        "lgbm-chi2",
        "lgbm-cvar",
        "xgb-kl",
        "xgb-chi2",
        "xgb-cvar",
    ],
)
@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_tree_models_store_expected_robust_objective(
    monkeypatch,
    module,
    model_class,
    expected_objective: Callable[[np.ndarray, float], float],
    kind: str,
):
    """All tree variants evaluate robust risk from fitted raw margins."""
    epsilon = 0.1
    model = model_class(eps=epsilon, kind=kind)
    assert model.robust_obj is None

    class FittedBooster:
        def predict(self, data, **kwargs):
            if module is lgbm_module:
                assert kwargs == {"raw_score": True}
            else:
                assert kwargs == {"output_margin": True}
            return RAW_TREE_SCORES.copy()

    fitted_booster = FittedBooster()
    if module is lgbm_module:
        monkeypatch.setattr(
            module.lightgbm,
            "train",
            lambda config, dtrain, num_boost_round: fitted_booster,
        )
    else:
        monkeypatch.setattr(
            module.xgb,
            "train",
            lambda config, dtrain, num_boost_round, obj: fitted_booster,
        )

    model.update({"num_boost_round": 1})
    model.fit(TREE_X, TREE_LABELS)

    expected = expected_objective(
        _tree_losses(RAW_TREE_SCORES, kind), epsilon
    )
    assert isinstance(model.robust_obj, float)
    assert model.robust_obj == pytest.approx(expected)


@pytest.fixture(scope="module")
def cvxpy_solver():
    """Select an installed conic solver for the exact linear objective test."""
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver in cp.installed_solvers():
            return solver
    pytest.skip("A conic CVXPY solver is required")


def test_exact_linear_model_stores_solved_objective(cvxpy_solver):
    """An exact linear model exposes the objective evaluated at its solution."""
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    y = np.array([0.0, 1.0, 1.0, 3.0])
    model = CVaRDRO(
        input_dim=2,
        model_type="ols",
        alpha=0.5,
        solver=cvxpy_solver,
    )

    assert model.robust_obj is None
    model.fit(X, y)

    fitted_losses = model._loss(X, y)
    expected = model.threshold_val + np.mean(
        np.maximum(fitted_losses - model.threshold_val, 0)
    ) / model.alpha
    assert isinstance(model.robust_obj, float)
    assert model.robust_obj == pytest.approx(expected, abs=1e-5)


class ConstantObjectiveNNDRO(BaseNNDRO):
    """Small neural model whose training objective has a known value."""

    def _criterion(self, outputs, labels):
        return outputs.sum() * 0 + 2.5


def test_neural_model_stores_final_epoch_objective():
    """Neural fitting records the robust criterion averaged over the epoch."""
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    model = ConstantObjectiveNNDRO(3, 2, model_type="linear")

    assert model.robust_obj is None
    model.fit(
        X,
        y,
        train_ratio=0.75,
        batch_size=3,
        epochs=2,
        verbose=False,
    )

    assert isinstance(model.robust_obj, float)
    assert model.robust_obj == pytest.approx(2.5)
