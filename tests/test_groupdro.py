"""Tests for the linear and neural Group DRO implementations."""

import cvxpy as cp
import numpy as np
import pytest
import torch

from src.dro.linear_model.base import DataValidationError as LinearDataError
from src.dro.linear_model.base import ParameterError as LinearParameterError
from src.dro.linear_model.group_dro import GroupDRO
from src.dro.neural_model.base_nn import DataValidationError as NeuralDataError
from src.dro.neural_model.base_nn import ParameterError as NeuralParameterError
from src.dro.neural_model.groupdro_nn import GroupNNDRO


@pytest.fixture(scope="module")
def cvxpy_solver():
    """Use an installed open-source solver so the tests do not require MOSEK."""
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver in cp.installed_solvers():
            return solver
    pytest.skip("GroupDRO tests require CLARABEL, ECOS, or SCS")


def test_linear_groupdro_minimizes_worst_group_loss(cvxpy_solver):
    """The fitted robust loss is the largest empirical group mean loss."""
    # The first feature is deliberately constant. The categorical second
    # feature lets the linear model fit a separate mean for each group.
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    y = np.array([0.0, 2.0, 0.0, 10.0, 14.0, 10.0])

    model = GroupDRO(
        input_dim=2,
        group_idx=1,
        model_type="ols",
        solver=cvxpy_solver,
    )
    result = model.fit(X, y)

    assert result["group_values"] == [0.0, 1.0]
    assert result["group_losses"][0] <= result["robust_loss"]
    assert result["group_losses"][1] == pytest.approx(32 / 9, abs=1e-4)
    assert result["robust_loss"] == pytest.approx(32 / 9, abs=1e-4)
    assert result["robust_loss"] == pytest.approx(max(result["group_losses"]))
    assert model.robust_obj == pytest.approx(result["robust_loss"])
    assert model.predict(X).shape == (len(X),)


def test_linear_groupdro_classification_and_update(cvxpy_solver):
    """Classification retains the common fit, predict, score, and update API."""
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [-2.0, 1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
    model = GroupDRO(2, group_idx=0, model_type="svm", solver=cvxpy_solver)

    model.update({"group_idx": 1})
    result = model.fit(X, y)
    accuracy, f1 = model.score(X, y)

    assert model.group_idx == 1
    assert len(result["group_losses"]) == 2
    assert set(np.unique(model.predict(X))).issubset({-1, 1})
    assert 0 <= accuracy <= 1
    assert 0 <= f1 <= 1


def test_linear_groupdro_validates_group_feature(cvxpy_solver):
    """Invalid indices, categories, and classification labels fail clearly."""
    with pytest.raises(LinearParameterError, match="group_idx"):
        GroupDRO(2, group_idx=2, solver=cvxpy_solver)

    model = GroupDRO(2, group_idx=1, solver=cvxpy_solver)
    with pytest.raises(LinearDataError, match="finite"):
        model.fit(np.array([[0.0, 0.0], [1.0, np.nan]]), np.array([-1, 1]))
    with pytest.raises(LinearDataError, match=r"\{-1, \+1\}"):
        model.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))


def test_neural_groupdro_moves_weight_to_high_loss_group():
    """Exponentiated ascent increases the higher-loss group's probability."""
    model = GroupNNDRO(
        input_dim=2,
        num_classes=2,
        group_idx=1,
        model_type="linear",
        step_size=1.0,
    )
    model.group_values_ = np.array([0.0, 1.0], dtype=np.float32)
    model._group_values_tensor = torch.tensor([0.0, 1.0])
    model.adv_probs = torch.tensor([0.5, 0.5])
    model.current_inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )

    # Group 0 is confidently correct and group 1 is confidently incorrect.
    outputs = torch.tensor(
        [[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = model._criterion(outputs, labels)
    loss.backward()

    assert model.adv_probs[1] > model.adv_probs[0]
    assert model.adv_probs.sum().item() == pytest.approx(1.0)
    assert outputs.grad is not None


def test_neural_groupdro_classification_fit_uses_common_api():
    """A tabular classification fit discovers groups and returns base metrics."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(48, 4)).astype(np.float32)
    X[:24, 3] = 0
    X[24:, 3] = 1
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int64)
    torch.manual_seed(7)

    model = GroupNNDRO(4, 2, group_idx=3, model_type="linear", step_size=0.1)
    metrics = model.fit(
        X,
        y,
        train_ratio=0.75,
        lr=1e-2,
        batch_size=12,
        epochs=1,
        verbose=False,
    )

    assert set(metrics) == {"acc", "f1"}
    assert model.group_values_.tolist() == [0.0, 1.0]
    assert model.group_weights_.sum() == pytest.approx(1.0)
    assert isinstance(model.robust_obj, float)
    assert np.isfinite(model.robust_obj)
    assert model.predict(X[:5]).shape == (5,)
    accuracy, f1 = model.score(X, y)
    assert 0 <= accuracy <= 1
    assert 0 <= f1 <= 1


def test_neural_groupdro_regression_fit_predict_and_score():
    """Regression uses raw network outputs for validation and prediction."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(48, 3)).astype(np.float32)
    X[:24, 2] = 0
    X[24:, 2] = 1
    y = (1.5 * X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)
    torch.manual_seed(11)

    model = GroupNNDRO(
        3,
        1,
        group_idx=2,
        task_type="regression",
        model_type="linear",
    )
    metrics = model.fit(
        X,
        y,
        train_ratio=0.75,
        batch_size=12,
        epochs=1,
        verbose=False,
    )

    assert isinstance(metrics["mse"], float)
    assert model.predict(X[:5]).shape == (5,)
    assert isinstance(model.score(X, y), float)
    assert model.group_weights_.sum() == pytest.approx(1.0)


def test_neural_groupdro_validates_configuration_and_data():
    """Neural Group DRO rejects invalid optimization and group definitions."""
    with pytest.raises(NeuralParameterError, match="group_idx"):
        GroupNNDRO(2, 2, group_idx=-1)
    with pytest.raises(NeuralParameterError, match="step_size"):
        GroupNNDRO(2, 2, group_idx=1, step_size=0)
    with pytest.raises(NeuralParameterError, match="tabular feature"):
        GroupNNDRO(2, 2, group_idx=1, model_type="resnet")

    model = GroupNNDRO(2, 2, group_idx=1, model_type="linear")
    model.update({"group_idx": 0, "step_size": 0.2})
    assert model.group_idx == 0
    assert model.step_size == pytest.approx(0.2)

    with pytest.raises(NeuralDataError, match="finite"):
        model.fit(
            np.array([[np.nan, 0.0], [1.0, 1.0]]),
            np.array([0, 1]),
            epochs=1,
            verbose=False,
        )
    with pytest.raises(NeuralDataError, match="Classification labels"):
        model.fit(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([0, 2]),
            epochs=1,
            verbose=False,
        )
