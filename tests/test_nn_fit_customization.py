"""Integration tests for user-supplied neural DRO training components."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dro.neural_model import base_nn
from src.dro.neural_model.base_nn import DataValidationError
from src.dro.neural_model.fdro_nn import CVaRNNDRO, Chi2NNDRO
from src.dro.neural_model.hrdro_nn import HRNNDRO
from src.dro.neural_model.wdro_nn import WNNDRO


class TrackingLoader(DataLoader):
    """Record that fit iterates over the loaders passed by the caller."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def make_loaders(batch_size=2):
    train_x = torch.tensor(
        [[0.1, 0.9], [0.8, 0.2], [-0.4, 0.7], [0.7, -0.3]],
        dtype=torch.float32,
    )
    train_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    val_x = torch.tensor([[0.2, 0.6], [0.6, 0.2]], dtype=torch.float32)
    val_y = torch.tensor([0, 1], dtype=torch.long)
    train_loader = TrackingLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    val_loader = TrackingLoader(TensorDataset(val_x, val_y), batch_size=2)
    return train_loader, val_loader


def replace_network(estimator):
    torch.manual_seed(7)
    network = torch.nn.Linear(2, 2)
    estimator.update_user_mode(
        input_dim=2, num_classes=2, model=network,
        task_type="classification", device=torch.device("cpu"),
    )
    return network


@pytest.mark.parametrize("estimator_class", [Chi2NNDRO, CVaRNNDRO])
@pytest.mark.parametrize(
    ("scheduler_step", "expected_lr"),
    [("epoch", 0.025), ("batch", 0.00625)],
)
def test_fdro_fit_uses_supplied_loaders_sgd_and_scheduler(
    estimator_class, scheduler_step, expected_lr
):
    train_loader, val_loader = make_loaders()
    estimator = estimator_class(
        input_dim=2, num_classes=2, model_type="linear", size=0.2, reg=0.1
    )
    network = replace_network(estimator)
    initial_weight = network.weight.detach().clone()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    metrics = estimator.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step=scheduler_step,
        epochs=2,
        verbose=False,
    )

    assert train_loader.iterations == 2
    assert val_loader.iterations == 1
    assert not torch.allclose(network.weight.detach(), initial_weight)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert np.isfinite(estimator.robust_obj)
    assert set(metrics) == {"acc", "f1"}


@pytest.mark.parametrize("estimator_class", [Chi2NNDRO, CVaRNNDRO])
def test_fdro_fit_accepts_singleton_final_training_batch(estimator_class):
    train_loader, val_loader = make_loaders(batch_size=3)
    estimator = estimator_class(
        input_dim=2, num_classes=2, model_type="linear", size=0.2, reg=0.1
    )
    network = replace_network(estimator)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    metrics = estimator.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=1,
        verbose=False,
    )

    assert train_loader.iterations == 1
    assert val_loader.iterations == 1
    assert np.isfinite(estimator.robust_obj)
    assert set(metrics) == {"acc", "f1"}


def test_hrdro_fit_accepts_supplied_training_components():
    # Disable adversarial examples to isolate HR-DRO's fitting path; its
    # CVXPY objective still runs for the caller's loader and network.
    # The final batch is shorter, so HR-DRO must resize its CVXPY problem.
    train_loader, val_loader = make_loaders(batch_size=3)
    estimator = HRNNDRO(
        input_dim=2, num_classes=2, model_type="linear",
        epsilon=0.0, train_batch_size=3, device=torch.device("cpu"),
    )
    network = replace_network(estimator)
    initial_weight = network.weight.detach().clone()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    with pytest.warns(UserWarning, match="Batch size changed"):
        metrics = estimator.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            verbose=False,
        )

    assert train_loader.iterations == 1
    assert val_loader.iterations == 1
    assert not torch.allclose(network.weight.detach(), initial_weight)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert estimator.train_batch_size == 1
    assert np.isfinite(estimator.robust_obj)
    assert set(metrics) == {"acc", "f1"}


def test_plateau_scheduler_receives_epoch_training_loss():
    class RecordingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, optimizer):
            self.seen_losses = []
            super().__init__(optimizer)

        def step(self, metrics, epoch=None):
            self.seen_losses.append(float(metrics))
            super().step(metrics, epoch=epoch)

    train_loader, val_loader = make_loaders()
    estimator = Chi2NNDRO(input_dim=2, num_classes=2, model_type="linear")
    network = replace_network(estimator)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    scheduler = RecordingPlateau(optimizer)

    estimator.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=1,
        verbose=False,
    )

    assert scheduler.seen_losses == pytest.approx([estimator.robust_obj])


def test_regression_fit_uses_supplied_loaders_and_optimizer():
    train_x = torch.tensor(
        [[0.1, 0.9], [0.8, 0.2], [-0.4, 0.7], [0.7, -0.3]],
        dtype=torch.float32,
    )
    train_y = torch.tensor([0.2, 0.7, -0.3, 0.6], dtype=torch.float32)
    val_x = torch.tensor([[0.2, 0.6], [0.6, 0.2]], dtype=torch.float32)
    val_y = torch.tensor([0.3, 0.5], dtype=torch.float32)
    train_loader = TrackingLoader(TensorDataset(train_x, train_y), batch_size=2)
    val_loader = TrackingLoader(TensorDataset(val_x, val_y), batch_size=2)
    estimator = Chi2NNDRO(
        input_dim=2, num_classes=1, task_type="regression", model_type="linear"
    )
    network = torch.nn.Linear(2, 1)
    estimator.update_user_mode(
        input_dim=2, num_classes=1, model=network,
        task_type="regression", device=torch.device("cpu"),
    )
    initial_weight = network.weight.detach().clone()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    metrics = estimator.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=1,
        verbose=False,
    )

    assert train_loader.iterations == 1
    assert val_loader.iterations == 1
    assert not torch.allclose(network.weight.detach(), initial_weight)
    assert np.isfinite(estimator.robust_obj)
    with torch.no_grad():
        expected_mse = torch.mean((network(val_x).squeeze(-1) - val_y) ** 2).item()
    assert metrics["mse"] == pytest.approx(expected_mse)


def test_fit_requires_loader_pair():
    estimator = Chi2NNDRO(input_dim=2, num_classes=2, model_type="linear")
    train_loader, _ = make_loaders()

    with pytest.raises(ValueError, match="val_loader|both"):
        estimator.fit(train_loader=train_loader, epochs=1, verbose=False)


def test_fit_rejects_optimizer_for_replaced_network():
    estimator = CVaRNNDRO(input_dim=2, num_classes=2, model_type="linear")
    stale_optimizer = torch.optim.SGD(estimator.model.parameters(), lr=0.1)
    replace_network(estimator)
    train_loader, val_loader = make_loaders()

    with pytest.raises(ValueError, match="optimizer|model|parameter"):
        estimator.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=stale_optimizer,
            epochs=1,
            verbose=False,
        )


def test_custom_wnndro_model_does_not_inherit_preset_image_shape_check(monkeypatch):
    class StubResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            return self.fc(inputs)

    monkeypatch.setattr(base_nn.models, "resnet18", lambda weights: StubResNet())
    estimator = WNNDRO(
        input_dim=2, num_classes=2, model_type="resnet", epsilon=0.0,
        device=torch.device("cpu"),
    )
    features = torch.tensor(
        [[0.1, 0.2], [0.2, 0.3], [0.7, 0.6], [0.8, 0.9]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    with pytest.raises(DataValidationError, match="Image models require 4D input"):
        estimator._validate_input_shape(features)

    estimator.update_user_mode(2, 2, torch.nn.Linear(2, 2))
    metrics = estimator.fit(features, labels, epochs=1, batch_size=2, verbose=False)

    assert set(metrics) == {"acc", "f1"}
