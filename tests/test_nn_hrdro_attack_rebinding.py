"""Regression tests for attacks after replacing an HR-DRO network."""

import torch
import torchattacks

from src.dro.neural_model._attack_utils import run_norm_ball_attack
from src.dro.neural_model.hrdro_nn import HRNNDRO


def test_update_user_mode_rebinds_main_and_tabular_pgdl2_attacks():
    torch.manual_seed(7)
    estimator = HRNNDRO(
        input_dim=2,
        num_classes=2,
        model_type="linear",
        device=torch.device("cpu"),
        train_batch_size=2,
        adversarial_params={
            "method": "PGD", "norm": "l2", "steps": 2, "step_size": 0.05,
        },
    )
    old_attack = estimator.attack
    old_tabular_attack = estimator._tabular_attack
    replacement = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 2))

    estimator.update_user_mode(
        input_dim=2,
        num_classes=2,
        model=replacement,
        device=torch.device("cpu"),
    )

    assert estimator.model is replacement
    assert estimator.attack is not old_attack
    assert estimator._tabular_attack is not old_tabular_attack
    assert isinstance(estimator.attack, torchattacks.PGDL2)
    assert isinstance(estimator._tabular_attack, torchattacks.PGDL2)
    assert estimator.attack.model is replacement
    assert estimator._tabular_attack.model.model is replacement

    inputs = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    image_adversarial = estimator.attack(inputs.reshape(2, 1, 1, 2), labels)
    tabular_adversarial = run_norm_ball_attack(
        estimator.attack, estimator._tabular_attack, inputs, labels
    )

    assert image_adversarial.shape == (2, 1, 1, 2)
    assert tabular_adversarial.shape == inputs.shape
    for adversarial in (image_adversarial.reshape_as(inputs), tabular_adversarial):
        assert torch.isfinite(adversarial).all()
        assert (adversarial - inputs).norm(p=2, dim=1).max() <= estimator.epsilon + 1e-6
