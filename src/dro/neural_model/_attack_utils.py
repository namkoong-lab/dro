"""Adapters for norm-ball attacks on image and tabular neural inputs."""

import torch
import torchattacks


class _TabularAttackModel(torch.nn.Module):
    """Present a tabular model with the four-dimensional input PGDL2 expects."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images.flatten(start_dim=1))


def make_norm_ball_attacks(model, method, norm, epsilon, step_size, steps):
    """Construct Torchattacks attacks for native and tabular input layouts."""
    if norm not in {'l2', 'l-inf'}:
        raise ValueError(f"Unsupported adversarial norm: {norm}")
    if method not in {'PGD', 'FGSM', 'FFGSM'}:
        raise ValueError(f"Unsupported adversarial method: {method}")
    if method != 'PGD' and norm != 'l-inf':
        raise ValueError(f"{method} requires adversarial_norm='l-inf'")

    # Torchattacks (Kim, 2020): https://github.com/Harry24k/adversarial-attacks-pytorch
    def construct(attack_model):
        if method == 'PGD':
            attack_cls = torchattacks.PGDL2 if norm == 'l2' else torchattacks.PGD
            return attack_cls(attack_model, eps=epsilon, alpha=step_size, steps=steps)
        if method == 'FGSM':
            return torchattacks.FGSM(attack_model, eps=epsilon)
        return torchattacks.FFGSM(attack_model, eps=epsilon, alpha=step_size)

    attack = construct(model)
    tabular_attack = construct(_TabularAttackModel(model)) if norm == 'l2' else None
    return attack, tabular_attack


def run_norm_ball_attack(attack, tabular_attack, inputs, labels):
    """Use a four-dimensional view for PGDL2 on a tabular input matrix."""
    if tabular_attack is not None and inputs.ndim == 2:
        # Keep the wrapper's mode aligned with the underlying model. Torchattacks
        # restores the wrapper's original mode after the attack.
        tabular_attack.model.train(attack.model.training)
        images = inputs.reshape(inputs.shape[0], 1, 1, inputs.shape[1])
        return tabular_attack(images, labels).reshape_as(inputs)
    return attack(inputs, labels)
