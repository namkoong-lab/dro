import unittest

import numpy as np
import torch
import torchattacks

from src.dro.neural_model._attack_utils import run_norm_ball_attack
from src.dro.neural_model.hrdro_nn import HRNNDRO
from src.dro.neural_model.wdro_nn import WNNDRO


class TestNormBallAttacks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.tensor(
            [[0.2, 0.3], [0.3, 0.2], [0.7, 0.8], [0.8, 0.7]],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 0, 1, 1])

    def make_model(self, norm, method):
        return WNNDRO(
            input_dim=2,
            num_classes=2,
            model_type="linear",
            device=torch.device("cpu"),
            epsilon=0.1,
            adversarial_norm=norm,
            adversarial_method=method,
            adversarial_steps=2,
            adversarial_step_size=0.05,
        )

    def test_fgsm_dispatches_to_single_step_attack(self):
        model = self.make_model("l-inf", "FGSM")
        self.assertIsInstance(model.attack, torchattacks.FGSM)

        with torch.enable_grad():
            clean = self.inputs.clone().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(model.model(clean), self.labels)
            gradient = torch.autograd.grad(loss, clean)[0]
        expected = (self.inputs + model.epsilon * gradient.sign()).clamp(0, 1)
        actual = run_norm_ball_attack(
            model.attack, model._tabular_attack, self.inputs, self.labels
        )
        torch.testing.assert_close(actual, expected)
        self.assertLessEqual((actual - self.inputs).abs().max().item(), model.epsilon + 1e-6)

    def test_pgdl2_trains_with_two_dimensional_tabular_inputs(self):
        model = self.make_model("l2", "PGD")
        self.assertIsInstance(model.attack, torchattacks.PGDL2)
        adversarial = run_norm_ball_attack(
            model.attack, model._tabular_attack, self.inputs, self.labels
        )
        self.assertEqual(adversarial.shape, self.inputs.shape)
        self.assertLessEqual(
            (adversarial - self.inputs).norm(p=2, dim=1).max().item(),
            model.epsilon + 1e-6,
        )
        self.assertTrue(((adversarial >= 0) & (adversarial <= 1)).all())

        metrics = model.fit(
            self.inputs.numpy(), self.labels.numpy(), train_ratio=0.75,
            batch_size=2, epochs=1, verbose=False,
        )
        self.assertTrue(np.isfinite(model.robust_obj))
        self.assertTrue(np.isfinite(metrics["acc"]))

    def test_fgsm_trains_with_two_dimensional_tabular_inputs(self):
        model = self.make_model("l-inf", "FGSM")
        model.fit(
            self.inputs.numpy(), self.labels.numpy(), train_ratio=0.75,
            batch_size=2, epochs=1, verbose=False,
        )
        self.assertTrue(np.isfinite(model.robust_obj))

    def test_ffgsm_uses_its_own_constructor(self):
        model = self.make_model("l-inf", "FFGSM")
        self.assertIsInstance(model.attack, torchattacks.FFGSM)
        adversarial = run_norm_ball_attack(
            model.attack, model._tabular_attack, self.inputs, self.labels
        )
        self.assertEqual(adversarial.shape, self.inputs.shape)
        self.assertLessEqual((adversarial - self.inputs).abs().max().item(), model.epsilon + 1e-6)

    def test_tabular_attack_restores_eval_mode(self):
        model = self.make_model("l2", "PGD")
        model.model.eval()
        run_norm_ball_attack(
            model.attack, model._tabular_attack, self.inputs, self.labels
        )
        self.assertFalse(model.model.training)

    def test_custom_model_rebinds_norm_ball_attacks(self):
        for norm, method in (
            ("l-inf", "FGSM"), ("l-inf", "PGD"), ("l2", "PGD"),
        ):
            with self.subTest(norm=norm, method=method):
                estimator = self.make_model(norm, method)
                original_attack = estimator.attack
                custom_model = torch.nn.Sequential(
                    torch.nn.Linear(2, 8), torch.nn.ReLU(),
                    torch.nn.Linear(8, 2),
                )

                estimator.update_user_mode(
                    input_dim=2, num_classes=2, model=custom_model,
                    device=torch.device("cpu"),
                )

                self.assertIs(estimator.model, custom_model)
                self.assertIsNot(estimator.attack, original_attack)
                self.assertIs(estimator.attack.model, custom_model)
                if norm == "l2":
                    self.assertIs(estimator._tabular_attack.model.model, custom_model)

                adversarial = run_norm_ball_attack(
                    estimator.attack, estimator._tabular_attack,
                    self.inputs, self.labels,
                )
                self.assertEqual(adversarial.shape, self.inputs.shape)
                self.assertTrue(torch.isfinite(adversarial).all())

    def test_hrdro_pgdl2_uses_the_same_tabular_adapter(self):
        model = HRNNDRO(
            input_dim=2, num_classes=2, model_type="linear",
            device=torch.device("cpu"), train_batch_size=2,
            adversarial_params={
                "method": "PGD", "norm": "l2", "steps": 2,
                "step_size": 0.05,
            },
        )
        adversarial = run_norm_ball_attack(
            model.attack, model._tabular_attack, self.inputs, self.labels
        )
        self.assertEqual(adversarial.shape, self.inputs.shape)
        self.assertLessEqual(
            (adversarial - self.inputs).norm(p=2, dim=1).max().item(),
            model.epsilon + 1e-6,
        )
        self.assertEqual(model.predict(self.inputs.numpy()).shape, (len(self.inputs),))

    def test_invalid_method_or_norm_is_rejected(self):
        for norm, method in (
            ("l1", "PGD"), ("l-inf", "unknown"), ("l2", "FGSM"),
        ):
            with self.subTest(norm=norm, method=method), self.assertRaises(ValueError):
                self.make_model(norm, method)


if __name__ == "__main__":
    unittest.main()
