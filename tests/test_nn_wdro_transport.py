import unittest

import numpy as np
import torch

from src.dro.neural_model.wdro_nn import WNNDRO


class TestWNNDROTransportPenalty(unittest.TestCase):
    def make_regression_model(self, input_bounds=None):
        model = WNNDRO(
            input_dim=1,
            num_classes=1,
            task_type="regression",
            model_type="linear",
            device=torch.device("cpu"),
            epsilon=0.0,  # The transport mode does not use the norm-ball radius.
            adversarial_steps=40,
            adversarial_step_size=0.1,
            robust_objective="transport_penalty",
            gamma=2.0,
            input_bounds=input_bounds,
        )
        with torch.no_grad():
            model.model.layer1.weight.fill_(1.0)
            model.model.layer1.bias.zero_()
        return model

    def test_inner_attack_and_loss_match_quadratic_optimum(self):
        # For f(x') = (x' - 1)^2 - 2(x' - 0)^2, the maximizer is x' = -1
        # and the penalized value is 2 (the unpenalized loss is 4).
        model = self.make_regression_model()
        inputs = torch.tensor([[0.0]])
        labels = torch.tensor([1.0])

        adversarial = model._transport_attack(inputs, labels)
        self.assertAlmostEqual(adversarial.item(), -1.0, delta=0.02)
        self.assertIsNone(model.attack)

        model.current_inputs = inputs
        robust_loss = model._criterion(model.model(inputs), labels)
        self.assertAlmostEqual(robust_loss.item(), 2.0, delta=0.03)
        robust_loss.backward()
        gradient = model.model.layer1.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().item(), 0.0)

    def test_input_bounds_limit_transport_attack(self):
        model = self.make_regression_model(input_bounds=(0.0, 1.0))
        adversarial = model._transport_attack(
            torch.tensor([[0.0]]), torch.tensor([1.0])
        )
        self.assertEqual(adversarial.item(), 0.0)

    def test_inner_search_keeps_clean_candidate_when_step_overshoots(self):
        model = self.make_regression_model()
        model.adversarial_steps = 1
        model.adversarial_step_size = 2.0
        inputs = torch.tensor([[0.0]])
        labels = torch.tensor([1.0])
        model.current_inputs = inputs

        # The first update reaches x' = -4, whose penalized value is -7.
        # The original input has value 1 and must remain an option.
        robust_loss = model._criterion(model.model(inputs), labels)
        self.assertAlmostEqual(robust_loss.item(), 1.0)

    def test_transport_mode_trains_through_base_loop(self):
        inputs = np.array(
            [[0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [0.3, 0.3],
             [0.7, 0.7], [0.8, 0.9], [0.9, 0.8], [1.0, 1.0]],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        model = WNNDRO(
            input_dim=2,
            num_classes=2,
            model_type="linear",
            device=torch.device("cpu"),
            robust_objective="transport_penalty",
            gamma=2.0,
            adversarial_steps=2,
            adversarial_step_size=0.01,
            input_bounds=(0.0, 1.0),
        )

        model.fit(inputs, labels, train_ratio=0.75, batch_size=3, epochs=1, verbose=False)
        self.assertTrue(np.isfinite(model.robust_obj))

    def test_default_norm_ball_mode_still_trains(self):
        model = WNNDRO(
            input_dim=4,
            num_classes=2,
            model_type="linear",
            device=torch.device("cpu"),
            epsilon=0.1,
            adversarial_steps=1,
            adversarial_step_size=0.05,
        )
        # Torchattacks PGDL2 expects image-shaped tensors.
        model.model = torch.nn.Sequential(torch.nn.Flatten(), model.model)
        model._init_adversarial_attack()
        inputs = torch.tensor(
            [[[[0.2, 0.3], [0.3, 0.2]]], [[[0.7, 0.8], [0.8, 0.7]]]]
        )
        labels = torch.tensor([0, 1])
        model.current_inputs = inputs

        loss = model._criterion(model.model(inputs), labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.model[1].layer1.weight.grad)

    def test_invalid_transport_parameters(self):
        for options in (
            {"robust_objective": "unknown"},
            {"gamma": 0.0},
            {"input_bounds": (1.0, 0.0)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                WNNDRO(input_dim=1, num_classes=2, **options)


if __name__ == "__main__":
    unittest.main()
