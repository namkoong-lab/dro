import math
import unittest

import torch
import torch.nn.functional as F

from src.dro.neural_model import CVaRNNDRO, Chi2NNDRO, WNNDRO


class TestNeuralRadiusCertificates(unittest.TestCase):
    def test_wasserstein_quadratic_certificate(self):
        # For x=0, y=1, f(x)=x, and lambda=2, the penalized inner
        # maximizer is x'=-1: loss=4, squared transport cost=1, gap=0.
        model = WNNDRO(
            input_dim=1, num_classes=1, task_type="regression",
            model_type="linear", robust_objective="transport_penalty",
            gamma=2.0, device=torch.device("cpu"),
        )
        with torch.no_grad():
            model.model.layer1.weight.fill_(1.0)
            model.model.layer1.bias.zero_()
            original = torch.tensor([[0.0]])
            adversarial = torch.tensor([[-1.0]])
            losses = F.mse_loss(
                model.model(adversarial).reshape(-1),
                torch.tensor([1.0]), reduction="none",
            )
            costs = (adversarial - original).flatten(1).square().sum(dim=1)
        result = model.certify_radius(1.2, losses, costs, [0.0])
        self.assertAlmostEqual(result["epsilon_cert"], 1.0)
        self.assertAlmostEqual(result["loss_bound"], 4.0)
        self.assertAlmostEqual(result["input_upper_bound"], 4.4)
        # The exact worst-case loss at squared-cost budget one is four.
        self.assertAlmostEqual((1 + math.sqrt(result["epsilon_cert"])) ** 2,
                               result["loss_bound"])

        self.assertIsNone(model.certify_radius(1.2, [4.0], [1.0], [3.0]))
        with self.assertRaises(ValueError):
            model.certify_radius(0.5, [4.0], [1.0], [0.0])
        with self.assertRaises(ValueError):
            model.certify_radius(1.2, [4.0], [1.0], [-0.1])

    def test_norm_ball_requires_explicit_transport_penalty(self):
        model = WNNDRO(input_dim=1, num_classes=2, epsilon=0.1)
        with self.assertRaises(ValueError):
            model.certify_radius(1.2, [4.0], [1.0], [0.0])
        self.assertAlmostEqual(
            model.certify_radius(
                1.2, [4.0], [1.0], [0.0], transport_penalty=2.0,
            )["epsilon_cert"], 1.0,
        )

    def test_chi2_certificate_and_invalid_candidate(self):
        model = Chi2NNDRO(input_dim=1, num_classes=2, size=0.5)
        result = model.certify_radius([0.0, 1.0], [0.25, 0.75], eta=0.0)
        self.assertAlmostEqual(result["epsilon_cert"], 0.0625)
        self.assertAlmostEqual(result["loss_bound"], 0.75)
        # Exact two-point chi-square robust loss is 1/2 + sqrt(epsilon/2).
        exact = 0.5 + math.sqrt(result["epsilon_cert"] / 2)
        self.assertLessEqual(exact, result["loss_bound"] + 1e-12)
        self.assertIsNone(model.certify_radius([0.0, 1.0], [0.5, 0.5], eta=0.0))
        with self.assertRaises(ValueError):
            model.certify_radius([0.0, 1.0], [0.0, 1.0], eta=0.0, epsilon=0.1)

    def test_cvar_certificate_and_invalid_candidate(self):
        model = CVaRNNDRO(input_dim=1, num_classes=2, size=0.5)
        result = model.certify_radius([0.0, 1.0], [0.25, 0.75], eta=0.0)
        self.assertAlmostEqual(result["alpha_cert"], 2 / 3)
        self.assertAlmostEqual(result["loss_bound"], 0.75)
        # At alpha=2/3 the high-loss point has maximum weight 3/4.
        self.assertAlmostEqual(1 / (2 * result["alpha_cert"]),
                               result["loss_bound"])
        self.assertIsNone(model.certify_radius([0.0, 1.0], [0.5, 0.5], eta=1.0))
        with self.assertRaises(ValueError):
            model.certify_radius([0.0, 1.0], [0.0, 1.0], eta=0.0, alpha=0.75)


if __name__ == "__main__":
    unittest.main()
