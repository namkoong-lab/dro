import unittest
import numpy as np
import torch
from src.dro.neural_model.sinkhorn_nn import SinkhornNNDRO, SinkhornNNDROError
from src.dro.neural_model.base_nn import DataValidationError


class TestSinkhornNNDRO(unittest.TestCase):
    """Unit tests for Sinkhorn Neural DRO model."""

    def setUp(self):
        """Initialize test fixtures with deterministic seed."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Multi-class classification data (3 classes)
        self.X_cls = np.random.randn(150, 5)
        self.y_cls = np.random.randint(0, 3, 150)

        # Regression data
        self.X_reg = np.random.randn(100, 5)
        self.y_reg = np.random.randn(100)

    # ------------------------------------------------------------------ #
    #  Initialization Tests
    # ------------------------------------------------------------------ #

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        self.assertEqual(model.reg_param, 1e-3)
        self.assertEqual(model.lambda_param, 1e2)
        self.assertEqual(model.k_sample_max, 5)
        self.assertEqual(model.optimization_type, "SG")
        self.assertEqual(model.task_type, "classification")

    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        model = SinkhornNNDRO(
            input_dim=10,
            num_classes=5,
            reg_param=0.01,
            lambda_param=50.0,
            k_sample_max=3,
            optimization_type="MLMC",
            model_type="linear",
        )
        self.assertEqual(model.reg_param, 0.01)
        self.assertEqual(model.lambda_param, 50.0)
        self.assertEqual(model.k_sample_max, 3)
        self.assertEqual(model.optimization_type, "MLMC")

    def test_invalid_reg_param(self):
        """Test rejection of non-positive reg_param."""
        with self.assertRaises(ValueError):
            SinkhornNNDRO(input_dim=5, num_classes=3, reg_param=0)
        with self.assertRaises(ValueError):
            SinkhornNNDRO(input_dim=5, num_classes=3, reg_param=-0.1)

    def test_invalid_lambda_param(self):
        """Test rejection of non-positive lambda_param."""
        with self.assertRaises(ValueError):
            SinkhornNNDRO(input_dim=5, num_classes=3, lambda_param=0)
        with self.assertRaises(ValueError):
            SinkhornNNDRO(input_dim=5, num_classes=3, lambda_param=-1.0)

    def test_invalid_k_sample_max(self):
        """Test rejection of invalid k_sample_max."""
        with self.assertRaises(ValueError):
            SinkhornNNDRO(input_dim=5, num_classes=3, k_sample_max=0)

    def test_invalid_optimization_type(self):
        """Test rejection of invalid optimization_type."""
        with self.assertRaises(ValueError):
            SinkhornNNDRO(
                input_dim=5, num_classes=3, optimization_type="INVALID"
            )

    # ------------------------------------------------------------------ #
    #  Configuration Update Tests
    # ------------------------------------------------------------------ #

    def test_update_reg_param(self):
        """Test dynamic update of reg_param."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        model.update({"reg": 0.05})
        self.assertEqual(model.reg_param, 0.05)

    def test_update_lambda_param(self):
        """Test dynamic update of lambda_param."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        model.update({"lambda": 200.0})
        self.assertEqual(model.lambda_param, 200.0)

    def test_update_k_sample_max(self):
        """Test dynamic update of k_sample_max."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        model.update({"k_sample_max": 8})
        self.assertEqual(model.k_sample_max, 8)

    def test_update_optimization_type(self):
        """Test dynamic update of optimization_type."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        model.update({"optimization_type": "RTMLMC"})
        self.assertEqual(model.optimization_type, "RTMLMC")

    def test_update_invalid_reg(self):
        """Test rejection of invalid reg update."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        with self.assertRaises(ValueError):
            model.update({"reg": -0.1})

    def test_update_invalid_lambda(self):
        """Test rejection of invalid lambda update."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        with self.assertRaises(ValueError):
            model.update({"lambda": 0})

    def test_update_invalid_optimization_type(self):
        """Test rejection of invalid optimization_type update."""
        model = SinkhornNNDRO(input_dim=5, num_classes=3)
        with self.assertRaises(ValueError):
            model.update({"optimization_type": "BAD"})

    # ------------------------------------------------------------------ #
    #  Training Tests — Classification (SG / MLMC / RTMLMC)
    # ------------------------------------------------------------------ #

    def test_sg_classification(self):
        """Test SG optimization for classification."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=3,
            task_type="classification",
            model_type="mlp",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=2,
            optimization_type="SG",
        )
        metrics = model.fit(self.X_cls, self.y_cls, epochs=2, verbose=False)
        self.assertIn("acc", metrics)
        self.assertIn("f1", metrics)

        preds = model.predict(self.X_cls[:10])
        self.assertEqual(preds.shape, (10,))

    def test_mlmc_classification(self):
        """Test MLMC optimization for classification."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=3,
            task_type="classification",
            model_type="mlp",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=3,
            optimization_type="MLMC",
        )
        metrics = model.fit(self.X_cls, self.y_cls, epochs=2, verbose=False)
        self.assertIn("acc", metrics)

    def test_rtmlmc_classification(self):
        """Test RTMLMC optimization for classification."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=3,
            task_type="classification",
            model_type="mlp",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=3,
            optimization_type="RTMLMC",
        )
        metrics = model.fit(self.X_cls, self.y_cls, epochs=2, verbose=False)
        self.assertIn("acc", metrics)

    # ------------------------------------------------------------------ #
    #  Training Tests — Regression (SG / MLMC / RTMLMC)
    # ------------------------------------------------------------------ #

    def test_sg_regression(self):
        """Test SG optimization for regression."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=1,
            task_type="regression",
            model_type="linear",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=2,
            optimization_type="SG",
        )
        metrics = model.fit(self.X_reg, self.y_reg, epochs=2, verbose=False)
        self.assertIn("mse", metrics)

    def test_mlmc_regression(self):
        """Test MLMC optimization for regression."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=1,
            task_type="regression",
            model_type="linear",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=3,
            optimization_type="MLMC",
        )
        metrics = model.fit(self.X_reg, self.y_reg, epochs=2, verbose=False)
        self.assertIn("mse", metrics)

    def test_rtmlmc_regression(self):
        """Test RTMLMC optimization for regression."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=1,
            task_type="regression",
            model_type="linear",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=3,
            optimization_type="RTMLMC",
        )
        metrics = model.fit(self.X_reg, self.y_reg, epochs=2, verbose=False)
        self.assertIn("mse", metrics)

    # ------------------------------------------------------------------ #
    #  Model Type Tests
    # ------------------------------------------------------------------ #

    def test_linear_model_type(self):
        """Test with linear model architecture."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=3,
            model_type="linear",
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=2,
        )
        model.fit(self.X_cls, self.y_cls, epochs=2, verbose=False)
        preds = model.predict(self.X_cls[:5])
        self.assertEqual(preds.shape, (5,))

    # ------------------------------------------------------------------ #
    #  Scoring Tests
    # ------------------------------------------------------------------ #

    def test_score_classification(self):
        """Test scoring interface for classification."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=3,
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=2,
        )
        model.fit(self.X_cls, self.y_cls, epochs=2, verbose=False)
        acc, f1 = model.score(self.X_cls, self.y_cls)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(f1, 0.0)

    # ------------------------------------------------------------------ #
    #  Label Validation Tests
    # ------------------------------------------------------------------ #

    def test_label_dimension_mismatch(self):
        """Test rejection when class count exceeds num_classes."""
        model = SinkhornNNDRO(
            input_dim=5,
            num_classes=2,  # 3 classes in data
            reg_param=0.1,
            lambda_param=10.0,
            k_sample_max=2,
        )
        with self.assertRaises(DataValidationError):
            model.fit(self.X_cls, self.y_cls, epochs=1)

    def test_invalid_task_type(self):
        """Test rejection of invalid task type."""
        with self.assertRaises(ValueError):
            SinkhornNNDRO(
                input_dim=5, num_classes=3, task_type="invalid"
            )


if __name__ == "__main__":
    unittest.main()
