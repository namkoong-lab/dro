import unittest
import numpy as np
import torch
from src.dro.neural_model.fdro_nn import Chi2NNDRO, CVaRNNDRO
from src.dro.neural_model.base_nn import DataValidationError

class TestNeuralDROModels(unittest.TestCase):
    """Unit tests for Neural DRO model implementations."""

    def setUp(self):
        """Initialize test fixtures with deterministic seed."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Multi-class classification data (3 classes)
        self.X_cls = np.random.randn(150, 5)  # 150 samples, 5 features
        self.y_cls = np.random.randint(0, 3, 150)  # Labels {0,1,2}
        
        # Regression data
        self.X_reg = np.random.randn(100, 5)
        self.y_reg = np.random.randn(100)
        
    # region Common Tests
    def _validate_model_interface(self, model, is_classification=True):
        """Validate standard model interface and outputs."""
        self.assertIsInstance(model.robust_obj, float)
        self.assertTrue(np.isfinite(model.robust_obj))

        # Test predictions for the task being exercised.
        inputs = self.X_cls[:5] if is_classification else self.X_reg[:5]
        preds = model.predict(inputs)
        if is_classification:
            self.assertEqual(preds.shape, (5,))
            score = model.score(self.X_cls, self.y_cls)[0]
            self.assertIsInstance(score, float)
        else:
            self.assertEqual(preds.shape, (5,))
            self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def _test_invalid_parameters(self, model_class):
        """Test parameter validation for DRO models."""
        # Invalid size parameter
        with self.assertRaises(ValueError):
            model_class(input_dim=5, num_classes=3, size=-0.1)

        # Invalid regularization
        with self.assertRaises(ValueError):
            model_class(input_dim=5, num_classes=3, reg=-0.1)

        # Invalid max iterations
        with self.assertRaises(ValueError):
            model_class(input_dim=5, num_classes=3, max_iter=0)

    def test_chi2nndro_regularized_classification(self):
        """Test Chi2NNDRO for multi-class classification."""
        model = Chi2NNDRO(
            input_dim=5,
            num_classes=3,
            task_type="classification",
            model_type='mlp',
            size=0.1,
            reg=0.1
        )
        model.fit(self.X_cls, self.y_cls, epochs=2)
        self._validate_model_interface(model)


    def test_chi2nndro_unregularized_classification(self):
        """Test Chi2NNDRO for multi-class classification."""
        model = Chi2NNDRO(
            input_dim=5,
            num_classes=3,
            task_type="classification",
            model_type='mlp',
            size=0.1,
            reg=0.0
        )
        model.fit(self.X_cls, self.y_cls, epochs=2)
        self._validate_model_interface(model)

    def test_chi2nndro_regression(self):
        """Test Chi2NNDRO for regression tasks."""
        model = Chi2NNDRO(
            input_dim=5,
            num_classes=1,
            task_type="regression",
            model_type='linear',
            size=0.2
        )
        model.fit(self.X_reg, self.y_reg, epochs=2)
        self._validate_model_interface(model, is_classification=False)

    def test_chi2nndro_parameter_validation(self):
        """Test Chi2NNDRO parameter validation."""
        self._test_invalid_parameters(Chi2NNDRO)

    def test_zero_radius_chi2_matches_erm_loss_and_gradient(self):
        for task in ('classification', 'regression'):
            for reg in (0.0, 0.2):
                with self.subTest(task=task, reg=reg):
                    model = Chi2NNDRO(
                        input_dim=2, num_classes=2, task_type=task,
                        model_type='linear', size=0.0, reg=reg,
                    )
                    if task == 'classification':
                        outputs = torch.tensor([[1.0, -0.5], [0.2, 0.8], [-0.4, 0.1]], requires_grad=True)
                        labels = torch.tensor([0, 0, 1])
                        expected = torch.nn.functional.cross_entropy(outputs, labels)
                    else:
                        outputs = torch.tensor([[0.5], [2.0], [-0.3]], requires_grad=True)
                        labels = torch.tensor([1.0, -0.5, 0.2])
                        expected = torch.nn.functional.mse_loss(outputs.squeeze(), labels)
                    actual = model._criterion(outputs, labels)
                    torch.testing.assert_close(actual, expected)
                    actual_grad = torch.autograd.grad(actual, outputs, retain_graph=True)[0]
                    expected_grad = torch.autograd.grad(expected, outputs)[0]
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_zero_radius_chi2_training(self):
        for task, X, y in (
            ('classification', self.X_cls, self.y_cls),
            ('regression', self.X_reg, self.y_reg),
        ):
            with self.subTest(task=task):
                model = Chi2NNDRO(
                    input_dim=5, num_classes=3, task_type=task,
                    model_type='linear', size=0.0,
                )
                model.fit(X[:16], y[:16], epochs=1)
                self.assertTrue(np.isfinite(model.robust_obj))

    def test_cvarnndro_classification(self):
        """Test CVaRNNDRO for multi-class classification."""
        model = CVaRNNDRO(
            input_dim=5,
            num_classes=3,
            task_type="classification",
            size=0.1,
            reg=0.1
        )
        model.fit(self.X_cls, self.y_cls, epochs=2)
        self._validate_model_interface(model)

    def test_cvarnndro_size_validation(self):
        """Test CVaR size parameter validation."""
        with self.assertRaises(ValueError):
            CVaRNNDRO(input_dim=10, num_classes=3, size=1.1)

    def test_cvarnndro_parameter_validation(self):
        """Test CVaRNNDRO regularization and iteration validation."""
        self._test_invalid_parameters(CVaRNNDRO)

    def test_update_rebuilds_fdro_models(self):
        """Both f-DRO variants apply training and robust-loss settings."""
        config = {
            "lr": 0.02,
            "batch_size": 8,
            "train_epochs": 3,
            "layer_num": 3,
            "hidden_size": 7,
            "dropout_ratio": 0.2,
            "size": 0.4,
            "reg": 0.3,
        }

        for model_class in (Chi2NNDRO, CVaRNNDRO):
            model = model_class(input_dim=5, num_classes=3)
            original_model = model.model
            model.update(config)

            self.assertEqual(model.lr, config["lr"])
            self.assertEqual(model.batch_size, config["batch_size"])
            self.assertEqual(model.train_epochs, config["train_epochs"])
            self.assertEqual(model.layer_num, config["layer_num"])
            self.assertEqual(model.size, config["size"])
            self.assertEqual(model.reg, config["reg"])
            self.assertIsNot(model.model, original_model)
            self.assertEqual(model.model(torch.zeros(2, 5)).shape, (2, 3))

    def test_cvarnndro_regression(self):
        """Test CVaRNNDRO for regression tasks."""
        model = CVaRNNDRO(
            input_dim=5,
            num_classes=1,
            task_type="regression",
            model_type='linear',
            size=0.2
        )
        model.fit(self.X_reg, self.y_reg, epochs=2)
        self._validate_model_interface(model, is_classification=False)

    def test_invalid_task_type(self):
        """Test invalid task type handling."""
        with self.assertRaises(ValueError):
            model = Chi2NNDRO(input_dim=5, num_classes=3, task_type="invalid")
            model.fit(self.X_cls, self.y_cls)

    def test_label_dimension_mismatch(self):
        """Test label dimension validation."""
        model = Chi2NNDRO(input_dim=5, num_classes=2)  # Should have 3 classes
        with self.assertRaises(DataValidationError):
            model.fit(self.X_cls, self.y_cls)
