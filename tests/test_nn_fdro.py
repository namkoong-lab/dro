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
        # Test prediction shape
        preds = model.predict(self.X_cls[:5])
        if is_classification:
            self.assertEqual(preds.shape, (5,))
        else:
            self.assertEqual(preds.shape, (5,))

        # Test scoring
        score = model.score(self.X_cls, self.y_cls)[0]
        self.assertIsInstance(score, float)

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

    def test_chi2nndro_classification(self):
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


    def test_chi2nndro_classification(self):
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
