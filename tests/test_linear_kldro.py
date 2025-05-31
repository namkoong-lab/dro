import unittest
import numpy as np
from src.dro.linear_model.kl_dro import KLDRO, KLDROError
from src.dro.linear_model.base import ParameterError

class TestKLDROModel(unittest.TestCase):
    """Unit tests for Kullback-Leibler DRO model implementation."""

    def setUp(self):
        """Initialize test fixtures with deterministic random seed."""
        np.random.seed(42)
        self.default_model = KLDRO(
            input_dim=5,
            model_type='svm',
            eps=0.5
        )
        self.valid_X = np.random.randn(100, 5)
        self.valid_y = np.sign(np.random.randn(100))

    def test_kernel_update(self):
        """Test dynamic kernel parameter updates."""
        model = KLDRO(input_dim=5, kernel='linear')
        
        # Valid update
        model.update_kernel({'metric': 'rbf'})
        self.assertEqual(model.kernel, 'rbf')
        
        # Invalid update
        with self.assertRaises(ValueError):
            model.update_kernel({'metric': 'invalid'})

    def test_rbf_kernel_fit(self):
        """Test model fitting with RBF kernel."""
        model = KLDRO(
            input_dim=5,
            kernel='rbf',
            eps=0.1
        )
        params = model.fit(self.valid_X, self.valid_y)
        
        # Verify solution structure
        self.assertIn('theta', params)
        self.assertEqual(len(params['theta']), 100)  # For RBF, theta matches sample size

    def test_valid_initialization(self):
        """Test successful model creation with valid parameters."""
        model = KLDRO(input_dim=3, model_type='logistic', eps=0.5)
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.eps, 0.5)

    def test_invalid_input_dimension(self):
        """Test initialization with invalid feature dimension."""
        with self.assertRaises(ParameterError) as context:
            KLDRO(input_dim=-1)
        self.assertIn("Input dimension must be a positive integer.", str(context.exception))

    def test_excessive_eps_value(self):
        """Test initialization stability with extreme epsilon values."""
        with self.assertRaises(KLDROError) as context:
            KLDRO(input_dim=5, eps=150.0)
        self.assertIn("eps >100.0 may cause system instability (got 150.0)", str(context.exception))
    
    def test_valid_parameter_update(self):
        """Test successful robustness parameter update."""
        self.default_model.update({'eps': 0.5})
        self.assertEqual(self.default_model.eps, 0.5)

    def test_invalid_parameter_type_update(self):
        """Test parameter update with invalid data types."""
        with self.assertRaises(KLDROError) as context:
            self.default_model.update({'eps': 'invalid'})
        self.assertIn("eps must be numeric", str(context.exception))
    
    def test_successful_svm_fit(self):
        """Test basic SVM model fitting with valid data."""
        params = self.default_model.fit(self.valid_X, self.valid_y)
        self.assertIn('theta', params)
        self.assertEqual(len(params['theta']), 5)
        self.assertIsInstance(params['dual'], float)

    def test_lad_regression_fit(self):
        """Test LAD regression model fitting."""
        model = KLDRO(input_dim=5, model_type='lad')
        y_reg = np.random.randn(100)
        params = model.fit(self.valid_X, y_reg)
        self.assertTrue(np.isfinite(params['theta']).all())

    def test_data_dimension_mismatch(self):
        """Test fitting with dimensionally inconsistent data."""
        with self.assertRaises(KLDROError) as context:
            self.default_model.fit(np.random.randn(100, 3), self.valid_y)
        self.assertIn("Expected input with 5 features", str(context.exception))

    def test_invalid_labels_classification(self):
        """Test classification with invalid label values."""
        invalid_y = np.random.randint(0, 2, 100)  # 0/1 labels instead of ±1
        with self.assertRaises(KLDROError) as context:
            self.default_model.fit(self.valid_X, invalid_y)
        self.assertIn("classification labels not in {-1, +1}", str(context.exception))
    
    def test_worst_distribution_calculation(self):
        """Test worst-case distribution properties."""
        model = KLDRO(input_dim=5, model_type='logistic', eps=0.5)
        dist_info = model.worst_distribution(self.valid_X, self.valid_y)
        
        weights = dist_info['weight']
        self.assertTrue(np.allclose(weights.sum(), 1.0))
        self.assertTrue(np.all(weights >= 0))

    def test_zero_epsilon_behavior(self):
        """Test ERM equivalence when epsilon is zero."""
        model = KLDRO(input_dim=5, eps=0.0)
        dist_info = model.worst_distribution(self.valid_X, self.valid_y)
        weights = dist_info['weight']
        assert np.std(weights) <= 1e-3

    def test_no_intercept_model(self):
        """Test model fitting without intercept term."""
        model = KLDRO(input_dim=5, fit_intercept=False)
        params = model.fit(self.valid_X, self.valid_y)
        assert params["b"] == 0.0
        