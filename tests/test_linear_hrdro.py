import unittest
import numpy as np
from src.dro.linear_model.hr_dro import HR_DRO_LR, HRDROError
from src.dro.linear_model.base import DataValidationError

class TestHRDROLinearRegression(unittest.TestCase):
    """Unit tests for HR_DRO_LR class."""

    def setUp(self):
        """Initialize test fixtures."""
        np.random.seed(42)
        self.default_model = HR_DRO_LR(
            input_dim=5,
            model_type='svm',
            fit_intercept=True,
            solver='MOSEK',
            r=0.0,
            alpha=0.05,
            epsilon=0.5,
            epsilon_prime=1.0
        )
        self.valid_X = np.random.randn(100, 5)
        self.valid_y = np.sign(np.random.randn(100))

    def test_valid_initialization(self):
        """Test successful model initialization with valid parameters."""
        model = HR_DRO_LR(input_dim=3, model_type='lad', fit_intercept=False)
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.model_type, 'lad')

    def test_invalid_input_dim(self):
        """Test initialization with invalid input dimensions."""
        with self.assertRaises(HRDROError) as context:
            HR_DRO_LR(input_dim=0)
        self.assertIn("input_dim must be ≥ 1", str(context.exception))

    def test_unsupported_model_type(self):
        """Test initialization with unsupported model types."""
        with self.assertRaises(HRDROError) as context:
            HR_DRO_LR(input_dim=3, model_type='ols')
        self.assertIn("HR DRO does not support OLS", str(context.exception))
    
    def test_valid_parameter_update(self):
        """Test successful parameter updates through update()."""
        self.default_model.update({'r': 0.8, 'epsilon_prime': 0.5})
        self.assertEqual(self.default_model.r, 0.8)
        self.assertEqual(self.default_model.epsilon_prime, 0.5)

    def test_invalid_alpha_update(self):
        """Test parameter updates with invalid alpha values."""
        with self.assertRaises(ValueError) as context:
            self.default_model.update({'alpha': 1.1})
        self.assertIn("alpha must be in (0, 1]", str(context.exception))
    
    
    def test_successful_fit_svm(self):
        """Test successful SVM model fitting."""
        params = self.default_model.fit(self.valid_X, self.valid_y)
        self.assertIn('theta', params)
        self.assertEqual(len(params['theta']), 5)

    def test_successful_fit_lad(self):
        """Test successful LAD model fitting."""
        lad_model = HR_DRO_LR(input_dim=5, model_type='lad')
        params = lad_model.fit(self.valid_X, self.valid_y)
        self.assertAlmostEqual(np.linalg.norm(params['theta']), 0.0, delta=1e-1)

    def test_data_dimension_mismatch(self):
        """Test fitting with dimensionally mismatched data."""
        with self.assertRaises(DataValidationError) as context:
            self.default_model.fit(np.random.randn(100, 3), self.valid_y)
        self.assertIn("Expected input with 5 features", str(context.exception))

    def test_zero_epsilon_behavior(self):
        """Test model behavior when disabling Wasserstein constraints."""
        model = HR_DRO_LR(input_dim=5, epsilon=0.0)
        params = model.fit(self.valid_X, self.valid_y)
        self.assertTrue(any(params['theta']))  # Should return non-trivial solution

    def test_no_intercept_model(self):
        """Test model fitting without intercept term."""
        model = HR_DRO_LR(input_dim=5, fit_intercept=False)
        params = model.fit(self.valid_X, self.valid_y)
        
        assert params["b"] == 0
