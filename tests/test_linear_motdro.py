import unittest
import numpy as np
from src.dro.linear_model.mot_dro import MOTDRO, MOTDROError

class TestMOTDROModel(unittest.TestCase):
    """Unit tests for MOT-DRO model implementation."""

    def setUp(self):
        """Initialize test fixtures with deterministic seed."""
        np.random.seed(42)
        self.valid_X = np.random.randn(100, 5)
        self.valid_y = np.sign(np.random.randn(100))  # Binary labels in {-1, +1}
        self.default_model = MOTDRO(
            input_dim=5,
            model_type='svm',
            fit_intercept=True
        )
    def test_valid_initialization(self):
        """Test successful model creation with valid parameters."""
        model = MOTDRO(input_dim=3, model_type='lad')
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.model_type, 'lad')

    def test_invalid_model_type(self):
        """Test initialization with unsupported model types."""
        with self.assertRaises(MOTDROError) as context:
            MOTDRO(input_dim=5, model_type='ols')
        self.assertIn("does not support OLS", str(context.exception))
    
    def test_theta_update_coupling(self):
        """Test parameter coupling between theta1 and theta2."""
        self.default_model.update({'theta1': 2.0})
        self.assertAlmostEqual(self.default_model.theta2, 2.0)
        self.assertAlmostEqual(self.default_model.theta1, 2.0)

        self.default_model.update({'theta2': 3.0})
        self.assertAlmostEqual(self.default_model.theta1, 1.5)

    def test_invalid_theta_update(self):
        """Test parameter updates with invalid theta values."""
        with self.assertRaises(AssertionError) as context:
            self.default_model.update({'theta1': 0.5})

    def test_invalid_eps_update(self):
        with self.assertRaises(MOTDROError) as context:
            self.default_model.update({'eps': -0.5})

    def test_invalid_p_update(self):
        with self.assertRaises(MOTDROError) as context:
            self.default_model.update({'p': 0.5})

    def test_invalid_square_update(self):
        with self.assertRaises(MOTDROError) as context:
            self.default_model.update({'square': 1.5})

    def test_valid_update(self):
        self.default_model.update({"eps": 0.5, "p": 1, "square": 2})
        
    
    def test_successful_svm_fit(self):
        """Test basic SVM fitting with valid binary labels."""
        params = self.default_model.fit(self.valid_X, self.valid_y)
        self._validate_output_structure(params)
        self.assertEqual(len(params['theta']), 5)

    def test_invalid_label_values(self):
        """Test classification with 0/1 labels instead of ±1."""
        invalid_y = np.random.choice([0, 1], 100)
        with self.assertRaises(MOTDROError) as context:
            self.default_model.fit(self.valid_X, invalid_y)
        self.assertIn("classification labels", str(context.exception))

    def test_feature_dimension_mismatch(self):
        """Test fitting with inconsistent feature dimensions."""
        with self.assertRaises(MOTDROError) as context:
            self.default_model.fit(np.random.randn(100, 3), self.valid_y)
        self.assertIn("Expected input with 5 features", str(context.exception))
    
    def test_lad_regression_fit(self):
        """Test LAD regression model fitting."""
        model = MOTDRO(input_dim=5, model_type='lad')
        y_reg = np.random.randn(100)
        params = model.fit(self.valid_X, y_reg)
        self.assertTrue(np.isfinite(params['theta']).all())

    def test_no_intercept_model(self):
        """Test model fitting without intercept term."""
        model = MOTDRO(input_dim=5, fit_intercept=False)
        params = model.fit(self.valid_X, self.valid_y)
        assert params["b"] == 0.0
        
        
    def _validate_output_structure(self, params: dict):
        """Validate output dictionary structure and value ranges."""
        mandatory_keys = {'theta', 'b'}
        self.assertTrue(mandatory_keys.issubset(params.keys()))
        self.assertEqual(len(params['theta']), self.default_model.input_dim)
        