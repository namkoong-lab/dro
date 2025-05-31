import unittest
import numpy as np
from src.dro.linear_model.mmd_dro import MMD_DRO, MMDDROError
from src.dro.linear_model.base import ParameterError

class TestMMDDROModel(unittest.TestCase):
    """Unit tests for MMD-DRO model implementation."""

    def setUp(self):
        """Initialize test fixtures with deterministic seed."""
        np.random.seed(42)
        self.valid_X = np.random.randn(100, 5)
        self.valid_y = np.sign(np.random.randn(100))  # Binary labels in {-1, +1}
        self.default_model = MMD_DRO(
            input_dim=5,
            model_type='svm',
            sampling_method='bound'
        )

    
    def test_valid_initialization(self):
        """Test successful model creation with valid parameters."""
        model = MMD_DRO(input_dim=4, model_type='logistic', sampling_method='hull')
        self.assertEqual(model.input_dim, 4)
        self.assertEqual(model.sampling_method, 'hull')

    def test_invalid_model_type(self):
        """Test initialization with unsupported model type."""
        with self.assertRaises(ParameterError) as context:
            MMD_DRO(input_dim=5, model_type='invalid_type')
        self.assertIn("model_type", str(context.exception))

    def test_negative_input_dim(self):
        """Test initialization with invalid feature dimension."""
        with self.assertRaises(ParameterError) as context:
            MMD_DRO(input_dim=-1, model_type='svm')
        self.assertIn("positive integer.", str(context.exception))

    def test_invalid_sampling(self):
        with self.assertRaises(MMDDROError) as context:
            MMD_DRO(input_dim = 3, model_type = 'svm',  sampling_method = 'invalid')
        self.assertIn("Invalid sampling method", str(context.exception))
    
    def test_valid_parameter_update(self):
        """Test successful parameter updates."""
        self.default_model.update({'eta': 0.5, 'sampling_method': 'hull'})
        self.assertEqual(self.default_model.eta, 0.5)
        self.assertEqual(self.default_model.sampling_method, 'hull')

    def test_invalid_eta_update(self):
        """Test parameter update with non-positive eta."""
        with self.assertRaises(ValueError) as context:
            self.default_model.update({'eta': -0.1})
        with self.assertRaises(ValueError) as context:
            self.default_model.update({'n_certify_ratio': -0.1})
        self.assertIn("must be a positive float", str(context.exception))
    
    def test_successful_svm_fit(self):
        """Test basic SVM fitting with valid binary labels."""
        params = self.default_model.fit(self.valid_X, self.valid_y)
        self._validate_output_structure(params)
        self.assertTrue(np.isfinite(params['theta']).all())
    

    def test_invalid_label_values(self):
        """Test classification with 0/1 labels instead of ±1."""
        invalid_y = np.random.choice([0, 1], 100)
        with self.assertRaises(MMDDROError) as context:
            self.default_model.fit(self.valid_X, invalid_y)
        self.assertIn("classification labels", str(context.exception))

    def test_data_dimension_mismatch(self):
        """Test fitting with inconsistent feature dimensions."""
        with self.assertRaises(ValueError) as context:
            self.default_model.fit(np.random.randn(100, 3), self.valid_y)
        
    def test_hull_sampling_behavior(self):
        """Test model fitting with hull sampling method."""
        model = MMD_DRO(input_dim=5, model_type='svm', sampling_method='hull')
        params = model.fit(self.valid_X, self.valid_y)
        self._validate_output_structure(params)

    def test_hull_sampling_regression_behavior(self):
        """Test model fitting with hull sampling method with regression"""
        model = MMD_DRO(input_dim=5, model_type='ols', sampling_method='hull')
        valid_y = np.random.randn(100)
        params = model.fit(self.valid_X, valid_y)
        params = model.fit(self.valid_X, valid_y, accelerate=False)

        model = MMD_DRO(input_dim=5, model_type='lad', sampling_method='hull')
        valid_y = np.random.randn(100)
        params = model.fit(self.valid_X, valid_y)
        params = model.fit(self.valid_X, valid_y, accelerate=False)

        model = MMD_DRO(input_dim=5, model_type='logistic', sampling_method='bound', fit_intercept=False)
        valid_y = np.sign(np.random.randn(100))
        params = model.fit(self.valid_X, valid_y)
        params = model.fit(self.valid_X, valid_y, accelerate=False)

        self._validate_output_structure(params)

    def test_unsupported_sampling_method(self):
        """Test invalid sampling method handling."""
        with self.assertRaises(MMDDROError) as context:
            self.default_model.update({'sampling_method': 'invalid'})
        self.assertIn("must be either 'bound' or 'hull'", str(context.exception))

    def _validate_output_structure(self, params: dict):
        """Validate output dictionary structure and value ranges."""
        self.assertIn('theta', params)
        self.assertEqual(len(params['theta']), self.default_model.input_dim)
        self.assertTrue(all(isinstance(x, float) for x in params['theta']))
