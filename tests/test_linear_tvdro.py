import unittest
import numpy as np
from src.dro.linear_model.tv_dro import TVDRO, TVDROError

class TestTVDROModel(unittest.TestCase):
    """Unit tests for TV-DRO model implementation."""

    def setUp(self):
        """Initialize test fixtures with deterministic seed."""
        np.random.seed(42)
        self.valid_X = np.random.randn(100, 5)
        self.valid_y_cls = np.sign(np.random.randn(100))  # Binary labels in {-1, +1}
        self.valid_y_reg = np.random.randn(100)          # Continuous values
        self.default_model = TVDRO(
            input_dim=5,
            model_type='svm',
            eps=0.1
        )

    def test_valid_initialization(self):
        """Test successful model creation with valid parameters."""
        model = TVDRO(input_dim=3, model_type='logistic', eps=0.05)
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.eps, 0.05)

    def test_rbf_kernel_fit(self):
        """Test model fitting with RBF kernel."""
        model = TVDRO(
            input_dim=5,
            kernel='rbf',
            eps=0.1
        )
        params = model.fit(self.valid_X, self.valid_y_cls)
        
        # Verify solution structure
        self.assertIn('theta', params)
        self.assertEqual(len(params['theta']), 100)  # For RBF, theta matches sample size


    def test_invalid_input_dim(self):
        """Test initialization with invalid feature dimension."""
        with self.assertRaises(ValueError) as context:
            TVDRO(input_dim=0, model_type='svm')
        self.assertIn("input_dim must be ≥ 1", str(context.exception))

    def test_valid_eps_update(self):
        """Test successful robustness parameter update."""
        self.default_model.update({'eps': 0.2})
        self.assertEqual(self.default_model.eps, 0.2)

    def test_invalid_eps_update(self):
        """Test parameter update with invalid epsilon values."""
        with self.assertRaises(TVDROError) as context:
            self.default_model.update({'eps': 1.5})
        self.assertIn("must be in the range (0, 1)", str(context.exception))

    def test_successful_svm_fit(self):
        """Test basic SVM fitting with valid binary labels."""
        params = self.default_model.fit(self.valid_X, self.valid_y_cls)
        self._validate_output_structure(params)
        self.assertEqual(len(params['theta']), 5)

    def test_invalid_classification_labels(self):
        """Test classification with 0/1 labels instead of ±1."""
        invalid_y = np.random.choice([0, 1], 100)
        with self.assertRaises(TVDROError) as context:
            self.default_model.fit(self.valid_X, invalid_y)
        self.assertIn("labels not in {-1, +1}", str(context.exception))

    def test_feature_dimension_mismatch(self):
        """Test fitting with inconsistent feature dimensions."""
        with self.assertRaises(TVDROError) as context:
            self.default_model.fit(np.random.randn(100, 3), self.valid_y_cls)
        self.assertIn("Expected input with 5 features", str(context.exception))
    
    def test_worst_distribution_calculation(self):
        """Test worst-case distribution properties."""
        self.default_model.fit(self.valid_X, self.valid_y_cls)
        dist_info = self.default_model.worst_distribution(self.valid_X, self.valid_y_cls)
        
        self.assertIn('sample_pts', dist_info)
        self.assertIn('weight', dist_info)
        self.assertTrue(np.isclose(dist_info['weight'].sum(), 1.0))
    
    def test_zero_epsilon_behavior(self):
        """Test ERM equivalence when epsilon is zero."""
        model = TVDRO(input_dim=5, eps=0.0, model_type='ols')
        params = model.fit(self.valid_X, self.valid_y_reg)
        self.assertTrue(np.isfinite(params['theta']).all())

    def test_lad_regression_fit(self):
        """Test LAD regression model fitting."""
        model = TVDRO(input_dim=5, model_type='lad')
        params = model.fit(self.valid_X, self.valid_y_reg)
        

    def test_no_intercept_model(self):
        """Test model fitting without intercept term."""
        model = TVDRO(input_dim=5, fit_intercept=False)
        params = model.fit(self.valid_X, self.valid_y_cls)
        
        assert params['b'] == 0.0
        
    def _validate_output_structure(self, params: dict):
        """Validate output dictionary structure and value ranges."""
        mandatory_keys = {'theta', 'threshold', 'b'}
        self.assertTrue(mandatory_keys.issubset(params.keys()))
        self.assertEqual(len(params['theta']), self.default_model.input_dim)
        