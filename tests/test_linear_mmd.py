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
        self.assertEqual(model.kernel, 'rbf')

    def test_supported_kernel_initialization(self):
        """Test the supported MMD kernels and the polynomial alias."""
        for kernel in ('rbf', 'laplacian', 'linear'):
            model = MMD_DRO(input_dim=4, kernel=kernel)
            self.assertEqual(model.kernel, kernel)

        model = MMD_DRO(
            input_dim=4,
            kernel='poly',
            kernel_gamma=0.25,
            kernel_degree=3,
            kernel_coef0=1.5,
        )
        self.assertEqual(model.kernel, 'polynomial')
        self.assertEqual(model.kernel_gamma, 0.25)
        self.assertEqual(model.kernel_degree, 3)
        self.assertEqual(model.kernel_coef0, 1.5)

    def test_invalid_kernel_configuration(self):
        """Test validation of kernel names and shape parameters."""
        with self.assertRaises(MMDDROError):
            MMD_DRO(input_dim=4, kernel='invalid')
        with self.assertRaises(ValueError):
            MMD_DRO(input_dim=4, kernel_gamma=0)
        with self.assertRaises(ValueError):
            MMD_DRO(input_dim=4, kernel='polynomial', kernel_degree=0)
        with self.assertRaises(ValueError):
            MMD_DRO(input_dim=4, kernel='polynomial', kernel_coef0=-1)

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
        self.default_model.update({
            'eta': 0.5,
            'sampling_method': 'hull',
            'kernel': 'polynomial',
            'kernel_gamma': 0.2,
            'kernel_degree': 3,
            'kernel_coef0': 2.0,
        })
        self.assertEqual(self.default_model.eta, 0.5)
        self.assertEqual(self.default_model.sampling_method, 'hull')
        self.assertEqual(self.default_model.kernel, 'polynomial')
        self.assertEqual(self.default_model.kernel_gamma, 0.2)
        self.assertEqual(self.default_model.kernel_degree, 3)
        self.assertEqual(self.default_model.kernel_coef0, 2.0)

    def test_update_kernel_aliases(self):
        """Test the common update_kernel API and polynomial aliases."""
        self.default_model.update_kernel({
            'metric': 'poly',
            'kernel_gamma': 0.25,
            'degree': 3,
            'coef0': 1.5,
            'n_components': 20,
        })
        self.assertEqual(self.default_model.kernel, 'polynomial')
        self.assertEqual(self.default_model.kernel_gamma, 0.25)
        self.assertEqual(self.default_model.kernel_degree, 3)
        self.assertEqual(self.default_model.kernel_coef0, 1.5)
        self.assertEqual(self.default_model.n_components, 20)

    def test_kernel_matrix_choices(self):
        """Test formulas and positive semidefiniteness of all MMD kernels."""
        zeta = np.array([
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
        ])

        polynomial_model = MMD_DRO(
            input_dim=2,
            kernel='polynomial',
            kernel_gamma=0.5,
            kernel_degree=2,
            kernel_coef0=1.0,
        )
        polynomial_gram = polynomial_model._kernel_matrix(zeta)
        expected = (0.5 * (zeta @ zeta.T) + 1.0) ** 2
        np.testing.assert_allclose(polynomial_gram, expected)

        default_polynomial = MMD_DRO(input_dim=2, kernel='polynomial')
        self.assertEqual(default_polynomial._kernel_parameters(zeta)['gamma'], 1 / 3)

        for kernel in ('rbf', 'laplacian', 'polynomial', 'linear'):
            model = MMD_DRO(input_dim=2, kernel=kernel)
            gram = model._kernel_matrix(zeta)
            np.testing.assert_allclose(gram, gram.T)
            self.assertGreaterEqual(np.linalg.eigvalsh(gram).min(), -1e-10)

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
        self.assertIsInstance(self.default_model.robust_obj, float)
        self.assertTrue(np.isfinite(self.default_model.robust_obj))

    def test_successful_polynomial_kernel_fit(self):
        """Test the Nyström fitting path with a polynomial MMD kernel."""
        model = MMD_DRO(
            input_dim=5,
            model_type='svm',
            kernel='polynomial',
            kernel_gamma=0.2,
            kernel_degree=2,
            kernel_coef0=1.0,
        )
        params = model.fit(self.valid_X, self.valid_y)
        self._validate_output_structure(params)
        self.assertTrue(np.isfinite(params['theta']).all())
        predictions = model.predict(self.valid_X)
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
    

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
