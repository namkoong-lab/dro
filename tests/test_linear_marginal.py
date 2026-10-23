import unittest
import numpy as np
from src.dro.linear_model.marginal_dro import MarginalCVaRDRO, MarginalCVaRDROError


class _FixedLossMarginalDRO(MarginalCVaRDRO):
    """Isolate transport risk; the predictor penalty is minimized at zero."""

    def _cvx_loss(self, X, y, theta, b, design_matrix=None):
        return y + theta[0] ** 2


class TestMarginalCVaRDRO(unittest.TestCase):
    """Unit tests for Marginal CVaR-DRO model."""

    def setUp(self):
        """Initialize test fixtures with deterministic seed."""
        np.random.seed(42)
        self.valid_X = np.random.randn(100, 5)
        self.valid_y = np.sign(np.random.randn(100))  # Binary labels in {-1, +1}
        self.default_model = MarginalCVaRDRO(
            input_dim=5,
            model_type='svm',
            alpha=0.95,
            L=5.0,
            p=2
        )
        self.default_model.update({"control_name": [0, 2]})

    def test_valid_initialization(self):
        """Test successful model creation with valid parameters."""
        model = MarginalCVaRDRO(
            input_dim=3,
            model_type='lad',
            alpha=0.9,
            L=10.0,
            p=1
        )
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.p, 1)

    def test_invalid_alpha_initialization(self):
        """Test initialization with out-of-range alpha."""
        with self.assertRaises(ValueError) as context:
            MarginalCVaRDRO(input_dim=5, alpha=1.5)
        self.assertIn("alpha must be in (0, 1]", str(context.exception))
    
    def test_valid_control_name_update(self):
        """Test valid feature index update."""
        self.default_model.update({"control_name": [1, 3]})
        self.assertEqual(self.default_model.control_name, [1, 3])

    def test_invalid_control_name_update(self):
        """Test update with out-of-bound feature indices."""
        with self.assertRaises(MarginalCVaRDROError) as context:
            self.default_model.update({"control_name": [5, 7]})
        self.assertIn("must be in the range [0, input_dim - 1]", str(context.exception))

    def test_negative_L_update(self):
        """Test parameter update with invalid L value."""
        with self.assertRaises(MarginalCVaRDROError) as context:
            self.default_model.update({"L": -1.0})
        self.assertIn("must be positive", str(context.exception))
    
    def test_success_update(self):
        self.default_model.update({"p": 1, "alpha": 0.8})

    def test_large_alpha_update(self):
        """Test parameter update with invalid L value."""
        with self.assertRaises(MarginalCVaRDROError) as context:
            self.default_model.update({"alpha": 1.2})
        self.assertIn("must be in the range (0, 1].", str(context.exception))

    def test_small_p_update(self):
        with self.assertRaises(MarginalCVaRDROError) as context:
            self.default_model.update({"p": 0.75})
        self.assertIn("must be >= 1.", str(context.exception))        

    def test_kernel_update(self):
        """Test dynamic kernel parameter updates."""
        model = MarginalCVaRDRO(input_dim=5, kernel='linear', alpha = 0.8)
        
        # Valid update
        model.update_kernel({'metric': 'rbf'})
        self.assertEqual(model.kernel, 'rbf')
        
        # Invalid update
        with self.assertRaises(ValueError):
            model.update_kernel({'metric': 'invalid'})

    def test_rbf_kernel_fit(self):
        """Test model fitting with RBF kernel."""
        model = MarginalCVaRDRO(
            input_dim=5,
            kernel='rbf',
            alpha = 0.8
        )
        params = model.fit(self.valid_X, self.valid_y)
        
        # Verify solution structure
        self.assertIn('theta', params)
        self.assertIsInstance(model.robust_obj, float)
        self.assertTrue(np.isfinite(model.robust_obj))
        self.assertEqual(len(params['theta']), 100)  # For RBF, theta matches sample size

    def test_successful_svm_fit(self):
        """Test basic SVM fitting with valid binary labels."""
        params = self.default_model.fit(self.valid_X, self.valid_y)
        params = self.default_model.fit(self.valid_X, self.valid_y, accelerate=False)
        self._validate_output_structure(params)
        self.assertTrue(np.isfinite(params['theta']).all())

    def test_invalid_label_values(self):
        """Test classification with 0/1 labels instead of ±1."""
        invalid_y = np.random.choice([0, 1], 100)
        with self.assertRaises(MarginalCVaRDROError) as context:
            self.default_model.fit(self.valid_X, invalid_y)
        self.assertIn("classification labels not in {-1, +1}", str(context.exception))

    def test_data_dimension_mismatch(self):
        """Test fitting with inconsistent feature dimensions."""
        with self.assertRaises(MarginalCVaRDROError) as context:
            self.default_model.fit(np.random.randn(100, 3), self.valid_y)
        self.assertIn("Expected input with 5 features", str(context.exception))
    
    def test_empty_control_name(self):
        """Test model fitting without feature protection."""
        model = MarginalCVaRDRO(input_dim=5)
        self.default_model.update({"control_name": []})
        params = model.fit(self.valid_X, self.valid_y)
        self._validate_output_structure(params)

    def test_no_intercept_model(self):
        """Test model fitting without intercept term."""
        model = MarginalCVaRDRO(input_dim=5, fit_intercept=False)
        params = model.fit(self.valid_X, self.valid_y)
        assert params["b"] == 0.0

    def test_unaccelerated_two_sample_transport_risk(self):
        # For two losses [0, 1], the full symmetric transport constraints give
        # |q[1] - q[0]| <= L**(p-1) * distance**(p-1) / 2.
        # Only the first feature is controlled; the second must not affect cost.
        X = np.array([[0.0, 100.0], [1.0, -100.0]])
        losses = np.array([0.0, 1.0])
        for p, expected_risk in [(1, 0.75), (2, 0.60)]:
            with self.subTest(p=p):
                model = _FixedLossMarginalDRO(
                    input_dim=2, model_type='ols', fit_intercept=False,
                    alpha=0.5, L=0.4, p=p,
                )
                model.update({'control_name': [0]})
                model.fit(X, losses, accelerate=False)
                self.assertAlmostEqual(model.robust_obj, expected_risk, places=5)

    def test_unaccelerated_full_cost_and_permutation_invariance(self):
        # At this L the distance constraints are inactive, giving CVaR 1/30.
        # The old sparse cost allowed free flows and reduced this to mean 1/60.
        n = 60
        X = np.arange(n, dtype=float)[:, None]
        losses = np.zeros(n)
        losses[-1] = 1.0
        permutation = np.random.default_rng(7).permutation(n)
        for order in [np.arange(n), permutation]:
            model = _FixedLossMarginalDRO(
                input_dim=1, model_type='ols', fit_intercept=False,
                alpha=0.5, L=2.0, p=2,
            )
            model.fit(X[order], losses[order], accelerate=False)
            self.assertAlmostEqual(model.robust_obj, 1.0 / 30.0, places=5)

    def _validate_output_structure(self, params: dict):
        """Validate output dictionary structure and value ranges."""
        mandatory_keys = {'theta', 'threshold'}
        self.assertTrue(mandatory_keys.issubset(params.keys()))
        self.assertEqual(len(params['theta']), self.default_model.input_dim)
