import unittest
import numpy as np
import cvxpy as cp
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
        for p in (0.5, 1.5, 3, '2', 'l2', np.inf, -np.inf, np.nan,
                  None, True, False, [], {}, np.array([1, 2])):
            with self.subTest(p=p):
                with self.assertRaisesRegex(MOTDROError, "'p' must be 1, 2, or 'inf'"):
                    self.default_model.update({'p': p})
                self.assertEqual(self.default_model.p, 2)

    def test_valid_p_update(self):
        for p in (1, 1.0, 2, 2.0, 'inf'):
            with self.subTest(p=p):
                self.default_model.update({'p': p})
                self.assertEqual(self.default_model.p, p)

    def test_invalid_square_update(self):
        for square in (0, 1.5, 3, '2', np.inf, np.nan, None, True, False,
                       [], np.array([1, 2])):
            with self.subTest(square=square):
                with self.assertRaisesRegex(MOTDROError, "'square' must be 1 or 2"):
                    self.default_model.update({'square': square})
                self.assertEqual(self.default_model.square, 2)

    def test_valid_update(self):
        self.default_model.update({"eps": 0.5, "p": 1, "square": 2})
        self.assertEqual(self.default_model.eps, 0.5)
        self.assertEqual(self.default_model.p, 1)
        self.assertEqual(self.default_model.square, 2)

    def test_positive_radius_transport_options(self):
        """All supported norms and exponents solve the robust problem."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(24, 2))
        targets = {
            'svm': np.where(X[:, 0] + 0.5 * X[:, 1] >= 0, 1, -1),
            'lad': X @ np.array([1.5, -0.5]) + rng.normal(scale=0.1, size=24),
        }
        for model_type, y in targets.items():
            for p in (1, 2, 'inf'):
                for square in (1, 2):
                    with self.subTest(model_type=model_type, p=p, square=square):
                        model = MOTDRO(input_dim=2, model_type=model_type)
                        model.update({'eps': 0.1, 'p': p, 'square': square})
                        params = model.fit(X, y)
                        self.assertTrue(np.isfinite(params['theta']).all())
                        self.assertTrue(np.isfinite(model.robust_obj))
                        empirical_loss = model._loss(X, y).mean()
                        self.assertGreaterEqual(model.robust_obj + 1e-5, empirical_loss)

    def test_fit_rejects_invalid_direct_transport_settings(self):
        """Direct attribute edits receive the same error as update()."""
        for eps in (0, 0.1):
            for key, value in (('p', 1.5), ('square', 3)):
                with self.subTest(eps=eps, key=key):
                    model = MOTDRO(input_dim=5)
                    model.update({'eps': eps})
                    setattr(model, key, value)
                    with self.assertRaises(MOTDROError):
                        model.fit(self.valid_X, self.valid_y)

    def test_successful_svm_fit(self):
        """Test basic SVM fitting with valid binary labels."""
        params = self.default_model.fit(self.valid_X, self.valid_y)
        self._validate_output_structure(params)
        self.assertEqual(len(params['theta']), 5)
        self.assertIsInstance(self.default_model.robust_obj, float)
        self.assertTrue(np.isfinite(self.default_model.robust_obj))

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
        """At zero radius, the no-intercept objective is empirical hinge risk."""
        model = MOTDRO(input_dim=5, fit_intercept=False)
        self.assertEqual(model.solver, 'CLARABEL')
        self.assertEqual(model.eps, 0)
        params = model.fit(self.valid_X, self.valid_y)
        assert params["b"] == 0.0

        reference_theta = cp.Variable(self.valid_X.shape[1])
        reference_loss = cp.sum(cp.pos(
            1 - cp.multiply(self.valid_y, self.valid_X @ reference_theta)
        )) / len(self.valid_y)
        reference = cp.Problem(cp.Minimize(reference_loss))
        reference.solve(solver='CLARABEL')

        fitted_loss = np.mean(np.maximum(
            1 - self.valid_y * (self.valid_X @ np.asarray(params['theta'])), 0
        ))
        self.assertAlmostEqual(model.robust_obj, reference.value, delta=1e-5)
        self.assertAlmostEqual(fitted_loss, reference.value, delta=1e-5)
        
        
    def _validate_output_structure(self, params: dict):
        """Validate output dictionary structure and value ranges."""
        mandatory_keys = {'theta', 'b'}
        self.assertTrue(mandatory_keys.issubset(params.keys()))
        self.assertEqual(len(params['theta']), self.default_model.input_dim)
