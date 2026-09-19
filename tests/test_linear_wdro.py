import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics.pairwise import pairwise_kernels
from src.dro.linear_model.wasserstein_dro import (
    WassersteinDRO,
    WassersteinDROsatisficing,
    WassersteinDROError,
    WassersteinDROSatisificingError
)
import cvxpy as cp 

# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate standardized test datasets"""
    n_samples, n_features = 100, 5
    if request.param == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            random_state=42
        )
        y = np.sign(y - 0.5)  # Convert to ±1 labels
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42
        )
    return X, y

# --------------------------
# WassersteinDRO Core Tests
# --------------------------

class TestWassersteinDRO:
    """Test suite for WassersteinDRO base functionality"""
    
    @pytest.mark.parametrize("model_type", ['svm', 'logistic', 'ols', 'lad'])
    def test_initialization(self, model_type):
        """Validate constructor arguments and default states"""
        model = WassersteinDRO(
            input_dim=5,
            model_type=model_type,
            solver='MOSEK'
        )
        
        # Validate default parameters
        assert model.cost_matrix.shape == (5, 5)
        assert model.eps == 0
        assert model.p == 1
        assert model.kappa == 'inf'
    
    def test_invalid_initialization(self):
        """Test constructor parameter validation"""
        with pytest.raises(ValueError):
            WassersteinDRO(input_dim=0, model_type='svm')
            
    @pytest.mark.parametrize("config,valid", [
        ({'cost_matrix': np.diag([1,2,3])}, True),
        ({'eps': -0.1}, False),
        ({'eps': np.nan}, False),
        ({'eps': np.inf}, False),
        ({'p': 0.5}, False),
        ({'p': np.nan}, False),
        ({'p': np.inf}, True),
        ({'kappa': np.nan}, False),
        ({'kappa': np.inf}, True),
        ({'kappa': 'invalid'}, False)
    ])
    def test_config_updates(self, config, valid):
        """Test dynamic configuration validation"""
        model = WassersteinDRO(input_dim=3, model_type='svm')
        if valid:
            model.update(config)
        else:
            with pytest.raises((ValueError, TypeError)):
                model.update(config)
    
    @pytest.mark.parametrize("model_type", ['svm', 'logistic', 'ols', 'lad'])
    def test_fit_interface(self, model_type):
        """Validate fit method input/output contracts"""
        if model_type in {"svm", "logistic"}:
            X, y = make_classification(
                n_samples=100,
                n_features=5,
                n_informative=3,
                random_state=42
                )
            y = np.sign(y - 0.5)
        else:
             X, y = make_regression(
                    n_samples=100,
                    n_features=5,
                    random_state=42
                )
        
        model = WassersteinDRO(
            input_dim=X.shape[1],
            model_type=model_type,
            solver='MOSEK'
        )
        model.update({'eps': 0.1, 'p': 2})
        
        params = model.fit(X, y)
        
        # Validate output structure
        assert 'theta' in params
        assert isinstance(model.robust_obj, float)
        assert np.isfinite(model.robust_obj)
        assert isinstance(params['theta'], list)
        if model.fit_intercept:
            assert 'b' in params

    @pytest.mark.parametrize("model_type", ['svm'])
    def test_fit_kernel_interface(self, model_type):
        """Validate fit method input/output contracts"""
        if model_type in {"svm", "logistic"}:
            X, y = make_classification(
                n_samples=100,
                n_features=5,
                n_informative=3,
                random_state=42
                )
            y = np.sign(y - 0.5)
        else:
             X, y = make_regression(
                    n_samples=100,
                    n_features=5,
                    random_state=42
                )
        
        model = WassersteinDRO(
            input_dim=X.shape[1],
            model_type=model_type,
            solver='MOSEK'
        )
        model.update({'eps': 0.1, 'p': 2})
        model.update_kernel({'metric': 'rbf', 'kernel_gamma': 1})
        params = model.fit(X, y)
        model.update_kernel({'metric': 'rbf', 'kernel_gamma': 'scale', 'n_components': 5})
        params = model.fit(X, y)
        
        # Validate output structure
        assert 'theta' in params
        assert isinstance(params['theta'], list)
        if model.fit_intercept:
            assert 'b' in params
    
    @pytest.mark.parametrize("model_type", ['svm', 'lad', 'ols'])
    def test_zero_epsilon_worst_distribution(self, model_type):
        """A zero Wasserstein radius returns the empirical distribution."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = rng.normal(size=30)
        if model_type == 'svm':
            y = np.where(y >= 0, 1, -1)

        model = WassersteinDRO(
            input_dim=X.shape[1], model_type=model_type, solver='CLARABEL'
        )
        dist = model.worst_distribution(X, y)

        np.testing.assert_array_equal(dist['sample_pts'][0], X)
        np.testing.assert_array_equal(dist['sample_pts'][1], y)
        np.testing.assert_allclose(dist['weight'], np.full(X.shape[0], 1.0 / X.shape[0]))
        np.testing.assert_array_equal(dist['source_index'], np.arange(X.shape[0]))
        assert dist['gamma_used'] is None
        assert dist['transport_cost'] == pytest.approx(0)
        assert abs(dist['optimality_gap']) <= 1e-7
        assert dist['expected_loss'] == pytest.approx(dist['target_objective'])
        if model_type == 'ols':
            assert dist['target_objective'] == pytest.approx(model.robust_obj ** 2)
        else:
            assert dist['target_objective'] == pytest.approx(model.robust_obj)
        assert dist['certified'] is True
        assert dist['asymptotic'] is False

# --------------------------
# WassersteinDROsatisficing Tests  
# --------------------------

class TestWassersteinSatisficing:
    """Test suite for robust satisficing variant"""
    
    def test_satisficing_optimization(self, dataset):
        """Validate target ratio constraint enforcement"""
        X, y = dataset
        y = np.sign(y-0.5)
        model = WassersteinDROsatisficing(
            input_dim=X.shape[1],
            model_type='svm',
            solver='MOSEK'
        )
        model.update({'target_ratio': 1.2})
        
        params = model.fit(X, y)
        
        # Validate solution feasibility
        assert params['theta'] is not None
        assert isinstance(model.robust_obj, float)
        assert np.isfinite(model.robust_obj)
        assert 'b' in params
    
    @pytest.mark.parametrize("ratio,valid", [
        (-1.0, False),
        (1.5, True),
        (-0.8, False)
    ])
    def test_target_ratio_validation(self, ratio, valid):
        """Test target ratio boundary conditions"""
        if valid:
            WassersteinDROsatisficing(
                input_dim=5,
                model_type='svm'
            ).update({'target_ratio': ratio})
        else:
            with pytest.raises(AssertionError):
                WassersteinDROsatisficing(
                    input_dim=5,
                    model_type='svm'
                ).update({'target_ratio': ratio})

# --------------------------
# Cross-Cutting Concerns
# --------------------------

@pytest.mark.parametrize("solver", ['MOSEK'])
def test_solver_compatibility(dataset, solver):
    """Validate solver interoperability"""
    X, y = dataset
    y = np.sign(y-0.5)
    model = WassersteinDRO(
        input_dim=X.shape[1],
        model_type='svm',
        solver=solver
    )
    model.update({'eps': 0.1})
    
    params = model.fit(X, y)
    assert params['theta'] is not None

@pytest.mark.parametrize("p_value", [1, 2, 'inf'])
def test_wasserstein_order_handling(p_value):
    """Test different Wasserstein metric configurations"""
    model = WassersteinDRO(
        input_dim=3,
        model_type='svm'
    )
    model.update({'p': p_value})
    assert model.p == (float(p_value) if p_value != 'inf' else 'inf')

# --------------------------
# Error Condition Tests
# --------------------------

def test_dimension_mismatch_errors():
    """Validate dimensional consistency checks"""
    model = WassersteinDRO(input_dim=3, model_type='svm')
    X = np.random.randn(5, 4)
    y = np.random.randn(5)
    
    with pytest.raises(WassersteinDROError):
        model.fit(X, y)


# --------------------------
# Specialized Kernel Tests
# --------------------------

@pytest.mark.parametrize("kernel", ['linear', 'rbf'])
def test_kernel_support(kernel):
    """Validate kernelized implementation"""
    model = WassersteinDRO(
        input_dim=10,
        model_type='svm',
        kernel=kernel
    )
    X, y = make_classification(n_samples=10, n_features=10)
    y = np.sign(y-0.5)
    if kernel == 'rbf':
        model.update({'cost_matrix': np.eye(10)})
    
    params = model.fit(X, y)
    assert params['theta'] is not None


def test_rbf_kernel_ols_fit_uses_kernel_design_matrix():
    """Full-kernel OLS works when sample and input dimensions differ."""
    X, y = make_regression(
        n_samples=18,
        n_features=3,
        noise=0.2,
        random_state=42,
    )
    model = WassersteinDRO(
        input_dim=X.shape[1],
        model_type='ols',
        solver='CLARABEL',
        kernel='rbf',
    )
    model.update({'eps': 0.05, 'p': 2})
    model.update_kernel({'kernel_gamma': 0.7})

    params = model.fit(X, y)

    theta = np.asarray(params['theta'])
    kernel_matrix = pairwise_kernels(
        X,
        X,
        metric='rbf',
        gamma=model.kernel_gamma,
    )
    predictions = kernel_matrix @ theta + model.b
    empirical_rmse = np.linalg.norm(predictions - y) / np.sqrt(X.shape[0])
    rkhs_norm = np.sqrt(max(float(theta @ kernel_matrix @ theta), 0.0))

    assert theta.shape == (X.shape[0],)
    np.testing.assert_allclose(model.predict(X), predictions)
    assert model.robust_obj == pytest.approx(
        empirical_rmse + np.sqrt(model.eps) * rkhs_norm,
        abs=1e-5,
    )


def test_nystroem_kernel_ols_reuses_training_feature_map():
    """Nyström OLS uses one fitted feature map throughout the objective."""
    X, y = make_regression(
        n_samples=20,
        n_features=4,
        noise=0.2,
        random_state=7,
    )
    model = WassersteinDRO(
        input_dim=X.shape[1],
        model_type='ols',
        solver='CLARABEL',
        kernel='rbf',
    )
    model.update({'eps': 0.05, 'p': 2})
    model.update_kernel({
        'kernel_gamma': 0.4,
        'n_components': 6,
    })

    params = model.fit(X, y)

    theta = np.asarray(params['theta'])
    features = model.nystroem_transformer.transform(X)
    predictions = features @ theta + model.b
    empirical_rmse = np.linalg.norm(predictions - y) / np.sqrt(X.shape[0])
    penalty = np.linalg.norm(features @ theta)

    assert theta.shape == (model.n_components,)
    np.testing.assert_allclose(model.predict(X), predictions)
    assert model.robust_obj == pytest.approx(
        empirical_rmse + np.sqrt(model.eps) * penalty,
        abs=1e-5,
    )


# --------------------------
# WassersteinDRO Penalization Tests
# --------------------------

def test_penalization_lad_with_kappa(dataset):
    """Test regularization term for LAD model with finite kappa"""
    X, y = dataset
    model = WassersteinDRO(
        input_dim=X.shape[1], 
        model_type='lad',
        solver='MOSEK'
    )
    model.update({
        'kappa': 1.0,
        'p': 2
    })
    
    # Mock theta variable
    theta = np.random.randn(X.shape[1])
    penalty = model._penalization(theta)
    
    expr_str = str(penalty.expr).lower()
    assert "maximum" in expr_str 

# --------------------------
# Distance Computation Tests
# --------------------------

def test_distance_computation_with_y_ambiguity():
    """A binary label flip costs kappa, not kappa times |1 - (-1)|."""
    model = WassersteinDRO(input_dim=2, model_type='svm')
    model.update({'kappa': 1.5, 'p': 2})
    
    # Test with different labels
    dist = model._distance_compute(
        X_1=np.array([1.0, 2.0]),
        X_2=np.array([3.0, 4.0]),
        Y_1=1.0,
        Y_2=-1.0
    )
    assert dist.value == pytest.approx(np.sqrt(8) + 1.5)

# --------------------------
# Lipschitz Norm Tests
# --------------------------

def test_lipschitz_norm_for_ols():
    """Verify Lipschitz constant for OLS models"""
    model = WassersteinDRO(input_dim=3, model_type='ols')
    assert np.isinf(model._lipschitz_norm())

# --------------------------
# Satisficing Model Edge Cases
# --------------------------

@pytest.mark.parametrize("model_type", ['ols', 'svm', 'lad', 'logistic'])
def test_satisficing_lad_constraints(model_type):
    """Test constraint formulation for LAD satisficing model"""
    if model_type in {"lad", "ols"}:
        X, y = make_regression(
            n_samples=50,
            n_features=10,
            random_state=42,
        )
    else:
        X, y = make_classification(
            n_samples=50,
            n_features=10,
            random_state=42,
        )
        y = np.sign(y-0.5)
        if model_type == 'logistic':
            # A full RBF basis separates any distinct finite training set. For
            # unregularized logistic loss that drives the oracle target toward
            # zero without a finite minimizer, which makes this solver-focused
            # test numerically unstable. Conflicting duplicate observations
            # retain the full-RBF path while giving the loss a positive floor.
            X[1] = X[0]
            y[0], y[1] = -1, 1

    model = WassersteinDROsatisficing(
        input_dim=10,
        model_type=model_type,
        solver='MOSEK'
    )
    model.update({
        'kappa': 0.5,
        'target_ratio': 1.5
    })
    
    params = model.fit(X, y)
    assert 'theta' in params  # Verify solution exists
    model.update_kernel({'metric': 'rbf', 'kernel_gamma': 1})
    params = model.fit(X, y)
    # Nystroem samples basis components from NumPy's global RNG. Keep this
    # solver test reproducible across local and xdist runs.
    np.random.seed(1)
    model.update_kernel({'metric': 'rbf', 'kernel_gamma': 'scale', 'n_components': 5})
    params = model.fit(X, y)
    assert 'theta' in params  # Verify solution exists


@pytest.mark.parametrize("model_type", ['lad'])
def test_satisficing_lad_constraints2(model_type):
    """Test constraint formulation for LAD satisficing model"""
    if model_type in {"lad", "ols"}:
        X, y = make_regression(n_samples=50, n_features=10)
    else:
        X, y = make_classification(n_samples=50, n_features=10)
        y = np.sign(y-0.5)

    model = WassersteinDROsatisficing(
        input_dim=10,
        model_type=model_type,
        solver='MOSEK',
        kernel='linear'
    )
    model.update({
        'kappa': 'inf',
        'target_ratio': 1.5,
        'cost_matrix': np.identity(10)
    })
    
    params = model.fit(X, y)
    assert 'theta' in params  # Verify solution exists


    
# --------------------------
# Configuration Validation
# --------------------------

def test_invalid_cost_matrix_type():
    """Test non-array cost matrix rejection"""
    model = WassersteinDRO(input_dim=2, model_type='svm')
    with pytest.raises(TypeError):
        model.update({'cost_matrix': [[1,0],[0,1]]})  # Not numpy array

def test_invalid_p_value_update():
    """Test invalid Wasserstein order rejection"""
    model = WassersteinDRO(input_dim=3, model_type='svm')
    with pytest.raises(ValueError):
        model.update({'p': 0.5})  # p must be ≥1


def test_lad_rejects_zero_kappa():
    """Free continuous-target transport makes robust LAD unbounded."""
    model = WassersteinDRO(input_dim=2, model_type='lad')
    with pytest.raises(ValueError, match='strictly positive'):
        model.update({'kappa': 0})


def test_ols_rejects_finite_kappa():
    """The OLS reformulation fixes targets and therefore requires kappa=inf."""
    model = WassersteinDRO(input_dim=2, model_type='ols', solver='CLARABEL')
    with pytest.raises(ValueError, match="kappa must be 'inf'"):
        model.update({'kappa': 1.0})

# --------------------------
# Worst-case Distribution
# --------------------------

def test_p():
    X, y = make_classification(n_samples=50, n_features=10)
    y = np.sign(y-0.5)
    model = WassersteinDROsatisficing(input_dim=10, model_type='svm')
    model.update({"p":2})
    
    model.fit(X, y)

    with pytest.raises(Warning):
        model.worst_distribution(X, y)



# --------------------------
# Distance Computation Tests
# --------------------------

def test_distance_with_infinite_kappa():
    """An infinite kappa prohibits rather than ignores a label change."""
    model = WassersteinDRO(input_dim=2, model_type='svm')
    model.update({'kappa': 'inf', 'p': 2})

    with pytest.raises(WassersteinDROError, match='prohibited'):
        model._distance_compute(
            X_1=np.array([1.0, 2.0]),
            X_2=np.array([3.0, 4.0]),
            Y_1=1.0,
            Y_2=-1.0
        )

# --------------------------
# Worst-case Distribution (OLS Exact)
# --------------------------

@pytest.mark.parametrize("p", [1, 1.5, 2, 'inf'])
def test_exact_worst_distribution_ols(p, monkeypatch):
    """OLS recovery attains the squared robust objective exactly."""
    X = np.array([[-1.0], [2.0]])
    y = np.array([0.0, 1.0])
    model = WassersteinDRO(
        input_dim=1, model_type='ols', fit_intercept=True, solver='CLARABEL'
    )
    model.update({'eps': 0.25, 'p': p, 'cost_matrix': np.array([[4.0]])})

    def fake_fit(X_fit, y_fit):
        model.theta = np.array([2.0])
        model.b = -0.5
        # R=2.5, L=||Sigma^(-1/2) theta||=1, sqrt(eps)=0.5.
        model.robust_obj = 3.0
        return {'theta': [2.0], 'b': -0.5}

    monkeypatch.setattr(model, 'fit', fake_fit)
    dist = model.worst_distribution(X, y)

    np.testing.assert_allclose(dist['sample_pts'][0], [[-1.25], [2.25]])
    np.testing.assert_array_equal(dist['sample_pts'][1], y)
    np.testing.assert_allclose(dist['weight'], [0.5, 0.5])
    np.testing.assert_array_equal(dist['source_index'], [0, 1])
    assert dist['transport_cost'] == pytest.approx(model.eps)
    assert dist['expected_loss'] == pytest.approx(9.0)
    assert dist['target_objective'] == pytest.approx(model.robust_obj ** 2)
    assert dist['optimality_gap'] == pytest.approx(0.0)
    assert dist['gamma_used'] is None
    assert dist['kappa_used'] == 'inf'
    assert dist['certified'] is True
    assert dist['asymptotic'] is False


@pytest.mark.parametrize("p", [1, 1.5, 2, 'inf'])
def test_ols_dual_direction_with_spd_cost(p):
    """The recovery direction attains the weighted dual norm."""
    sigma = np.array([[3.0, 0.5], [0.5, 1.0]])
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    transform = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    theta = np.array([1.25, -0.75])
    model = WassersteinDRO(input_dim=2, model_type='ols', solver='CLARABEL')
    model.update({'p': p, 'cost_matrix': sigma})
    model.theta = theta

    direction, slope = model._feature_dual_direction()

    primal_order = np.inf if p == 'inf' else p
    if p == 1:
        dual_order = np.inf
    elif p == 'inf':
        dual_order = 1
    else:
        dual_order = p / (p - 1)
    expected_slope = np.linalg.norm(
        np.linalg.solve(transform.T, theta), ord=dual_order
    )
    assert np.linalg.norm(transform @ direction, ord=primal_order) == pytest.approx(1)
    assert theta @ direction == pytest.approx(expected_slope)
    assert slope == pytest.approx(expected_slope)


def test_exact_worst_distribution_ols_matches_fitted_objective():
    """Positive-radius recovery integrates with the fitted robust RMSE."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3))
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.3
    y += rng.normal(scale=0.4, size=X.shape[0])
    cost_matrix = np.array([
        [2.0, 0.1, 0.2],
        [0.1, 1.0, 0.05],
        [0.2, 0.05, 1.5],
    ])
    model = WassersteinDRO(input_dim=3, model_type='ols', solver='CLARABEL')
    model.update({'eps': 0.2, 'p': 2, 'cost_matrix': cost_matrix})

    dist = model.worst_distribution(X, y)

    np.testing.assert_array_equal(dist['sample_pts'][1], y)
    np.testing.assert_allclose(dist['weight'], np.full(X.shape[0], 1 / X.shape[0]))
    assert dist['transport_cost'] == pytest.approx(model.eps)
    assert dist['expected_loss'] == pytest.approx(model.robust_obj ** 2)
    assert dist['target_objective'] == pytest.approx(model.robust_obj ** 2)
    assert dist['certified'] is True
    assert dist['asymptotic'] is False


def test_exact_worst_distribution_ols_zero_residual(monkeypatch):
    """Zero empirical residual uses a finite common shift without division."""
    X = np.zeros((2, 1))
    y = np.zeros(2)
    model = WassersteinDRO(
        input_dim=1, model_type='ols', fit_intercept=False, solver='CLARABEL'
    )
    model.update({'eps': 0.25, 'p': 2})

    def fake_fit(X_fit, y_fit):
        model.theta = np.array([1.0])
        model.b = 0.0
        model.robust_obj = 0.5
        return {'theta': [1.0], 'b': 0.0}

    monkeypatch.setattr(model, 'fit', fake_fit)
    dist = model.worst_distribution(X, y)

    np.testing.assert_allclose(dist['sample_pts'][0], [[0.5], [0.5]])
    np.testing.assert_array_equal(dist['sample_pts'][1], y)
    assert dist['transport_cost'] == pytest.approx(0.25)
    assert dist['expected_loss'] == pytest.approx(0.25)
    assert dist['target_objective'] == pytest.approx(model.robust_obj ** 2)
    assert dist['certified'] is True


def test_exact_worst_distribution_ols_zero_slope(monkeypatch):
    """When theta is zero, feature transport cannot increase OLS loss."""
    X = np.array([[-1.0], [2.0]])
    y = np.array([0.0, 1.0])
    model = WassersteinDRO(input_dim=1, model_type='ols', solver='CLARABEL')
    model.update({'eps': 0.25, 'p': 2})

    def fake_fit(X_fit, y_fit):
        model.theta = np.array([0.0])
        model.b = 0.5
        model.robust_obj = 0.5
        return {'theta': [0.0], 'b': 0.5}

    monkeypatch.setattr(model, 'fit', fake_fit)
    dist = model.worst_distribution(X, y)

    np.testing.assert_array_equal(dist['sample_pts'][0], X)
    np.testing.assert_array_equal(dist['sample_pts'][1], y)
    assert dist['transport_cost'] == pytest.approx(0.0)
    assert dist['expected_loss'] == pytest.approx(0.25)
    assert dist['target_objective'] == pytest.approx(model.robust_obj ** 2)
    assert dist['certified'] is True


def test_ols_rejects_asymptotic_options():
    """Asymptotic tuning controls are not part of exact OLS recovery."""
    model = WassersteinDRO(input_dim=1, model_type='ols', solver='CLARABEL')
    with pytest.raises(WassersteinDROError, match='do not apply'):
        model.worst_distribution(
            np.array([[0.0]]),
            np.array([0.0]),
            asymptotic_options={'gamma': 0.1},
        )

# --------------------------
# Asymptotic Method Tests
# --------------------------

@pytest.mark.parametrize("model_type,gamma,kappa", [
    ('svm', 0.1, 1.0),
    ('logistic', 0.1, 1.0),
    ('lad', 0.2, 1.0),
    ('lad', 0.2, 'inf'),
])
def test_asymptotic_method_gamma(model_type, gamma, kappa):
    """A successful asymptotic result carries its numerical certificate."""
    
    if model_type in {"svm", "logistic"}:
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=3,
            random_state=42
            )
        y = np.sign(y - 0.5)
    else:
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            random_state=42
            )
    
    model = WassersteinDRO(
        input_dim=10,
        model_type=model_type,
        solver='MOSEK'
    )
    model.update({'kappa': kappa, 'eps': 0.5})
    
    dist = model.worst_distribution(
        X, y,
        asymptotic_options={'gamma': gamma},
    )
    assert np.isclose(sum(dist['weight']), 1.0, atol=1e-7)
    assert len(dist['source_index']) == len(dist['weight'])
    assert 0 < dist['gamma_used'] <= gamma
    assert dist['transport_cost'] <= model.eps + 1e-7
    assert dist['expected_loss'] == pytest.approx(
        dist['target_objective'] - dist['optimality_gap']
    )
    assert dist['optimality_gap'] <= (
        1e-5 + 1e-5 * max(1.0, abs(dist['target_objective']))
    )
    assert dist['certified'] is True
    assert dist['asymptotic'] is True

# --------------------------
# Satisficing Model Tests
# --------------------------

# def test_satisficing_target_ratio_enforcement():
#     """Verify target ratio constraint is enforced"""
#     X, y = make_regression(n_samples=100, n_features=10)
#     model = WassersteinDROsatisficing(
#         input_dim=10,
#         model_type='ols',
#         solver='MOSEK'
#     )
#     model.update({'target_ratio': 1.5})
    
#     params = model.fit(X, y)
#     empirical_loss = np.mean((X @ params['theta'] - y)**2)
#     assert empirical_loss <= 1.5 * model.fit_oracle(X, y)

# --------------------------
# Kernel Method Tests
# --------------------------

def test_rbf_kernel_support():
    """Validate RBF kernel implementation"""
    X, y = make_classification(n_samples=100, n_features=10)
    y = np.sign(y-0.5)
    model = WassersteinDRO(
        input_dim=10,
        model_type='svm',
        kernel='rbf'
    )
    model.update({
        'cost_matrix': np.eye(10),
        'eps': 0.1
    })
    
    params = model.fit(X, y)
    assert len(params['theta']) == 100 

# --------------------------
# Exception Handling
# --------------------------

@pytest.mark.parametrize("gamma", [0, -0.1, np.nan, np.inf, '0.1'])
def test_asymptotic_gamma_must_be_positive_and_finite(gamma, monkeypatch):
    """Positive-radius constructions cannot materialize the gamma=0 limit."""
    X = np.array([[0.0]])
    y = np.array([1])
    model = WassersteinDRO(
        input_dim=1,
        model_type='svm',
        fit_intercept=False,
        solver='MOSEK'
    )
    model.update({'kappa': 'inf', 'eps': 0.5, 'p': 2})

    def fake_fit(X_fit, y_fit):
        model.theta = np.array([1.0])
        model.b = 0.0
        model.robust_obj = 1.5
        return {'theta': [1.0], 'b': 0.0}

    monkeypatch.setattr(model, 'fit', fake_fit)

    with pytest.raises(WassersteinDROError):
        model.worst_distribution(
            X, y, asymptotic_options={'gamma': gamma}
        )


@pytest.mark.parametrize("kwargs", [
    {'objective_atol': -1e-5},
    {'objective_atol': np.inf},
    {'objective_atol': np.nan},
    {'objective_rtol': -1e-5},
    {'objective_rtol': np.inf},
    {'feasibility_tol': -1e-7},
    {'feasibility_tol': np.inf},
    {'feasibility_tol': np.nan},
    {'max_iter': 0},
    {'gamma_decay': 0},
    {'gamma_decay': 1},
])
def test_worst_distribution_certificate_parameter_validation(kwargs):
    """Certificate and refinement controls reject invalid ranges."""
    model = WassersteinDRO(input_dim=1, model_type='svm', solver='MOSEK')
    with pytest.raises(WassersteinDROError):
        model.worst_distribution(
            np.array([[0.0]]),
            np.array([1]),
            asymptotic_options=kwargs,
        )


@pytest.mark.parametrize("options", [1, 'gamma', [], {'unknown': 1}])
def test_worst_distribution_rejects_invalid_asymptotic_options(options):
    """The grouped asymptotic configuration is type- and key-checked."""
    model = WassersteinDRO(input_dim=1, model_type='svm', solver='MOSEK')
    with pytest.raises(WassersteinDROError, match='dictionary|Unknown'):
        model.worst_distribution(
            np.array([[0.0]]),
            np.array([1]),
            asymptotic_options=options,
        )


def test_worst_distribution_rejects_gamma_above_bound():
    """Finite-kappa classification reserves gamma from the radius budget."""
    model = WassersteinDRO(input_dim=1, model_type='svm', solver='MOSEK')
    model.update({'eps': 0.2, 'kappa': 1.0})
    with pytest.raises(WassersteinDROError, match='must not exceed'):
        model.worst_distribution(
            np.array([[0.0]]),
            np.array([1]),
            asymptotic_options={'gamma': 0.3},
        )


def test_worst_distribution_rejects_nonlinear_kernel():
    """The input-space recovery formula is not valid for kernel parameters."""
    model = WassersteinDRO(
        input_dim=1, model_type='svm', solver='MOSEK', kernel='rbf'
    )
    with pytest.raises(WassersteinDROError, match="kernel='linear'"):
        model.worst_distribution(np.array([[0.0]]), np.array([1]))


def test_classification_recession_direction_and_certificate(monkeypatch):
    """A positive-label atom must move against theta and pass both checks."""
    # The two nominal atoms share a location but have different labels. Their
    # provenance therefore cannot be inferred from the generated coordinates.
    X = np.array([[0.0], [0.0]])
    y = np.array([1, -1])
    model = WassersteinDRO(
        input_dim=1,
        model_type='svm',
        fit_intercept=False,
        solver='MOSEK'
    )
    model.update({'kappa': 'inf', 'eps': 0.5, 'p': 2})

    # Fix a classifier for which the inner robust value is known exactly:
    # max_Q E_Q[(1-X)_+] = 1 + eps when theta=1 and X=0 nominally.
    def fake_fit(X_fit, y_fit):
        model.theta = np.array([1.0])
        model.b = 0.0
        model.robust_obj = 1.0 + model.eps
        return {'theta': [1.0], 'b': 0.0}

    monkeypatch.setattr(model, 'fit', fake_fit)
    dist = model.worst_distribution(
        X,
        y,
        asymptotic_options={
            'objective_atol': 1e-5,
            'objective_rtol': 1e-5,
            'feasibility_tol': 1e-8,
        },
    )

    expected_keys = {
        'sample_pts', 'weight', 'source_index', 'gamma_used',
        'expected_loss', 'target_objective', 'optimality_gap',
        'transport_cost', 'certified', 'asymptotic', 'kappa_used',
    }
    assert expected_keys <= dist.keys()
    assert dist['gamma_used'] > 0

    worst_X = np.asarray(dist['sample_pts'][0])
    worst_y = np.asarray(dist['sample_pts'][1])
    weight = np.asarray(dist['weight'])
    source = np.asarray(dist['source_index'])

    assert np.all(weight >= 0)
    assert weight.sum() == pytest.approx(1)
    assert np.all((0 <= source) & (source < len(X)))
    assert set(source) == {0, 1}
    np.testing.assert_array_equal(worst_y, y[source])

    # Hinge/logistic loss grows when the signed margin decreases. For y=+1
    # and theta=+1, the nontrivial recession atom must therefore lie at x<0.
    far_index = int(np.argmin(worst_X[:, 0]))
    assert worst_X[far_index, 0] < X[0, 0]
    assert source[far_index] == 0

    explicit_transport_cost = np.sum(
        weight * np.abs(worst_X[:, 0] - X[source, 0])
    )
    expected_loss = np.sum(
        weight * np.maximum(1.0 - worst_y * worst_X[:, 0], 0.0)
    )
    assert dist['transport_cost'] == pytest.approx(
        explicit_transport_cost, abs=1e-8
    )
    assert dist['transport_cost'] <= model.eps + 1e-8
    assert dist['expected_loss'] == pytest.approx(expected_loss)
    assert dist['target_objective'] == pytest.approx(model.robust_obj)
    assert dist['optimality_gap'] == pytest.approx(
        dist['target_objective'] - dist['expected_loss']
    )
    assert dist['optimality_gap'] <= (
        1e-5 + 1e-5 * max(1.0, abs(dist['target_objective']))
    )
    assert dist['certified'] is True


def test_finite_kappa_label_flip_coupling_is_certified(monkeypatch):
    """Finite-kappa recovery charges one indicator cost per flipped label."""
    X = np.array([[0.0]])
    y = np.array([1])
    model = WassersteinDRO(
        input_dim=1,
        model_type='svm',
        fit_intercept=False,
        solver='MOSEK',
    )
    model.update({'kappa': 1.0, 'eps': 0.5, 'p': 2})

    # With theta=0 and b=1, the nominal hinge loss is zero and the flipped
    # loss is two. The fixed-decision worst-case value is therefore one.
    def fake_fit(X_fit, y_fit):
        model.theta = np.array([0.0])
        model.b = 1.0
        model.robust_obj = 1.0
        return {'theta': [0.0], 'b': 1.0}

    monkeypatch.setattr(model, 'fit', fake_fit)
    dist = model.worst_distribution(X, y)

    worst_X, worst_y = map(np.asarray, dist['sample_pts'])
    weight = np.asarray(dist['weight'])
    source = np.asarray(dist['source_index'])
    source_mass = np.bincount(source, weights=weight, minlength=len(X))
    indicator_cost = np.sum(weight * (worst_y != y[source]))
    expected_loss = np.sum(
        weight * np.maximum(1.0 - worst_y * (worst_X[:, 0] * 0.0 + 1.0), 0.0)
    )

    np.testing.assert_allclose(source_mass, np.full(len(X), 1.0 / len(X)))
    assert dist['transport_cost'] == pytest.approx(indicator_cost)
    assert dist['transport_cost'] <= model.eps + 1e-7
    assert dist['expected_loss'] == pytest.approx(expected_loss)
    assert abs(dist['optimality_gap']) <= (
        1e-5 + 1e-5 * max(1.0, abs(dist['target_objective']))
    )
    assert dist['certified'] is True


def test_uncertified_worst_distribution_is_not_returned(monkeypatch):
    """Objective disagreement must raise instead of returning a false WCD."""
    X = np.array([[0.0]])
    y = np.array([1])
    model = WassersteinDRO(
        input_dim=1,
        model_type='svm',
        fit_intercept=False,
        solver='MOSEK'
    )
    model.update({'kappa': 'inf', 'eps': 0.5, 'p': 2})

    def fake_fit(X_fit, y_fit):
        model.theta = np.array([1.0])
        model.b = 0.0
        # Deliberately inconsistent with the maximum value 1 + eps.
        model.robust_obj = 2.0
        return {'theta': [1.0], 'b': 0.0}

    monkeypatch.setattr(model, 'fit', fake_fit)
    with pytest.raises(WassersteinDROError, match='certif|objective|agree'):
        model.worst_distribution(
            X,
            y,
            asymptotic_options={
                'gamma': 0.1,
                'objective_atol': 1e-8,
                'objective_rtol': 0,
                'max_iter': 2,
            },
        )

# --------------------------
# Edge Case: Zero Epsilon
# --------------------------

# def test_zero_epsilon_case():
#     """Verify behavior when epsilon=0 (non-robust)"""
#     X, y = make_classification(n_samples=100, n_features=10)
#     model = WassersteinDRO(
#         input_dim=10,
#         model_type='svm',
#         solver='MOSEK'
#     )
#     model.update({'eps': 0.0})

#     params = model.fit(X, y)
#     dist = model.worst_distribution(X, y)
#     assert np.allclose(dist['sample_pts'][0], X)
