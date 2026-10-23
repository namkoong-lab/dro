import pytest
import numpy as np
import cvxpy as cp
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.cvar_dro import CVaRDRO, CVaRDROError

# --------------------------
# Test Data Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate test datasets for classification and regression tasks"""
    n_samples = 100
    n_features = 5
    
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
# Initialization Validation
# --------------------------

@pytest.mark.parametrize("alpha,valid", [
    (0.5, True),
    (1.0, True),
    (-0.1, False),
    (1.1, False),
    (0.0, False)
])
def test_initialization_validation(alpha, valid):
    """Test parameter validation during initialization"""
    if valid:
        CVaRDRO(input_dim=5, alpha=alpha)
    else:
        with pytest.raises(ValueError):
            CVaRDRO(input_dim=5, alpha=alpha)

# --------------------------
# Configuration Update Tests
# --------------------------

def test_config_update_mechanism():
    """Test dynamic configuration updates"""
    model = CVaRDRO(input_dim=3, alpha=0.9)
    
    # Valid alpha update
    model.update({"alpha": 0.95})
    assert np.isclose(model.alpha, 0.95)
    
    # Invalid alpha value
    with pytest.raises(CVaRDROError):
        model.update({"alpha": 1.5})
    
    # Unrelated keys should be ignored
    model.update({"invalid_key": 0.1})  # No error raised

# --------------------------
# Model Fitting Tests
# --------------------------

@pytest.mark.parametrize("model_type", ['svm', 'logistic', 'ols', 'lad'])
def test_fit_success(model_type):
    """Test successful fitting across model types"""
    if model_type in {"svm", "logistic"}:
        X, y = make_classification(n_samples=100, n_features=5)
        y = np.sign(y-0.5)
    else:
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            random_state=42
        )

    model = CVaRDRO(
        input_dim=X.shape[1],
        model_type=model_type,
        alpha=0.95
    )
    assert model.solver == 'CLARABEL'
    
    params = model.fit(X, y)
    
    # Validate parameter shapes
    assert len(params["theta"]) == X.shape[1]
    assert isinstance(model.robust_obj, float)
    assert np.isfinite(model.robust_obj)
    if model.fit_intercept:
        assert "b" in params
    assert "threshold" in params

    model.update_kernel({"metric": "rbf"})
    model.fit(X, y)

# --------------------------
# Input Validation Tests
# --------------------------

def test_input_dimension_mismatch():
    """Test feature dimension validation"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)
    model = CVaRDRO(input_dim=X.shape[1]+1, alpha=0.9)
    with pytest.raises(CVaRDROError):
        model.fit(X, y)

def test_sample_size_mismatch():
    """Test sample count validation"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)
    model = CVaRDRO(input_dim=X.shape[1], alpha=0.9)
    with pytest.raises(CVaRDROError):
        model.fit(X, y[:-1])  # Different sample count

# --------------------------
# Worst-case Distribution Tests
# --------------------------

def test_worst_distribution_properties():
    """Validate worst-case distribution properties"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)
    model = CVaRDRO(input_dim=X.shape[1], alpha=0.9)
    model.fit(X, y)
    
    dist = model.worst_distribution(X, y)
    weights = dist["weight"]
    
    # Validate probability distribution properties
    if len(weights) > 0:
        assert np.allclose(weights.sum(), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-8)  # Allow small numerical errors


def test_alpha_one_worst_distribution():
    """At alpha one, CVaR returns the empirical distribution."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    model = CVaRDRO(input_dim=X.shape[1], model_type='ols', alpha=1.0)

    dist = model.worst_distribution(X, y)

    np.testing.assert_array_equal(dist['sample_pts'][0], X)
    np.testing.assert_array_equal(dist['sample_pts'][1], y)
    np.testing.assert_allclose(dist['weight'], np.full(X.shape[0], 1.0 / X.shape[0]))

# --------------------------
# Solver Compatibility Tests
# --------------------------

def test_explicit_clarabel_solver():
    """An explicit Clarabel selection solves the CVaR formulation."""
    X, y = make_classification(n_samples=50, n_features=10)
    y = np.sign(y-0.5)
    model = CVaRDRO(
        input_dim=10,
        alpha=0.9,
        solver='CLARABEL'
    )
    params = model.fit(X, y)
    assert params["theta"] is not None


@pytest.mark.parametrize("model_type", ['logistic', 'ols', 'lad'])
def test_clarabel_and_scs_agree_on_objective(model_type):
    """Compare the same fitted objective across two conic solvers."""
    if 'SCS' not in cp.installed_solvers():
        pytest.skip("SCS is not installed")

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    if model_type in {'svm', 'logistic'}:
        y = np.array([-1, -1, 1, -1, 1, 1])
    else:
        y = np.array([-1.1, -0.8, 0.4, 1.1, 1.9, 2.8])

    objectives = {}
    for solver in ('CLARABEL', 'SCS'):
        kwargs = {} if solver == 'CLARABEL' else {'solver': solver}
        model = CVaRDRO(input_dim=1, model_type=model_type, alpha=0.8, **kwargs)
        assert model.solver == solver
        model.fit(X, y)
        objectives[solver] = model.robust_obj

    assert np.isfinite(objectives['CLARABEL'])
    assert objectives['CLARABEL'] == pytest.approx(
        objectives['SCS'], rel=5e-3, abs=5e-3
    )

# --------------------------
# Edge Case Handling
# --------------------------

def test_extreme_alpha_values():
    """Test boundary alpha values"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)
    
    # Near-1 alpha
    model = CVaRDRO(input_dim=5)
    model.update({"eps":0.999})
    model.fit(X, y)
    
    # Near-0 alpha
    with pytest.raises(ValueError):  # Should fail due to alpha <=0
        model = CVaRDRO(input_dim=5, alpha=-1e-9)

# --------------------------
# Threshold Validation
# --------------------------

def test_threshold_calculation(dataset):
    """Validate threshold value calculation"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)

    model = CVaRDRO(input_dim=X.shape[1], alpha=0.9)
    params = model.fit(X, y)
    
    assert params["threshold"] == model.threshold_val
