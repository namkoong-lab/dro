import pytest
import numpy as np
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
    
    params = model.fit(X, y)
    
    # Validate parameter shapes
    assert len(params["theta"]) == X.shape[1]
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

# --------------------------
# Solver Compatibility Tests
# --------------------------

def test_solver_fallback():
    """Test solver compatibility with open-source alternatives"""
    X, y = make_classification(n_samples=50, n_features=10)
    y = np.sign(y-0.5)
    model = CVaRDRO(
        input_dim=10,
        alpha=0.9,
        solver='MOSEK'  # Open-source solver
    )
    params = model.fit(X, y)
    assert params["theta"] is not None

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
    