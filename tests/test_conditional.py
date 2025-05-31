import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.conditional_dro import (  
    ConditionalCVaRDRO, 
    ConditionalCVaRDROError
)
from src.dro.linear_model.base import ParameterError, InstallError

# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate standardized test datasets"""
    n_samples = 100
    n_features = 5
    
    if request.param == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            random_state=42
        )
        y = np.where(y > 0, 1, -1)  # Convert to ±1 labels
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42
        )
        
    return X, y

# --------------------------
# Initialization Tests
# --------------------------

@pytest.mark.parametrize("input_dim,valid", [
    (5, True),
    (1, True),
    (0, False),
    (-1, False)
])
def test_initialization_validation(input_dim, valid):
    """Test input dimension validation during initialization"""
    if valid:
        ConditionalCVaRDRO(input_dim=input_dim)
    else:
        with pytest.raises(ValueError):
            ConditionalCVaRDRO(input_dim=input_dim)

@pytest.mark.parametrize("model_type,valid", [
    ('svm', True),
    ('logistic', True),
    ('ols', True),
    ('lad', True),
    ('invalid', False),
    (None, False)
])
def test_model_type_validation(model_type, valid):
    """Test model type parameter validation"""
    if valid:
        ConditionalCVaRDRO(input_dim=5, model_type=model_type)
    else:
        with pytest.raises(ParameterError):
            ConditionalCVaRDRO(input_dim=5, model_type=model_type)

# --------------------------
# Configuration Update Tests
# --------------------------

@pytest.mark.parametrize("control_indices,valid", [
    ([0, 2], True),
    ([4], True),  # input_dim=5, max index=4
    ([5], False),
    ([-1], False),
    (None, True)
])
def test_control_feature_validation(control_indices, valid):
    """Test control feature index validation"""
    model = ConditionalCVaRDRO(input_dim=5)
    if valid:
        model.update({'control_name': control_indices})
    else:
        with pytest.raises(ConditionalCVaRDROError):
            model.update({'control_name': control_indices})

@pytest.mark.parametrize("alpha,valid", [
    (0.5, True),
    (1.0, True),
    (0.0, False),
    (1.1, False),
    (-0.5, False)
])
def test_alpha_validation(alpha, valid):
    """Test CVaR alpha parameter validation"""
    model = ConditionalCVaRDRO(input_dim=5)
    if valid:
        model.update({'alpha': alpha})
    else:
        with pytest.raises(ConditionalCVaRDROError):
            model.update({'alpha': alpha})

# --------------------------
# Training Workflow Tests
# --------------------------

def test_fit_interface(dataset):
    """Verify basic training workflow and output structure"""
    X, y = dataset
    model = ConditionalCVaRDRO(
        input_dim=X.shape[1],
        model_type='svm' if np.unique(y).size == 2 else 'lad',
        fit_intercept=True
    )
    
    # Valid case
    params = model.fit(X, y)
    assert 'theta' in params
    assert 'b' in params
    assert len(params['theta']) == X.shape[1]

    model.update_kernel({"metric": "rbf"})
    model.fit(X, y)
    
def test_feature_dimension_mismatch():
    """Test input dimension mismatch handling"""
    model = ConditionalCVaRDRO(input_dim=5)
    X = np.random.randn(10, 3)  # Expected 5 features
    y = np.random.randn(10)
    
    with pytest.raises(ConditionalCVaRDROError):
        model.fit(X, y)

# --------------------------
# Control Feature Functionality Tests
# --------------------------

def test_controlled_feature_behavior():
    """Test model behavior with controlled features"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)
    control_indices = [0, 2]
    
    model = ConditionalCVaRDRO(
        input_dim=5,
    )
    model.update({"control_name":control_indices, "alpha":0.9})

    params = model.fit(X, y)
    
    
# --------------------------
# Numerical Stability Tests
# --------------------------

def test_extreme_alpha_values():
    """Test edge cases for CVaR alpha parameter"""
    X, y = make_regression(n_samples=10000, n_features=5)
    
    # Alpha approaching 1.0
    model = ConditionalCVaRDRO(input_dim=5, model_type='ols')
    model.update({"alpha":0.9})
    params = model.fit(X, y)
    assert not np.isnan(params['theta']).any()

# --------------------------
# Error Condition Tests
# --------------------------

def test_solver_failure_handling():
    """Test proper error reporting for solver failures"""
    X, y = make_classification(n_samples=10, n_features=5)
    with pytest.raises(InstallError):
        model = ConditionalCVaRDRO(input_dim=5, solver='INVALID_SOLVER')
        model.fit(X, y)
