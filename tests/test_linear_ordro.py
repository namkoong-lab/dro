import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.or_wasserstein_dro import ORWDRO, ORWDROError

# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification'])
def dataset(request):
    """Generate standardized test datasets"""
    n_samples = 100
    n_features = 10
    
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

@pytest.mark.parametrize("model_type,valid", [
    ('svm', True),
    ('lad', True),
    ('logistic', False),
    ('invalid', False)
])
def test_model_type_validation(model_type, valid):
    """Test constructor parameter validation"""
    if valid:
        ORWDRO(input_dim=5, model_type=model_type)
    else:
        with pytest.raises(ValueError):
            ORWDRO(input_dim=5, model_type=model_type)

# --------------------------
# Configuration Update Tests
# --------------------------

def test_parameter_update_mechanism():
    """Test dynamic parameter updates"""
    model = ORWDRO(input_dim=3, model_type='svm')
    
    # Valid updates
    model.update({
        'eps': 0.2,
        'eta': 0.1,
        'dual_norm': 2
    })
    assert model.eps == 0.2
    assert model.eta == 0.1
    assert model.dual_norm == 2
    
    # Invalid parameter types
    with pytest.raises(ValueError):
        model.update({'eta': -0.1})

# --------------------------
# Training Workflow Tests
# --------------------------

def test_fit_interface(dataset):
    """Validate fit method input/output contracts"""
    X, y = dataset
    model = ORWDRO(
        input_dim=X.shape[1],
        model_type='svm' if np.unique(y).size == 2 else 'lad'
    )
    
    # Valid case
    params = model.fit(X, y)
    assert 'theta' in params
    assert len(params['theta']) == X.shape[1]
    
    # Dimension mismatch
    with pytest.raises(ORWDROError):
        model.fit(X[:, :-1], y)

# # --------------------------
# # Worst-case Distribution Tests
# # --------------------------

def test_worst_distribution_structure(dataset):
    """Validate worst-case distribution output structure"""
    X, y = dataset
    model = ORWDRO(
        input_dim=X.shape[1],
        model_type='svm' if np.unique(y).size == 2 else 'lad',
        eps=0.1,
        eta=0.
    )
    model.fit(X, y)
    
    dist = model.worst_distribution(X, y)
    # print(dist)
    # assert 'sample_pts' in dist
    # assert 'weight' in dist
    # assert len(dist['weight']) == X.shape[0] * 2  # J=2 components
    # assert np.all(dist['weight'] >= 0)
    # assert np.isclose(sum(dist['weight']), 1.0, atol=1e-3)

# --------------------------
# Edge Case Tests
# --------------------------

# def test_vanilla_erm_case():
#     """Test degenerate case with eps=0 and eta=0"""
#     X, y = make_regression(n_samples=50, n_features=3)
#     model = ORWDRO(
#         input_dim=3,
#         model_type='lad',
#         eps=0.0,
#         eta=0.0
#     )
#     params = model.fit(X, y)
    
#     # Should match ordinary LAD
#     from sklearn.linear_model import QuantileRegressor
#     sk_model = QuantileRegressor(alpha=0.5, solver='highs')
#     sk_model.fit(X, y)
#     assert np.allclose(params['theta'], sk_model.coef_, atol=1e-2)

# --------------------------
# Error Condition Tests
# --------------------------

# def test_nan_input_handling():
#     """Test detection of invalid input values"""
#     model = ORWDRO(input_dim=2, model_type='svm')
#     X = np.array([[1.0, 2.0], [np.nan, 3.0]])
#     y = np.array([1, -1])
    
#     with pytest.raises(ORWDROError):
#         model.fit(X, y)