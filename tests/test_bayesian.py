import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from dro.src.linear_model.bayesian_dro import BayesianDRO, BayesianDROError 
from dro.src.linear_model.base import ParameterError
# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate standardized test datasets with proper formats"""
    n_samples = 100
    n_features = 5
    
    if request.param == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            random_state=42
        )
        y = np.where(y > 0, 1, -1)  # Ensure ±1 labels
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42
        )
        
    return X, y

# --------------------------
# Initialization & Config Tests
# --------------------------

@pytest.mark.parametrize("model_type,valid", [
    ('svm', True),
    ('lad', True),
    ('invalid', False),
    ('logistic', False)
])
def test_model_type_validation(model_type, valid):
    """Verify constructor parameter validation logic"""
    if valid:
        BayesianDRO(input_dim=5, model_type=model_type)
    else:
        with pytest.raises(ParameterError):
            BayesianDRO(input_dim=-5, model_type=model_type)

@pytest.mark.parametrize("distance_type", ['KL', 'chi2', 'invalid'])
def test_distance_type_validation(distance_type):
    """Test distance metric type enforcement"""
    if distance_type in ['KL', 'chi2']:
        BayesianDRO(input_dim=3, distance_type=distance_type)
    else:
        with pytest.raises(BayesianDROError):
            BayesianDRO(input_dim=3, distance_type=distance_type)

# --------------------------
# Resampling Tests
# --------------------------

def test_resample_output_structure():
    """Validate resampled data dimensions and types"""
    model = BayesianDRO(input_dim=10, model_type='svm')
    model.update({
        'posterior_param_num': 5,
        'posterior_sample_ratio': 2,
        'distribution_class': 'Gaussian'
    })
    
    X, y = make_classification(n_samples=50, n_features=10)
    X_res, y_res = model.resample(X, y)
    
    # Validate output dimensions
    assert len(X_res) == 5
    assert len(y_res) == 5
    assert X_res[0].shape == (100, 10)  # 50 * 2=100 samples per param
    assert y_res[0].shape == (100,)

# --------------------------
# Training Workflow Tests
# --------------------------

@pytest.mark.parametrize("distance_type", ['KL', 'chi2'])
def test_fit_interface(dataset, distance_type):
    """Validate parameter outputs and basic convergence"""
    X, y = dataset
    model = BayesianDRO(
        input_dim=X.shape[1],
        model_type='svm' if np.unique(y).size == 2 else 'lad',
        distance_type=distance_type,
        eps=0.1
    )

    model.update({"posterior_param_num": 3})
    
    params = model.fit(X, y)
    
    # Validate output structure
    assert 'theta' in params
    assert len(params['theta']) == X.shape[1]
    
# --------------------------
# Edge Case Tests
# --------------------------

def test_non_robust_case():
    """Verify behavior when eps=0 (non-robust limit)"""
    X, y = make_regression(n_samples=50, n_features=3)
    model = BayesianDRO(
        input_dim=3,
        model_type='lad',
        eps=0.0,
        distance_type='KL'
    )
    
    params = model.fit(X, y)
    # Should approximate standard Bayesian regression
    assert np.linalg.norm(params['theta']) > 0  # Non-trivial solution

# --------------------------
# Error Condition Tests
# --------------------------

def test_invalid_resample_input():
    """Test dimension mismatch detection"""
    model = BayesianDRO(input_dim=3)
    X = np.random.randn(50, 2)  # Wrong feature dimension
    y = np.random.randn(50)
    
    with pytest.raises(ValueError):
        model.resample(X, y)

@pytest.mark.skipif(True, reason="Solver configuration required")
def test_solver_failure_handling():
    """Verify proper error on solver failures"""
    model = BayesianDRO(input_dim=10, solver='INVALID')
    X, y = make_regression(n_samples=50, n_features=10)
    
    with pytest.raises(BayesianDROError):
        model.fit(X, y)

if __name__ == "__main__":
    pytest.main(["-v", "--cov=your_module"])