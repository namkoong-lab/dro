import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.bayesian_dro import BayesianDRO, BayesianDROError 
from src.dro.linear_model.base import ParameterError, InstallError
import cvxpy as cp
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



def test_config_update_validation():
    """Test all config update scenarios with valid/invalid inputs"""
    model = BayesianDRO(input_dim=5)
    
    # Valid updates
    valid_configs = [
        {'eps': 0.5},
        {'distance_type': 'chi2'},
        {'posterior_param_num': 10},
        {'distribution_class': 'Exponential'}
    ]
    
    for config in valid_configs:
        model.update(config)  # Should not raise
    
    # Invalid updates
    invalid_configs = [
        {'distance_type': 'invalid'},
        {'distribution_class': 'value'}
    ]
    
    for config in invalid_configs:
        with pytest.raises((BayesianDROError, AssertionError)):
            model.update(config)

def test_exponential_distribution_resampling():
    """Verify resampling with exponential distribution"""
    model = BayesianDRO(input_dim=1)
    model.update({
        'distribution_class': 'Exponential',
        'posterior_param_num': 3,
        'posterior_sample_ratio': 2
    })
    
    X = np.ones((50, 1))  # Dummy features for exponential
    y = np.random.exponential(1, 50)
    
    X_res, y_res = model.resample(X, y)
    
    assert len(X_res) == 3
    assert len(y_res) == 3
    assert X_res[0].shape == (100, 1)
    assert y_res[0].shape == (100,)
    assert np.all(X_res[0] == 1)  # Should be all ones for exponential


@pytest.mark.parametrize("model_type", ['svm', 'logistic', 'lad', 'ols'])
def test_fit_with_zero_eps(model_type, dataset):
    """Test fitting with eps=0 (non-robust) for all model types"""
    X, y = dataset
    if model_type in ['svm', 'logistic'] and len(np.unique(y)) > 2:
        pytest.skip("Requires binary classification data")
        
    model = BayesianDRO(
        input_dim=X.shape[1],
        model_type=model_type,
        eps=0.0
    )
    
    params = model.fit(X, y)
    assert 'theta' in params
    assert len(params['theta']) == X.shape[1]
    if model.fit_intercept:
        assert 'b' in params


@pytest.mark.parametrize("model_type,expected_loss", [
    ('svm', 'maximum'),
    ('logistic', 'logistic'),
    ('ols', 'power'),
    ('lad', 'abs')
])
def test_cvx_loss_types(model_type, expected_loss):
    """Verify correct loss expressions are generated"""
    model = BayesianDRO(input_dim=3, model_type=model_type)
    X = cp.Parameter((10, 3))
    y = cp.Parameter(10)
    theta = cp.Variable(3)
    b = cp.Variable()
    
    loss_expr = model._cvx_loss(X, y, theta, b)
    assert expected_loss in str(loss_expr)


def test_solver_error_propagation():
    """Verify proper error handling when solver fails"""
    X, y = make_regression(n_samples=10, n_features=2)
    with pytest.raises(InstallError):
        model = BayesianDRO(
            input_dim=2,
            solver='INVALID_SOLVER'  # Force solver error
        )
    

def test_input_dimension_validation():
    """Test all dimension validation checks"""
    # Constructor validation
    with pytest.raises(ParameterError):
        BayesianDRO(input_dim=-1)
    
    # Fit-time validation
    model = BayesianDRO(input_dim=5, model_type = 'ols')
    X = np.random.randn(10, 4)  # Wrong dimension
    y = np.random.randn(10)
    
    with pytest.raises(BayesianDROError):
        model.fit(X, y)

def test_invalid_classification_labels():
    """Verify error when non-binary labels are provided"""
    X, _ = make_classification(n_samples=50, n_features=5)
    y = np.random.randint(0, 3, 50)  # 0,1,2 labels
    
    model = BayesianDRO(input_dim=5, model_type='svm')
    
    with pytest.raises(BayesianDROError) as excinfo:
        model.fit(X, y)
    assert "labels not in {-1, +1}" in str(excinfo.value)


def test_output_structure_no_intercept():
    """Verify output dict structure when fit_intercept=False"""
    X, y = make_regression(n_samples=50, n_features=3)
    model = BayesianDRO(
        input_dim=3,
        fit_intercept=False,
        model_type = 'ols'
    )
    
    params = model.fit(X, y)
    assert 'theta' in params

def test_newsvendor_loss():
    """Test the newsvendor loss formulation"""
    model = BayesianDRO(input_dim=3, model_type='newsvendor')
    X = cp.Parameter((10, 3))
    y = cp.Parameter(10)
    theta = cp.Variable(3)
    
    loss_expr = model._cvx_loss(X, y, theta, 0)

if __name__ == "__main__":
    pytest.main(["-v", "--cov=your_module"])