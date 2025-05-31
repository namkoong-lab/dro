import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.sinkhorn_dro import SinkhornLinearDRO, SinkhornDROError 
from src.dro.linear_model.base import ParameterError
import torch 

# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate standardized test datasets"""
    n_samples = 200
    n_features = 5
    
    if request.param == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            random_state=42
        )
        y = (y > 0).astype(float)  # Convert to 0/1 labels
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42
        )
        
    return X, y.reshape(-1, 1)

# --------------------------
# Initialization Tests
# --------------------------

@pytest.mark.parametrize("model_type,valid", [
    ('svm', True),
    ('logistic', True),
    ('ols', True),
    ('lad', True),
    ('invalid', False)
])
def test_initialization_validation(model_type, valid):
    """Test constructor parameter validation"""
    if valid:
        SinkhornLinearDRO(input_dim=5, model_type=model_type)
    else:
        with pytest.raises(ParameterError):
            SinkhornLinearDRO(input_dim=-5, model_type=model_type)

# --------------------------
# Configuration Update Tests
# --------------------------

def test_config_update_mechanism():
    """Test dynamic configuration updates"""
    model = SinkhornLinearDRO(input_dim=3, model_type='svm')
    
    # Valid updates
    model.update({
        'reg': 0.01,
        'lambda': 50.0,
        'k_sample_max': 3
    })
    assert model.reg_param == 0.01
    assert model.lambda_param == 50.0
    assert model.k_sample_max == 3
    
    # Invalid parameter types
    with pytest.raises(ValueError):
        model.update({'reg': -0.1})

# --------------------------
# Prediction Interface Tests
# --------------------------

def test_predict_interface(dataset):
    """Validate prediction input/output contracts"""
    X, y = dataset
    model = SinkhornLinearDRO(input_dim=X.shape[1], model_type='ols')
    
    # Valid case
    preds = model.predict(X)
    assert preds.shape == (X.shape[0], 1)
    
    # Dimension mismatch
    with pytest.raises(ValueError):
        model.predict(X[:, :-1])

# --------------------------
# Scoring Function Tests
# --------------------------

def test_regression_scoring():
    """Test scoring for regression models"""
    X, y = make_regression(n_samples=100, n_features=5)
    model = SinkhornLinearDRO(input_dim=5, model_type='ols')
    
    # Mock perfect predictions
    mock_pred = y.reshape(-1, 1)
    model.predict = lambda x: mock_pred
    assert model.score(X, y) == pytest.approx(0.0)

def test_classification_scoring():
    """Test scoring for classification models"""
    X, y = make_classification(n_samples=100, n_features=5)
    y = 2 * y - 1
    model = SinkhornLinearDRO(input_dim=5, model_type='svm')
    
    # Mock perfect predictions
    mock_pred = y.reshape(1, -1)
    model.predict = lambda x: mock_pred
    acc, __ = model.score(X, y)
    assert acc == 1.0

# --------------------------
# Training Workflow Tests
# --------------------------

@pytest.mark.parametrize("optim_type", ['SG', 'MLMC', 'RTMLMC'])
def test_training_workflow(dataset, optim_type):
    """Validate end-to-end training workflow"""
    X, y = dataset
    model = SinkhornLinearDRO(
        input_dim=X.shape[1],
        model_type='ols' if y.ndim > 1 else 'svm',
        max_iter=10,
        learning_rate=0.1
    )
    
    params = model.fit(X, y, optimization_type=optim_type)
    assert 'theta' in params
    assert params['theta'].shape == (X.shape[1],)
    if model.fit_intercept:
        assert 'bias' in params

@pytest.mark.parametrize("optim_type", ['SG', 'MLMC', 'RTMLMC'])
def test_training_workflow2(optim_type):
    """Validate end-to-end training workflow"""
    X, y = make_classification(n_samples=100, n_features=10)
    
    model = SinkhornLinearDRO(
        input_dim=X.shape[1],
        model_type='logistic',
        max_iter=10,
        learning_rate=0.1
    )
    
    params = model.fit(X, y, optimization_type=optim_type)
    assert 'theta' in params
    assert params['theta'].shape == (X.shape[1],)
    if model.fit_intercept:
        assert 'bias' in params

@pytest.mark.parametrize("optim_type", ['SG', 'MLMC', 'RTMLMC'])
def test_training_workflow3(optim_type):
    """Validate end-to-end training workflow"""
    X, y = make_regression(n_samples=100, n_features=10)
    model = SinkhornLinearDRO(
        input_dim=X.shape[1],
        model_type='lad',
        max_iter=10,
        learning_rate=0.1
    )
    
    params = model.fit(X, y, optimization_type=optim_type)
    assert 'theta' in params
    assert params['theta'].shape == (X.shape[1],)
    if model.fit_intercept:
        assert 'bias' in params

# --------------------------
# Device Compatibility Tests
# --------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_execution():
    """Verify basic GPU execution capability"""
    model = SinkhornLinearDRO(input_dim=5, device='cuda')
    X = np.random.randn(10, 5).astype(np.float32)
    
    # Should not raise errors
    model.predict(X)

# --------------------------
# Error Condition Tests
# --------------------------

def test_invalid_optimization_type():
    """Test rejection of invalid optimization types"""
    model = SinkhornLinearDRO(input_dim=5)
    with pytest.raises(SinkhornDROError):
        model.fit(np.random.randn(10,5), np.random.randn(10), 'INVALID')

def test_nan_input_handling():
    """Test detection of invalid input values"""
    model = SinkhornLinearDRO(input_dim=2)
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(ValueError):
        model.predict(X)
