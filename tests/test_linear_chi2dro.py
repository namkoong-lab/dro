import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.chi2_dro import Chi2DRO, Chi2DROError  
from src.dro.linear_model.base import ParameterError 


@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    n_samples = 100
    n_features = 5
    
    if request.param == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            random_state=42
        )
        y = np.sign(y - 0.5)  
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42
        )
        
    return X, y


def test_invalid_initialization():
    with pytest.raises(ParameterError):
        Chi2DRO(input_dim=-1)
        


def test_config_update_validation():
    model = Chi2DRO(input_dim=5)
    
    model.update({"eps": 0.5})
    assert model.eps == 0.5
    
    with pytest.raises(Chi2DROError):
        model.update({"eps": "invalid"})
    
@pytest.mark.parametrize("model_type", ['svm', 'logistic', 'ols', 'lad'])



def test_fit_success(model_type):
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
    
    model = Chi2DRO(
        input_dim=X.shape[1],
        model_type=model_type,
        fit_intercept=True
    )
    model.update({"eps": 0.1})
    
    params = model.fit(X, y)
    
    assert len(params["theta"]) == X.shape[1]
    if model.fit_intercept:
        assert "b" in params

    model.update_kernel({"metric": "rbf"})
    model.fit(X, y)

def test_fit_failure(dataset):
    X, y = dataset
    
    model = Chi2DRO(input_dim=X.shape[1] + 1)
    with pytest.raises(Chi2DROError):
        model.fit(X, y)
        
    with pytest.raises(Chi2DROError):
        model.fit(X, y[:-1])

def test_worst_distribution(dataset):
    X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
    y = np.sign(y - 0.5) 

    model = Chi2DRO(input_dim=X.shape[1])
    model.update({"eps":0.1})
    model.fit(X, y)
    
    dist = model.worst_distribution(X, y)
    weights = dist["weight"]
    
    assert np.allclose(weights.sum(), 1.0, atol=1e-3)
    assert np.all(weights >= 0)


def test_solver_fallback():
    X, y = make_classification(n_samples=50, n_features=10)
    y = np.sign(y-0.5)
    
    model = Chi2DRO(
        input_dim=10,
        solver='MOSEK',
    )
    model.update({"eps":0.1})
    params = model.fit(X, y)
    
    assert params["theta"] is not None


def test_extreme_epsilon():
    X, y = make_classification(n_samples=100, n_features=5)
    y = np.sign(y-0.5)
    
    model = Chi2DRO(input_dim=5)
    model.update({"eps":1e6})
    model.fit(X, y)
    
    model.update({"eps": 1e-9})
    model.fit(X, y)
