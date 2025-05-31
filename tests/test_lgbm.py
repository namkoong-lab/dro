import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from src.dro.tree_model.lgbm import KLDRO_LGBM, CVaRDRO_LGBM, Chi2DRO_LGBM, NotFittedError 
import lightgbm

# --------------------------
# Test Data Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate test datasets for both classification and regression tasks"""
    n_samples = 500
    n_features = 5
    
    if request.param == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42
        )
        
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# KLDRO_LGBM Test Class
# --------------------------

class TestKLDRO_LGBM:
    """Test core functionality of KL-DRO LightGBM model"""
    
    def test_invalid_initialization(self):
        """Validate exception handling for invalid initialization parameters"""
        # Test invalid epsilon
        with pytest.raises(ValueError):
            KLDRO_LGBM(eps=-0.1)
            
        # Test invalid task type
        with pytest.raises(ValueError):
            KLDRO_LGBM(kind='invalid_task')
    
    def test_config_update_validation(self):
        """Verify configuration update validation logic"""
        model = KLDRO_LGBM()
        
        # Test non-dictionary config
        with pytest.raises(TypeError):
            model.update("invalid_config")
            
        # Test missing required parameter
        with pytest.raises(KeyError):
            model.update({"learning_rate": 0.1})
    
    def test_training_workflow(self):
        """End-to-end training pipeline validation"""
        X_train, y_train = make_classification(n_samples=100, random_state=42)        
        # Model initialization
        model = KLDRO_LGBM(eps=0.1)
        model.update({
            "num_boost_round": 10,
            "max_depth": 2,
            "learning_rate": 0.1
        })
        
        # Training validation
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        assert model.model is not None
        
        # Prediction validation
        preds = model.predict(X_train)
        assert preds.shape == y_train.shape
        
        # Classification output check
        if model.kind == 'classification':
            assert set(preds).issubset({0, 1})

    def test_training_workflow2(self):
        """End-to-end training pipeline validation"""
        X_train, y_train = make_regression(n_samples=100, random_state=42)         
        # Model initialization
        model = KLDRO_LGBM(eps=0.1, kind="regression")
        model.update({
            "num_boost_round": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eps":0.5
        })
        
        # Training validation
        model.fit(X_train, y_train)
        model.score(X_train, y_train)

        assert model.model is not None
        
        # Prediction validation
        preds = model.predict(X_train)
        assert preds.shape == y_train.shape
        
        
    def test_unfitted_error(self, dataset):
        """Verify proper error handling for untrained model usage"""
        X_train, X_test, _, _ = dataset
        model = KLDRO_LGBM()
        
        with pytest.raises(NotFittedError):
            model.predict(X_test)
            
        # Test training with empty config
        model = KLDRO_LGBM()
        with pytest.raises(RuntimeError):
            model.fit(X_train, np.random.randn(X_train.shape[0]))

# --------------------------
# CVaRDRO_LGBM Test Class  
# --------------------------

class TestCVaRDRO_LGBM:
    """Test core functionality of CVaR-DRO LightGBM model"""
    
    @pytest.mark.parametrize("eps", [0.0, 1.0, 1.5])
    def test_invalid_epsilon(self, eps):
        """Validate CVaR epsilon range constraints"""
        with pytest.raises(ValueError):
            CVaRDRO_LGBM(eps=eps)

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            CVaRDRO_LGBM(kind="")


        model = CVaRDRO_LGBM()
        with pytest.raises(TypeError):
            model.update(["config",1])
        model = CVaRDRO_LGBM()
        with pytest.raises(KeyError):
            model.update({'eps':1})
    
    def test_cvar_loss_mechanism(self):
        """Verify CVaR loss calculation logic"""
        model = CVaRDRO_LGBM(eps=0.2)
        X, y = make_classification(n_samples=100, random_state=42)
        model.update({"num_boost_round": 5, "max_depth": 2})
        model.fit(X, y)
        model.score(X, y)

        
        # Validate model type
        assert isinstance(model.model, lightgbm.Booster)
        
    def test_regression_support(self):
        """Validate regression task implementation"""
        X, y = make_regression(n_samples=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = CVaRDRO_LGBM(kind='regression', eps=0.1)
        model.update({"num_boost_round": 10, "learning_rate": 0.1, "eps":0.5})
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        
        preds = model.predict(X_test)
        assert preds.dtype == float

# --------------------------
# Chi2DRO_LGBM Test Class  
# --------------------------

class TestChi2DRO_LGBM:
    """Test core functionality of CVaR-DRO LightGBM model"""
    
    @pytest.mark.parametrize("eps", [0.0, -1.0])
    def test_invalid_epsilon(self, eps):
        """Validate CVaR epsilon range constraints"""
        with pytest.raises(ValueError):
            Chi2DRO_LGBM(eps=eps)

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            Chi2DRO_LGBM(kind="")

        model = Chi2DRO_LGBM()
        with pytest.raises(TypeError):
            model.update(["config",1])
        model = Chi2DRO_LGBM()
        with pytest.raises(KeyError):
            model.update({'eps':1})

    def test_cvar_loss_mechanism(self):
        """Verify CVaR loss calculation logic"""
        model = Chi2DRO_LGBM(eps=0.2)
        X, y = make_classification(n_samples=100, random_state=42)
        model.update({"num_boost_round": 5, "max_depth": 2})
        model.fit(X, y)
        model.score(X, y)
        
        # Validate model type
        assert isinstance(model.model, lightgbm.Booster)
        
    def test_regression_support(self):
        """Validate regression task implementation"""
        X, y = make_regression(n_samples=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = Chi2DRO_LGBM(kind='regression', eps=0.1)
        model.update({"num_boost_round": 10, "learning_rate": 0.1, "eps":0.5})
        model.fit(X_train, y_train)
        model.score(X_train, y_train)


        preds = model.predict(X_test)
        assert preds.dtype == float

# --------------------------
# Shared Functionality Tests
# --------------------------

def test_loss_calculation():
    """Validate base loss function implementations"""
    # Classification loss
    model = KLDRO_LGBM()
    preds = np.array([0.9, 0.1])
    labels = np.array([1, 0])
    loss = model.loss(preds, labels)
    assert np.allclose(loss, [-np.log(0.9), -np.log(0.9)], rtol=1e-3)
    
    # Regression loss
    model = KLDRO_LGBM(kind='regression')
    preds = np.array([3.0, 2.5])
    labels = np.array([2.5, 2.0])
    loss = model.loss(preds, labels)
    assert np.array_equal(loss, (0.5**2, 0.5**2))

def test_robustness_to_randomness():
    """Verify model stability across different random seeds"""
    X, y = make_classification(n_samples=1000, random_state=42)
    accuracies = []
    
    for seed in range(3):  # Multi-seed validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        model = KLDRO_LGBM(eps=0.1)
        model.update({"num_boost_round": 10, "max_depth": 3, "learning_rate": 0.1})
        model.fit(X_train, y_train)
        
        acc = (model.predict(X_test) == y_test).mean()
        accuracies.append(acc)
    
    # Validate performance stability
    assert np.std(accuracies) < 0.05, "Excessive performance variance across seeds"