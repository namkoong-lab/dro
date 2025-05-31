import pytest
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from src.dro.tree_model.xgb import KLDRO_XGB, CVaRDRO_XGB, Chi2DRO_XGB, NotFittedError

# --------------------------
# Test Data Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def xgb_dataset(request):
    """Generate test datasets for XGBoost models"""
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
# KLDRO_XGB Test Class
# --------------------------

class TestKLDRO_XGB:
    """Test KL-DRO XGBoost model functionality"""
    
    def test_initialization_validation(self):
        """Verify parameter validation during initialization"""
        with pytest.raises(ValueError):
            KLDRO_XGB(eps=-0.1)
            
        with pytest.raises(ValueError):
            KLDRO_XGB(kind='invalid_type')
    
    def test_config_handling(self):
        """Test configuration update mechanics"""
        model = KLDRO_XGB()
        
        # Test invalid config types
        with pytest.raises(TypeError):
            model.update("invalid_config")
            
        # Test missing required parameter
        with pytest.raises(KeyError):
            model.update({"learning_rate": 0.1})
            
        # Valid config test
        valid_config = {"num_boost_round": 10, "max_depth": 3}
        model.update(valid_config)
        assert model.config == valid_config
    
    def test_full_training_cycle(self):
        """End-to-end training workflow test"""
        X_train, y_train = make_classification(n_samples=100, random_state=42)
        
        model = KLDRO_XGB(eps=0.1)
        model.update({
            "num_boost_round": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eps":0.5
        })
        
        # Training validation
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        assert isinstance(model.model, xgb.Booster)
        
        # Prediction validation
        preds = model.predict(X_train)
        assert preds.shape == y_train.shape
        
        # Output type checks
        if model.kind == 'classification':
            assert np.array_equal(preds, preds.astype(bool))

    def test_regression_support(self):
        """Validate regression task implementation"""
        X, y = make_regression(n_samples=100, random_state=42)
        model = KLDRO_XGB(kind='regression', eps=0.1)
        model.update({"num_boost_round": 10, "learning_rate": 0.1})
        model.fit(X, y)
        model.score(X, y)
        
        preds = model.predict(X)
        assert preds.dtype == np.float32

    def test_error_handling(self):
        """Test error conditions and exceptions"""
        model = KLDRO_XGB()
        
        # Untrained prediction
        with pytest.raises(NotFittedError):
            model.predict(np.random.randn(10, 5))
            
        # Invalid input dimensions
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10), np.random.randn(10))

    def test_error_handling2(self):
        """Test error conditions and exceptions"""
        X, y = make_classification(n_samples=100, n_features=10)
        y = np.sign(y-0.5)
        model = KLDRO_XGB()
        model.config = None 
        # Untrained prediction
        with pytest.raises(RuntimeError):
            model.fit(X, y)

# --------------------------
# CVaRDRO_XGB Test Class
# --------------------------

class TestCVaRDRO_XGB:
    """Test CVaR-DRO XGBoost model functionality"""
    
    @pytest.mark.parametrize("eps", [-0.1, 0.0, 1.0, 1.5])
    def test_epsilon_validation(self, eps):
        """Verify CVaR epsilon parameter constraints"""
        if eps <= 0 or eps >= 1:
            with pytest.raises(ValueError):
                CVaRDRO_XGB(eps=eps)
    

    def test_update_validation(self):
        with pytest.raises(ValueError):
            CVaRDRO_XGB(kind = 'none')
        model = CVaRDRO_XGB()
        with pytest.raises(TypeError):
            model.update(["config",1])
        model = CVaRDRO_XGB()
        with pytest.raises(KeyError):
            model.update({'eps':1})

    def test_cvar_implementation(self):
        """Test CVaR-specific implementation details"""
        X_train, y_train = make_classification(n_samples=100, random_state=42)
        
        model = CVaRDRO_XGB(eps=0.2)
        model.update({
            "num_boost_round": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eps":0.5
        })
        
        model.fit(X_train, y_train)
        model.score(X_train, y_train)

        
    def test_regression_support(self):
        """Validate regression task implementation"""
        X, y = make_regression(n_samples=100, random_state=42)
        model = CVaRDRO_XGB(kind='regression', eps=0.1)
        model.update({"num_boost_round": 10, "learning_rate": 0.1})
        model.fit(X, y)
        model.score(X, y)

        
        preds = model.predict(X)
        assert preds.dtype == np.float32

# --------------------------
# Chi2DRO_XGB Test Class
# --------------------------

class TestChi2DRO_XGB:
    """Test CVaR-DRO XGBoost model functionality"""
    
    @pytest.mark.parametrize("eps", [-0.1, 0.0, 1.0, 1.5])
    def test_epsilon_validation(self, eps):
        """Verify CVaR epsilon parameter constraints"""
        if eps <= 0:
            with pytest.raises(ValueError):
                Chi2DRO_XGB(eps=eps)

    def test_update_validation(self):
        with pytest.raises(ValueError):
            Chi2DRO_XGB(kind = 'none')
        model = Chi2DRO_XGB()
        with pytest.raises(TypeError):
            model.update(["config",1])
        model = Chi2DRO_XGB()
        with pytest.raises(KeyError):
            model.update({'eps':1})





    def test_chi2_implementation(self):
        """Test Chi2-specific implementation details"""
        X_train, y_train = make_classification(n_samples=100, random_state=42)
        
        model = Chi2DRO_XGB(eps=0.2)
        model.update({
            "num_boost_round": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eps":0.5
        })
        
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        
    def test_regression_support(self):
        """Validate regression task implementation"""
        X, y = make_regression(n_samples=100, random_state=42)
        model = Chi2DRO_XGB(kind='regression', eps=0.1)
        model.update({"num_boost_round": 10, "learning_rate": 0.1})
        model.fit(X, y)
        model.score(X, y)
        
        preds = model.predict(X)
        assert preds.dtype == np.float32


# --------------------------
# Shared Functionality Tests
# --------------------------

def test_loss_calculations():
    """Verify loss function implementations"""
    # Classification loss
    model = KLDRO_XGB()
    preds = np.array([0.9, 0.1])
    labels = np.array([1, 0])
    loss = model.loss(preds, labels)
    assert np.allclose(loss, [-np.log(0.9), -np.log(0.9)], rtol=1e-3)
    
    # Regression loss
    model = KLDRO_XGB(kind='regression')
    preds = np.array([3.0, 2.5])
    labels = np.array([2.5, 2.0])
    loss = model.loss(preds, labels)
    assert np.array_equal(loss, (0.5**2, 0.5**2))

def test_dmatrix_handling():
    """Test DMatrix creation and error handling"""
    model = KLDRO_XGB()
    model.update({"num_boost_round": 2})
    
    # Invalid label dimensions
    with pytest.raises(ValueError):
        model.fit(np.random.randn(10, 2), np.random.randn(10, 2))

# --------------------------
# Performance Tests
# --------------------------

@pytest.mark.benchmark
def test_training_performance(benchmark):
    """Benchmark training performance"""
    X, y = make_classification(n_samples=100, random_state=42)
    model = KLDRO_XGB()
    model.update({"num_boost_round": 5, "max_depth": 4})
    
    benchmark(model.fit, X, y)
    assert benchmark.stats['mean'] < 100.0  # Adjust based on system capability
