import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.dro.tree_model.lgbm import KLDRO_LGBM  

@pytest.fixture(scope="module")
def fixed_data():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_config_update_basic():
    model = KLDRO_LGBM(eps=0.1)
    
    
    new_config = {
        "max_depth": 2,
        "learning_rate": 1.0,
        "num_boost_round": 4
    }
    model.update(new_config)
    
    updated_params = model.config
    assert updated_params["max_depth"] == 2
    assert updated_params["learning_rate"] == 1.0
    assert updated_params["num_boost_round"] == 4
    assert model.eps == 0.1 

def test_config_invalid_values():
    model = KLDRO_LGBM(eps=0.1)
    
    with pytest.raises(KeyError):
        model.update({"max_depth": 1})
        
    
def test_config_update_with_training(fixed_data):
    X_train, X_test, y_train, y_test = fixed_data
    
    model = KLDRO_LGBM(eps=0.1)
    
    init_config = {"max_depth": 1, "num_boost_round": 2}
    model.update(init_config)
    model.fit(X_train, y_train)
    init_acc = (model.predict(X_test) == y_test).mean()
    
    updated_config = {"max_depth": 5, "num_boost_round": 10}
    model.update(updated_config)
    model.fit(X_train, y_train)
    updated_acc = (model.predict(X_test) == y_test).mean()
    
    assert updated_acc >= init_acc - 0.01, "Not improved"
    
    assert model.config["max_depth"] == 5