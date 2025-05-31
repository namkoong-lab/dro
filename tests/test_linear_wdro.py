import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.dro.linear_model.wasserstein_dro import (
    WassersteinDRO,
    WassersteinDROsatisficing,
    WassersteinDROError,
    WassersteinDROSatisificingError
)
import cvxpy as cp 

# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def dataset(request):
    """Generate standardized test datasets"""
    n_samples, n_features = 100, 5
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
# WassersteinDRO Core Tests
# --------------------------

class TestWassersteinDRO:
    """Test suite for WassersteinDRO base functionality"""
    
    @pytest.mark.parametrize("model_type", ['svm', 'logistic', 'ols', 'lad'])
    def test_initialization(self, model_type):
        """Validate constructor arguments and default states"""
        model = WassersteinDRO(
            input_dim=5,
            model_type=model_type,
            solver='MOSEK'
        )
        
        # Validate default parameters
        assert model.cost_matrix.shape == (5, 5)
        assert model.eps == 0
        assert model.p == 1
        assert model.kappa == 'inf'
    
    def test_invalid_initialization(self):
        """Test constructor parameter validation"""
        with pytest.raises(ValueError):
            WassersteinDRO(input_dim=0, model_type='svm')
            
    @pytest.mark.parametrize("config,valid", [
        ({'cost_matrix': np.diag([1,2,3])}, True),
        ({'eps': -0.1}, False),
        ({'p': 0.5}, False),
        ({'kappa': 'invalid'}, False)
    ])
    def test_config_updates(self, config, valid):
        """Test dynamic configuration validation"""
        model = WassersteinDRO(input_dim=3, model_type='svm')
        if valid:
            model.update(config)
        else:
            with pytest.raises((ValueError, TypeError)):
                model.update(config)
    
    @pytest.mark.parametrize("model_type", ['svm', 'logistic', 'ols', 'lad'])
    def test_fit_interface(self, model_type):
        """Validate fit method input/output contracts"""
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
        
        model = WassersteinDRO(
            input_dim=X.shape[1],
            model_type=model_type,
            solver='MOSEK'
        )
        model.update({'eps': 0.1, 'p': 2})
        
        params = model.fit(X, y)
        
        # Validate output structure
        assert 'theta' in params
        assert isinstance(params['theta'], list)
        if model.fit_intercept:
            assert 'b' in params

    @pytest.mark.parametrize("model_type", ['svm'])
    def test_fit_kernel_interface(self, model_type):
        """Validate fit method input/output contracts"""
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
        
        model = WassersteinDRO(
            input_dim=X.shape[1],
            model_type=model_type,
            solver='MOSEK'
        )
        model.update({'eps': 0.1, 'p': 2})
        model.update_kernel({'metric': 'rbf', 'kernel_gamma': 1})
        params = model.fit(X, y)
        model.update_kernel({'metric': 'rbf', 'kernel_gamma': 'scale', 'n_components': 5})
        params = model.fit(X, y)
        
        # Validate output structure
        assert 'theta' in params
        assert isinstance(params['theta'], list)
        if model.fit_intercept:
            assert 'b' in params
    
    @pytest.mark.parametrize("compute_type", ['asymp', 'exact'])
    def test_worst_distribution(self, dataset, compute_type):
        """Validate worst-case distribution properties"""
        X, y = dataset
        y = np.sign(y-0.5)
        model = WassersteinDRO(
            input_dim=X.shape[1],
            model_type='svm',
            solver='MOSEK'
        )
        model.update({'eps': 0.5, 'kappa': 1.0})
        model.fit(X, y)
        
        dist = model.worst_distribution(X, y, compute_type=compute_type)
        
        # Validate output structure
        assert 'sample_pts' in dist
        assert 'weight' in dist
        if len(dist['weight']) > 0:
            assert np.isclose(sum(dist['weight']), 1.0, atol=1e-3)

# --------------------------
# WassersteinDROsatisficing Tests  
# --------------------------

class TestWassersteinSatisficing:
    """Test suite for robust satisficing variant"""
    
    def test_satisficing_optimization(self, dataset):
        """Validate target ratio constraint enforcement"""
        X, y = dataset
        y = np.sign(y-0.5)
        model = WassersteinDROsatisficing(
            input_dim=X.shape[1],
            model_type='svm',
            solver='MOSEK'
        )
        model.update({'target_ratio': 1.2})
        
        params = model.fit(X, y)
        
        # Validate solution feasibility
        assert params['theta'] is not None
        assert 'b' in params
    
    @pytest.mark.parametrize("ratio,valid", [
        (-1.0, False),
        (1.5, True),
        (-0.8, False)
    ])
    def test_target_ratio_validation(self, ratio, valid):
        """Test target ratio boundary conditions"""
        if valid:
            WassersteinDROsatisficing(
                input_dim=5,
                model_type='svm'
            ).update({'target_ratio': ratio})
        else:
            with pytest.raises(AssertionError):
                WassersteinDROsatisficing(
                    input_dim=5,
                    model_type='svm'
                ).update({'target_ratio': ratio})

# --------------------------
# Cross-Cutting Concerns
# --------------------------

@pytest.mark.parametrize("solver", ['MOSEK'])
def test_solver_compatibility(dataset, solver):
    """Validate solver interoperability"""
    X, y = dataset
    y = np.sign(y-0.5)
    model = WassersteinDRO(
        input_dim=X.shape[1],
        model_type='svm',
        solver=solver
    )
    model.update({'eps': 0.1})
    
    params = model.fit(X, y)
    assert params['theta'] is not None

@pytest.mark.parametrize("p_value", [1, 2, 'inf'])
def test_wasserstein_order_handling(p_value):
    """Test different Wasserstein metric configurations"""
    model = WassersteinDRO(
        input_dim=3,
        model_type='svm'
    )
    model.update({'p': p_value})
    assert model.p == (float(p_value) if p_value != 'inf' else 'inf')

# --------------------------
# Error Condition Tests
# --------------------------

def test_dimension_mismatch_errors():
    """Validate dimensional consistency checks"""
    model = WassersteinDRO(input_dim=3, model_type='svm')
    X = np.random.randn(5, 4)
    y = np.random.randn(5)
    
    with pytest.raises(WassersteinDROError):
        model.fit(X, y)


# --------------------------
# Specialized Kernel Tests
# --------------------------

@pytest.mark.parametrize("kernel", ['linear', 'rbf'])
def test_kernel_support(kernel):
    """Validate kernelized implementation"""
    model = WassersteinDRO(
        input_dim=10,
        model_type='svm',
        kernel=kernel
    )
    X, y = make_classification(n_samples=10, n_features=10)
    y = np.sign(y-0.5)
    if kernel == 'rbf':
        model.update({'cost_matrix': np.eye(10)})
    
    params = model.fit(X, y)
    assert params['theta'] is not None


# --------------------------
# WassersteinDRO Penalization Tests
# --------------------------

def test_penalization_lad_with_kappa(dataset):
    """Test regularization term for LAD model with finite kappa"""
    X, y = dataset
    model = WassersteinDRO(
        input_dim=X.shape[1], 
        model_type='lad',
        solver='MOSEK'
    )
    model.update({
        'kappa': 1.0,
        'p': 2
    })
    
    # Mock theta variable
    theta = np.random.randn(X.shape[1])
    penalty = model._penalization(theta)
    
    expr_str = str(penalty.expr).lower()
    assert "maximum" in expr_str 

# --------------------------
# Distance Computation Tests
# --------------------------

def test_distance_computation_with_y_ambiguity():
    """Validate distance calculation with Y-component"""
    model = WassersteinDRO(input_dim=2, model_type='svm')
    model.update({'kappa': 1.5, 'p': 2})
    
    # Test with different labels
    dist = model._distance_compute(
        X_1=np.array([1.0, 2.0]),
        X_2=np.array([3.0, 4.0]),
        Y_1=1.0,
        Y_2=-1.0
    )
    assert 'abs' in str(dist.expr).lower()  # Verify Y component exists

# --------------------------
# Lipschitz Norm Tests
# --------------------------

def test_lipschitz_norm_for_ols():
    """Verify Lipschitz constant for OLS models"""
    model = WassersteinDRO(input_dim=3, model_type='ols')
    assert np.isinf(model._lipschitz_norm())

# --------------------------
# Satisficing Model Edge Cases
# --------------------------

@pytest.mark.parametrize("model_type", ['ols', 'svm', 'lad', 'logistic'])
def test_satisficing_lad_constraints(model_type):
    """Test constraint formulation for LAD satisficing model"""
    if model_type in {"lad", "ols"}:
        X, y = make_regression(n_samples=50, n_features=10)
    else:
        X, y = make_classification(n_samples=50, n_features=10)
        y = np.sign(y-0.5)

    model = WassersteinDROsatisficing(
        input_dim=10,
        model_type=model_type,
        solver='MOSEK'
    )
    model.update({
        'kappa': 0.5,
        'target_ratio': 1.5
    })
    
    params = model.fit(X, y)
    assert 'theta' in params  # Verify solution exists
    model.update_kernel({'metric': 'rbf', 'kernel_gamma': 1})
    print('=========')
    params = model.fit(X, y)
    print(model.kernel)
    print('========')
    model.update_kernel({'metric': 'rbf', 'kernel_gamma': 'scale', 'n_components': 5})
    params = model.fit(X, y)
    assert 'theta' in params  # Verify solution exists


@pytest.mark.parametrize("model_type", ['lad'])
def test_satisficing_lad_constraints2(model_type):
    """Test constraint formulation for LAD satisficing model"""
    if model_type in {"lad", "ols"}:
        X, y = make_regression(n_samples=50, n_features=10)
    else:
        X, y = make_classification(n_samples=50, n_features=10)
        y = np.sign(y-0.5)

    model = WassersteinDROsatisficing(
        input_dim=10,
        model_type=model_type,
        solver='MOSEK',
        kernel='linear'
    )
    model.update({
        'kappa': 'inf',
        'target_ratio': 1.5,
        'cost_matrix': np.identity(10)
    })
    
    params = model.fit(X, y)
    assert 'theta' in params  # Verify solution exists


    
# --------------------------
# Configuration Validation
# --------------------------

def test_invalid_cost_matrix_type():
    """Test non-array cost matrix rejection"""
    model = WassersteinDRO(input_dim=2, model_type='svm')
    with pytest.raises(TypeError):
        model.update({'cost_matrix': [[1,0],[0,1]]})  # Not numpy array

def test_invalid_p_value_update():
    """Test invalid Wasserstein order rejection"""
    model = WassersteinDRO(input_dim=3, model_type='svm')
    with pytest.raises(ValueError):
        model.update({'p': 0.5})  # p must be ≥1

# --------------------------
# Worst-case Distribution
# --------------------------

def test_unsupported_compute_type():
    """Verify error for invalid computation methods"""
    X, y = make_classification(n_samples=50, n_features=10)
    y = np.sign(y-0.5)
    model = WassersteinDRO(input_dim=10, model_type='svm')
    model.fit(X, y)
    
    with pytest.raises(WassersteinDROError):
        model.worst_distribution(X, y, compute_type='invalid')

def test_p():
    X, y = make_classification(n_samples=50, n_features=10)
    y = np.sign(y-0.5)
    model = WassersteinDROsatisficing(input_dim=10, model_type='svm')
    model.update({"p":2})
    
    model.fit(X, y)

    with pytest.raises(Warning):
        model.worst_distribution(X, y)



# --------------------------
# Distance Computation Tests
# --------------------------

def test_distance_with_infinite_kappa():
    """Test distance calculation when kappa is infinite"""
    model = WassersteinDRO(input_dim=2, model_type='svm')
    model.update({'kappa': 'inf', 'p': 2})
    
    dist = model._distance_compute(
        X_1=np.array([1.0, 2.0]),
        X_2=np.array([3.0, 4.0]),
        Y_1=1.0,
        Y_2=-1.0
    )
    assert 'abs' not in str(dist)  

# --------------------------
# Worst-case Distribution (Exact)
# --------------------------

def test_exact_worst_distribution_ols():
    """Validate OLS exact worst-case distribution"""
    X, y = make_regression(n_samples=100, n_features=10)
    model = WassersteinDRO(
        input_dim=10,
        model_type='ols',
        solver='MOSEK'
    )
    model.update({'eps': 0.5, 'p': 2})
    
    with pytest.raises(WassersteinDROError):
        dist = model.worst_distribution(X, y, compute_type='exact')

# def test_exact_worst_distribution_lad():
#     """Validate OLS exact worst-case distribution"""
#     X, y = make_classification(n_samples=100, n_features=10)
#     y = np.sign(y-0.5)
#     model = WassersteinDRO(
#         input_dim=10,
#         model_type='lad',
#         solver='MOSEK'
#     )
#     model.update({'eps': 0.5, 'p': 2})
    
#     dist = model.worst_distribution(X, y, compute_type='exact')

def test_asymp_worst_distribution_ols():
    """Validate OLS asypm worst-case distribution"""
    X, y = make_regression(n_samples=100, n_features=10)
    model = WassersteinDRO(
        input_dim=10,
        model_type='ols',
        solver='MOSEK'
    )
    model.update({'eps': 0.5, 'p': 2})
    
    with pytest.raises(WassersteinDROError):
        dist = model.worst_distribution(X, y, compute_type='asymp')

# --------------------------
# Asymptotic Method Tests
# --------------------------

@pytest.mark.parametrize("model_type,gamma", [
    ('svm', 0.1),
    ('lad', 0.2)
])
def test_asymptotic_method_gamma(model_type, gamma):
    """Test asymptotic method with different gamma values"""
    
    if model_type in {"svm", "logistic"}:
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=3,
            random_state=42
            )
        y = np.sign(y - 0.5)
    else:
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            random_state=42
            )
    
    model = WassersteinDRO(
        input_dim=10,
        model_type=model_type,
        solver='MOSEK'
    )
    model.update({'kappa': 1.0, 'eps': 0.5})
    
    dist = model.worst_distribution(
        X, y,
        compute_type='asymp',
        gamma=gamma
    )
    assert np.isclose(sum(dist['weight']), 1.0, atol=1e-3)

# --------------------------
# Satisficing Model Tests
# --------------------------

# def test_satisficing_target_ratio_enforcement():
#     """Verify target ratio constraint is enforced"""
#     X, y = make_regression(n_samples=100, n_features=10)
#     model = WassersteinDROsatisficing(
#         input_dim=10,
#         model_type='ols',
#         solver='MOSEK'
#     )
#     model.update({'target_ratio': 1.5})
    
#     params = model.fit(X, y)
#     empirical_loss = np.mean((X @ params['theta'] - y)**2)
#     assert empirical_loss <= 1.5 * model.fit_oracle(X, y)

# --------------------------
# Kernel Method Tests
# --------------------------

def test_rbf_kernel_support():
    """Validate RBF kernel implementation"""
    X, y = make_classification(n_samples=100, n_features=10)
    y = np.sign(y-0.5)
    model = WassersteinDRO(
        input_dim=10,
        model_type='svm',
        kernel='rbf'
    )
    model.update({
        'cost_matrix': np.eye(10),
        'eps': 0.1
    })
    
    params = model.fit(X, y)
    assert len(params['theta']) == 100 

# --------------------------
# Exception Handling
# --------------------------

def test_invalid_kappa_for_asymptotic():
    """Test invalid kappa in asymptotic method"""
    X, y = make_classification(n_samples=100, n_features=10)
    model = WassersteinDRO(
        input_dim=10,
        model_type='svm',
        solver='MOSEK'
    )
    model.update({'kappa': 'inf', 'eps': 0.5})
    
    with pytest.raises(WassersteinDROError):
        model.worst_distribution(X, y, compute_type='asymp')

# --------------------------
# Edge Case: Zero Epsilon
# --------------------------

# def test_zero_epsilon_case():
#     """Verify behavior when epsilon=0 (non-robust)"""
#     X, y = make_classification(n_samples=100, n_features=10)
#     model = WassersteinDRO(
#         input_dim=10,
#         model_type='svm',
#         solver='MOSEK'
#     )
#     model.update({'eps': 0.0})
    
#     params = model.fit(X, y)
#     dist = model.worst_distribution(X, y, compute_type='exact')
#     assert np.allclose(dist['sample_pts'][0], X)  