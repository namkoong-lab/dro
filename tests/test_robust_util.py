import pytest
import torch
import numpy as np
from scipy import optimize
from src.dro.neural_model.fdro_utils import (  # Replace with actual import
    RobustLoss, chi_square_doro, cvar_doro,bisection,
    DROError, ParameterError, NumericalError
)

# --------------------------
# Test Fixtures
# --------------------------

@pytest.fixture(params=['classification', 'regression'])
def sample_data(request):
    """Generate sample data for different task types"""
    torch.manual_seed(42)
    batch_size = 32
    
    if request.param == 'classification':
        return (
            torch.randn(batch_size, 10),  # logits
            torch.randint(0, 2, (batch_size,))  # binary labels
        )
    return (
        torch.randn(batch_size, 1),      # regression outputs
        torch.randn(batch_size, 1)       # regression targets
    )

# --------------------------
# Parameter Validation Tests
# --------------------------

@pytest.mark.parametrize("geometry,valid", [
    ('cvar', True),
    ('chi-square', True),
    ('invalid', False),
    ('CVAR', True),  # case insensitivity check
    (None, False)
])
def test_robustloss_geometry_validation(geometry, valid):
    """Test geometry parameter validation"""
    if valid:
        RobustLoss(size=0.5, reg=0.1, geometry=geometry)
    else:
        with pytest.raises(ParameterError):
            RobustLoss(size=0.5, reg=0.1, geometry=geometry)

@pytest.mark.parametrize("size,geometry,valid", [
    (1.1, 'cvar', False),
    (0.5, 'cvar', True),
    (-0.1, 'chi-square', False),
    (5.0, 'chi-square', True)
])
def test_size_validation(size, geometry, valid):
    """Test size parameter bounds"""
    if valid:
        RobustLoss(size=size, reg=0.1, geometry=geometry)
    else:
        with pytest.raises(ParameterError):
            RobustLoss(size=size, reg=0.1, geometry=geometry)

# --------------------------
# Forward Pass Tests
# --------------------------

def test_robustloss_output_shape(sample_data):
    """Verify output tensor properties"""
    outputs, targets = sample_data
    model = RobustLoss(size=0.5, reg=0.1, geometry='cvar', 
                      is_regression=len(targets.shape)>1)
    
    loss = model(outputs, targets)
    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32

def test_robustloss_output_shape2(sample_data):
    """Verify output tensor properties"""
    outputs, targets = sample_data
    model = RobustLoss(size=0.5, reg=0.0, geometry='cvar', 
                      is_regression=len(targets.shape)>1)
    
    loss = model(outputs, targets)
    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32


# --------------------------
# Weight Computation Tests
# --------------------------

def test_hard_cvar_weights():
    """Verify hard thresholding CVaR weight logic"""
    v = torch.tensor([3.0, 2.0, 1.0, 0.0])
    model = RobustLoss(size=0.5, reg=0.0, geometry='cvar')
    
    weights = model._hard_cvar(v)
    expected = torch.tensor([0.5, 0.5, 0.0, 0.0])  # Top 2 of 4 samples
    assert torch.allclose(weights, expected)

# --------------------------
# Bisection Method Tests
# --------------------------

def test_bisection_convergence():
    """Verify bisection root finding logic"""
    result = bisection(
        eta_min=0.0,
        eta_max=2.0,
        f=lambda x: x - 1.5,  # Root at 1.5
        tol=1e-1
    )
    assert abs(result - 1.5) < 1e-1

# --------------------------
# DORO Function Tests
# --------------------------

@pytest.mark.parametrize("contamination", [0.1, 0.3])
def test_chi_square_doro(contamination):
    """Verify χ²-DORO contamination handling"""
    outputs = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    loss = chi_square_doro(outputs, targets, alpha=0.9, eps=contamination)
    
    assert 0.0 <= loss.item() <= 10.0  # Sanity check

def test_cvar_doro_extreme_values():
    """Test CVaR-DORO with extreme losses"""
    outputs = torch.tensor([[10.0, -10.0]] * 100)  # High confidence incorrect
    targets = torch.ones(100).long()
    loss = cvar_doro(outputs, targets, eps=0.1, alpha=0.9)
    
    assert 0.0 < loss.item() < 100.0

# --------------------------
# Numerical Stability Tests
# --------------------------

def test_near_zero_regularization():
    model = RobustLoss(size=0.5, reg=1e-9, geometry='chi-square', is_regression=False)
    outputs = torch.randn(5, 3)  
    targets = torch.tensor([0, 2, 1, 0, 1], dtype=torch.long)  
    loss = model(outputs, targets)
    assert not torch.isnan(loss)

    model = RobustLoss(size=0.5, reg=1e-9, geometry='chi-square', is_regression=True)
    outputs = torch.randn(5, 1)  
    targets = torch.randn(5, 1)  
    loss = model(outputs, targets)
    assert not torch.isnan(loss)

    model = RobustLoss(size=100000, reg=1e-5, geometry='chi-square', is_regression=True)
    outputs = torch.randn(5, 1)  
    targets = torch.randn(5, 1)  
    loss = model(outputs, targets)
    assert not torch.isnan(loss)