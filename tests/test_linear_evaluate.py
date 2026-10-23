import numpy as np
import pytest

from src.dro.linear_model.chi2_dro import Chi2DRO


@pytest.mark.parametrize('model_type,kernel', [
    ('svm', 'linear'),
    ('lad', 'linear'),
    ('svm', 'rbf'),
    ('lad', 'rbf'),
    ('ols', 'rbf'),
    ('logistic', 'rbf'),
])
def test_evaluate_rejects_unsupported_loss_or_kernel(model_type, kernel):
    model = Chi2DRO(input_dim=2, model_type=model_type, kernel=kernel)
    with pytest.raises(NotImplementedError, match='only linear-kernel OLS and logistic'):
        model.evaluate(np.zeros((4, 2)), np.ones(4))


@pytest.mark.parametrize('model_type', ['ols', 'logistic'])
def test_evaluate_supported_losses_remain_bias_corrected(model_type):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    if model_type == 'logistic':
        y = np.where(y >= 0, 1.0, -1.0)
    model = Chi2DRO(input_dim=2, model_type=model_type)
    model.load({'theta': [0.2, -0.3]})
    estimate = model.evaluate(X, y)
    assert np.isfinite(estimate)
    assert estimate > np.mean(model._loss(X, y))
