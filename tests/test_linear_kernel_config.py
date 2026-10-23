import numpy as np
import pytest

from src.dro.linear_model.base import BaseLinearDRO
from src.dro.linear_model.mmd_dro import MMD_DRO
from src.dro.linear_model.wasserstein_dro import WassersteinDRO


@pytest.mark.parametrize('model_class', [BaseLinearDRO, MMD_DRO])
def test_n_components_can_be_reset_to_none(model_class):
    model = model_class(input_dim=2)
    model.update_kernel({'n_components': 4})
    model.update_kernel({})
    assert model.n_components == 4

    model.update_kernel({'n_components': None})
    assert model.n_components is None


@pytest.mark.parametrize('model_class', [BaseLinearDRO, MMD_DRO])
@pytest.mark.parametrize('invalid', [4.5, '4', [4]])
def test_n_components_rejects_other_nonintegers(model_class, invalid):
    model = model_class(input_dim=2)
    with pytest.raises((TypeError, ValueError), match='n_components'):
        model.update_kernel({'n_components': invalid})


def test_refit_full_kernel_after_nystroem_fit():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 2))
    y = np.where(X[:, 0] + 0.3 * X[:, 1] >= 0, 1.0, -1.0)
    model = WassersteinDRO(input_dim=2, model_type='svm')
    model.update({'eps': 0.1, 'p': 2})
    model.update_kernel({'metric': 'rbf', 'kernel_gamma': 0.5, 'n_components': 4})
    model.fit(X, y)
    assert model.theta.shape == (4,)

    model.update_kernel({'n_components': None})
    model.fit(X, y)
    assert model.theta.shape == (len(X),)
    assert np.isfinite(model.robust_obj)

    reference = WassersteinDRO(input_dim=2, model_type='svm')
    reference.update({'eps': 0.1, 'p': 2})
    reference.update_kernel({'metric': 'rbf', 'kernel_gamma': 0.5})
    reference.fit(X, y)
    assert model.robust_obj == pytest.approx(reference.robust_obj, abs=1e-6)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
