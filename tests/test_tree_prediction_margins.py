"""Classification predictions must use a consistent raw-margin boundary."""
import numpy as np
import pytest
from scipy.special import expit
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_limits
import xgboost as xgb

from src.dro.tree_model.xgb import KLDRO_XGB, Chi2DRO_XGB, CVaRDRO_XGB
from src.dro.tree_model.lgbm import KLDRO_LGBM, Chi2DRO_LGBM, CVaRDRO_LGBM

WRAPPERS = [
    (KLDRO_XGB, 'output_margin'), (Chi2DRO_XGB, 'output_margin'),
    (CVaRDRO_XGB, 'output_margin'), (KLDRO_LGBM, 'raw_score'),
    (Chi2DRO_LGBM, 'raw_score'), (CVaRDRO_LGBM, 'raw_score'),
]


@pytest.mark.parametrize('wrapper,raw_key', WRAPPERS)
def test_classification_raw_margin_boundary(wrapper, raw_key):
    margins = np.array([-1.0, -0.1, 0.0, 0.1, 0.4, 1.0])

    class FixedBooster:
        def predict(self, data, **kwargs):
            assert kwargs.get(raw_key) is True
            return margins

    model = wrapper()
    model.model = FixedBooster()
    np.testing.assert_array_equal(
        model.predict(np.zeros((len(margins), 2))), [0, 0, 0, 1, 1, 1]
    )


@pytest.mark.parametrize('wrapper,raw_key', WRAPPERS)
def test_regression_predictions_remain_unthresholded(wrapper, raw_key):
    values = np.array([-0.2, 0.1, 0.4])

    class FixedBooster:
        def predict(self, data, **kwargs):
            assert kwargs.get(raw_key) is False
            return values

    model = wrapper(kind='regression')
    model.model = FixedBooster()
    np.testing.assert_array_equal(model.predict(np.zeros((3, 2))), values)


@pytest.mark.parametrize('wrapper,raw_key', WRAPPERS)
def test_trained_classifier_matches_sigmoid_boundary(wrapper, raw_key):
    X, y = make_classification(
        n_samples=80, n_features=4, n_informative=3, n_redundant=0,
        random_state=7,
    )
    config = {'num_boost_round': 4, 'max_depth': 2, 'learning_rate': 0.1}
    if raw_key == 'output_margin':
        # Exercise XGBoost's probability-transforming objective as well.
        config.update({'objective': 'binary:logistic', 'nthread': 1, 'seed': 7})
    else:
        config.update({'num_threads': 1, 'seed': 7})
    model = wrapper()
    model.update(config)
    with threadpool_limits(limits=1):
        model.fit(X, y)
        data = xgb.DMatrix(X) if raw_key == 'output_margin' else X
        margins = model.model.predict(data, **{raw_key: True})
        np.testing.assert_array_equal(model.predict(X), (expit(margins) > 0.5).astype(int))
