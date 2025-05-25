import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from dro.src.tree_model.lgbm import KLDRO_LGBM  

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

model = KLDRO_LGBM(eps=0.1)

init_config = {"max_depth": 1, "num_boost_round": 2}
model.update(init_config)
# model.fit(X, y)

import lightgbm
dtrain = lightgbm.Dataset(X, y)
num_boost_round = init_config["num_boost_round"]
del init_config["num_boost_round"]

print('check')
init_config['verbosity'] = -1
init_config['objective'] =lambda preds, dtrain: model._kl_dro_loss(preds, dtrain, 0.1)
model = lightgbm.train(init_config, dtrain, num_boost_round=num_boost_round)


# import lightgbm as lgb
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=100, n_features=10)
# dtrain = lgb.Dataset(X, label=y)

# params = {
#     'objective': 'binary',
#     'verbosity': -1,
#     'num_threads': 1,
# }

# model = lgb.train(params, dtrain, num_boost_round=10)