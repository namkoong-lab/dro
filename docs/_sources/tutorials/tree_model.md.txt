# Tree-DRO

Besides linear models and neural networks, `dro` package also integrates tree-based ensemble models as backbone models for DRO, including XGBoost and LightGBM. 

Given the non-convexity of the optimization of tree-ensemble models, we only implement KL-DRO and CVaR-DRO in an approximate way. Specifically, we directly apply the DRO-kind loss functions to XGBoost and LightGBM.