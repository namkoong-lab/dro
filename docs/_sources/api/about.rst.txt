.. _about:

About
======

.. image:: ./logo.png
   :width: 800px
   :align: center

`dro` is a python package that implements typical DRO methods on linear loss (SVM, logistic regression, and linear regression) for supervised learning tasks. It is built based on the convex optimization solver ``cvxpy``. The `dro` package supports different kinds of distance metrics :math:`d(\cdot,\cdot)` as well as different kinds of base models (e.g., linear regression, logistic regression, SVM, tree-ensembles, neural networks...). Furthermore, it integrates different synthetic data generating mechanisms from recent research papers.

Without specified, our DRO model is to solve the following optimization problem:

.. math::
   \min_{\theta} \max_{P: P \in U} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))],

where :math:`U` is the so-called ambiguity set and typically of the form :math:`U = \{P: d(P, \hat P_n) \leq \epsilon\}` and :math:`\hat P_n := \frac{1}{n}\sum_{i = 1}^n \delta_{(X_i, Y_i)}` is the empirical distribution of training samples :math:`\{(X_i, Y_i)\}_{i = 1}^n`. And :math:`\epsilon` is the hyperparameter.

As for the latest ``v0.2.2`` version, `dro` supports:

(1) Synthetic data generation
-------------------------------

.. list-table:: Synthetic Data Generation Modules
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Function Name
     - Description
   * - :file:`dro.src.data.dataloader_classification`
     - ``classification_basic``
     - Basic classification task
   * - 
     - ``classification_DN21``
     - Following Section 3.1.1 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
   * - 
     - ``classification_SNVD20``
     - Following Section 5.1 of "Certifying Some Distributional Robustness with Principled Adversarial Training"
   * - 
     - ``classification_LWLC``
     - Following Section 4.1 (Classification) of "Distributionally Robust Optimization with Data Geometry"
   * - :file:`dro.src.data.dataloader_regression`
     - ``regression_basic``
     - Basic regression task
   * - 
     - ``regression_DN20_1``
     - Following Section 3.1.2 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
   * - 
     - ``regression_DN20_2``
     - Following Section 3.1.3 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
   * - 
     - ``regression_DN20_3``
     - Following Section 3.3 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
   * - 
     - ``regression_LWLC``
     - Following Section 4.1 (Regression) of "Distributionally Robust Optimization with Data Geometry"

(2) Linear DRO models
---------------------

.. list-table:: Linear DRO Models
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Class Name
     - Description
   * - :file:`dro.src.linear_dro.base`
     - ``BaseLinearDRO``
     - Base class for linear DRO methods
   * - :file:`dro.src.linear_dro.chi2_dro`
     - ``Chi2DRO``
     - Linear chi-square divergence-based DRO
   * - :file:`dro.src.linear_dro.kl_dro`
     - ``KLDRO``
     - Kullback-Leibler divergence-based DRO
   * - :file:`dro.src.linear_dro.cvar_dro`
     - ``CVaRDRO``
     - CVaR DRO
   * - :file:`dro.src.linear_dro.tv_dro`
     - ``TVDRO``
     - Total Variation DRO
   * - :file:`dro.src.linear_dro.marginal_dro`
     - ``MarginalCVaRDRO``
     - Marginal-X CVaR DRO
   * - :file:`dro.src.linear_dro.mmd_dro`
     - ``MMD_DRO``
     - Maximum Mean Discrepancy DRO
   * - :file:`dro.src.linear_dro.conditional_dro`
     - ``ConditionalCVaRDRO``
     - Y|X (ConditionalShiftBased) CVaR DRO
   * - :file:`dro.src.linear_dro.hr_dro`
     - ``HR_DRO_LR``
     - Holistic Robust DRO on linear models
   * - :file:`dro.src.linear_dro.wasserstein_dro`
     - ``WassersteinDRO``
     - Wasserstein DRO
   * - 
     - ``WassersteinDROsatisficing``
     - Robust satisficing version of Wasserstein DRO
   * - :file:`dro.src.linear_dro.sinkhorn_dro`
     - ``SinkhornLinearDRO``
     - Sinkhorn DRO on linear models
   * - :file:`dro.src.linear_dro.mot_dro`
     - ``MOTDRO``
     - Optimal Transport DRO with Conditional Moment Constraints
   * - :file:`dro.src.linear_dro.or_wasserstein_dro`
     - ``ORWDRO``
     - Outlier-Robust Wasserstein DRO

(3) NN DRO models
-----------------

.. list-table:: Neural Network DRO Models
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Class Name
     - Description
   * - :file:`dro.src.neural_model.base_nn`
     - ``BaseNNDRO``
     - Base model for neural-network-based DRO
   * - :file:`dro.src.neural_model.fdro_nn`
     - ``Chi2NNDRO``
     - Chi-square Divergence-based Neural DRO Model
   * - :file:`dro.src.neural_model.wdro_nn`
     - ``WNNDRO``
     - Wasserstein Neural DRO with Adversarial Robustness
   * - :file:`dro.src.neural_model.hrdro_nn`
     - ``HRNNDRO``
     - Holistic Robust NN DRO



(4) Tree-Ensembles DRO models
------------------------------

.. list-table:: Tree-Ensembles DRO Models
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Class Name
     - Description
   * - :file:`dro.src.tree_model.xgb`
     - ``KLDRO_XGB``
     - KL-DRO for XGBoost
   * - :file:`dro.src.tree_model.xgb`
     - ``CVaRDRO_XGB``
     - CVaR-DRO for XGBoost
   * - :file:`dro.src.tree_model.lgbm`
     - ``KLDRO_LGBM``
     - KL-DRO for Light GBM
   * - :file:`dro.src.tree_model.lgbm`
     - ``CVaRDRO_LGBM``
     - CVaR-DRO for Light GBM