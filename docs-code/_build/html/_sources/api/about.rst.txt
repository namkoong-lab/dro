.. _about:

About
======

.. image:: ./logo-new.png
   :width: 800px
   :align: center

``dro`` |release| is a Python package for distributionally robust optimization
in supervised learning. It supports linear and kernel models (including SVM,
logistic regression, and linear regression), neural networks, and tree ensembles
with several choices of ambiguity set. It also includes synthetic data
generators, runnable examples, and tools for model evaluation.

The convex linear formulations use ``cvxpy`` with ``CLARABEL`` by default.
Commercial solvers such as ``MOSEK`` are optional. See the
:doc:`installation guide <install>` for solver selection and the
:doc:`tutorials <tutorials>` for formulations and usage.

Without specified, our DRO model is to solve the following optimization problem:

.. math::
   \min_{\theta} \max_{P: P \in U} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))],

where :math:`U` is the so-called ambiguity set and typically of the form :math:`U = \{P: d(P, \hat P_n) \leq \epsilon\}` and :math:`\hat P_n := \frac{1}{n}\sum_{i = 1}^n \delta_{(X_i, Y_i)}` is the empirical distribution of training samples :math:`\{(X_i, Y_i)\}_{i = 1}^n`. And :math:`\epsilon` is the hyperparameter.

In release |release|, ``dro`` supports:

Synthetic data generation
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Python Module
     - Function Name
     - Description
   * - :file:`dro.data.dataloader_classification`
     - ``classification_basic``
     - Basic classification task
   * - 
     - ``classification_DN21``
     - Following Section 3.1.1 of `[1] <about-reference-1_>`_
   * - 
     - ``classification_SNVD20``
     - Following Section 5.1 of `[2] <about-reference-2_>`_
   * - 
     - ``classification_LWLC``
     - Following Section 4.1 (Classification) of `[3] <about-reference-3_>`_
   * - :file:`dro.data.dataloader_regression`
     - ``regression_basic``
     - Basic regression task
   * - 
     - ``regression_DN20_1``
     - Following Section 3.1.2 of `[1] <about-reference-1_>`_
   * - 
     - ``regression_DN20_2``
     - Following Section 3.1.3 of `[1] <about-reference-1_>`_
   * - 
     - ``regression_DN20_3``
     - Following Section 3.3 of `[1] <about-reference-1_>`_
   * - 
     - ``regression_LWLC``
     - Following Section 4.1 (Regression) of `[3] <about-reference-3_>`_


Linear DRO models
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Class Name
     - Description
   * - :file:`dro.linear_model.base`
     - ``BaseLinearDRO``
     - Base class for linear DRO methods
   * - :file:`dro.linear_model.chi2_dro`
     - ``Chi2DRO``
     - Linear chi-square divergence-based DRO
   * - :file:`dro.linear_model.kl_dro`
     - ``KLDRO``
     - Kullback-Leibler divergence-based DRO
   * - :file:`dro.linear_model.cvar_dro`
     - ``CVaRDRO``
     - CVaR DRO
   * - :file:`dro.linear_model.tv_dro`
     - ``TVDRO``
     - Total Variation DRO
   * - :file:`dro.linear_model.marginal_dro`
     - ``MarginalCVaRDRO``
     - Marginal-X CVaR DRO
   * - :file:`dro.linear_model.mmd_dro`
     - ``MMD_DRO``
     - Maximum Mean Discrepancy DRO
   * - :file:`dro.linear_model.conditional_dro`
     - ``ConditionalCVaRDRO``
     - Y|X (ConditionalShiftBased) CVaR DRO
   * - :file:`dro.linear_model.hr_dro`
     - ``HR_DRO_LR``
     - Holistic Robust DRO on linear models
   * - :file:`dro.linear_model.wasserstein_dro`
     - ``WassersteinDRO``
     - Wasserstein DRO
   * - 
     - ``WassersteinDROsatisficing``
     - Robust satisficing version of Wasserstein DRO
   * - :file:`dro.linear_model.sinkhorn_dro`
     - ``SinkhornLinearDRO``
     - Sinkhorn DRO on linear models
   * - :file:`dro.linear_model.mot_dro`
     - ``MOTDRO``
     - Optimal Transport DRO with Conditional Moment Constraints
   * - :file:`dro.linear_model.or_wasserstein_dro`
     - ``ORWDRO``
     - Outlier-Robust Wasserstein DRO
   * - :file:`dro.linear_model.group_dro`
     - ``GroupDRO``
     - Linear Group DRO minimizing the worst empirical group loss

NN DRO models
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Class Name
     - Description
   * - :file:`dro.neural_model.base_nn`
     - ``BaseNNDRO``
     - Base model for neural-network-based DRO
   * - :file:`dro.neural_model.fdro_nn`
     - ``Chi2NNDRO``
     - Chi-square Divergence-based Neural DRO Model
   * - :file:`dro.neural_model.wdro_nn`
     - ``WNNDRO``
     - Wasserstein Neural DRO with Adversarial Robustness
   * - :file:`dro.neural_model.hrdro_nn`
     - ``HRNNDRO``
     - Holistic Robust NN DRO
   * - :file:`dro.neural_model.sinkhorn_nn`
     - ``SinkhornNNDRO``
     - Sinkhorn DRO for neural networks
   * - :file:`dro.neural_model.groupdro_nn`
     - ``GroupNNDRO``
     - Neural Group DRO with adversarial group reweighting



Tree-Ensembles DRO models
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Python Module
     - Class Name
     - Description
   * - :file:`dro.tree_model.xgb`
     - ``KLDRO_XGB``
     - KL-DRO for XGBoost
   * - :file:`dro.tree_model.xgb`
     - ``CVaRDRO_XGB``
     - CVaR-DRO for XGBoost
   * - :file:`dro.tree_model.xgb`
     - ``Chi2DRO_XGB``
     - Chi2-DRO for XGBoost
   * - :file:`dro.tree_model.lgbm`
     - ``KLDRO_LGBM``
     - KL-DRO for Light GBM
   * - :file:`dro.tree_model.lgbm`
     - ``CVaRDRO_LGBM``
     - CVaR-DRO for Light GBM
   * - :file:`dro.tree_model.lgbm`
     - ``Chi2DRO_LGBM``
     - Chi2-DRO for Light GBM

References
----------

.. _about-reference-1:

[1] John C. Duchi and Hongseok Namkoong.
`Learning Models with Uniform Performance via Distributionally Robust Optimization <https://doi.org/10.1214/20-AOS2004>`_.
*The Annals of Statistics*, 49(3):1378–1406, 2021.
Institute of Mathematical Statistics.

.. _about-reference-2:

[2] Aman Sinha, Hongseok Namkoong, and John C. Duchi.
`Certifying Some Distributional Robustness with Principled Adversarial Training <https://openreview.net/forum?id=Hk6kPgZA->`_.
*International Conference on Learning Representations (ICLR)*, 2018.
OpenReview.net.

.. _about-reference-3:

[3] Jiashuo Liu, Jiayun Wu, Bo Li, and Peng Cui.
`Distributionally Robust Optimization with Data Geometry <https://proceedings.neurips.cc/paper_files/paper/2022/hash/da535999561b932f56efdd559498282e-Abstract-Conference.html>`_.
*Advances in Neural Information Processing Systems*, 35:33689–33701, 2022.
Curran Associates, Inc.
