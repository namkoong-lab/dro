# Formulation

Given the empirical distribution $\hat P$ from the training data $\{(x_i, y_i)\}_{i \in [N]}$, we consider the following (distance-based) distributionally robust optimization formulations under the machine learning context. In general, DRO optimizes over the worst-case loss and satisfies the following structure:

$$
\min_{f \in \mathcal{F}}\max_{Q \in \mathcal{P}}\mathbb{E}_Q[\ell(f(X), Y)],
$$

where $\mathcal{P}$ is denoted as the ambiguity set. Usually, it satisfies the following structure:

$$
\mathcal{P}(d, \epsilon) = \{Q: d(Q, \hat P) \leq \epsilon\}. 
$$


Here, $d(\cdot, \cdot)$ is a notion of distance between probability measures and $\epsilon$ captures the size of the ambiguity set.

Given each function class $\mathcal{F}$, we classify all the models into the following cases, where each case can be further classified given each distance type $d$.

We design our package based on the principle pipeline "Data -> Model -> Evaluation / Diagnostics" and discuss them one by one as follows:

## Data Module
## Synthetic Data Generation
Following the general pipeline of "Data -> Model -> Evaluation / Diagnostics", we first integrate different kinds of synthetic data generating mechanisms into `dro`, including:

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Python Module</th>
    <th class="tg-0pky">Function Name</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="4"><br><br><br><br>dro.src.data.dataloader_classification</td>
    <td class="tg-0pky">classification_basic</td>
    <td class="tg-0pky">Basic classification task</td>
  </tr>
  <tr>
    <td class="tg-0pky">classification_DN21</td>
    <td class="tg-0pky">Following Section 3.1.1 of <br>"Learning Models with Uniform Performance via Distributionally Robust Optimization"</td>
  </tr>
  <tr>
    <td class="tg-0pky">classification_SNVD20</td>
    <td class="tg-0pky">Following Section 5.1 of <br>"Certifying Some Distributional Robustness with Principled Adversarial Training"</td>
  </tr>
  <tr>
    <td class="tg-0lax">classification_LWLC</td>
    <td class="tg-0lax">Following Section 4.1 (Classification) of <br>"Distributionally Robust Optimization with Data Geometry"</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="5"><br><br><br><br><br>dro.src.data.dataloader_regression</td>
    <td class="tg-0lax">regression_basic</td>
    <td class="tg-0lax">Basic regression task</td>
  </tr>
  <tr>
    <td class="tg-0lax">regression_DN20_1</td>
    <td class="tg-0lax">Following Section 3.1.2 of <br>"Learning Models with Uniform Performance via Distributionally Robust Optimization"</td>
  </tr>
  <tr>
    <td class="tg-0lax">regression_DN20_2</td>
    <td class="tg-0lax">Following Section 3.1.3 of <br>"Learning Models with Uniform Performance via Distributionally Robust Optimization"</td>
  </tr>
  <tr>
    <td class="tg-0lax">regression_DN20_3</td>
    <td class="tg-0lax">Following Section 3.3 of <br>"Learning Models with Uniform Performance via Distributionally Robust Optimization"</td>
  </tr>
  <tr>
    <td class="tg-0lax">regression_LWLC</td>
    <td class="tg-0lax">Following Section 4.1 (Regression) <br>of "Distributionally Robust Optimization with Data Geometry"</td>
  </tr>
</tbody></table>

## Model Module
Models expose the robust empirical optimization objective from the most recent fit as ``model.robust_obj`` whenever it can be computed; otherwise, the value is ``None``.

### Linear and Kernel Models

#### Model and Loss Setup

For linear models, $f(X) = \theta^{\top}X + b$. Across the linear module, the coefficient vector $\theta = (\theta_1,\ldots, \theta_p)$ is stored as ``theta`` and the intercept $b$ as ``b``. Set ``model_type`` to the name in parentheses for one of the following losses.

Classification, where $Y \in \{-1, 1\}$:

* SVM (hinge) loss (``svm``): $\ell(f(X), Y) = \max\{1 - Y f(X), 0\}$.
* Logistic loss (``logistic``): $\ell(f(X), Y) = \log(1 + \exp(-Y f(X)))$.

Regression:

* Least absolute deviation (``lad``): $\ell(f(X), Y) = |Y - f(X)|$.
* Ordinary least squares (``ols``): $\ell(f(X), Y) = (Y - f(X))^2$.

The linear models use built-in ``cvxpy`` solvers; our tests use ``MOSEK``.

#### Kernel Setup

Kernelized distributionally robust regression and classification are configured through ``.update_kernel()`` and support the same four loss types. The predictor becomes $f(X) = \sum_{i \in [N]}\alpha_i K(x, x_i)$, where $K(\cdot,\cdot)$ is the kernel and $\{\alpha_i\}_{i \in [N]}$ are fitted parameters.

The kernel interface follows scikit-learn:

* ``metric``: ``additive_chi2``, ``chi2``, ``linear``, ``poly``, ``polynomial``, or ``rbf``.
* ``kernel_gamma``: a positive gamma value, ``scale``, or ``auto``.
* ``n_components``: ``None`` for exact fitting, or an integer for a Nystroem approximation when $n$ is large.

#### Supported DRO Methods

* WDRO: basic Wasserstein DRO and satisficing Wasserstein DRO.
* Standard $f$-DRO: KL-DRO, $\chi^2$-DRO, and TV-DRO.
* Generalized $f$-DRO: CVaR-DRO, Marginal DRO (CVaR), and Conditional DRO (CVaR).
* MMD-DRO.
* Bayesian-based DRO: Bayesian-PDRO and PDRO.
* Mixed-DRO: Sinkhorn-DRO, HR-DRO, MOT-DRO, and Outlier-Robust Wasserstein DRO (OR-Wasserstein DRO).

### Neural Network Models (Approximate Fitting)

The neural module implements $\chi^2$-DRO, CVaR-DRO, Wasserstein DRO through adversarial training, and Holistic Robust DRO. Supported architectures are linear models, vanilla MLP, AlexNet, and ResNet18. Users can also supply their own architecture through the `update` function in `BaseNNDRO`.

### Tree-based Ensemble Models (Approximate Fitting)

The tree module supports KL-DRO, CVaR-DRO, and $\chi^2$-DRO with LightGBM and XGBoost.

### Evaluation and Diagnostics

Some linear DRO models provide ``worst_distribution`` to inspect worst-case model performance. The ``evaluate`` function in `BaseLinearDRO` estimates true model performance from fitted data.


## Reference
* Daniel Kuhn, Soroosh Shafiee, and Wolfram Wiesemann. Distributionally robust optimization. arXiv
preprint arXiv:2411.02549, 2024.
