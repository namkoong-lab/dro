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
### Exact Fitting: Linear
We discuss the implementations of different classification and regression losses,

Classification:
* SVM Loss (``svm``): $\ell(f(X), Y) = \max\{1 - Y (\theta^{\top}X + b), 0\}.$
* Logistic Loss (``logistic``): $\ell(f(X), Y) = \log(1 + \exp(-Y(\theta^{\top}X + b))).$

Note that in classification tasks, $Y \in \{-1, 1\}$.

Regression:
* Least Absolute Deviation (``lad``): $\ell(f(X), Y) = |Y - \theta^{\top}X - b|$.
* Ordinary Least Squares (``ols``): $\ell(f(X), Y) = (Y - \theta^{\top} X - b)^2$. 

Above, we designate the ``model_type`` as the names in parentheses.

And our package can support ($\ell_2$ linear regression), $\max\{1 - Y \theta^{\top}X, 0\}$ (SVM loss) and etc. 


Across the linear module, we designate the vector $\theta = (\theta_1,\ldots, \theta_p)$ as ``theta`` and $b$ as ``b``.

Besides this, we support other loss types.

Solvers support: The built-in solvers in ``cvxpy`` (where we set ``Mosek`` during our test).


We support DRO methods including:
* WDRO: (Basic) Wasserstein DRO, Satisificing Wasserstein DRO;
* Standard $f$-DRO: KL-DRO, $\chi^2$-DRO, TV-DRO;
* Generalized $f$-DRO: CVaR-DRO, Marginal DRO (CVaR), Conditional DRO (CVaR);
* MMD-DRO;
* Bayesian-based DRO: Bayesian-PDRO, PDRO;
* Mixed-DRO: Sinkhorn-DRO, HR-DRO, MOT-DRO, Outlier-Robust Wasserstein DRO (OR-Wasserstein DRO).

### Exact or Approximate Fitting: Kernel
We allow kernelized distributionally robust regression or classification via ``.update_kernel()``. We mimic the standard scikit-learn kernel interface with the following hyperparameters:
* metric: standard kernel metrics when calculating kernel between instances in a feature array, including ``additive_chi2``, ``chi2``, ``linear``, ``poly``, ``polynomial``, ``rbf``;
* kernel_gamma:  Parameter gamma of the pairwise kernel specified by metric. It should be positive, or ``scale``, ``auto``.
* n_components: Exact fitting -- ``None``; Approximate fitting -- int, which denotes the reduced number of data points to construct the kernel mapping in Nystroem approximation (recommend to use when $n$ is large).

### Approximate Fitting: Neural Network
Given the complexity of neural networks, many of the explicit optimization algorithms are not applicable. And we implement four DRO methods in an "approximate" way, including:
* $\chi^2$-DRO;
* CVaR-DRO;
* Wasserstein DRO: we approximate it via adversarial training;
* Holistic Robust DRO.

Furthermore, the model architectures supported in `dro` include:
* Linear Models;
* Vanilla MLP;
* AlexNet;
* ResNet18.
  
And the users could also use their own model architecture (please refer to the `update` function in `BaseNNDRO`).


### Approximate Fitting: Tree-based Ensemble DRO
Due to the popular use of tree-based ensemble models in many applciations, we implement two DRO methods in an "approximate" way as a preliminary setting, including:
* KL-DRO
* CVaR-DRO

The current model architectures supported in `dro` include:
* LightGBM
* XGBoost

### Evaluation
In some of linear DRO models, we provide additional interfaces for understanding the worst-case model performance (refer to the ``worst_distribution`` function in each derivative DRO) and evaluating the true model performance in terms of the true MSE estimated from the fitted data (refer to `evaluate` function in `BaseLinearDRO`). 


## Reference
* Daniel Kuhn, Soroosh Shafiee, and Wolfram Wiesemann. Distributionally robust optimization. arXiv
preprint arXiv:2411.02549, 2024.