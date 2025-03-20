# Formulation

Given the empirical distribution $\hat P$ from the data $\{(x_i, y_i)\}$, we consider the following (distance-based) distributionally robust optimization formulations under the machine learning context. In general, DRO optimizes over the worst-case loss and satisfies the following structure:
$$\min_{f \in \mathcal{F}}\max_{Q \in \mathcal{P}}\mathbb{E}_Q[\ell(f(X), Y)],$$
where $\mathcal{P}$ is denoted as the ambiguity set. Usually, it satisfies the following structure:
$$\mathcal{P}(d, \epsilon) = \{Q: d(Q, \hat P) \leq \epsilon\}. $$
Here, $d(\cdot, \cdot)$ is a notion of distance between probability measures and $\epsilon$ captures the size of the ambiguity set.

Given each function class $\mathcal{F}$, we classify all the models into the following cases, where each case can be further classified given each distance type $d$.

## Holistic Model Pipeline

Data -> Model -> Evaluation / Diagnostics

## Linear
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

Solvers support:


We support DRO methods including:
* WDRO: (Basic) Wasserstein DRO, Satisificing Wasserstein DRO;
* Standard $f$-DRO: KL-DRO, $\chi^2$-DRO, TV-DRO;
* Generalized $f$-DRO: CVaR-DRO, Marginal DRO (CVaR), Conditional DRO (CVaR);
* MMD-DRO;
* Bayesian-based DRO: Bayesian-PDRO, PDRO
* Mixed-DRO: Sinkhorn-DRO, Holistic DRO, MOT-DRO, Outlier-Robust Wasserstein DRO (OR-Wasserstein DRO).

## Neural Network




## Reference:
* Daniel Kuhn, Soroosh Shafiee, and Wolfram Wiesemann. Distributionally robust optimization. arXiv
preprint arXiv:2411.02549, 2024.