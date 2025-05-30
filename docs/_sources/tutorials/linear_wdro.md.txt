# Wasserstein DRO
In Wasserstein DRO, 
$\mathcal{P}(d, \epsilon) = \{Q: W(Q, \hat P)\leq \epsilon\}$.

Specifically, Wasserstein distance is defined as follows:
For $Z_1 = (X_1, Y_1), Z_2 = (X_2, Y_2)$, 

$$
W(P_1, P_2) = \inf_{\pi \sim (P_1, P_2)}\mathbb{E}_{\pi}[d(Z_1, Z_2)],
$$
where for ``lad``, ``svm``, ``logistic``, the inner distance is captured by the norm: $d((X_1, Y_1), (X_2, Y_2)) = \|(X_1 - X_2, Y_1 - Y_2)\|.$
For ``ols``, the inner distance is captured by the norm square: 
$d((X_1, Y_1), (X_2, Y_2)) = \|(X_1 - X_2, Y_1 - Y_2)\|^2$. 

No matter in each case, the norm is defined on the product space $\mathcal{X} \times \mathbb{R}$ by:

$$
\|(x, y)\| = \|x\|_{\Sigma, p} + \kappa |y|.
$$
Here $\|x\|_{\Sigma, p} = \|\Sigma^{1/2}x\|_p$. Furthermore, we can show that the dual norm $\|(\cdot, \cdot)\|_*$ is given by:

$$
\|(u,v)\|_* = \max\{\|u\|_{\Sigma, q}, \frac{v}{\kappa}\}
$$

with $\frac{1}{p} + \frac{1}{q} = 1$.

The norm is defined by the parameter tuple $(\Sigma, p, \kappa)$,

## Standard Version

### Hyperparameter
* $\Sigma$: the feature importance perturbation matrix with the dimension being the (dimX, dimX).
* $p$: Norm parameter for controlling the perturbation moment of X.
* $\kappa$: Robustness parameter for the perturbation of Y.
* $\epsilon$: Ambiguity size of Wasserstein ball. 

For OLS, Theorem 1 in [1] there provides the corresponding reformulations. We set $\kappa = \infty$, i.e., not allow changes in $Y$. 

For loss functions in LAD, Logistic, and SVM models, [2] provide to reformulate the problem.

### Worst case Distribution
When we set ``compute_type == asymp``, [2] provides asymptotic worst-case distribution for linear regression and classification (Theorems 9 and 20), where we take $\gamma \in (0, 1]$ as the hyperparameter to tune in each case.

``compute_type == exact`` is depreciated right now.

## Robust Satisficing Version

For the Satisficing Wasserstein-DRO model [3], we solve the following constrained optimization problem, where DRO is set as the constraint counterpart:

$$
\max {\epsilon,\quad \text{s.t.}~E_{(X,Y) \sim P}[\ell_{tr}(\theta;(X, Y))] \leq \tau + \epsilon W(P, \widehat P), \forall P}.
$$

For (approximated) regression / classification, we can show the optimization problem above is equivalent to:

$$
\max \{\|\theta\|_{\Sigma^{-1/2},p},\quad \text{s.t.}~ E_{(X,Y) \sim \widehat P}[\ell_{tr}(\theta;(X, Y))] \leq \tau\}.
$$

### Hyperparameter
In the satisfying version,  we do not set $\epsilon$ as the hyperparameter but as an optimization goal such that to minimize the worst-case performance. 
* $\Sigma$: the feature importance perturbation matrix with the dimension being the (dimX, dimX).
* $p$: Norm parameter for controlling the perturbation moment of X.
* $\kappa$: Robustness parameter for the perturbation of Y.
* $\tau$: $\tau \geq 1$ is set as the multiplication of the best empirical performance with $E_{(X, Y)\sim \hat P_n}[\ell(\theta_{ERM};(X, Y))]$. 



## Reference
* [1] Blanchet, Jose, et al. "Data-driven optimal transport cost selection for distributionally robust optimization." 2019 winter simulation conference (WSC). IEEE, 2019.
* [2] Shafieezadeh-Abadeh, Soroosh, Daniel Kuhn, and Peyman Mohajerin Esfahani. "Regularization via mass transportation." Journal of Machine Learning Research 20.103 (2019): 1-68.
* [3] Long, Daniel Zhuoyu, Melvyn Sim, and Minglong Zhou. "Robust satisficing." Operations Research 71.1 (2023): 61-82.