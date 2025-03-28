## DRO: A Package for Distributionally Robust Optimization in Machine Learning

> <a href="https://ljsthu.github.io">Jiashuo Liu*</a>, <a href="https://wangtianyu61.github.io">Tianyu Wang*</a>, <a href="https://pengcui.thumedialab.com">Peng Cui</a>, <a href="https://hsnamkoong.github.io">Hongseok Namkoong</a>, <a href="https://web.stanford.edu/~jblanche/">Jose Blanchet</a>

> Tsinghua University, Columbia University, Stanford University


`DRO` is a python package that implements 12 typical DRO methods on linear loss (SVM, logistic regression, and linear regression) for supervised learning tasks. It is built based on the convex optimization solver `cvxpy`. Without specified, our DRO model is to solve the following optimization problem:
$$\min_{\theta} \max_{P: P \in U} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))],$$
where $U$ is the so-called ambiguity set and typically of the form $U = \{P: d(P, \hat P_n) \leq \epsilon\}$ and $\hat P_n := \frac{1}{n}\sum_{i = 1}^n \delta_{(X_i, Y_i)}$ is the empirical distribution of training samples $\{(X_i, Y_i)\}_{i = 1}^n$. And our package can support $\ell(\theta;(X,Y)) = (Y - \theta^{\top} X)^2$ ($\ell_2$ linear regression), $\max\{1 - Y \theta^{\top}X, 0\}$ (SVM loss) and etc. And $\epsilon$ is the hyperparameter 

## Implemented Algorithms
We support DRO methods including:
* WDRO: (Basic) Wasserstein DRO, Satisificing Wasserstein DRO
* $f$-DRO: KL-DRO, $\chi^2$-DRO, TV-DRO, CVaR-DRO, Marginal DRO (CVaR), Conditional DRO (CVaR)
* MMD-DRO
* Bayesian-based DRO: Bayesian-PDRO, PDRO
* Mixed-DRO: Sinkhorn-DRO, Holistic DRO, Unified-DRO ($L_2$ / $L_{\infty}$ cost), Outlier-Robust Wasserstein DRO (OR-Wasserstein DRO)

Then we give some high-level construction of each method and corresponding references as follows. 

### Wasserstein-DRO
For the basic Wasserstein-DRO model,
* we apply $d(P, Q) = W_c(P, Q)$, where $W_c$ is the Wasserstein distance with the induced cost function inside it being $c((X_1, Y_1), (X_2, Y_2)) = X_1^{\top} \Lambda X_2$, the default $\Lambda$ is taken as the unit matrix and $c(\cdot, \cdot)$ then becomes squared Euclidean distance. That is, we only allow the perturbation of $X$ but not $Y$.   
* Reference: Blanchet, Jose, et al. "Data-driven optimal transport cost selection for distributionally robust optimization." 2019 winter simulation conference (WSC). IEEE, 2019., where Theorem 1 there provides the corresponding reformulations.

For the augmented Wasserstein-DRO model,
* we still use $d(P, Q)$ being the Wasserstein distance $W_c$ but now we allow the change of $y$. More specifically, we set $c((X_1, Y_1), (X_2, Y_2)) = X_1^{\top}\Lambda X_2 + \kappa|Y_1 - Y_2|$, and we introduce another hyperparameter $\kappa$ to adjust the change in $Y$. Note that when $\kappa = \infty$, it reduces to the previous case.
* Reference:  Shafieezadeh-Abadeh, Soroosh, Daniel Kuhn, and Peyman Mohajerin Esfahani. "Regularization via mass transportation." Journal of Machine Learning Research 20.103 (2019): 1-68, where they provide various loss functions to reformulate the problem.

For the Satisficing Wasserstein-DRO model, 
* we solve the following constrained optimization problem, where DRO is set as the constraint counterpart:
$$\max \epsilon, \text{s.t.}~\min_{\theta} \max_{P: P \in U_{\epsilon}} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))] \geq \tau,$$
where we do not set $\epsilon$ as the hyperparameter but as an optimization goal such that to minimize the worst-case performance. Instead, we set $\tau$ as the hyperparameter, which is set as the multiplication of the best empirical performance with $E_{(X, Y)\sim \hat P_n}[\ell(\theta_{ERM};(X, Y))]$. To solve this optimization problem, at each binary search, we prefix $\epsilon \in [0, M]$ and compute the corresponding left-hand objective value and then reduce the potential $\epsilon$ half by half.
* Reference: Long, Daniel Zhuoyu, Melvyn Sim, and Minglong Zhou. "Robust satisficing." Operations Research 71.1 (2023): 61-82.
### (General) f-DRO
Models listed here can all be formulated with the distance being $d(P, Q) = E_{Q}[f(\frac{dP}{dQ})]$ unless specified.

For KL-DRO problem, 
* we apply $f(x) = x \log x - (x - 1)$.
* Reference: Hu, Zhaolin, and L. Jeff Hong. "Kullback-Leibler divergence constrained distributionally robust optimization." Available at Optimization Online 1.2 (2013): 9.

For $\chi^2$-DRO problem, 
* we apply $f(x) = (x - 1)^2$.
* Reference: Duchi, John, and Hongseok Namkoong. "Variance-based regularization with convex objectives." Journal of Machine Learning Research 20.68 (2019): 1-55.

For TV-DRO problem, 
* we apply $f(x) = |x - 1|$.
* Reference: Ruiwei Jiang, Yongpei Guan (2018) Risk-Averse Two-Stage Stochastic Program with Distributional Ambiguity. Operations Research 66(5):1390-1405.

For the CVaR-DRO problem,
* we apply $f(x) = 0$ if $x \in [\frac{1}{\alpha}, \alpha]$ and $\infty$ otherwise (an augmented definition of the standard $f$-DRO problem). Here $\alpha$ denotes the worst-case ratio. 
* Reference: R Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal of risk, 2: 21–42, 2000.

Especially, if we only consider the shifts in the marginal distribution $X$, then we obtain the marginal-CVaR model,
* Reference: Duchi, John, Tatsunori Hashimoto, and Hongseok Namkoong. "Distributionally robust losses for latent covariate mixtures." Operations Research 71.2 (2023): 649-664.

### MMD-DRO
* We set $d(P, Q)$ as the kernel distance with the Gaussian kernel.
* Reference: Zhu, Jia-Jie, et al. "Kernel distributionally robust optimization: Generalized duality theorem and stochastic approximation." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.
### Bayesian-based DRO
Bayesian-PDRO:
* Reference: Shapiro, Alexander, Enlu Zhou, and Yifan Lin. "Bayesian distributionally robust optimization." SIAM Journal on Optimization 33.2 (2023): 1279-1304.

### Sinkhorn-DRO
* Reference: Wang, Jie, Rui Gao, and Yao Xie. "Sinkhorn distributionally robust optimization." arXiv preprint arXiv:2109.11926 (2021).

### Holistic-DRO
* Reference: Bennouna, Amine, and Bart Van Parys. "Holistic robust data-driven decisions." arXiv preprint arXiv:2207.09560 (2022).
### MOT-DRO
* Reference: Blanchet, Jose, et al. "Unifying Distributionally Robust Optimization via Optimal Transport Theory." arXiv preprint arXiv:2308.05414 (2023).

## Base Models
Currently we only focus on linear models, including logistic regression, linear SVM, and linear regression. And further version will incorporate neural network implementations via some approximations.

## Install

```
pip install dro
```

#### Usage
```
from dro import fetch_method
method = fetch_method("name", is_regression, input_dim)
```

### Reference
