# MMD-DRO
In MMD-DRO [1], $\mathcal{P}(d, \eta) = \{Q: d(Q, \hat P)\leq \eta\}$.  
Here, $d(P, Q)$ as the kernel distance with the Gaussian kernel, i.e., $\|\mu_P - \mu_Q\|_{\mathcal H}$, which is defined as:

$$\|\mu_P - \mu_Q\|_{\mathcal H}^2 = \mathbb E_{x, x' \sim P}[k(x, x')] + \mathbb E_{y, y' \sim Q}[k(y, y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)], $$

with $k(x, y) = \exp(-\|x - y\|_2^2 / (2\sigma^2))$.

In the computation, we apply Equation (7) 3.1.1 in [1], which requires the following hyperparameters:
## hyperparameters
* $\eta$, the size of the MMD ambigiuty set, denoted as ``eta``.

Besides, MMD-DRO requires constructing ambiguity sets supported on some $\{(\tilde X_j, \tilde Y_j)\}_{j \in [M]}\subseteq \mathcal{X} \times \mathcal{Y}$, where setting $(\tilde X_j, \tilde Y_j) = (X_j, Y_j)$ for $j \in [n]$ and creates new data which leads to additional input parameters: 
* ``sampling_method``, chosen from ``bound`` or ``hull``. When ``sampling_method == bound``, we set each new $(\tilde X, \tilde Y)$ uniformly sampled from $[-1, 1]^{d + 1}$; When ``sampling_method == hull``, we set each new $(\tilde X, \tilde Y)$ uniformly sampled from the range of each individual component for regression problems while setting $X$ sampled from $[-1,1]^d$ and $Y$ samplefor classification problems. 
* ``n_certify_ratio``, the additional number of samples created, i.e., the size $\frac{M - n}{n}$. 


## Reference
* [1] Zhu, Jia-Jie, et al. "Kernel distributionally robust optimization: Generalized duality theorem and stochastic approximation." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.
