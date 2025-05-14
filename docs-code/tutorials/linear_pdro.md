# Bayesian (Parametric) DRO
In the context of general stochastic optimization $\ell(\theta;\xi)$ (where $\xi = (x, y)$ in the supervised learning setup), Bayesian DRO [1] is formulated as the following nested structure:

$$\min_{\theta \in \Theta}\mathbb{E}_{\zeta \sim \zeta_N}[\sup_{Q \in \mathcal{Q}_{\zeta}(d, \epsilon)} \mathbb{E}_{\xi \sim Q}[\ell(\theta;\xi)]],$$
where $\zeta_N$ denotes the posterior distribution of the parametric distribution of $\xi$  given $\{\xi_i\}_{i \in [N]}$. And $\mathcal{Q}_{\zeta}(d, \epsilon) = \{Q: d(Q, P_{\zeta}) \leq \epsilon\}$ with $P_{\zeta}$ denotes the distribution parametrized by $\zeta$. 


In practice, we approximate the outer expectation $\zeta \sim \zeta_N$ via finite samples generated from the posterior distribution of $\zeta_N$. In this sense, the optimization problem can be reformulated as a optimization problem with finite variables.

Note that the idea of parametric distributions is often utilized in operational settings (e.g., newsvendor [1], portfolio optimization [2]), where we can modify ``_cvx_loss`` to incorporate that. Specifically, we consider two preliminary settings. We implement the `Exponential` distribution class used in the numerical experiments in [1] where we use the rate $\zeta$ in the exponential distribution as the parameter and set $\Gamma(1, 1)$ as the default prior. To accomodate machine learning problems, we also consider ``Gaussian`` distribution class for $\xi$ with a normal prior on the mean vector $\mu$ and an inverse-Wishart prior on the covariance matrix $\Sigma$. Formally, 

$$\Sigma \sim IW(d+2, I), \mu | \Sigma \sim N(0, \Sigma).$$

The NIW Prior is chosen such that it is conjugate to the multivariate Gaussian likelihood, allowing for closed-form posterior updates of $\mu$ and $\Sigma$ after observing data.

Despite the computable prior, ${P}_{\zeta}$ is still a continuous distribution where the optimization problem involves continuous integral and is not tractable. Therefore, we need to conduct Monte carlo sampling for each sampled $\zeta \sim \zeta_N$. 

## Hyperparameters
* $\epsilon$: Robustness parameter, denoted as ``eps``.
* ``distance_type``: The choice of $d$, where we only implement ``KL`` and ``chi2``.
* ``distribution_class``: The choice of distribution class in the parametric family, where we only implement ``Gaussian`` and ``Exponential``.
* ``posterior_param_num``: The number of sampled $\zeta$ for $\zeta_N$, where we restrict it to be smaller than 100 for computation.
* ``posterior_sample_ratio``: The number of Monte Carlo sampled (relative to $N$) for the number of samples for each $P_{\zeta}$.

## Reference
* [1] Shapiro, Alexander, Enlu Zhou, and Yifan Lin. "Bayesian distributionally robust optimization." SIAM Journal on Optimization 33.2 (2023): 1279-1304.
* [2] G. Iyengar, H. Lam, and T. Wang. Hedging complexity in generalization via a parametric
distributionally robust optimization framework. arXiv preprint arXiv:2212.01518, 2022.
