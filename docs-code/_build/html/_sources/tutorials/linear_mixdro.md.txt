# Hybrid-distance DRO
This guide covers Hybrid-distance DRO formulations: Sinkhorn-DRO, Holistic Robust DRO, MOT-DRO, and outlier-robust Wasserstein DRO. They combine transport-based distances with entropy, divergence, moment, or contamination constraints. Let $\widehat P$ denote the empirical training distribution.


## Sinkhorn-DRO
In Sinkhorn-DRO [1], $\mathcal{P}(W_{\epsilon};{\rho,\epsilon})= \{P: W_{\epsilon}(P,\hat{P})\leq \rho \}$. Here $W_{\epsilon}(\cdot,\cdot)$ denotes the Sinkhorn Distance, defined as:

$$
W_{\epsilon}(P,Q) = \inf_{\gamma \in \Pi(P,Q)}\mathbb{E}_{(x,y)\sim \gamma}[c(x,y)]+\epsilon\cdot H(\gamma\vert \mu\otimes\nu),
$$

where $\mu,\nu$ are reference measures satisfying $P\ll \mu$ and $Q\ll \nu$.

### Hyperparameters
* `reg_param`: entropic regularization strength $\epsilon$, also used as the Gaussian perturbation variance. Please refer to Remark 8 in [1].
* `lambda_param`: dual scaling parameter $\lambda$. The robust loss uses the product `lambda_param * reg_param`; see Equation (14) in [1].
* k_sample_max: $l$ in Equation (14) in [1], where $2^l$ points are sampled to approximate the sub-gradient.



## Holistic-DRO
In Holistic-DRO [2], $\mathcal{P}(LP_{\mathcal N}, D_{KL}; \alpha, r) = \{P: P,Q\in\mathcal{P}, LP_{\mathcal N}(\hat{P},Q)\leq \alpha, D_{KL}(Q\|P)\leq r \}$, which depends on two metrics divergence:
* Levy-Prokhorov metric $LP_{\mathcal N}(P,Q) = \inf_{\gamma\in\Pi(P,Q)} \mathbb{I}(\xi-\xi'\notin \mathcal{N})d\gamma(\xi, \xi')$, where $\mathcal N$ denotes the perturbed ball of each sample.
* KL-divergence $D_{KL}(Q\|P) = \int_Q \log \frac{dQ}{dP}dQ$.

We support linear losses (SVM for classification and LAD for regression), where we follow Appendix D.1 and D.2 in [2], and we set the worst-case domain $\Sigma = \{(X_i, Y_i): i \in [n]\} + B_2(0,\epsilon') \times \{0\}$ and $\mathcal N = B_2(0, \epsilon) \times \{0\}$.
### Hyperparameters
* $r$: Robustness parameter for the KL-DRO, denoted as ``r`` in the model config.
* $\alpha$: Robustness parameter for the Levy-Prokhorov metric DRO, denoted as ``alpha`` in the model config.
* $\epsilon$: Robustness parameter for the model noise (perturbed ball size), denoted as ``epsilon`` in the model config.
* $\epsilon'$: Domain parameter, denoted as ``epsilon_prime`` in the model config.

## MOT-DRO
In MOT-DRO [3], $\mathcal{P}(M_c;\epsilon) = \{(Q, \delta): M_c((Q, \delta), \tilde P) \leq \epsilon\}$
uses the OT-discrepancy with moment constraints, defined as:

$$
M_c(P,Q)= \inf_\pi \mathbb{E}_\pi[c((Z,W),(\hat Z, \hat W))],
$$

where $\pi_{(Z,W)}=P, \pi_{(\hat Z, \hat W)}=Q$, and $\mathbb{E}_\pi[W]=1$.
Taking the cost function as

$$
c((z,w), (\hat z, \hat w))=\theta_1\cdot w \cdot \|\hat x - x\|_p^s +\theta_2\cdot (\phi(w)-\phi(\hat w))_+,
$$

where $\tilde{P} =\hat{P} \otimes \delta_1$. Here $x$ is the feature component of
$z=(x,y)$; targets $y$ remain fixed. The norm order $p$ and transport exponent
$s$ are independent, configured through `p` and `square`, respectively.

We support linear losses (SVM for classification and LAD for regression), where we follow Theorem 5.2 and Corollary 5.1 in [3].

### Hyperparameters
* $\theta_1$ (or $\theta_2$): relative penalty of feature transport or likelihood perturbation. The constructor sets both to `1`. Updating either through `theta1` or `theta2` requires a value greater than `1` and sets the other so that $\frac{1}{\theta_1} + \frac{1}{\theta_2} = 1$.
* $\epsilon$: robustness radius for the OT ambiguity set, configured through `eps`.
* $p$: feature perturbation norm order, configured through `p`. Supported values are `1`, `2`, and `'inf'`; the default is `2`.
* $s$: transport exponent, configured through `square`. Supported values are `1` and `2`; the default is `2`.

For squared Euclidean feature transport, use:

```python
from dro.linear_model.mot_dro import MOTDRO

model = MOTDRO(input_dim=2, model_type='lad')
model.update({'eps': 0.1, 'p': 2, 'square': 2})
```

## Outlier-Robust Wasserstein DRO
In Outlier-Robust Wassersteuin DRO (OR-WDRO) [4], $\mathcal{P}(W_p^{\eta};\epsilon) = \{Q: W_p^{\eta}(Q, \widehat P)\leq \epsilon\}$, where:

$$
W_p^{\eta}(P, Q) = \inf_{Q' \in \mathcal{P}(R^d), \|Q - Q'\|_{TV}\leq \eta} W_p(P, Q'),
$$

where $p$ is the $p$-Wasserstein distance and $\eta \in [0, 0.5)$ denotes the corruption ratio.

### Hyperparameters
* $p$: Norm parameter for controlling the perturbation moment of X. We only allow the dual norm $\frac{p}{p - 1}$ in $\{1, 2\}$.
* $\eta$: Contamination level $[0, 0.5)$.

We only consider SVM (for classification) and LAD (for regression) based on the convex reformulation of Theorem 2 in [4]. Note that the model also requires the input of $\sigma$, which we take $\sigma = \sqrt{d_x}$ as default.

See the {doc}`Hybrid-distance DRO notebook <../api/notebooks/mixed_dro_tutorial>`
for end-to-end Sinkhorn, MOT, OR-WDRO, and Holistic DRO examples.

## Reference
* [1] Wang, Jie, Rui Gao, and Yao Xie. "Sinkhorn distributionally robust optimization." arXiv preprint arXiv:2109.11926 (2021).
* [2] Bennouna, Amine, and Bart Van Parys. "Holistic robust data-driven decisions." arXiv preprint arXiv:2207.09560 (2022).
* [3] Jose Blanchet, Daniel Kuhn, Jiajin Li, Bahar Taskesen. "Unifying Distributionally Robust Optimization via Optimal Transport Theory." arXiv preprint arXiv:2308.05414 (2023).
* [4] Nietert, Sloan, Ziv Goldfeld, Soroosh Shafiee, "Outlier-Robust Wasserstein DRO." NeurIPS 2023.
