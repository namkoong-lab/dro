# DRO with a mixture of distance metrics
Recall the training distribution as $\hat P$, we further implement several DRO methods based on a mixture of distance metrics, including Sinkhorn-DRO, Holistic Robust DRO, and DRO based on OT discrepancy with moment constraints (MOT-DRO).


## Sinkhorn-DRO
In Sinkhorn-DRO [1], $\mathcal{P}(W_{\epsilon};{\rho,\epsilon})= \{P: W_{\epsilon}(P,\hat{P})\leq \rho \}$. Here $W_{\epsilon}(\cdot,\cdot)$ denotes the Sinkhorn Distance, defined as:

$$
W_{\epsilon}(P,Q) = \inf_{\gamma \in \Pi(P,Q)}\mathbb{E}_{(x,y)\sim \gamma}[c(x,y)]+\epsilon\cdot H(\gamma\vert \mu\otimes\nu),
$$

where $\mu,\nu$ are reference measures satisfying $P\ll \mu$ and $Q\ll \nu$.

### Hyperparameters
* 

## Holistic-DRO
In Holistic-DRO [2], $\mathcal{P}(LP_{\mathcal N}, D_{KL}; \alpha, r) = \{P: P,Q\in\mathcal{P}, LP_{\mathcal N}(\hat{P},Q)\leq \alpha, D_{KL}(Q\|P)\leq r \}$, which depends on two metrics divergence: 
* Levy-Prokhorov metric $LP_{\mathcal N}(P,Q) = \inf_{\gamma\in\Pi(P,Q)} \inf \mathbb{I}(\xi-\xi'\notin \mathcal{N})d\gamma(\xi, \xi')$, 
* KL-divergence $D_{KL}(Q\|P) = \int_Q \log \frac{dQ}{dP}dQ$.

We support linear losses (SVM for classification and LAD for regression), where we follow Appendix D.1 and D.2 in [2].
### Hyperparameters
* 

## MOT-DRO
In MOT-DRO [3], $\mathcal{P}(M_c;\epsilon) = \{(Q, \delta): M_c((Q, \delta), \tilde P) \leq \epsilon\}$
uses the OT-discrepancy with moment constraints, defined as:

$$
M_c(P,Q)= \inf_\pi \mathbb{E}_\pi[c((Z,W),(\hat Z, \hat W))],
$$

where $\pi_{(Z,W)}=P, \pi_{(\hat Z, \hat W)}=Q$, and $\mathbb{E}_\pi[W]=1$.
Taking the cost function as

$$
c((z,w), (\hat z, \hat w))=\theta_1\cdot w \cdot d(z,\hat z)+\theta_2\cdot (\phi(w)-\phi(\hat w))_+,
$$

where $\tilde{P} =\hat{P} \otimes \delta_1$.

### Hyperparameters
* 
## Outlier-Robust Wasserstein DRO
In Outlier-Robust Wassersteuin DRO (OR-WDRO) [4], $\mathcal{P}(W_p^{\eta};\epsilon) = \{Q: W_p^{\eta}(Q, \hat P)\leq \epsilon\}$, where:

$$
W_p^{\eta}(P, Q) = \inf_{Q' \in \mathcal{P}(R^d), \|Q - Q'\|_{TV}\leq \eta} W_p(P, Q'), 
$$

where $p$ is the $p$-Wasserstein distance and $\eta \in [0, 0.5)$ denotes the corruption ratio.

### Hyperparameters
* $p$: Norm parameter for controlling the perturbation moment of X. We only allow the dual norm $\frac{p}{p - 1}$ in $\{1, 2\}$.
* $\eta$: Contamination level $[0, 0.5)$.

We only consider SVM (for classification) and LAD (for regression) based on the convex reformulation of Theorem 2 in [4]. Note that the model also requires the input of $\sigma$, which we take $\sigma = \sqrt{d_x}$ as default.
## Reference
* [1] Wang, Jie, Rui Gao, and Yao Xie. "Sinkhorn distributionally robust optimization." arXiv preprint arXiv:2109.11926 (2021).
* [2] Bennouna, Amine, and Bart Van Parys. "Holistic robust data-driven decisions." arXiv preprint arXiv:2207.09560 (2022).
* [3] Jose Blanchet, Daniel Kuhn, Jiajin Li, Bahar Taskesen. "Unifying Distributionally Robust Optimization via Optimal Transport Theory." arXiv preprint arXiv:2308.05414 (2023).
* [4] Nietert, Sloan, Ziv Goldfeld, Soroosh Shafiee, "Outlier-Robust Wasserstein DRO." NeurIPS 2023.