# DRO with a mixture of distance metrics
We further implement several DRO methods based on a mixture of distance metrics, including Sinkhorn-DRO, Holistic Robust DRO, and DRO based on OT discrepancy with moment constraints (MOT-DRO).


## Sinkhorn-DRO
The objective function of Sinkhorn-DRO is:

$$
\min_\theta \sup_{P\in \mathcal{B}_{\rho,\epsilon}(P_{tr})} \mathbb{E}_P[\ell(f_\theta(X),Y)],
$$

where $\mathcal{B}_{\rho,\epsilon}(P_{tr})= \{P: W_{\epsilon}(P,P_{tr})\leq \rho \}$. Here $W_{\epsilon}(\cdot,\cdot)$ denotes the Sinkhorn Distance, defined as:

$$
W_{\epsilon}(P,Q) = \inf_{\gamma \in \Pi(P,Q)}\mathbb{E}_{(x,y)\sim \gamma}[c(x,y)]+\epsilon\cdot H(\gamma\vert \mu\otimes\nu),
$$

where $\mu,\nu$ are reference measures satisfying $P\ll \mu$ and $Q\ll \nu$.


Reference: Wang, Jie, Rui Gao, and Yao Xie. "Sinkhorn distributionally robust optimization." arXiv preprint arXiv:2109.11926 (2021).


## Holistic-DRO
Holist Robust DRO is based on two metrics, convex pseudo divergence metric $LP_{\mathcal N}(\cdot,\cdot)$

$$
LP_{\mathcal N}(P, Q) =  \inf_{\gamma\in\Pi(P,Q)} \inf \mathbb{I}(\xi-\xi'\notin \mathcal{N})d\gamma(\xi, \xi'),
$$

and KL-divergence $D_{KL}(\cdot\|\cdot)$. The objective function is:

$$
\min_\theta\sup_{P\in \mathcal{B}_{\alpha,r}(P_{tr})}\mathbb{E}_{P}[\ell(f_\theta(X),Y)],
$$

where the uncertainty set is defined as:

$$
\mathcal{B}_{\alpha,r}(P_{tr}) = \{P: P,Q\in\mathcal{P}, LP_{\mathcal N}(P_{tr},Q)\leq \alpha, D_{KL}(Q\|P)\leq r \}.
$$

For linear models, we follow Appendix D.1 and D.2, and code for neural networks is based on the official codes.

Reference: Bennouna, Amine, and Bart Van Parys. "Holistic robust data-driven decisions." arXiv preprint arXiv:2207.09560 (2022).

## MOT-DRO
MOT-DRO uses the OT-discrepancy with moment constraints, defined as:

$$
M_c(P,Q)= \inf_\pi \mathbb{E}_\pi[c((Z,W),(\hat Z, \hat W))],
$$

where $\pi_{(Z,W)}=P, \pi_{(\hat Z, \hat W)}=Q$, and $\mathbb{E}_\pi[W]=1$.
Taking the cost function as

$$
c((z,w), (\hat z, \hat w))=\theta_1\cdot w \cdot d(z,\hat z)+\theta_2\cdot (\phi(w)-\phi(\hat w))_+,
$$

the objective function of MOT-DRO is:

$$
\min_\theta\sup_{P: M_c(P,\hat{P}_{train})\leq \epsilon} \mathbb{E}_P[\ell(f_\theta(X),Y)],
$$

where $\hat{P}_{train}=P_{train}\otimes \delta_1$.


Reference: Jose Blanchet, Daniel Kuhn, Jiajin Li, Bahar Taskesen. "Unifying Distributionally Robust Optimization via Optimal Transport Theory." arXiv preprint arXiv:2308.05414 (2023).
