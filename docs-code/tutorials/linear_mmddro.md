# MMD-DRO
In MMD-DRO [1], $\mathcal{P}(d, \eta) = \{Q: d(Q, \hat P)\leq \eta\}$.
Here, $d(P, Q)$ is the kernel distance $\|\mu_P - \mu_Q\|_{\mathcal H}$, which is defined as:

$$\|\mu_P - \mu_Q\|_{\mathcal H}^2 = \mathbb E_{x, x' \sim P}[k(x, x')] + \mathbb E_{y, y' \sim Q}[k(y, y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)], $$

The kernel $k$ controls which differences between distributions are emphasized. The default is the Gaussian RBF kernel,

$$k(x,y)=\exp(-\gamma\|x-y\|_2^2),$$

but the implementation also supports Laplacian, polynomial, and linear kernels.

In the computation, we apply Equation (7) 3.1.1 in [1], which requires the following hyperparameters:
## Hyperparameters

* $\eta$, the size of the MMD ambiguity set, denoted as ``eta``.
* ``kernel``, the kernel defining the MMD ambiguity set. Choose from:
  * ``rbf`` (default): $\exp(-\gamma\|x-y\|_2^2)$. When ``kernel_gamma=None``, $\gamma$ is the reciprocal of the median nonzero squared pairwise distance.
  * ``laplacian``: $\exp(-\gamma\|x-y\|_1)$. When ``kernel_gamma=None``, $\gamma$ is the reciprocal of the median nonzero $\ell_1$ distance. This is another characteristic kernel and is often useful for less smooth similarity.
  * ``polynomial`` (alias ``poly``): $(\gamma x^\top y+c_0)^d$. The defaults are degree $d=2$, offset $c_0=1$, and $\gamma=1/p$, where $p$ is the dimension of the joint sample $(X,Y)$. Tune ``kernel_degree`` over small positive integers such as 2 or 3; large degrees can make the Gram matrix poorly scaled.
  * ``linear``: $x^\top y$. It has no kernel-specific parameters and captures differences in first moments only.
* ``kernel_gamma``, an optional positive numeric value overriding the automatic scale for RBF, Laplacian, and polynomial kernels.
* ``kernel_degree``, a positive integer used by the polynomial kernel (default: ``2``).
* ``kernel_coef0``, a non-negative offset used by the polynomial kernel (default: ``1.0``). Non-negative values together with positive ``kernel_gamma`` and an integer degree preserve a positive-semidefinite polynomial kernel.

Besides, MMD-DRO requires constructing ambiguity sets supported on some $\{(\tilde X_j, \tilde Y_j)\}_{j \in [M]}\subseteq \mathcal{X} \times \mathcal{Y}$, where setting $(\tilde X_j, \tilde Y_j) = (X_j, Y_j)$ for $j \in [n]$ and creates new data which leads to additional input parameters:

* ``sampling_method``, chosen from ``bound`` or ``hull``. When ``sampling_method == bound``, we set each new $(\tilde X, \tilde Y)$ uniformly sampled from $[-1, 1]^{d + 1}$; when ``sampling_method == hull``, each regression coordinate is sampled within its observed range, while classification labels are sampled from $\{-1,+1\}$.
* ``n_certify_ratio``, the additional number of samples created, i.e., the size $\frac{M - n}{n}$.

For example, the default RBF kernel needs no explicit kernel settings:

```python
from dro.linear_model.mmd_dro import MMD_DRO

model = MMD_DRO(input_dim=X.shape[1], model_type="svm")
model.update({"eta": 0.1, "sampling_method": "hull"})
params = model.fit(X, y)
```

To use an inhomogeneous quadratic polynomial kernel:

```python
model = MMD_DRO(
    input_dim=X.shape[1],
    model_type="svm",
    kernel="polynomial",
    kernel_degree=2,
    kernel_coef0=1.0,
    kernel_gamma=None,  # defaults to 1 / (input_dim + 1)
)
model.update({"eta": 0.1, "sampling_method": "hull"})
params = model.fit(X, y)
```

Because MMD-DRO applies the ambiguity-set kernel to the joint samples $(X,Y)$, standardizing continuous features and targets before fitting is recommended, especially for polynomial kernels.

See the {doc}`kernel DRO notebook <../api/notebooks/kernel_dro_tutorial>` for
an end-to-end MMD-DRO example and ambiguity-set kernel selection guidance.


## Reference
* [1] Zhu, Jia-Jie, et al. "Kernel distributionally robust optimization: Generalized duality theorem and stochastic approximation." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.
