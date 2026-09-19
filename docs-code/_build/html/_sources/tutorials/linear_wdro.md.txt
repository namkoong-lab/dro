# Wasserstein DRO
In Wasserstein DRO,
$\mathcal{P}(d, \epsilon) = \{Q: W(Q, \hat P)\leq \epsilon\}$.

Specifically, Wasserstein distance is defined as follows. For
$Z_1 = (X_1, Y_1)$ and $Z_2 = (X_2, Y_2)$,

$$
W(P_1, P_2) = \inf_{\pi \sim (P_1, P_2)}\mathbb{E}_{\pi}[d(Z_1, Z_2)],
$$
For ``lad``, the ground cost is

$$
d((x,y),(x',y'))=\|x-x'\|_{\Sigma,p}+\kappa|y-y'|.
$$

LAD requires $\kappa>0$ (or $\kappa=\infty$); at $\kappa=0$, continuous
target changes are free and the robust absolute loss is unbounded.

For binary classification with ``svm`` or ``logistic``, the label space is
discrete and the implementation uses

$$
d((x,y),(x',y'))=\|x-x'\|_{\Sigma,p}
    +\kappa\mathbf{1}_{\{y\ne y'\}}.
$$

Thus a label flip costs $\kappa$, independent of whether labels are encoded as
$-1$ and $1$. The ``svm`` and ``logistic`` models use this same ground cost;
they differ in their loss functions (hinge versus logistic), not in their
transport distances. Setting $\kappa=\infty$ forbids label changes. For
``ols``, changes in $Y$ are prohibited and the ground cost is

$$
d((x,y),(x',y'))=
\begin{cases}
\|x-x'\|_{\Sigma,p}^2,&y=y',\\
+\infty,&y\ne y'.
\end{cases}
$$

Here $\|x\|_{\Sigma,p}=\|\Sigma^{1/2}x\|_p$. For the continuous product norm
used by ``lad``, the dual norm is

$$
\|(u,v)\|_* = \max\left\{\|u\|_{\Sigma^{-1},q},
    \frac{|v|}{\kappa}\right\},
$$

where $1/p+1/q=1$.

## Standard Version

### Hyperparameter
* $\Sigma$: the feature importance perturbation matrix with the dimension being the (dimX, dimX).
* $p$: Norm parameter for controlling the perturbation moment of X.
* $\kappa$: Robustness parameter for the perturbation of Y.
* $\epsilon$: Ambiguity size of Wasserstein ball.

For OLS, Theorem 1 in [1] there provides the corresponding reformulations. We set $\kappa = \infty$, i.e., not allow changes in $Y$.

For loss functions in LAD, Logistic, and SVM models, [2] provide to reformulate the problem.

### Worst-case distribution

#### OLS: an exact worst-case distribution

The OLS construction follows the quadratic-loss, quadratic-transport
reformulation in [1, 5]. Let $q$ be conjugate to $p$, and define the fitted
residuals, empirical root mean squared error, and dual slope by

$$
r_i=\theta^\top x_i+b-y_i,\qquad
R=\left(\frac1n\sum_{i=1}^n r_i^2\right)^{1/2},\qquad
L=\|\Sigma^{-1/2}\theta\|_q.
$$

The worst-case expected squared loss is

$$
\sup_{Q:W(Q,\widehat P_n)\leq\epsilon}
\mathbb E_Q[(\theta^\top X+b-Y)^2]
=\left(R+\sqrt{\epsilon}\,L\right)^2.
$$

To see both the upper bound and how it is attained, choose a dual-norm
maximizing direction $v$ such that

$$
\|\Sigma^{1/2}v\|_p=1,
\qquad \theta^\top v=L.
$$

For any feasible feature displacement $\Delta X$, Minkowski's and Hölder's
inequalities give

$$
\sqrt{\mathbb E[(r+\theta^\top\Delta X)^2]}
\leq R+L\sqrt{\mathbb E[\|\Sigma^{1/2}\Delta X\|_p^2]}
\leq R+\sqrt{\epsilon}\,L.
$$

When $R>0$, equality is attained by keeping the uniform weights and targets
unchanged and transporting each feature vector to

$$
x_i^\star=x_i+\frac{\sqrt{\epsilon}}{R}r_i v,
\qquad
Q^\star=\frac1n\sum_{i=1}^n\delta_{(x_i^\star,y_i)}.
$$

Indeed,

$$
\frac1n\sum_i\|\Sigma^{1/2}(x_i^\star-x_i)\|_p^2=\epsilon
$$

and the new residuals are
$r_i(1+\sqrt{\epsilon}L/R)$, which gives expected loss
$(R+\sqrt{\epsilon}L)^2$. If $R=0<L$, every atom can instead be shifted by
$\sqrt{\epsilon}v$. If $L=0$ or $\epsilon=0$, the empirical distribution is
already worst-case. Thus OLS recovery is exact and needs no $\gamma$ or
escaping atom.

The OLS ``fit`` method stores the robust RMSE $R+\sqrt{\epsilon}L$ in
``robust_obj``. Because ``worst_distribution`` evaluates squared residuals,
its OLS ``expected_loss`` and ``target_objective`` are expressed in MSE units
and are certified against ``robust_obj**2``. Call the exact recovery without
asymptotic options:

```python
wc = model.worst_distribution(X, y)
```

#### SVM, logistic, and LAD: asymptotic recovery

``worst_distribution`` implements the asymptotic constructions in Theorems 9
and 20 of [2] for linear ``lad``, ``svm``, and ``logistic`` models. These
constructions form a family $Q_\gamma$; they do not in general produce an
attained worst-case distribution. A strictly positive $\gamma$ materializes a
finite distribution, and its expected loss approaches the optimized WDRO value
as $\gamma\downarrow0$. At the same time, some mass vanishes and its destination
moves to infinity. Consequently, $\gamma=0$ denotes a limiting objective in the
theory and is not a valid finite-distribution parameter.

For positive Wasserstein radius, ``gamma=None`` chooses a valid positive
starting value automatically. A user-supplied starting value must satisfy

- $0<\gamma\leq\min\{\epsilon,1\}$ for ``svm`` and ``logistic`` with finite
  $\kappa$;
- $0<\gamma\leq1$ for ``svm`` and ``logistic`` with $\kappa=\infty$, and for
  ``lad``.

The automatic starting value is one tenth of the applicable upper bound.

A smaller $\gamma$ usually reduces the optimality gap, but also places an atom
farther away and can make the computation less numerically stable. For
$\epsilon=0$, the method returns the empirical distribution exactly and no
asymptotic construction is needed. These controls are passed together in the
``asymptotic_options`` dictionary. The certification loop reduces the starting
value by ``gamma_decay`` for at most ``max_iter`` attempts. The objective checks
use ``objective_atol`` and ``objective_rtol``; transport feasibility uses
``feasibility_tol``.

For classification, the escaping atom is moved in a label-aware recession
direction: the direction must increase the asymptotic loss, not merely the
linear score. For hinge and logistic loss it solves
$\max_{\|d\|_{\Sigma,p}\leq1}-y_a\theta^\top d$, where $(x_a,y_a)$ is the
empirical anchor. Under Euclidean feature cost this direction is proportional
to $-y_a\theta$. This distinction is essential because moving in the same
feature direction for both labels lowers the loss for one of the two classes.

The returned candidate is certified before it is exposed as an approximation
to a worst-case distribution. Every generated atom retains the index of the
empirical sample from which its mass was moved. These source indices specify an
explicit coupling with cost

$$
C(Q_\gamma)=\sum_j q_j\,
    d\!\left(z_j,\widehat z_{\operatorname{source}(j)}\right).
$$

The implementation requires both

$$
C(Q_\gamma)\leq\epsilon+\text{feasibility tolerance}
$$

and agreement, within ``objective_atol`` and ``objective_rtol``, between the
candidate's expected loss and the optimized WDRO objective. If either check
fails, the method tries a smaller positive $\gamma$. It raises an error rather
than returning an uncertified candidate if no attempt succeeds.

The returned dictionary contains ``sample_pts`` and ``weight`` together with
the following audit metadata:

- ``source_index``: empirical source of every returned atom;
- ``gamma_used`` and ``kappa_used``: effective construction parameters;
- ``expected_loss`` and ``target_objective``: the two certified objective
  values;
- ``optimality_gap`` and ``transport_cost``: the objective and feasibility
  diagnostics;
- ``certified`` and ``asymptotic``: whether the checks passed and whether the
  construction represents an asymptotic sequence.

For example:

```python
wc = model.worst_distribution(
    X,
    y,
    asymptotic_options={
        "gamma": None,
        "objective_atol": 1e-5,
        "objective_rtol": 1e-5,
        "feasibility_tol": 1e-7,
        "max_iter": 20,
        "gamma_decay": 0.5,
    },
)
```

Finite $\kappa$ permits classification-label flips at cost $\kappa$ per unit of
transported mass. With $\kappa=\infty$, labels remain fixed and the implementation
uses a label-preserving recession construction; it does not approximate
$\infty$ with an arbitrary large finite number.

This recovery method assumes a linear kernel, unbounded feature support, and
one of the supported convex Lipschitz losses. More generally, the asymptotic
construction also requires the loss's recession growth to attain its Lipschitz
slope, as explained in [4]. OLS uses the separate exact construction above.

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

See the {doc}`Wasserstein DRO notebook <../api/notebooks/WassersteinDRO_tutorial>`
for end-to-end classification, regression, and robust-satisficing examples.



## Reference
* [1] Blanchet, Jose, et al. "Data-driven optimal transport cost selection for distributionally robust optimization." 2019 winter simulation conference (WSC). IEEE, 2019.
* [2] Shafieezadeh-Abadeh, Soroosh, Daniel Kuhn, and Peyman Mohajerin Esfahani. "Regularization via mass transportation." Journal of Machine Learning Research 20.103 (2019): 1-68.
* [3] Long, Daniel Zhuoyu, Melvyn Sim, and Minglong Zhou. "Robust satisficing." Operations Research 71.1 (2023): 61-82.
* [4] Shafiee, Soroosh, Liviu Aolaritei, Florian Dörfler, and Daniel Kuhn. "Nash equilibria, regularization, and computation in optimal transport-based distributionally robust optimization." *Operations Research* 74.3 (2026): 1689–1709.
* [5] Blanchet, Jose, Yang Kang, and Karthyek Murthy. "Robust Wasserstein profile inference and applications to machine learning." *Journal of Applied Probability* 56.3 (2019): 830–857.
