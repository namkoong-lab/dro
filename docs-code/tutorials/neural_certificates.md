# NN-DRO: Robustness Certificates

Neural DRO training uses approximate optimization, but the fitted network can
still admit a bound on its worst-case loss. This tutorial explains how to
construct such a certificate for Wasserstein, $\chi^2$, and CVaR ambiguity
sets, how to account for numerical and mini-batch errors, and when a smaller
ambiguity set can be certified from a larger-set calculation.

See the {doc}`NN-DRO model guide <neural_model>` for the model and training interfaces. The
calculations below describe post-training evaluation; they do not assume that
a training loss or a solver stopping tolerance is already a certificate.
In the current neural base class, `robust_obj` is an average of final-epoch
training-batch objectives [4]. It is not the full-data certificate of the
final frozen network defined here.

## 1. Certificate Guarantees
Fix a fitted network $\widehat\theta$ and a reference dataset
$z_1,\ldots,z_N$, with empirical distribution
$\widehat P=N^{-1}\sum_i\delta_{z_i}$. Write
$\ell_{\widehat\theta}(z)=\ell(f_{\widehat\theta}(x),y)$ and
$v_i=\ell_{\widehat\theta}(z_i)$. For an ambiguity set
$\mathcal P_\epsilon$ centered at $\widehat P$, the quantity to certify is

$$
R_{\epsilon}
=\sup_{Q\in\mathcal P_\epsilon}
\mathbb E_Q[\ell_{\widehat\theta}(Z)].
$$

**Objective.** A valid upper bound $U$ establishes $R_{\epsilon}\le U$. If $U$ is below some
loss threshold, every distribution in the specified ambiguity set satisfies that threshold. A feasible adversarial distribution gives a lower bound $L$,
so $[L,U]$ is an interval for the empirical worst-case loss.

**Radius.** The target radius $\epsilon_0$ must be specified to state which ambiguity set
the guarantee covers. One may train or evaluate at a larger radius
$\epsilon_1\ge\epsilon_0$. For nested ambiguity sets, a **verified upper
bound** $U(\epsilon_1)$ gives

$$
R_{\epsilon_0}\le R_{\epsilon_1}\le U(\epsilon_1).
$$

CVaR uses the tail fraction $\alpha$ instead of a radius; a smaller $\alpha$
gives a larger set. An approximate inner adversarial value is generally a
lower estimate of $R_{\epsilon_1}$, so enlarging the training radius alone
does not provide an upper bound. To use that approximate value itself as a
certificate at the prespecified $\epsilon_0$, the radius margin must cover
a certified numerical gap. The checks below give this condition for each
formulation. Otherwise, compute an upper bound directly at $\epsilon_0$.

All bounds below concern the final frozen network and the same reference
dataset, loss, and admissible support at both parameter settings. They certify
empirical worst-case loss conditional on that dataset. A population confidence statement
needs an additional sampling bound; if $\epsilon_0$ is chosen using the same
evaluation data, that bound must also account for the selection. A literal
numerical certificate also requires verified feasibility and conservative
rounding of computed upper and lower values.

## 2. Wasserstein DRO

### Formulation

Let $\mathcal T_c(P,Q)$ be the minimum expected transport cost over couplings
of $P$ and $Q$. Define

$$
R_\epsilon^W
=\sup_{Q:\,\mathcal T_c(\widehat P,Q)\le\epsilon}
\mathbb E_Q[\ell_{\widehat\theta}(Z)].
$$

Here $\epsilon$ is a transport-cost budget. If $c(z,z')=d(z,z')^p$, then
$\mathcal T_c=W_p^p$ and the ordinary $W_p$ radius is
$\epsilon^{1/p}$. For label-preserving shifts, use infinite transport cost when
labels differ and maximize only over the input coordinates.

For any transport penalty $\lambda\ge0$, weak duality gives [1]

$$
R_\epsilon^W\le\lambda\epsilon+\frac1N\sum_{i=1}^N\phi_i,
\qquad
\phi_i=\sup_z\{\ell_{\widehat\theta}(z)-\lambda c(z,z_i)\}.
$$

The supremum must cover the support allowed by the target ambiguity set.
An attack restricted to a smaller per-example ball does not upper-bound this
supremum without an additional argument. In particular, a per-example attack
radius is not interchangeable with a finite-order Wasserstein average budget.
In `WNNDRO` [4], the code's `epsilon` in `norm_ball` mode is such a
per-example attack radius, whereas $\epsilon$ here is a transport-cost
budget. In `transport_penalty` mode, `gamma` is the coefficient $\lambda$
for squared Euclidean cost. Either attack supplies candidate inputs, but a
certificate still needs valid global inner error bounds for the specified
support and cost.

### Primal-dual certificate

Suppose $\lambda>0$, and an attack returns $\widetilde z_i$. Define
$g_i(z)=\ell_{\widehat\theta}(z)-\lambda c(z,z_i)$. Assume a certified bound

$$
0\le\phi_i-g_i(\widetilde z_i)\le e_i
$$

is available. For example, an independently established upper bound
$u_i\ge\phi_i$ gives $e_i=u_i-g_i(\widetilde z_i)$. Put

$$
\widehat A=\frac1N\sum_i\ell_{\widehat\theta}(\widetilde z_i),
\qquad
\widehat c=\frac1N\sum_i c(\widetilde z_i,z_i),
\qquad
\bar e=\frac1N\sum_i e_i.
$$

Substitution into weak duality yields the certificate

$$
{R_\epsilon^W\le U_W(\epsilon)
:=\widehat A+\lambda(\epsilon-\widehat c)+\bar e.}
$$

This derivation requires no global optimality of $\widehat\theta$ or
$\lambda$. The attack coupling has cost $\widehat c$, so
$\widehat A$ is also a feasible lower bound at any budget
$\epsilon\ge\widehat c$.
It need not be a lower bound at a smaller budget.

### Radius certificate

Take the user-specified transport-cost budget $\epsilon$ and suppose the
attack coupling is feasible at that budget, $\widehat c\le\epsilon$. With
$\lambda>0$ and certified $e_i$, define the verified upper-minus-lower gap

$$
\delta_W=U_W(\epsilon)-\widehat A
=\lambda(\epsilon-\widehat c)+\bar e\ge0.
$$

The largest budget at which this calculation certifies the **observed
adversarial loss** $\widehat A$ as an upper bound is

$$
\epsilon_{\mathrm{cert}}^W
=\epsilon-\frac{\delta_W}{\lambda}
=\widehat c-\frac{\bar e}{\lambda}\le\epsilon,\qquad
R_{\epsilon_{\mathrm{cert}}^W}^W\le\widehat A.
$$

Use this formula only if $\epsilon_{\mathrm{cert}}^W\ge0$; otherwise it
does not certify $\widehat A$ at any nonnegative budget. In particular, a
prespecified target $\epsilon_0\le\epsilon$ is covered at this loss level
exactly when $\widehat c\ge\epsilon_0+\bar e/\lambda$. The realized cost
and error bound, rather than the nominal training radius, determine this
margin. Even when the condition fails, $U_W(\epsilon_0)$ remains a valid
upper bound. If $c=d^p$, the corresponding ordinary $W_p$ radius is
$(\epsilon_{\mathrm{cert}}^W)^{1/p}$.

## 3. $\chi^2$-DRO

### Formulation

For the neural reweighting convention used in [2], let
$u_N=(1/N,\ldots,1/N)$ and

$$
D_{\chi^2}(q,u_N)
=\frac{1}{2N}\sum_i(Nq_i-1)^2
=\frac N2\sum_iq_i^2-\frac12.
$$

The empirical robust loss is

$$
R_\epsilon^{\chi^2}
=\max_{q\in\Delta_N:\,D_{\chi^2}(q,u_N)\le\epsilon}
\sum_iq_iv_i,
\qquad \epsilon\ge0,
$$

where $\Delta_N=\{q\ge0:\sum_iq_i=1\}$. This is a convex optimization
problem in the weights for any fixed network, including a nonconvex neural
architecture. If another convention omits the factor $1/2$ in the divergence,
rescale its radius before using the formulas below.

### Primal-dual certificate

For any scalar $\eta$, define

$$
S_2(\eta)=\sqrt{\frac1N\sum_i(v_i-\eta)_+^2},
\qquad
U_{\chi^2}(\epsilon;\eta)=\eta+\sqrt{1+2\epsilon}\,S_2(\eta).
$$

Every feasible $q$ satisfies $N\sum_iq_i^2\le1+2\epsilon$. Hence

$$
\sum_iq_iv_i
\le\eta+\sum_iq_i(v_i-\eta)_+
\le\eta+\sqrt{N\sum_iq_i^2}\,S_2(\eta)
\le U_{\chi^2}(\epsilon;\eta).
$$

The middle inequality is Cauchy-Schwarz. A verified feasible candidate
$\widetilde q$ therefore gives

$$
{
L_{\chi^2}:=\sum_i\widetilde q_iv_i
\le R_\epsilon^{\chi^2}
\le U_{\chi^2}(\epsilon;\eta).
}
$$

Optimizing the one-dimensional convex upper bound tightens it, but any
$\eta$ already gives a valid upper bound. The difference $U-L$ measures
the uncertainty in evaluating this fitted network's robust loss, not the
suboptimality of the network weights. A bisection residual is not itself
this objective gap.

In the neural implementation [3,4], `size` is $\epsilon$, while `reg`
multiplies a separate $\chi^2$ penalty on the adversarial weights. If that
penalty has coefficient $\tau\ge0$, a certified upper bound on the
regularized inner maximum needs a correction of at most $\tau\epsilon$
to upper-bound the unregularized $R_\epsilon^{\chi^2}$, because
$D_{\chi^2}\le\epsilon$ on the feasible set. Include any numerical error
in that bound. The displayed $L_{\chi^2}$ and $U_{\chi^2}$ instead use
unregularized losses evaluated on the final network.

### Radius certificate

At the user-specified radius $\epsilon$, take a verified feasible
$\widetilde q$, its **unregularized** loss $L_{\chi^2}=\widetilde q^\top v$,
and any fixed $\eta$. Write

$$
\delta_{\chi^2}=U_{\chi^2}(\epsilon;\eta)-L_{\chi^2}\ge0.
$$

When $S_2(\eta)>0$ and
$\delta_{\chi^2}\le S_2(\eta)(\sqrt{1+2\epsilon}-1)$, the largest radius
certified by this fixed-$\eta$ bound at loss level $L_{\chi^2}$ is

$$
\epsilon_{\mathrm{cert}}^{\chi^2}
=\frac12\left[
\left(\sqrt{1+2\epsilon}
-\frac{\delta_{\chi^2}}{S_2(\eta)}\right)^2-1
\right]\in[0,\epsilon],\qquad
R_{\epsilon_{\mathrm{cert}}^{\chi^2}}^{\chi^2}
\le L_{\chi^2}.
$$

Indeed, this choice makes
$U_{\chi^2}(\epsilon_{\mathrm{cert}}^{\chi^2};\eta)=L_{\chi^2}$.
If the gap exceeds the stated limit, this fixed-$\eta$ bound cannot certify
$L_{\chi^2}$ even at radius zero; do not clip a negative result to zero.
When $S_2(\eta)=0$, the upper bound is the constant $\eta$: it certifies
$L_{\chi^2}$ through the input radius only if $L_{\chi^2}=\eta$.
When $\epsilon_{\mathrm{cert}}^{\chi^2}$ exists, a prespecified target
$\epsilon_0\le\epsilon$ is covered at this loss level if
$\epsilon_0\le\epsilon_{\mathrm{cert}}^{\chi^2}$. A different $\eta$ or
direct evaluation of $U_{\chi^2}$ at $\epsilon_0$ may give a useful bound
even when this inverse check fails.

## 4. CVaR-DRO

### Formulation

Here $\alpha\in(0,1]$ is the upper-tail fraction, following the neural
implementation in [2,3]:

$$
R_\alpha^{\mathrm{CVaR}}
=\max_{q\in\Delta_N:\,q_i\le1/(\alpha N)}q^\top v.
$$

Smaller $\alpha$ permits more concentrated weights and increases robustness.
At $\alpha=1$ this reduces to the empirical mean. CVaR describes a
likelihood-ratio ambiguity set, not a metric distance ball. Its parameter
should not be interpreted as a Wasserstein radius or a $\chi^2$ radius.

### Primal-dual certificate
For any $\eta$, put

$$
S_1(\eta)=\frac1N\sum_i(v_i-\eta)_+,
\qquad
U_{\mathrm{CVaR}}(\alpha;\eta)=\eta+\frac{S_1(\eta)}{\alpha}.
$$

Every feasible $q$ satisfies

$$
q^\top v
\le\eta+\sum_iq_i(v_i-\eta)_+
\le\eta+\frac1{\alpha N}\sum_i(v_i-\eta)_+.
$$

Consequently, a verified feasible $\widetilde q$ gives

$$
{
L_{\mathrm{CVaR}}:=\widetilde q^\top v
\le R_\alpha^{\mathrm{CVaR}}
\le U_{\mathrm{CVaR}}(\alpha;\eta).
}
$$

Sorting the losses gives the exact empirical optimum in real arithmetic:
allocate weight $1/(\alpha N)$ to the largest losses, with any remaining
fractional weight on the next loss. This also covers noninteger $\alpha N$;
if $\alpha N<1$, the optimum is the largest loss.

In the neural implementation [3,4], `size` is $\alpha$, while `reg`
multiplies a KL penalty on the adversarial weights. With coefficient
$\tau\ge0$, this penalty is at most $\tau\log(1/\alpha)$ because
$D_{\mathrm{KL}}(q,u_N)=\sum_iq_i\log(Nq_i)\le\log(1/\alpha)$ [2].
A certified upper bound on the regularized inner maximum needs this
correction, plus any numerical error, to upper-bound the unregularized
$R_\alpha^{\mathrm{CVaR}}$. The displayed $L_{\mathrm{CVaR}}$ and
$U_{\mathrm{CVaR}}$ instead use unregularized losses.

### Radius certificate

At the user-specified tail fraction $\alpha$, take a verified feasible
$\widetilde q$, its **unregularized** loss
$L_{\mathrm{CVaR}}=\widetilde q^\top v$, and any fixed $\eta$. Set

$$
\delta_{\mathrm{CVaR}}
=U_{\mathrm{CVaR}}(\alpha;\eta)-L_{\mathrm{CVaR}}\ge0.
$$

If $S_1(\eta)>0$ and
$\delta_{\mathrm{CVaR}}\le S_1(\eta)(1/\alpha-1)$, define

$$
\alpha_{\mathrm{cert}}
=\left(\frac1\alpha
-\frac{\delta_{\mathrm{CVaR}}}{S_1(\eta)}\right)^{-1}
=\frac{S_1(\eta)}{L_{\mathrm{CVaR}}-\eta}
\in[\alpha,1],\qquad
R_{\alpha_{\mathrm{cert}}}^{\mathrm{CVaR}}
\le L_{\mathrm{CVaR}}.
$$

A larger $\alpha_{\mathrm{cert}}$ means a smaller CVaR ambiguity set.
It is the smallest tail fraction certified by this fixed-$\eta$ bound at
the loss level $L_{\mathrm{CVaR}}$. This value makes
$U_{\mathrm{CVaR}}(\alpha_{\mathrm{cert}};\eta)=L_{\mathrm{CVaR}}$.
If the gap exceeds the stated limit, this fixed-$\eta$ bound cannot
certify $L_{\mathrm{CVaR}}$ at any $\alpha_{\mathrm{cert}}\le1$.
When $S_1(\eta)=0$, the upper bound is the constant $\eta$: it certifies
$L_{\mathrm{CVaR}}$ through the input tail fraction only if
$L_{\mathrm{CVaR}}=\eta$. When $\alpha_{\mathrm{cert}}$ exists, a
prespecified target $\alpha_0\ge\alpha$ is covered at this loss level if
$\alpha_0\ge\alpha_{\mathrm{cert}}$. Direct evaluation of
$U_{\mathrm{CVaR}}(\alpha_0;\eta)$ remains available in every case.

## 5. Examples
For each of the Wasserstein, Chi2, and CVaR NN DRO, the interface `certify_radius(...)` is set for the post-training empirical calculation. It returns a dictionary containing `epsilon_cert` (or `alpha_cert` for CVaR), `loss_bound`, `input_upper_bound`, and `gap`.
It returns `None` when the supplied lower loss cannot be certified at a valid reduced parameter. The inputs must come from the **same frozen model and complete reference dataset**.

```python
import numpy as np
import torch
import torch.nn.functional as F
from dro.neural_model import WNNDRO, Chi2NNDRO, CVaRNNDRO

# Analytic Wasserstein example: f(x)=x, y=1, x=0, squared cost,
# and penalty lambda=2. The global penalized maximizer is x'=-1.
w = WNNDRO(input_dim=1, num_classes=1, task_type="regression",
           model_type="linear", robust_objective="transport_penalty",
           gamma=2.0, device=torch.device("cpu"))
with torch.no_grad():
    w.model.layer1.weight.fill_(1.0)
    w.model.layer1.bias.zero_()
    x, y, x_adv = torch.tensor([[0.0]]), torch.tensor([1.0]), torch.tensor([[-1.0]])
    adv_losses = F.mse_loss(w.model(x_adv).reshape(-1), y, reduction="none")
    costs = (x_adv - x).flatten(1).square().sum(dim=1)
# Here g(x')=(x'-1)^2-2(x')^2 has global maximum at x'=-1,
# so the verified inner error bound is zero.
w_cert = w.certify_radius(1.2, adv_losses, costs, certified_inner_errors=[0.0])
print(w_cert["epsilon_cert"], w_cert["loss_bound"])  # 1.0, 4.0

# Illustrative full-data losses and feasible adversarial weights for a
# frozen chi-square or CVaR network. Replace these with values from your
# final model and complete reference dataset.
losses = np.array([0.0, 1.0])
weights = np.array([0.25, 0.75])
chi2 = Chi2NNDRO(input_dim=1, num_classes=2, size=0.5)
chi2_cert = chi2.certify_radius(losses, weights, eta=0.0)
print(chi2_cert["epsilon_cert"])  # 0.0625

cvar = CVaRNNDRO(input_dim=1, num_classes=2, size=0.5)
cvar_cert = cvar.certify_radius(losses, weights, eta=0.0)
print(cvar_cert["alpha_cert"])  # 0.666666...
```

For a general `WNNDRO` run, a finite number of attack steps does not
establish `certified_inner_errors`. Supply those bounds only after proving
them for the chosen network, cost, and support. For $\chi^2$ and CVaR, we verify the adversarial weights and evaluate unregularized per-example losses on the full reference dataset, even if training used `reg`.

## Reference

* [1] Aman Sinha, Hongseok Namkoong, Riccardo Volpi, and John Duchi.
  [Certifying Some Distributional Robustness with Principled Adversarial
  Training](https://arxiv.org/abs/1710.10571). ICLR 2018; extended manuscript
  2020. See the Wasserstein dual bound and smooth penalized-adversary analysis.
* [2] Daniel Levy, Yair Carmon, John C. Duchi, and Aaron Sidford.
  [Large-Scale Methods for Distributionally Robust
  Optimization](https://papers.nips.cc/paper/2020/hash/64986d86a17424eeac96b08a6d519059-Abstract.html).
  NeurIPS 2020. See Proposition 1 for batch bias, Claim 1 for CVaR
  regularization error, and Appendix A for dual representations.
* [3] Daniel Levy and collaborators.
  [fast-dro: robust_losses.py](https://github.com/daniellevy/fast-dro/blob/main/robust_losses.py).
  Reference implementation of CVaR and $\chi^2$ robust losses, their
  regularizers, and scalar dual computations.
* [4] `dro` implementation:
  [wdro_nn.py](https://github.com/namkoong-lab/dro/blob/main/src/dro/neural_model/wdro_nn.py),
  [fdro_utils.py](https://github.com/namkoong-lab/dro/blob/main/src/dro/neural_model/fdro_utils.py),
  and [base_nn.py](https://github.com/namkoong-lab/dro/blob/main/src/dro/neural_model/base_nn.py).
  Parameter conventions refer to the source inspected on October 16, 2026.

The smaller-set formulas above follow directly from the displayed upper
bounds. They are not theorems attributed verbatim to these references.
