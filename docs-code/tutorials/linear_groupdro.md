# Group DRO for Linear Models

Group distributionally robust optimization (Group DRO) is useful when the
training data is divided into known groups and average performance can hide a
poorly performing subgroup. Instead of minimizing the average loss over all
samples, Group DRO minimizes the largest empirical mean loss among the observed
groups.

## Formulation

Let $g_i \in \mathcal{G}$ be the group category for sample $i$, and let $n_g$
be the number of training samples in group $g$. The linear implementation
solves

$$
\min_{\theta,b}\max_{g\in\mathcal{G}}
\frac{1}{n_g}\sum_{i:g_i=g}
\ell(\theta,b;x_i,y_i).
$$

The maximum is represented by an epigraph variable $t$. For every observed
group, the CVXPY problem adds the constraint

$$
\frac{1}{n_g}\sum_{i:g_i=g}\ell_i(\theta,b)\leq t,
$$

and minimizes $t$. This is an exact convex formulation for each loss supported
by `BaseLinearDRO`.

## Defining groups

Set `group_idx` to the zero-based column index in `X` that contains group
membership. The column must contain finite numeric categories. Categories do
not need to be consecutive: values such as `0`, `2`, and `10` define three
groups.

The group column remains in the design matrix during fitting and prediction.
Consequently, the fitted model may use that value as a predictive feature. This
keeps the same `fit(X, y)` interface as the other linear estimators, but it is an
important modeling choice when the group column represents a protected or
sensitive attribute.

Every category present in the training data contributes one group-average
constraint. A small group therefore has the same opportunity to determine the
objective as a large group; it is not down-weighted by its sample count.

## Supported losses

Choose the loss with `model_type`:

- `"svm"`: hinge loss for binary labels in $\{-1,+1\}$.
- `"logistic"`: logistic loss for binary labels in $\{-1,+1\}$.
- `"ols"`: squared loss for regression.
- `"lad"`: absolute loss for regression.

Linear and kernelized predictors are supported through the common
`BaseLinearDRO` machinery. For a kernelized model, call `update_kernel()` before
`fit()` just as with the other linear DRO estimators.

## Hyperparameters

`GroupDRO` accepts the following constructor arguments:

- `input_dim`: number of columns in `X`, including the group column.
- `group_idx`: required index of the finite categorical group feature.
- `model_type`: one of `"svm"`, `"logistic"`, `"ols"`, or `"lad"`.
- `fit_intercept`: whether to learn an intercept; the default is `True`.
- `solver`: an installed CVXPY solver. The package default is `"MOSEK"`;
  `"CLARABEL"` or `"SCS"` can be used when available.
- `kernel`: kernel name passed to `BaseLinearDRO`; the default is `"linear"`.

The group feature can be changed before refitting with
`model.update({"group_idx": new_index})`.

## Fitting and diagnostics

The public call follows the other linear models:

```python
from dro.linear_model import GroupDRO

model = GroupDRO(
    input_dim=X.shape[1],
    group_idx=2,
    model_type="svm",
    solver="CLARABEL",
)
result = model.fit(X, y)
predictions = model.predict(X)
accuracy, f1 = model.score(X, y)
```

In addition to `theta` and `b`, `fit()` returns:

- `group_values`: sorted categories found in `X[:, group_idx]`.
- `group_losses`: fitted empirical losses aligned with `group_values`.
- `robust_loss`: the largest value in `group_losses`.

The same values remain available as `group_values_`, `group_losses_`, and
`robust_loss_` on the fitted estimator.

## Practical guidance

- Ensure every group has enough observations for a meaningful empirical loss.
  Group DRO cannot correct an unreliable group estimate caused by extremely
  sparse data.
- Select hyperparameters using worst-group validation performance when that is
  the deployment goal. Average validation accuracy can favor a different
  model.
- Scale continuous features before fitting, especially for logistic loss or
  when group sizes are highly imbalanced.
- Solver status and tolerances matter. The implementation accepts both optimal
  and optimal-inaccurate CVXPY solutions and reports the losses evaluated at
  the returned parameters.

See the {doc}`Group DRO notebook <../api/notebooks/groupdro_tutorial>` for a
complete synthetic classification example using both the linear and neural
interfaces.

## Reference

Sagawa, Shiori, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang.
"Distributionally Robust Neural Networks for Group Shifts: On the Importance
of Regularization for Worst-Case Generalization." ICLR, 2020.
