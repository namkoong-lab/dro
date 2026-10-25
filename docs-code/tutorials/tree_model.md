# Tree DRO

The `dro` package provides KL-DRO, $\chi^2$-DRO, and CVaR-DRO wrappers for
XGBoost and LightGBM. These methods fit tree ensembles with custom boosting
gradients that place more emphasis on observations with large losses. Tree
construction is non-convex, so fitting is approximate; `fit()` does not solve
the distributionally robust objective to global optimality.

## Formulation

For binary classification, let $\ell_i(f)$ be the log loss on training example
$i$, and let $n$ be the number of examples. The two boosting backends use the
same three DRO formulations:

### KL-DRO

The implemented KL objective is the entropic risk

$$
R_{\mathrm{KL},\varepsilon}(f)
=\frac{1}{\varepsilon}\log\left(
  \frac{1}{n}\sum_{i=1}^{n}e^{\varepsilon\ell_i(f)}
\right),\qquad \varepsilon>0.
$$

During boosting, the loss of example $i$ receives a weight proportional to
$e^{\varepsilon\ell_i}$. Equivalently, this objective uses a KL *penalty* on
adversarial sample weights; `eps` is an inverse temperature, not the radius of
a hard KL-divergence ball. Larger `eps` concentrates more weight on high-loss
examples. As `eps` approaches zero, the objective approaches mean log loss.

### $\chi^2$-DRO

This method uses a $\chi^2$ ball around the uniform empirical probabilities:

$$
R_{\chi^2,\varepsilon}(f)
=\max_{p\in\Delta_n}\sum_{i=1}^{n}p_i\ell_i(f)
\quad\text{subject to}\quad
n\sum_{i=1}^{n}\left(p_i-\frac{1}{n}\right)^2\leq\varepsilon,
$$

where $\Delta_n=\{p_i\geq0,\ \sum_i p_i=1\}$. The booster uses the resulting
adversarial weights $np_i$ to scale sample gradients. The implementation first
computes a closed-form candidate and solves a constrained problem when that
candidate has a negative weight. Larger `eps` permits a greater shift of
probability toward high-loss examples.

### CVaR-DRO

For CVaR, `eps` is a loss quantile level rather than a divergence radius. The
reported empirical objective is

$$
R_{\mathrm{CVaR},\varepsilon}(f)
=t+\frac{1}{n(1-\varepsilon)}
  \sum_{i=1}^{n}\bigl(\ell_i(f)-t\bigr)_+,
\qquad t=\operatorname{Quantile}_{\varepsilon}
  (\ell_1(f),\ldots,\ell_n(f)).
$$

Increasing `eps` focuses the reported objective on a smaller upper tail of
losses. During boosting, examples above the current quantile have their
gradients and Hessians multiplied by $1/(1-\varepsilon)$; examples at or below
it retain their ordinary gradients. This is a heuristic training rule, so the
boosting updates are not exact derivatives of the displayed CVaR objective.

## Objective and booster configuration

Configure the robust objective through the model class and `eps`, and pass
tree-construction settings to `update()`. The DRO wrapper supplies the custom
training objective during `fit()`.

### Robust objective

The model class selects KL-DRO, $\chi^2$-DRO, or CVaR-DRO and the XGBoost or
LightGBM backend. The formulations above define the corresponding objectives.

- `eps` controls the selected DRO formulation. The default is `0.1` for KL
  and $\chi^2$, and `0.2` for CVaR. KL and $\chi^2$ constructors require
  `eps > 0`; CVaR requires `0 < eps < 1`. Its meaning differs across the three
  formulations, so the same numerical value does not imply the same amount of
  robustness.
- `kind` selects `"classification"` or `"regression"`; the default is
  `"classification"`. The custom boosting gradients currently apply a sigmoid
  to raw scores and use binary-classification gradients in both modes. Use
  `"classification"` with labels in $\{0,1\}$ for the objectives described
  above. Although `"regression"` is accepted, its training updates do not
  implement the squared-loss DRO objective reported after fitting.

### Booster configuration

Pass a dictionary of XGBoost or LightGBM settings to `update()` before fitting:

- `max_depth` limits tree depth.
- `learning_rate` scales each boosting step.
- `num_boost_round` sets the number of boosting iterations and is required.

These settings control tree construction. The class and `eps` determine the
robust objective used to train the ensemble.

Choose `eps` using held-out data and the performance measure relevant to the
deployment setting. Tree depth, learning rate, and boosting rounds should be
tuned as well: a larger robustness parameter cannot compensate for a poorly
fitted or overfitted booster.

## Fitting and diagnostics

Select a class for both the DRO formulation and the boosting backend, then
call `update()` before `fit()`. For example:

```python
from dro.tree_model.xgb import KLDRO_XGB

model = KLDRO_XGB(eps=0.01, kind="classification")
model.update({
    "max_depth": 2,
    "learning_rate": 0.1,
    "num_boost_round": 50,
})
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy, f1 = model.score(X_test, y_test)
print(model.robust_obj)
```

The corresponding class names are `Chi2DRO_XGB`, `CVaRDRO_XGB`,
`KLDRO_LGBM`, `Chi2DRO_LGBM`, and `CVaRDRO_LGBM`. `robust_obj` is `None`
before fitting and is the selected robust objective evaluated on the training
losses afterward. It is a training diagnostic; compare it only when the data,
loss, and DRO formulation are held fixed. Use held-out data to estimate
predictive performance. `score()` returns accuracy and macro F1 for
classification. Classification uses raw booster margins with a threshold of
zero, equivalent to a sigmoid probability threshold of 0.5.

See the {doc}`tree DRO notebook <../api/notebooks/tree_dro_tutorial>` for
end-to-end XGBoost and LightGBM examples.
