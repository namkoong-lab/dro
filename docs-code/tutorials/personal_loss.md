# Personalization

In general, when rewritting the function for a new class, one way is to apply the following protocol code:
```python
from types import MethodType
def _loss(self, X, y):
    xxx
def _cvx_loss(self, X, y, theta, b):
    xxx

model = XXDRO(...)
model._loss = MethodType(_loss, model)
model._cvx_loss = MethodType(_cvx_loss, model)
```
if we want to modify the ``self._cvx_loss`` and ``_loss`` functions in the model class.

## 1. Linear (Exact) DRO Methods
In DRO models that are solved exactly, for each particular DRO type, we change ``_loss`` and ``_cvx_loss`` in each class.

As a general high-level example, consider a convex piecewise-affine loss of the
prediction residual

$$
z = \theta^\top x + b - y, \qquad
L(z) = \max_{j=1,\ldots,J}\{a_j z + c_j\}.
$$

The arrays of slopes ``a`` and offsets ``c`` specify the loss. This family is
broader than a single newsvendor loss: absolute, pinball, and
epsilon-insensitive losses are all special cases. The
{doc}`personalized-loss notebook <../api/notebooks/personalize_loss_tutorial>`
implements this parameterization for both $f$-DRO and Wasserstein DRO.

### $f$-DRO
In KLDRO, Chi2DRO, CVaRDRO, TVDRO (and corresponding BayesianDRO), the
ambiguity set reweights the per-sample losses. Therefore, implementing the
piecewise-affine example only requires overriding ``_loss`` and
``_cvx_loss``.

### Wasserstein DRO
For Wasserstein DRO, also override ``_penalization`` so that it matches the
Lipschitz modulus of the personalized loss. For the loss above and transport
cost

$$
\lVert A(x-x')\rVert_p + \kappa |y-y'|,
$$

the notebook uses

$$
\max_j |a_j|\,
\max\left\{\lVert A^{-1}\theta\rVert_q,\;1/\kappa\right\},
\qquad 1/p+1/q=1,
$$

with the second term omitted when labels cannot move
(``kappa='inf'``). This follows the tractable piecewise-affine regression
formulation in Theorem 4 of
[Regularization via Mass Transportation](https://jmlr.org/papers/v20/17-633.html).

### Remark
We remark that for more complicated losses, e.g., losses with a mixture of distances, we have not implemented the personalize loss yet.

Note that we have not implemented the personalized constraint module yet $(e.g., for $\theta$). Stay tuned for that.


## 2. NN-Based DRO Methods

### 2.1 Personalized Loss

For `f-DRO` and `WDRO` methods, our package supports personalized loss functions.

For example, a decision-aware regression model can use the unbalanced L1 loss

$$
c_{\mathrm{under}}(y-\hat y)^+
+ c_{\mathrm{over}}(\hat y-y)^+,
$$

where the two coefficients encode the different downstream costs of
under-predicting and over-predicting. The final section of the personalized-loss
notebook shows how to use this per-sample loss in both the neural $f$-DRO and
neural Wasserstein hooks.

#### $f$-DRO
To integrate a custom loss function:

1. Create a new `RobustLoss` instance (from `fdro_utils.py`), and re-write the `self._compute_individual_loss()` function to user-specified forms.
2. Create a new `Chi2NNDRO` or `CVaRNNDRO` instance (from `fdro_nn.py`), and re-write the `self._criterion()` function with the newly-modified `RobustLoss` instance above.


#### WDRO
When personalizing the loss function for WDRO, please:

1. Create a new `WNNDRO` instance (from `wdro_nn.py`).
2. Re-write the `self._loss()` function.


### 2.2 Personalized Model Architecture
Users could pass their own model via `self.update()` function. Note that the personalized model must be written via `PyTorch` and is a sub-class of `torch.nn.Module`.


## 3. Tree-Based DRO Methods

For tree-based DRO methods, users can rewrite the `self.loss()` function to change loss functions. To change the DRO type, adjust ``self._kl_dro_loss()`` (or ``self._cvar_dro_loss()``) if the base model is ``KLDRO_XX`` (or ``CVaRDRO_XX``), respectively.

See the {doc}`personalized-loss notebook <../api/notebooks/personalize_loss_tutorial>`
for end-to-end linear and neural customization examples.
