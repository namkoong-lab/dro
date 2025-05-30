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

### $f$-DRO
In KLDRO, Chi2DRO, CVaRDRO, TVDRO (and corresponding BayesianDRO), we only need to rewrite ``_loss`` and ``_cvx_loss``.

### Wasserstein DRO
To adjust Wasserstein DRO, besides modifying the `_cvx_loss` (and ``_loss``) functions, we also need to modify the `_penalization` function to adjust the regularization component, where the regularization component denotes the additional part besides the empirical objective in the Wasserstein DRO objective after the problem reformulation. 

### Remark
We remark that for more complicated losses, e.g., losses with a mixture of distances, we have not implemented the personalize loss yet.

Note that we have not implemented the personanlized constraint module yet $(e.g., for $\theta$). Stay tuned for that.


## 2. NN-Based DRO Methods

### 2.1 Personalized Loss 

For `f-DRO` and `WDRO` methods, our package supports personalized loss functions.

#### $f$-DRO
If the user would like to integrate his/her own loss functions, please 
1. Create a new `RobustLoss` instance (from `fdro.utils.py`), and re-write the `self._compute_individual_loss()` function to user-specified forms.
2. Create a new `Chi2NNDRO` or `CVaRNNDRO` instance (from `fdro_nn.py`), and re-write the `self._criterion()` function with the newly-modified `RobustLoss` instance above.


#### WDRO
When personalizing the loss function for WDRO, please:
1. Create a new `WNNDRO` instance (from `wdro_nn.py`).
2. Re-write the `self._loss()` function.


### 2.2 Personalized Model Architecture
Users could pass their own model via `self.update()` function. Note that the personalized model must be written via `PyTorch` and is a sub-class of `torch.nn.Module`.


## 3. Tree-Based DRO Methods

For tree-based DRO methods, users could simply rewrite the `self.loss()` function to change loss functions. To change  the DRO type, one need to adjust ``self._kl_dro_loss()`` (or  ``self._cvar_dro_loss()``) if their base model is ``KLDRO_XX`` (or ``CVaRDRO_XX``) respectively. 
