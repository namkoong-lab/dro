# Personalization

## 1. Linear DRO Methods

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

For tree-based DRO methods, users could simply rewrite the `self.loss()` function
