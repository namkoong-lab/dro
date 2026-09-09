# NN-DRO

For $\chi^2$-DRO, CVaR-DRO, Wasserstein DRO, Holistic Robust DRO, Sinkhorn DRO, and Group DRO, we implement their `neural-network` version, where the backbone model is (by default) MLP.

We support the following NN architectures:
- `linear`: Linear Model
- `mlp`: MLP
- `alexnet`: AlexNet
- `resnet`: ResNet-18

Furthermore, users can use their own model architectures via the `update()` function.
Implementation details are as follows:

## 1. $\chi^2$-DRO and CVaR-DRO
We follow [1] to implement these two $f$-DROs. Our code is largely based on https://github.com/daniellevy/fast-dro.

### Hyperparameters
- size: the size of the uncertainty set
- reg: the strength of the $l_2$-regularization


[1] Large-Scale Methods for Distributionally Robust Optimization. Daniel Levy, Yair Carmon, John Duchi, and Aaron Sidford. NeurIPS 2020.

See the {doc}`general NN DRO notebook <../api/notebooks/neural_dro_tutorial>`
for companion neural $f$-DRO material.


## 2. Wasserstein DRO
For WDRO on neural networks, the main challenge is the perturbation. In our package, the perturbation step is implemented via recent advanced adversarial attack techniques which slightly perturb the data points to increase the prediction error. And the general procedure follows [2].


### Hyperparameters
- epsilon: Dual parameter. Coefficient of the penalty during adversarial training.
- adversarial_steps: Num of steps of the inner adversarial attacking.
- adversarial_step_size: Learning rate of the inner adversarial attacking.


[2] Certifying some distributional robustness with principled adversarial training. Aman Sinha, Hongseok Namkoong, Riccardo Volpi, and John Duchi. ICLR 2018.

See the {doc}`general NN DRO notebook <../api/notebooks/neural_dro_tutorial>`
for companion neural Wasserstein DRO material.


## 3. Holistic Robust DRO
We follow [3], and our code is largely based on https://github.com/RyanLucas3/HR_Neural_Networks.

### Hyperparameters
* $r$: Robustness parameter for the KL-DRO, denoted as ``r`` in the model config.
* $\alpha$: Robustness parameter for the Levy-Prokhorov metric DRO, denoted as ``alpha`` in the model config.
* $\epsilon$: Robustness parameter for the model noise (perturbed ball size), denoted as ``epsilon`` in the model config.


[3] Certified Robust Neural Networks: Generalization and Corruption Resistance. Amine Bennouna, Ryan Lucas, and Bart Van Parys. ICML 2023.

See the {doc}`general NN DRO notebook <../api/notebooks/neural_dro_tutorial>`
for companion Holistic Robust DRO material.


## 4. Sinkhorn DRO
Sinkhorn DRO uses an entropic-regularized Wasserstein distance and Gaussian input perturbations to construct a smooth robust loss. The implementation supports standard stochastic gradient (SG), multilevel Monte Carlo (MLMC), and randomized truncated MLMC (RTMLMC) optimization [4].

### Hyperparameters
- `reg_param`: Entropic regularization strength $\varepsilon > 0$. It also sets the perturbation variance. Default: `1e-3`.
- `lambda_param`: Positive loss-scaling factor $\lambda$. Together, `lambda_param * reg_param` controls the scale of the log-sum-exp robust loss. Default: `1e2`.
- `k_sample_max`: Maximum Monte Carlo level. SG uses $2^{\mathtt{k\_sample\_max}}$ perturbation samples, so increasing this value improves the sampling approximation but increases computation exponentially. Default: `5`.
- `optimization_type`: Stochastic estimator used during training. Choose `"SG"`, `"MLMC"`, or `"RTMLMC"`. Default: `"SG"`.

The usual neural-model arguments are also available: `task_type` (`"classification"` or `"regression"`), `model_type` (`"mlp"`, `"linear"`, `"resnet"`, or `"alexnet"`), and `device`.

### Hyperparameter setup
The following classification setup uses all of the Sinkhorn defaults explicitly. `X` may be a NumPy array or PyTorch tensor, and `y` should contain class indices.

```python
from dro.neural_model import SinkhornNNDRO

model = SinkhornNNDRO(
    input_dim=X.shape[1],
    num_classes=2,
    task_type="classification",
    model_type="mlp",
    reg_param=1e-3,
    lambda_param=1e2,
    k_sample_max=5,
    optimization_type="SG",
)

metrics = model.fit(
    X,
    y,
    train_ratio=0.8,
    lr=1e-3,
    batch_size=32,
    epochs=100,
)
```

For faster initial experiments, reduce `k_sample_max` (for example, to `3`) and increase it for the final run after selecting the other hyperparameters. The `lr`, `batch_size`, `epochs`, and `train_ratio` training settings are passed to `fit()`, rather than to the constructor.

An existing Sinkhorn model can be reconfigured with `update()`. Note that the update keys are `reg` and `lambda`, whereas the corresponding constructor arguments are `reg_param` and `lambda_param`.

```python
model.update({
    "reg": 1e-2,
    "lambda": 50.0,
    "k_sample_max": 4,
    "optimization_type": "MLMC",
})
```

See the {doc}`Sinkhorn NN notebook <../api/notebooks/sinkhorn-nn>` for classification, regression, custom-model, and optimizer examples.


[4] [Sinkhorn Distributionally Robust Optimization](https://arxiv.org/abs/2109.11926). Jie Wang, Rui Gao, and Yao Xie.


## 5. Group DRO

Group DRO targets uniform performance across a finite set of observed groups.
For group-average losses $L_g(\theta)$, its objective is

$$
\min_\theta\max_{g\in\mathcal{G}}L_g(\theta).
$$

The neural implementation follows the stochastic exponentiated-gradient
method in [5]. It maintains adversarial group probabilities $q_g$ and updates
them after each mini-batch:

$$
q_g \leftarrow
\frac{q_g\exp(\eta_q L_g)}
{\sum_j q_j\exp(\eta_q L_j)}.
$$

The network minimizes the resulting weighted group loss. Groups with larger
loss receive increasing adversarial probability, so training focuses on the
current worst-performing groups. If a mini-batch omits a category, the loss
renormalizes the current weights over the groups present in that batch; no
gradient is fabricated for an absent group.

### Group feature and supported models

`group_idx` is the zero-based index of a finite categorical column in the
two-dimensional input matrix `X`. The category column remains an input to the
network. `GroupNNDRO` therefore supports the tabular `"linear"` and `"mlp"`
architectures. `"resnet"` and `"alexnet"` are not supported because an image
tensor does not provide one scalar feature column per sample.

Classification targets must be integer class indices from `0` through
`num_classes - 1`. Regression targets are numeric, and prediction returns the
raw one-dimensional network output.

### Hyperparameters

- `group_idx`: required column containing the group category.
- `step_size`: adversarial exponentiated-gradient step $\eta_q>0$. The default
  is `0.01`. Larger values move weight toward high-loss groups more quickly,
  while smaller values produce smoother, slower changes.
- `input_dim`: number of input columns, including the group feature.
- `num_classes`: output class count for classification; regression uses one
  output.
- `task_type`: `"classification"` or `"regression"`.
- `model_type`: `"mlp"` or `"linear"`.
- `device`: PyTorch device used for training.

The standard training arguments remain on `fit()`: `train_ratio`, `lr`,
`batch_size`, `epochs`, and `verbose`. This preserves the same call pattern as
the other neural DRO estimators.

```python
from dro.neural_model import GroupNNDRO

model = GroupNNDRO(
    input_dim=X.shape[1],
    num_classes=2,
    group_idx=2,
    task_type="classification",
    model_type="mlp",
    step_size=0.05,
)

metrics = model.fit(
    X,
    y,
    train_ratio=0.8,
    lr=1e-3,
    batch_size=32,
    epochs=100,
)
```

After fitting, `group_values_` gives the category order and `group_weights_`
contains the final adversarial probabilities in the same order.
`batch_group_losses_` contains the most recently observed mini-batch loss for
each group. Both `group_idx` and `step_size` can be changed before refitting:

```python
model.update({"group_idx": 1, "step_size": 0.02})
```

Regularization and early stopping are especially important for
over-parameterized networks: a model can drive every training-group loss near
zero while still generalizing poorly on a minority group. Monitor held-out
worst-group performance rather than only aggregate validation accuracy.

See the {doc}`Group DRO notebook <../api/notebooks/groupdro_tutorial>` for an
end-to-end linear and neural example.

[5] [Distributionally Robust Neural Networks for Group Shifts: On the
Importance of Regularization for Worst-Case
Generalization](https://arxiv.org/abs/1911.08731). Shiori Sagawa, Pang Wei Koh,
Tatsunori B. Hashimoto, and Percy Liang, ICLR 2020. The exponentiated
group-weight update is adapted from the authors' [reference
implementation](https://github.com/kohpangwei/group_DRO).
