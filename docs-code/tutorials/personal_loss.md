# Personalization

In general, when rewritting the function for a new class, one way is to apply the following protocol code:
```python
from types import MethodType
def _loss(self, X, y):
    xxx
def _cvx_loss(self, X, y, theta, b, design_matrix=None):
    features = X if design_matrix is None else design_matrix
    # Construct the custom loss using features @ theta + b.
    xxx

model = XXDRO(...)
model._loss = MethodType(_loss, model)
model._cvx_loss = MethodType(_cvx_loss, model)
```
if we want to modify the ``self._cvx_loss`` and ``_loss`` functions in the model class.
Wasserstein DRO passes a precomputed `design_matrix` to `_cvx_loss`; use it when
provided so the custom loss uses the same linear or kernel coordinates as fitting.

## 1. Linear (Convex) DRO Methods
For linear DRO models formulated in CVXPY, customize both ``_loss`` and
``_cvx_loss`` for the chosen loss function.

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
Personalized loss hooks for Hybrid-distance DRO formulations have not yet been implemented.

Note that we have not implemented the personalized constraint module yet $(e.g., for $\theta$). Stay tuned for that.


## 2. NN-Based DRO Methods

For post-training bounds on the robust loss of a fitted neural model, see the
{doc}`NN-DRO robustness certificates guide <neural_certificates>`.

### 2.1 Personalized loss

For example, a decision-aware regression model can use the unbalanced L1 loss

$$
c_{\mathrm{under}}(y-\hat y)^+
+ c_{\mathrm{over}}(\hat y-y)^+,
$$

where the coefficients encode the different costs of under-predicting and
over-predicting. The {doc}`personalized-loss notebook <../api/notebooks/personalize_loss_tutorial>`
shows the corresponding per-sample loss for neural $f$-DRO and Wasserstein DRO.

#### Neural $f$-DRO

For `Chi2NNDRO` and `CVaRNNDRO`, subclass `RobustLoss` from `fdro_utils.py` and
override `_compute_individual_loss()` to return one loss per sample. Bind an
instance to the model by overriding `_criterion(outputs, labels)`; it must
return the scalar robust objective. The notebook demonstrates this for both
chi-squared and CVaR geometry.

#### Neural Wasserstein DRO

For `WNNDRO`, override `_loss(outputs, labels)` to return one loss per sample.
With `robust_objective='transport_penalty'`, the model uses this loss in both
the inner perturbation search and the outer training objective. With
`robust_objective='norm_ball'` and positive `epsilon`, Torchattacks generates
perturbations using its classification cross-entropy loss; `_loss` changes the
outer objective but does not change that attack objective. A custom norm-ball
inner loss requires a matching custom attack. The Torchattacks path also
expects classification labels, so use the transport-penalty path for the
notebook's continuous regression targets when perturbations are enabled.

#### HR-DRO

`HRNNDRO` currently computes cross-entropy or MSE inside `_criterion` and
uses Torchattacks when `epsilon > 0`. Replacing that loss requires a matching
change to the per-sample losses passed to its CVXPY weight problem and, if the
attack objective should change, a custom attack. Its training controls below
do not change the loss or attack objective. The example in the next section
replaces the architecture, per-sample loss, and training attack.

### 2.2 Personalized model architecture

Pass a custom `torch.nn.Module` through `update_user_mode(...)` before fitting.
For HR-DRO and WNNDRO norm-ball mode, this also binds the adversarial attack to
the replacement network.
The `update(config)` methods of neural $f$-DRO rebuild a preset network; they
are not the custom-network interface. Create a PyTorch optimizer only **after**
the final network has been installed, so it references that network's
parameters.

The following example replaces the default MLP with a small residual network.
Its `forward` method must accept a batch of two-feature inputs and return two
logits per sample for binary classification.

```python
import torch
from torch import nn
from dro.neural_model.fdro_nn import Chi2NNDRO

class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = nn.Linear(2, 2)
        self.hidden = nn.Sequential(
            nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.skip(x) + self.hidden(x)

torch.manual_seed(7)
positions = torch.linspace(0.05, 0.95, steps=40)
X = torch.stack((positions, 1 - positions), dim=1)
y = (positions >= 0.5).long()

device = torch.device('cpu')
model = Chi2NNDRO(input_dim=2, num_classes=2, device=device)
model.update_user_mode(
    input_dim=2, num_classes=2, model=ResidualClassifier(),
    task_type='classification', device=device,
)
metrics = model.fit(X, y, lr=0.05, epochs=10, batch_size=8, verbose=False)
predictions = model.predict(X[:5])
```

Replace `ResidualClassifier` with any compatible `nn.Module`. For a custom
regression model, return one output per sample and set `task_type='regression'`
and `num_classes=1`.

For HR-DRO, the following code continues from the same `X`, `y`, and
`ResidualClassifier`. It doubles the loss for class 1. A custom projected
gradient attack maximizes that same loss within an $\ell_\infty$ ball of radius
`epsilon`, and the attacked per-sample losses are passed to the robust
weighting problem. The raw-gradient step lets the class-specific cost affect
the perturbation size; projection enforces the radius. Inputs are clipped to
`[0, 1]` as in the notebook's toy data.

```python
import cvxpy as cp
import torch.nn.functional as F
from dro.neural_model.hrdro_nn import HRNNDRO

def cost_sensitive_losses(outputs, labels):
    return (1.0 + labels.float()) * F.cross_entropy(
        outputs, labels, reduction='none'
    )

class CostSensitivePGD:
    def __init__(self, model, epsilon, step_size, steps):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.steps = steps

    def __call__(self, inputs, labels):
        original = inputs.detach()
        perturbed = original.clone()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                for _ in range(self.steps):
                    perturbed.requires_grad_(True)
                    loss = cost_sensitive_losses(
                        self.model(perturbed), labels
                    ).sum()
                    gradient = torch.autograd.grad(loss, perturbed)[0]
                    perturbed = perturbed.detach() + self.step_size * gradient
                    perturbed = torch.maximum(
                        torch.minimum(perturbed, original + self.epsilon),
                        original - self.epsilon,
                    ).clamp(0, 1).detach()
        finally:
            self.model.train(was_training)
        return perturbed

class CostSensitiveHRNNDRO(HRNNDRO):
    def _init_adversarial_attack(self):
        params = self.adversarial_params
        if params['method'] != 'PGD' or params['norm'] != 'l-inf':
            raise ValueError('This example supports l-inf PGD only')
        self.attack = CostSensitivePGD(
            self.model, self.epsilon, params['step_size'], params['steps']
        )
        self._tabular_attack = None

    def _criterion(self, outputs, labels):
        adversarial_inputs = self.attack(self.current_inputs, labels)
        losses = cost_sensitive_losses(self.model(adversarial_inputs), labels)
        batch_size = labels.shape[0]
        if batch_size != self.train_batch_size:
            self.train_batch_size = batch_size
            self._init_optimization_problem()

        self.loss_param.value = losses.detach().cpu().numpy()
        self.worst_case.value = losses.max().detach().item()
        try:
            self.problem.solve(solver=cp.CLARABEL, verbose=False)
        except cp.SolverError:
            self.loss_param.value += self.numerical_eps
            self.problem.solve(solver=cp.SCS, verbose=False)

        weights = torch.as_tensor(
            self.p.value, dtype=losses.dtype, device=losses.device
        )
        return (weights[:batch_size] * losses).sum() + weights[-1] * losses.max()

hr_model = CostSensitiveHRNNDRO(
    input_dim=2, num_classes=2, alpha=0.1, r=0.01,
    epsilon=0.05, train_batch_size=8, device=device,
    adversarial_params={
        'method': 'PGD', 'norm': 'l-inf', 'step_size': 0.02, 'steps': 3,
    },
)
hr_model.update_user_mode(
    input_dim=2, num_classes=2, model=ResidualClassifier(),
    task_type='classification', device=device,
)
assert hr_model.attack.model is hr_model.model
hr_metrics = hr_model.fit(
    X, y, lr=0.05, epochs=3, batch_size=8, verbose=False
)
adversarial_inputs = hr_model.attack(X[:5], y[:5])
```

The attack uses the true labels for each training batch. For labeled
adversarial evaluation, call `hr_model.attack(X_eval, y_eval)` with the true
labels; the inherited `HRNNDRO.predict()` instead supplies dummy labels. This
example overrides HR-DRO internals because the class does not currently expose
custom-loss or custom-attack hooks. Check those internals when upgrading the
package.

### 2.3 Data loaders, optimizers, and schedulers

`Chi2NNDRO`, `CVaRNNDRO`, `HRNNDRO`, and `WNNDRO` accept these optional controls
through `fit(...)`. Supply both `train_loader` and `val_loader` as loaders of
`(inputs, labels)` batches, and omit `X` and `y` on this path. Pass a PyTorch
optimizer built from `model.model.parameters()`; a scheduler also requires an
explicit optimizer. `scheduler_step='epoch'` steps it after each epoch, while
`scheduler_step='batch'` steps it after each training batch. Without external
loaders, `fit(X, y, ...)` retains the existing internal split and Adam default.

```python
from torch.utils.data import DataLoader, TensorDataset
import torch
from dro.neural_model.fdro_nn import Chi2NNDRO

# X_train/X_val are float tensors; y_train/y_val are long class labels.
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

device = torch.device('cpu')
model = Chi2NNDRO(input_dim=X_train.shape[1], num_classes=2, device=device)
custom_network = torch.nn.Linear(X_train.shape[1], 2)
model.update_user_mode(
    input_dim=X_train.shape[1], num_classes=2, model=custom_network,
    task_type='classification', device=device,
)
optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
metrics = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    scheduler_step='epoch',
    epochs=10,
)
```

The same fitting arguments apply to `CVaRNNDRO` and `HRNNDRO` instances.
For HR-DRO, set `train_batch_size` to the loader's usual batch size when
constructing the model. A shorter final batch is handled by rebuilding its
CVXPY problem for that batch size. Torchattacks clips adversarial inputs to
`[0, 1]`, so prepare inputs accordingly when `epsilon > 0`. The notebook
includes complete chi-squared, CVaR, and HR-DRO fitting examples.


## 3. Tree-Based DRO Methods

For tree-based DRO methods, `loss(preds, labels)` supplies the per-example
losses used for robust sample weighting and the reported objective. When a
personalized loss also changes its prediction derivatives, update the booster
objective's gradients and Hessians as well (for example, `_kl_dro_loss()` in
`KLDRO_XGB`).

See the {doc}`personalized-loss notebook <../api/notebooks/personalize_loss_tutorial>`
for end-to-end linear, neural, and cost-sensitive XGBoost examples.
