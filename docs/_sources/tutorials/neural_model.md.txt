# NN-DRO

For $\chi^2$-DRO, CVaR-DRO, Wasserstein DRO, and Holistic Robust DRO, we implement their `neural-network` version, where the backbone model is (by default) MLP.

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


[1] Large-Scale Methods for Distributionally Robust Optimization. Daniel Levy, Yair Carmon, John Duchi, and Aaron Sidford. 


## 2. Wasserstein DRO
For WDRO on neural networks, the main challenge is the perturbation. In our package, the perturbation step is implemented via recent advanced adversarial attack techniques which slightly perturb the data points to increase the prediction error. And the general procedure follows [2].


### Hyperparameters
- epsilon: Dual parameter. Coefficient of the penalty during adversarial training.
- adversarial_steps: Num of steps of the inner adversarial attacking.
- adversarial_step_size: Learning rate of the inner adversarial attacking.


[2] Certifying some distributional robustness with principled adversarial training. Aman Sinha, Hongseok Namkoong, Riccardo Volpi, and John Duchi.


## 3. Holistic Robust DRO
We follow [3], and our code is largely based on https://github.com/RyanLucas3/HR_Neural_Networks.

### Hyperparameters
* $r$: Robustness parameter for the KL-DRO, denoted as ``r`` in the model config.
* $\alpha$: Robustness parameter for the Levy-Prokhorov metric DRO, denoted as ``alpha`` in the model config.
* $\epsilon$: Robustness parameter for the model noise (perturbed ball size), denoted as ``epsilon`` in the model config.


[3] Certified Robust Neural Networks: Generalization and Corruption Resistance. Amine Bennouna, Ryan Lucas, and Bart Van Parys.