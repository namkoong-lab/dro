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

[1] Large-Scale Methods for Distributionally Robust Optimization. Daniel Levy, Yair Carmon, John Duchi, and Aaron Sidford. 


## 2. Wasserstein DRO
For WDRO on neural networks, the main challenge is the perturbation. In our package, the perturbation step is implemented via recent advanced adversarial attack techniques which slightly perturb the data points to increase the prediction error. And the general procedure follows [2].



[2] Certifying some distributional robustness with principled adversarial training. Aman Sinha, Hongseok Namkoong, Riccardo Volpi, and John Duchi.


## 3. Holistic Robust DRO
We follow [3], and our code is largely based on https://github.com/RyanLucas3/HR_Neural_Networks.

[3] Certified Robust Neural Networks: Generalization and Corruption Resistance. Amine Bennouna, Ryan Lucas, and Bart Van Parys.