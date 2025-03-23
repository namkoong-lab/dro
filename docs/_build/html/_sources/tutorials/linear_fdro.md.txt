# f-divergence DRO

We describe the $f$-divergence DRO here.


## Standard (Generalized)
Models listed here can all be formulated with the distance being $d(P, Q) = E_{Q}[f(\frac{dP}{dQ})]$ where $f(\cdot)$ satisfies some properties [1]. We do not apply them due to numerical stability. Instead, we choose the following more commonly used $f$-divergences and compute their stable tractable reformulation.

### Formulation

For KL-DRO problem, we apply $f(x) = x \log x - (x - 1)$ and follow the reformulation in Theorem 4 of [2] to fit the model.


For $\chi^2$-DRO problem, we apply $f(x) = (x - 1)^2$ and follow the reformulation in Lemma 1 of [3] to fit the model. 

For TV-DRO problem, we apply $f(x) = |x - 1|$ and follow the reformulation in Theorem 1 of [4] to fit the model.

For the CVaR-DRO problem, we apply $f(x) = 0$ if $x \in [\frac{1}{\alpha}, \alpha]$ and $\infty$ otherwise (an augmented definition of the standard $f$-DRO problem). Here $\alpha$ denotes the worst-case ratio. We follow the reformulation in Theorem 2 of [5] to fit the model.


### Hyperparameters
Across all the above models except CVaR-DRO problem, the only model-specific hyperparameter is the ambiguity size ``eps``. $\epsilon \geq 0$ (and reduce to ERM when $\epsilon = 0$). Specifically for TV-DRO problem, since TV-Distance is bounded between [0, 1], the corresponding $\epsilon \in [0, 1]$. In CVaR-DRO problem, the model-specific hyperparameter is the worst-case ratio ``alpha``, that takes values in (0, 1] (and reduce to ERM when $\alpha = 0$). 

### Worst-case illustration
The above formulations are all based on joint perturbed probability. 

### Evaluation

ADD DATA-DRIVEN EVALUATION from [6].


## Partial DRO
The ambiguity set here are based on (joint) cannot be written as the standard (generalized) f-divergence DRO format. Instead, we directly use $Q(\alpha)$.

If we only consider the shifts in the marginal distribution $X$, 
$Q(\alpha) = \{Q_0: P_X = \alpha Q_0 + (1-\alpha) Q_1, \text{for some}~\alpha \geq \alpha_0~\text{and distribution}~Q_1~\text{and}~\mathcal{X}\}$, we obtain the marginal-CVaR model. Specifically, we follow the formulation of (27) in [7] to fit the model.

If we consider the shift in the conditional distribution $Y|X$, 
$Q(\alpha) = \{Q_0: P_{Y|X} = \alpha Q_0 + (1-\alpha)Q_1, \text{for some}~\alpha \geq \alpha_0~\text{and distribution}~Q_1~\text{and}~\mathcal{Y}\}$, we obtain the conditional-CVaR model. Specifically, we follow the formulation of Theorem 2 in [8] to fit the model where approximating $\alpha(x) = \theta^{\top}x$. 


## Reference:
* [1] A. Ben-Tal, D. den Hertog, A. D. Waegenaere, B. Melenberg, and G. Rennen. Robust solutions of optimization problems affected by uncertain probabilities. Management Science,
59(2):341–357, 2013.
* [2] Hu, Zhaolin, and L. Jeff Hong. "Kullback-Leibler divergence constrained distributionally robust optimization." Available at Optimization Online 1.2, 2013.
* [3] Duchi, John C., and Hongseok Namkoong. "Learning models with uniform performance via distributionally robust optimization." The Annals of Statistics 49 (3): 1378-1406. 2021.
* [4] Ruiwei Jiang, Yongpei Guan. Risk-Averse Two-Stage Stochastic Program with Distributional Ambiguity. Operations Research 66(5):1390-1405, 2018.
* [5] R Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal of risk, 2: 21–42, 2000.
* [6] Iyengar G, Lam H, Wang T. Optimizer's Information Criterion: Dissecting and Correcting Bias in Data-Driven Optimization. arXiv preprint arXiv:2306.10081, 2023.
* [7] Duchi, John, Tatsunori Hashimoto, and Hongseok Namkoong. "Distributionally robust losses for latent covariate mixtures." arXiv:2007.13982, 2020.
* [8] Sahoo R, Lei L, Wager S. Learning from a biased sample. arXiv preprint arXiv:2209.01754, 2022.



