# f-divergence DRO

## Standard (Generalized)

## Partial DRO

Models listed here can all be formulated with the distance being $d(P, Q) = E_{Q}[f(\frac{dP}{dQ})]$ unless specified.

For KL-DRO problem, 
* we apply $f(x) = x \log x - (x - 1)$.
* Reference: Hu, Zhaolin, and L. Jeff Hong. "Kullback-Leibler divergence constrained distributionally robust optimization." Available at Optimization Online 1.2 (2013): 9.

For $\chi^2$-DRO problem, 
* we apply $f(x) = (x - 1)^2$.
* Reference: Duchi, John, and Hongseok Namkoong. "Variance-based regularization with convex objectives." Journal of Machine Learning Research 20.68 (2019): 1-55.

For TV-DRO problem, 
* we apply $f(x) = |x - 1|$.
* Reference: Ruiwei Jiang, Yongpei Guan (2018) Risk-Averse Two-Stage Stochastic Program with Distributional Ambiguity. Operations Research 66(5):1390-1405.

For the CVaR-DRO problem,
* we apply $f(x) = 0$ if $x \in [\frac{1}{\alpha}, \alpha]$ and $\infty$ otherwise (an augmented definition of the standard $f$-DRO problem). Here $\alpha$ denotes the worst-case ratio. 
* Reference: R Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal of risk, 2: 21–42, 2000.

Especially, if we only consider the shifts in the marginal distribution $X$, then we obtain the marginal-CVaR model,
* Reference: Duchi, John, Tatsunori Hashimoto, and Hongseok Namkoong. "Distributionally robust losses for latent covariate mixtures." Operations Research 71.2 (2023): 649-664.

