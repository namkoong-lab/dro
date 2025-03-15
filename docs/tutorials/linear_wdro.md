# Wasserstein DRO

For $Z_1 = (X_1, Y_1), Z_2 = (X_2, Y_2)$, 

$W(P_1, P_2) = \inf_{\pi \sim (P_1, P_2)}\mathbb{E}_{\pi}[d(X, Y)]$


```python

```


or the basic Wasserstein-DRO model,
* we apply $d(P, Q) = W_c(P, Q)$, where $W_c$ is the Wasserstein distance with the induced cost function inside it being $c((X_1, Y_1), (X_2, Y_2)) = X_1^{\top} \Lambda X_2$, the default $\Lambda$ is taken as the unit matrix and $c(\cdot, \cdot)$ then becomes squared Euclidean distance. That is, we only allow the perturbation of $X$ but not $Y$.   
* Reference: Blanchet, Jose, et al. "Data-driven optimal transport cost selection for distributionally robust optimization." 2019 winter simulation conference (WSC). IEEE, 2019., where Theorem 1 there provides the corresponding reformulations.

For the augmented Wasserstein-DRO model,
* we still use $d(P, Q)$ being the Wasserstein distance $W_c$ but now we allow the change of $y$. More specifically, we set $c((X_1, Y_1), (X_2, Y_2)) = X_1^{\top}\Lambda X_2 + \kappa|Y_1 - Y_2|$, and we introduce another hyperparameter $\kappa$ to adjust the change in $Y$. Note that when $\kappa = \infty$, it reduces to the previous case.
* Reference:  Shafieezadeh-Abadeh, Soroosh, Daniel Kuhn, and Peyman Mohajerin Esfahani. "Regularization via mass transportation." Journal of Machine Learning Research 20.103 (2019): 1-68, where they provide various loss functions to reformulate the problem.

For the Satisficing Wasserstein-DRO model, 
* we solve the following constrained optimization problem, where DRO is set as the constraint counterpart:
$$\max \epsilon, \text{s.t.}~\min_{\theta} \max_{P: P \in U_{\epsilon}} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))] \geq \tau,$$
where we do not set $\epsilon$ as the hyperparameter but as an optimization goal such that to minimize the worst-case performance. Instead, we set $\tau$ as the hyperparameter, which is set as the multiplication of the best empirical performance with $E_{(X, Y)\sim \hat P_n}[\ell(\theta_{ERM};(X, Y))]$. To solve this optimization problem, at each binary search, we prefix $\epsilon \in [0, M]$ and compute the corresponding left-hand objective value and then reduce the potential $\epsilon$ half by half.
* Reference: Long, Daniel Zhuoyu, Melvyn Sim, and Minglong Zhou. "Robust satisficing." Operations Research 71.1 (2023): 61-82.



## Reference
* 