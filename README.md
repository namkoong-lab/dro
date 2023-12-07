### DRO Package

> <a href="https://ljsthu.github.io">Jiashuo Liu*</a>, <a href="https://wangtianyu61.github.io">Tianyu Wang*</a>, <a href="https://pengcui.thumedialab.com">Peng Cui</a>, <a href="https://hsnamkoong.github.io">Hongseok Namkoong</a>

> Tsinghua University, Columbia University


`DRO` is a python package that implements 12 typical DRO methods on linear models (SVM, logistic regression, and linear regression). It is built based on `cvxpy`. Implemented DRO methods include:
* $f$-DRO
    * CVaR-DRO
    * KL-DRO
    * TV-DRO
    * Marginal DRO (CVaR)
* Wasserstein DRO
    * Wasserstein DRO
    * Augmented Wasserstein DRO
    * Regularized Wasserstein DRO
* MMD-DRO
* Sinkhorn-DRO
* Holistic DRO
* Unified-DRO
    * $L_2$ cost
    * $L_{inf}$ cost

Current version only contains linear models. And further version will incorporate neural network implementations via some approximations.