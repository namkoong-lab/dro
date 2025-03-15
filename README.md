## DRO: A Package for Distributionally Robust Optimization in Machine Learning

> <a href="https://ljsthu.github.io">Jiashuo Liu*</a>, <a href="https://wangtianyu61.github.io">Tianyu Wang*</a>, <a href="https://pengcui.thumedialab.com">Peng Cui</a>, <a href="https://hsnamkoong.github.io">Hongseok Namkoong</a>, <a href="https://web.stanford.edu/~jblanche/">Jose Blanchet</a>

> Tsinghua University, Columbia University, Stanford University


`DRO` is a python package that implements 12 typical DRO methods on linear loss (SVM, logistic regression, and linear regression) for supervised learning tasks. It is built based on the convex optimization solver `cvxpy`. Without specified, our DRO model is to solve the following optimization problem:
$$\min_{\theta} \max_{P: P \in U} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))],$$
where $U$ is the so-called ambiguity set and typically of the form $U = \{P: d(P, \hat P_n) \leq \epsilon\}$ and $\hat P_n := \frac{1}{n}\sum_{i = 1}^n \delta_{(X_i, Y_i)}$ is the empirical distribution of training samples $\{(X_i, Y_i)\}_{i = 1}^n$. And $\epsilon$ is the hyperparameter 


## Base Models
Currently we only focus on linear models, including logistic regression, linear SVM, and linear regression. And further version will incorporate neural network implementations via some approximations.

## Install

```
pip install dro
```

#### Usage
```
from dro import fetch_method
method = fetch_method("name", is_regression, input_dim)
```

### Reference
