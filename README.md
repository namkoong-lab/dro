## DRO: A Python Package for Distributionally Robust Optimization in Machine Learning

`DRO` is a python package that implements typical DRO methods on linear loss (SVM, logistic regression, and linear regression) for supervised learning tasks. It is built based on the convex optimization solver `cvxpy`. Without specified, our DRO model is to solve the following optimization problem:
$$\min_{\theta} \max_{P: P \in U} E_{(X,Y) \sim P}[\ell(\theta;(X, Y))],$$
where $U$ is the so-called ambiguity set and typically of the form $U = \{P: d(P, \hat P_n) \leq \epsilon\}$ and $\hat P_n := \frac{1}{n}\sum_{i = 1}^n \delta_{(X_i, Y_i)}$ is the empirical distribution of training samples $\{(X_i, Y_i)\}_{i = 1}^n$. And $\epsilon$ is the hyperparameter. 


## Install

```
pip install dro
```

## Examples
Please refer to our <a href="https://python-dro.org/api/example.html">examples</a>.

#### Documentation
Please refer to https://python-dro.org for more details!
