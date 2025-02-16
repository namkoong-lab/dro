import numpy as np 
from sklearn.datasets import make_regression
from ucimlrepo import fetch_ucirepo   
import pandas as pd 

def regression_basic(num_samples=100, d=1, noise=0.1, seed=42):
    X, y = make_regression(n_samples=num_samples, n_features=d, noise=noise, random_state=seed)
    return X, y


def regression_DN20_1(num_samples, d=5, noise=0.01, seed=42):
    """
    Following Section 3.1.2 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
    link: https://arxiv.org/pdf/1810.08750
    """
    np.random.seed(seed)

    X = np.random.randn(num_samples, d)
    eps = np.random.randn(num_samples)*noise
    theta_star = np.random.randn(d)
    theta_star /= np.linalg.norm(theta_star)

    y = X @ theta_star+eps 
    y_noisy = np.where(X[:,0]>1.645, y+X[:,0], y)

    return X, y_noisy


def regression_DN20_2(num_samples, prob=0.1, noise=0.01, seed=42):
    """
    Following Section 3.1.3 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
    link: https://arxiv.org/pdf/1810.08750

    prob: the minority group ratio
    """

    np.random.seed(seed)
    X = np.random.randn(num_samples, 2)
    theta_star1 = np.array([1.0, 0.1]).T
    theta_star2 = np.array([1.0, 1.0]).T
    
    eps = np.random.randn(num_samples)*noise
    G = np.random.uniform(low=0, high=1, size=num_samples)
    y = np.where(G<prob, X@theta_star1+eps, X@theta_star2+eps)

    return X, y

def regression_DN20_3(save_dir="./data/", download=True):
    """
    Following Section 3.3 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
    link: https://arxiv.org/pdf/1810.08750

    Data is from UCI repository: https://archive.ics.uci.edu/dataset/183/communities+and+crime
    """
    if download:
        communities_and_crime = fetch_ucirepo(id=183) 
        X = communities_and_crime.data.features 
        y = communities_and_crime.data.targets 
        X = X.drop(columns=['communityname'])
        X_values = X.apply(pd.to_numeric, errors='coerce')
        X_filled = X_values.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col, axis=0)
        X_values = X_filled.to_numpy()
        y = y.to_numpy()
        np.savez(f'{save_dir}crime.npz', X=X_values, y=y)
    else:
        try:
            data = np.load(f'{save_dir}crime.npz')
            X = data["X"]
            y = data["y"]
        except Exception as e :
            print(e)
            print("Please set download=True and retry!")
    return X, y

def regression_LWLC(n1=100000, n2=1000, ps=5, pvb=1, pv=4, r=1.7, scramble=0):
    """
    Following Section 4.1 (Regression) of "Distributionally Robust Optimization with Data Geometry"
    link: https://proceedings.neurips.cc/paper_files/paper/2022/file/da535999561b932f56efdd559498282e-Paper-Conference.pdf

    n1: total number of samples in the pool
    n2: number of samples required
    r: controls the spurious correlation between pvb and y
    scramble: whether to mix S and V
    """
    
    S = np.random.normal(0, 2, [n1, ps])

    Z = np.random.normal(0, 1, [n1, ps + 1])
    for i in range(ps):
        S[:, i:i + 1] = 0.8 * Z[:, i:i + 1] + 0.2 * Z[:, i + 1:i + 2]

    beta = np.zeros((ps, 1))
    for i in range(ps):
        beta[i] = (-1) ** i * (i % 3 + 1) * 1.0

    noise = np.random.normal(0, 0.5, [n1, 1])

    Y = np.dot(S, beta) + noise + 0.1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]
    V = np.random.normal(Y, 2, [n1, pvb + pv])
    V[:, :pv] = np.random.normal(0, 2, [n1, pv])
    index_pre = np.ones([n1, 1], dtype=bool)
    for i in range(pvb):
        D = np.abs(V[:, pv + i:pv + i + 1] * np.sign(r) - Y)
        pro = np.power(np.abs(r), -D * 5)
        selection_bias = np.random.random([n1, 1])
        index_pre = index_pre & (
                    selection_bias < pro)
    index = np.where(index_pre == True)
    S_re = S[index[0], :]
    V_re = V[index[0], :]
    Y_re = Y[index[0]]
    n, _ = S_re.shape
    index_s = np.random.permutation(n)

    X_re = np.hstack((S_re, V_re))
    
    X = X_re[index_s[0:n2], :]
    y = Y_re[index_s[0:n2], :]

    from scipy.stats import ortho_group
    S = np.float32(ortho_group.rvs(size=1, dim=X.shape[1], random_state=1))
    if scramble == 1:
        X = np.matmul(X, S)
    return X, y
