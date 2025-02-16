import numpy as np 
from src.data.draw_utils import draw_classification

def classification_basic(d=2, k=2, num_samples=500, radius=5, seed=42, visualize=False, save_dir="./visualization.png"):
    np.random.seed(seed)
    samples_per_class = num_samples // k 
    X, y = [], []

    centers = np.random.randn(k, d)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) 
    centers *= radius

    for i, center in enumerate(centers):
        class_points = np.random.randn(samples_per_class, d) + center 
        X.append(class_points)
        y.append(np.full(samples_per_class, i)) 
    
    X = np.vstack(X)
    y = np.hstack(y)

    if visualize:
        draw_classification(X, y, save_dir)

    return X, y


def classification_DN21(d, num_samples=100, seed=42, visualize=False, save_dir="./visualization.png"):
    """
    Following Section 3.1.1 of "Learning Models with Uniform Performance via Distributionally Robust Optimization"
    link: https://arxiv.org/pdf/1810.08750
    """

    np.random.seed(seed)

    X = np.random.randn(num_samples, d)

    theta_star = np.random.randn(d)
    theta_star /= np.linalg.norm(theta_star)

    y_clean = np.sign(X @ theta_star)

    flip_mask = np.random.rand(num_samples) < 0.1  # 10% 概率翻转标签
    y_noisy = np.where(flip_mask, -y_clean, y_clean)

    if visualize:
        draw_classification(X, y_noisy, save_dir)

    return X, y_noisy

def classification_SNVD20(num_samples=500, seed=42, visualize=False, save_dir="./visualization.png"):
    """
    Following Section 5.1 of "Certifying Some Distributional Robustness with Principled Adversarial Training"
    link: https://arxiv.org/pdf/1710.10571
    """
    np.random.seed(seed)
    X = np.random.randn(num_samples, 2)
    norms = np.linalg.norm(X, axis=1)
    y = np.sign(norms-np.sqrt(2))
    lower_bound = np.sqrt(2) / 1.3
    upper_bound = 1.3 * np.sqrt(2)
    mask = (norms < lower_bound) | (norms > upper_bound) 

    X_filtered = X[mask]
    y_filtered = y[mask]

    if visualize:
        draw_classification(X_filtered, y_filtered, save_dir)

    return X_filtered, y_filtered


def classification_LWLC(num_data=10000, d=5, bias=0.5, scramble=1, sigma_s=3.0, sigma_v=0.3, high_dimension=300, seed=42, visualize=False, save_dir="./visualization.png"):
    """
    Following Section 4.1 (Classification) of "Distributionally Robust Optimization with Data Geometry"
    link: https://proceedings.neurips.cc/paper_files/paper/2022/file/da535999561b932f56efdd559498282e-Paper-Conference.pdf
    """
    from scipy.stats import ortho_group
    S = np.float32(ortho_group.rvs(size=1, dim=high_dimension, random_state=1))

    np.random.seed(seed)
    y = np.random.choice([1, -1], size=(num_data, 1))
    X = np.random.randn(num_data, d * 2)
    
    X[:, :d] *= sigma_s
    X[:, d:] *= sigma_v
    flip = np.random.choice([1, -1], size=(num_data, 1), p=[bias, 1. - bias]) * y
    X[:, :d] += y
    X[:, d:] += flip
    if scramble == 1:
        X = np.tile(X,(1,high_dimension//(2*d)))
        X = np.matmul(X, S)

    if visualize:
        draw_classification(X, y, save_dir)

    return X, y


