import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import numpy as np 

class VisualizationError(Exception):
    """Base exception class for errors in the visualization.
    """
    pass

def draw_classification(X, y, save_dir = None, title = None, weight = None, scale = 20):
    """
    two dimensional projection of classification data (X, y)
    """
    if weight is None:
        weight = np.ones(X.shape[0])
    else:
        weight = weight * X.shape[0]

    if X.shape[1]>2:
        pca = PCA(n_components=2)
        X_2D = pca.fit_transform(X)
    else:
        X_2D = X
    if X.shape[1] == 1:
        raise VisualizationError()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], s = weight * scale, c = y, cmap=plt.cm.jet, edgecolors="k", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    if title is None:
        plt.title("Data Visualization")
    else:
        plt.title(title)

    plt.colorbar(label="Class Label")
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir)