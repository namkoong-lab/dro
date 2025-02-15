import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import numpy as np 

def draw_classification(X, y, save_dir="visualization.png"):
    if X.shape[1]>2:
        pca = PCA(n_components=2)
        X_2D = pca.fit_transform(X)
    else:
        X_2D = X
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap=plt.cm.jet, edgecolors="k", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Data Visualization")
    plt.colorbar(label="Class Label")
    plt.savefig(save_dir)