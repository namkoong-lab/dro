from src.data.dataloader_classification import *
from src.data.dataloader_regression import *



# X, y = classification_LWLC(num_data=10000, d=5, bias=0.5, scramble=1, sigma_s=3.0, sigma_v=0.3, high_dimension=300, seed=42, visualize=True, save_dir="./visualization.png")
X, y = regression_LWLC(scramble=1)
print(X.shape)
print(y.shape)