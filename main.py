import dro 
from dro.src.linear_model import Chi2DRO
from dro.src.data import classification_DN21
from dro.src.neural_model import Chi2NNDRO
import numpy as np 

if __name__ == "__main__":
    # Example usage
    X = np.random.randn(1000, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    model = Chi2NNDRO(input_dim=10, num_classes=2, model_type='mlp', task_type="classification")
    
    try:
        # Training
        metrics = model.fit(X, y, epochs=100)
        print(metrics)

        # Inference
        preds = model.predict(X[:5])
        print(f"Sample predictions: {preds}")

        # Evaluation
        acc = model.score(X, y)
        f1 = model.f1score(X, y)
        print(f"Final Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        
    except DROError as e:
        print(f"Error occurred: {str(e)}")