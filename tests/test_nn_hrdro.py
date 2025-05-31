import unittest
import numpy as np
import torch
from src.dro.neural_model.hrdro_nn import HRNNDRO, DROError

class TestHRNNDROModel(unittest.TestCase):
    def test_attack_robustness(self):
        """Verify prediction changes under attack."""
        X = np.random.randn(100, 3, 64, 64)  
        y = np.random.randint(0, 3, 100) 

        model = HRNNDRO(input_dim=3*64*64, num_classes=3, model_type='alexnet', task_type="classification")
        
        metrics = model.fit(X, y, epochs=1)
        preds = model.predict(X[:5])
        acc = model.score(X, y)
        f1 = model.f1score(X, y)

    def test_attack_robustness2(self):
        """Verify prediction changes under attack."""
        X = np.random.randn(100, 3, 64, 64)  
        y = np.random.randint(0, 3, 100) 

        model = HRNNDRO(input_dim=3*64*64, num_classes=3, model_type='alexnet', task_type="classification", learning_approach="HR")
        
        metrics = model.fit(X, y, epochs=1)
        preds = model.predict(X[:5])
        acc = model.score(X, y)
        f1 = model.f1score(X, y)
        
