import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
#from sklearn.metrics import f1_score

from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
from typing import Optional, Tuple, Union

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_FRAC = 0.8
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, num_classes)
        
    def forward(self, X, **kwargs):
        return self.dense0(X.float())


class BaseNNDRO:
    """Base class for Neural Network Distributionally Robust Optimization (DRO) models.
    
    This class supports both regression and binary classification tasks. 

    Attributes:
        input_dim (int): Dimensionality of the input features.
        model_type (str): Model type indicator ('svm' for SVM, 'logistic' for Logistic Regression, 'ols' for Linear Regression).
        theta (np.ndarray): Model parameters.
    """
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.model = MLP(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("accuracy", correct / total)

        return correct / total

