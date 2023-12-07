import numpy as np
import math
import cvxpy as cp
from sklearn.metrics import f1_score

# DRO + l2 linear regression
## regression: l2; 
## classification: logistic loss (currently only binary loss) with Y in {-1, 1}
## we unify the term ambigiuty size as eps, but we include the intercept in the feature X
 
class base_DRO:
    def __init__(self, input_dim, is_regression=0):
        self.input_dim = input_dim
        self.is_regression = is_regression
        self.theta = np.zeros(self.input_dim)
    def update(self, config):   
        # depend on different parameters in DRO models
        pass

    def fit(self, X, y):
        # solve different optimization problems
        pass
    
    def load(self, config):
        self.theta = np.array(config['theta']).reshape(self.input_dim)

    def predict(self, X):
        if self.is_regression == 2:
            return np.dot(X, self.theta)

        elif self.is_regression == 0:
            scores = self.theta.T @ X.T
            preds = scores.copy()
            preds[scores >= 0] = 1
            preds[scores < 0] = 0
            
            return preds

        # use OLS for classification
        elif self.is_regression == 1:
            scores = self.theta.T @ X.T
            preds = scores.copy()
            preds[scores >= 0.5] = 1
            preds[scores < 0.5] = 0
            return preds


    def score(self, X, y, weights = None):
        predictions = self.predict(X)    
        print("pred", predictions.mean())
        print("y", y.mean())
        if weights is not None:
            weights = weights / np.sum(weights)
        if self.is_regression == 2:
            err = np.average((predictions-y.reshape(predictions.shape))**2, weights = weights)
            return err  
        else:
            pred = (predictions.reshape(-1) == y.reshape(-1))
            acc = np.dot(weights.reshape(-1), pred.reshape(-1))
            # print(np.array([predictions.flatten() == y.flatten()]).reshape(-1, 1).shape)
            # acc = np.average(np.array([predictions.flatten() == y.flatten()])[0], weights = weights)
            f1 = f1_score(y, predictions, average='macro')
            return acc, f1

    def loss(self, X, y):
        # after obtaining theta
        if self.is_regression == 0:
            newy = 2*y - 1
            return np.maximum(1 - np.multiply(np.dot(X, self.theta), newy), 0)
        elif self.is_regression == 1 or self.is_regression == 2:
            return (y - X @ self.theta) ** 2
        else:
            #logistic
            raise NotImplementedError
            
    def cvx_loss(self, X, y, theta):
        if self.is_regression == 0:
            newy = 2*y - 1
            return cp.pos(1 - cp.multiply(newy, X @ theta))
        elif self.is_regression == 1 or self.is_regression == 2:
            return cp.power(y - X @ theta, 2)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    predictions = np.array([1,1,1,0])
    y = np.array([0,0,1,1])
    acc = np.average(np.array([predictions.flatten() == y.flatten()])[0], weights = None)
    f1 = f1_score(y, predictions, average='macro')
    print(acc, f1)
