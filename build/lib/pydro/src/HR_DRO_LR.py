import cvxpy as cp
import numpy as np
from sklearn.metrics import f1_score

class HR_DRO_LR:
    def __init__(self, r=1.0, alpha=1.0, epsilon=0.5, epsilon_prime=1.0, is_regression=0):
        self.r = r 
        self.alpha = alpha 
        self.epsilon = epsilon 
        self.epsilon_prime = epsilon_prime
        self.is_regression = is_regression
    
    def update(self, config={}):
        if 'r' in config.keys():
            self.r = config["r"]
        if 'alpha' in config.keys():
            self.alpha = config["alpha"]
        if 'epsilon' in config.keys():
            self.epsilon = config["epsilon"]
        if 'epsilon_prime' in config.keys():
            self.epsilon_prime = config["epsilon_prime"]
        
    def fit(self, X, Y):
        T = X.shape[0]
        theta = cp.Variable(X.shape[1])  # Define the size based on the problem
        w = cp.Variable(T)
        lambda_ = cp.Variable(nonneg=True)
        beta = cp.Variable(nonneg=True)
        eta = cp.Variable()
        temp = cp.Variable()

        # Objective
        objective = cp.Minimize(1/T * cp.sum(w) + lambda_ * (self.r - 1) + beta * self.alpha + eta)
        
        if self.is_regression == 1 or self.is_regression == 2:
            # Constraints
            constraints = []
            # Add constraints based on the problem
            for t in range(T):
                constraints.append(temp >= cp.abs(theta.T @ X[t] - Y[t]))
                constraints.append(w[t]>= cp.rel_entr(lambda_, (eta - cp.abs(theta.T @ X[t] - Y[t]) - self.epsilon * cp.norm(theta, 2)) ))
                constraints.append(w[t]>= cp.rel_entr(lambda_, (eta - temp - self.epsilon_prime * cp.norm(theta, 2)))-beta)
                constraints.append(eta >= cp.abs(theta.T @ X[t] - Y[t]) + self.epsilon_prime * cp.norm(theta, 2))
        elif self.is_regression == 0:
            Y = 2*Y - 1.0
            # Constraints
            constraints = []
            # Add constraints based on the problem
            constraints.append(eta>=1e-6)
            for t in range(T):
                constraints.append(temp <= Y[t]*(theta.T@X[t]))
                constraints.append(w[t] >= cp.rel_entr(lambda_, eta))
                constraints.append(w[t]>= cp.rel_entr(lambda_, (eta - 1 + Y[t]*(theta.T @ X[t])-self.epsilon*cp.norm(theta,2)) ))
                constraints.append(w[t]>= cp.rel_entr(lambda_, (eta - 1 + temp - self.epsilon_prime * cp.norm(theta, 2)))-beta)
                constraints.append(eta >= 1-Y[t]*(theta.T @ X[t])+self.epsilon*cp.norm(theta,2))
        else:
            raise NotImplementedError

        # Problem
        prob = cp.Problem(objective, constraints)

        # Solve
        prob.solve(solver=cp.MOSEK,
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8},
            verbose=True)


        self.theta = theta.value
        self.w = w.value 
        self.lambda_ = lambda_.value 
        self.beta = beta.value
        self.eta = eta.value 

        model_params = {}
        model_params["theta"] = self.theta.reshape(-1).tolist()
        return model_params
    
    def predict(self, X):
        scores = self.theta.T @ X.T
        preds = scores.copy()
        preds[scores >= 0] = 1
        preds[scores < 0] = 0
        return preds

    def score(self, X, y):
        # calculate accuracy of the given test data set
        predictions = self.predict(X)
        acc = np.mean([predictions.flatten() == y.flatten()])
        f1 = f1_score(y, predictions, average='macro')
        return acc, f1 


        
        

if __name__=="__main__":
    from sklearn.datasets import make_regression
    from  sklearn.linear_model import LinearRegression
    import numpy as np

    sample_size = 1000
    feature_size = 10
    X, y = make_regression(n_samples = sample_size, n_features = feature_size, noise = 1, random_state = 42)

    method = HR_DRO_LR()
    method.fit(X,y)
    print(method.theta)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    print(model.coef_)