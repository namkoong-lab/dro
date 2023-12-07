import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import math
from sklearn.metrics import f1_score

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")

def SDRO_eval(theta, Lambda, Reg, X, Y):
    n,d = X.shape

    ratio_1 = Lambda / (Lambda - 2 * np.linalg.norm(theta)**2)
    residual = np.mean((X@theta - Y)**2,0)
    obj_1 = ratio_1 * residual
    obj_2 = Lambda*Reg/2 * np.linalg.slogdet(np.eye(d) - theta@theta.T*2/Lambda)[1]
    return obj_1[0] - obj_2, np.sqrt(residual[0])




class Sinkhorn_DRO_Linear:
    def __init__(self, input_dim, reg_=1, lambda_=1, output_dim=1,
                    maxiter = 50, learning_rate = 1e-2, K_sample_max=5, is_regression=0):
        print(input_dim)
        self.model = LinearModel(input_dim, output_dim)
        self.lambda_ = lambda_
        self.reg_ = reg_
        self.maxiter_ = maxiter
        self.learning_rate = learning_rate
        self.K_sample_max = K_sample_max
        self.is_regression = is_regression
    
    def update(self, config={}):
        if "reg" in config.keys():
            self.reg_ = config["reg"]
        if "lambda" in config.keys():
            self.lambda_ = config["lambda"]
        if "k" in config.keys():
            self.K_sample_max = config["k"]
    
    def predict(self, X):
        X = torch.tensor(X).float()
        self.model.cpu()
        if self.is_regression:
            pred = self.model(X)
        else:
            pred = self.model(X)
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

        return pred.detach().numpy()
    
    def score(self, X, y):
        if self.is_regression:
            return np.mean(self.predict(X).reshape(-1)-y.reshape(-1))
        else:
            acc = np.mean([self.predict(X).flatten() == y.flatten()])
            f1 = f1_score(y, self.predict(X), average='macro')
            return acc, f1 
        
    def fit(self, X, y, optimization_type="SG"):
        X_tensor = torch.tensor(X, dtype=torch.float32)  
        Y_tensor = torch.tensor(y, dtype=torch.float32)  
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        if optimization_type == 'SG':
            self.SDRO_SG_solver(dataloader)
        elif optimization_type == 'MLMC':
            self.SDRO_MLMC_solver(dataloader)
        elif optimization_type == 'RTMLMC':
            self.SDRO_RTMLMC_solver(dataloader)
        else:
            raise ImplementationError("This optimization method is not implemented! Please choose one in \{SG, MLMC, RTMLMC\}")
        pass

        theta = self.model.linear.weight.cpu().detach().tolist()
        bias = self.model.linear.bias.cpu().detach().tolist()

        params = {}
        params['theta'] = theta 
        params['bias'] = bias 
        return params


    def SDRO_SG_solver(self, dataloader, device=torch.device("cuda:7")):
        """
        2-SDRO Approach with SG estimator for Regression Problem
        #   Input:
        # Feature: N samples of R^d [dim: N*d]
        #  Target: labels of N samples [dim: N*1]
        #   theta: initial guess for optimization
        #  Lambda: Lagrangian multiplier
        #     Reg: bandwidth
        #  Output:
        #   theta: optimized decision
        """
        iter = 0
        Lambda_Reg = self.lambda_ * self.reg_
        self.model.to(device)
        optimizer_theta = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        
        for epoch in range(self.maxiter_):
            for _, (data, target) in enumerate(dataloader):
                iter = iter + 1

                # generate stochastic samples
                N, d         = data.shape
                data, target = Variable(data), Variable(target)
                m            = int(2**self.K_sample_max)

                optimizer_theta.zero_grad()
                data_noise     = torch.randn([m, N, d]) * np.sqrt(self.reg_) + data.reshape([1,N,d])
                data_noise_vec = data_noise.reshape([-1,d])
                target_noise   = target.repeat(m,1).reshape(-1,1).to(device)

                haty  = self.model(data_noise_vec.to(device))

                obj_vec  = (haty - target_noise) ** 2
                obj_mat  = obj_vec.reshape([m, N])
                Residual = obj_mat / Lambda_Reg

                Loss_SDRO = (torch.logsumexp(Residual, dim=0, keepdim=True)-math.log(m)) * Lambda_Reg
                Loss_SDRO_avg = torch.mean(Loss_SDRO)

                Loss_SDRO_avg.backward()
                optimizer_theta.step()

                if iter % 10000 == 0:
                    print(f"Iter {iter} {Loss_SDRO_avg.data}")


    def SDRO_MLMC_solver(self, dataloader):
        iter       = 0
        Lambda_Reg = self.lambda_ * self.reg_
        
        optimizer_theta = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        N_ell_hist      = np.int_(2**(np.arange(self.K_sample_max) + 1))

        
        for epoch in range(self.maxiter_):
            for _, (data, target) in enumerate(dataloader):
                iter = iter + 1
                # generate stochastic samples
                N, d         = data.shape
                data, target = Variable(data), Variable(target)
                optimizer_theta.zero_grad()

                m_total = 0
                for K_sample in np.arange(self.K_sample_max):
                    m          = int(2**K_sample)
                    N_ell      = N_ell_hist[-K_sample-1]
                    data_ell   = data[:N_ell, :]
                    N_ell, d   = data_ell.shape
                    target_ell = target[:N_ell]

                    data_noise     = torch.randn([m, N_ell, d]) * np.sqrt(self.reg_) + data_ell.reshape([1,N_ell,d])
                    m_total       += m * N_ell
                    data_noise_vec = data_noise.reshape([-1,d])
                    target_noise   = target_ell.repeat(m,1)

                    haty     = self.model(data_noise_vec)
                    obj_vec  = (haty - target_noise) ** 2
                    obj_mat  = obj_vec.reshape([m, N_ell])
                    Residual = obj_mat / Lambda_Reg

                    if K_sample == 0:
                        Loss_SDRO = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                        Loss_SDRO_avg_K_sample = torch.mean(Loss_SDRO)
                        Loss_SDRO_avg_sum = Loss_SDRO_avg_K_sample
                    else:
                        m1 = int(m/2)
                        Residual_half   = Residual[:m1,:]
                        Residual_remain = Residual[m1:,:]
                        Loss_SDRO_1     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                        Loss_SDRO_avg_1 = torch.mean(Loss_SDRO_1)
                        Loss_SDRO_2     = (torch.log(torch.mean(torch.exp(Residual_half), dim=0)) + torch.log(torch.mean(torch.exp(Residual_remain), dim=0))) * Lambda_Reg
                        Loss_SDRO_avg_2 = torch.mean(Loss_SDRO_2)
                        Loss_SDRO_avg_K_sample   = Loss_SDRO_avg_1 - 0.5 * Loss_SDRO_avg_2                
                        Loss_SDRO_avg_sum = Loss_SDRO_avg_sum + Loss_SDRO_avg_K_sample

                Loss_SDRO_avg_sum.backward()
                optimizer_theta.step()
                
                # if torch.linalg.norm(self.model.linear.weight.data) > 0.95*np.sqrt(Lambda/2):
                #     self.model.linear.weight.data = self.model.linear.weight.data / torch.linalg.norm(self.model.linear.weight.data) * 0.95*np.sqrt(Lambda/2)
        
        

    def SDRO_RTMLMC_solver(self, dataloader):
        iter            = 0
        Lambda_Reg      = self.lambda_ * self.reg_
        optimizer_theta = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # sampling from truncated gemoetric distribution
        elements      = np.arange(self.K_sample_max)
        probabilities = (0.5) ** (elements)
        probabilities = probabilities / np.sum(probabilities)

        for epoch in range(self.maxiter_):
            for _, (data, target) in enumerate(dataloader):
                iter = iter + 1
                # generate stochastic samples
                N, d         = data.shape
                data, target = Variable(data), Variable(target)

                K_sample       = int(np.random.choice(list(elements), 1, list(probabilities)))
                m              = int(2**K_sample)
                optimizer_theta.zero_grad()
                data_noise     = torch.randn([m, N, d]) * np.sqrt(Reg) + data.reshape([1,N,d])
                data_noise_vec = data_noise.reshape([-1,d])
                target_noise   = target.repeat(m,1)

                haty       = data_noise_vec @ theta_torch.type(torch.float64)
                obj_vec    = (haty - target_noise) ** 2
                obj_mat    = obj_vec.reshape([m, N])
                Residual   = obj_mat / Lambda_Reg

                if m == 1:
                    Loss_SDRO = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                    Loss_SDRO_avg = torch.mean(Loss_SDRO)
                else:
                    m1 = int(2**(K_sample-1))
                    Residual_half   = Residual[:m1,:]
                    Residual_remain = Residual[m1:,:]
                    Loss_SDRO_1     = torch.log(torch.mean(torch.exp(Residual), dim=0)) * Lambda_Reg
                    Loss_SDRO_avg_1 = torch.mean(Loss_SDRO_1)
                    Loss_SDRO_2     = (torch.log(torch.mean(torch.exp(Residual_half), dim=0)) + torch.log(torch.mean(torch.exp(Residual_remain), dim=0))) * Lambda_Reg
                    Loss_SDRO_avg_2 = torch.mean(Loss_SDRO_2)
                    Loss_SDRO_avg   = Loss_SDRO_avg_1 - 0.5 * Loss_SDRO_avg_2
                
                Loss_SDRO_avg = 1/probabilities[K_sample]* Loss_SDRO_avg
                Loss_SDRO_avg.backward()
                optimizer_theta.step()
        
                # if torch.linalg.norm(self.model.linear.weight.data) > 0.95*np.sqrt(Lambda/2):
                #     self.model.linear.weight.data = self.model.linear.weight.data / torch.linalg.norm(self.model.linear.weight.data) * 0.95*np.sqrt(Lambda/2)
        



if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from  sklearn.linear_model import LinearRegression
    import numpy as np

    sample_size = 1000
    feature_size = 10
    X, y = make_regression(n_samples = sample_size, n_features = feature_size, noise = 1, random_state = 42)
    
    method = Sinkhorn_DRO_Linear(0.1, 1000.0, maxiter=2000, input_dim=feature_size)
    method.fit(X,y)
    print(method.model.linear.weight)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    print(model.coef_)