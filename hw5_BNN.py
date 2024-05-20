import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import os
from utils import *
from scipy.special import expit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_POSTERIOR_SAMPLES = 1000

'''
 result: 
    max train accuracy:  0.93828005
    max test accuracy:  0.93263996
    within 1000 epochs

'''

class BNN(nn.Module):
    '''
    Bayesian Neural Network for classification
    
    Assume that the variational sposterior and prior of weights is fully factorized
    '''
    def __init__(self, in_dim, hidden_dim, activation):
        super(BNN, self).__init__()
        self.d = in_dim
        self.h = hidden_dim
        self.activation = activation
        self.total_num_weights = self.d*self.h + self.h*1 # we can denote this as D

        self.mu = nn.Parameter(torch.zeros(self.total_num_weights,1)).float() # mean of the variational posterior
        self.p = nn.Parameter(torch.ones(self.total_num_weights,)).float() # parameter of the variance of the variational posterior

    def forward(self, X, W1, W2):
        '''
        X: shape (N,d)
        W1: shape (B,d,h)
        W2: shape (B,h,1)

        Return output of shape (B,N,1)
        '''
        X = X.unsqueeze(0).tile(dims=(W1.shape[0],1,1)) #(1,N,d)
        x = self.activation(torch.bmm(X,W1)) #(B,N,h)
        x = torch.matmul(x,W2) #(B,N,1)
        out = torch.sigmoid(x) #(B,N,1)
        return out
    
    def accuracy(self, X, y, W1, W2):
        '''
        X: shape (N,d)
        y: shape (N,1)
        W1: shape (B,d,h)
        W2: shape (B,h,1)

        Return accuracy of shape (B,)
        '''
        y = y.unsqueeze(0)
        out = self.forward(X, W1, W2) #(B,N,1)
        pred = torch.where(out>0.5, torch.ones_like(out), torch.zeros_like(out)) #(B,N,1)
        correct = (pred==y).int().float() #(B,N,1)
        sample_correct = torch.mean(correct, dim=1) #(B,1)
        return torch.mean(sample_correct)
    
    def predictive_likelihood(self, X, y, W1, W2):
        '''
        X: shape (N,d)
        y: shape (N,1)
        W1: shape (B,d,h)
        W2: shape (B,h,1)

        Return predictive likelihood of shape (B,)
        '''
        y = y.unsqueeze(0)
        out = self.forward(X, W1, W2) #(B,N,1)
        likelihood = torch.where(y==1, out, 1-out) #(B,N,1)
        sample_likelihood = torch.mean(likelihood, dim=1) #(B,1)
        return torch.mean(sample_likelihood)
    
    def sample_epsilon(self, num_samples):
        '''
        return epsilon of shape (total_num_weights,num_samples), each column is a vector with each component sampled from N(0,1)
        '''
        epsilon = torch.distributions.Normal(loc=0, scale=1).sample((self.total_num_weights,num_samples))
        return epsilon
    
    def cov_diagonal(self):
        '''
        return the diagonal matrix of the variance of the variational posterior
        '''
        return torch.diag(torch.log(1+torch.exp(self.p)))
    
    def sample_weights(self, Epsilon):
        '''
        Epsilon: shape (D,num_samples)
        mu: shape (D,1)
        covariance matrix of posterior: shape (D,D)

        Return W1, W2 of shape (num_samples, d, h), (num_samples, h,1)
        '''
        num_samples = Epsilon.shape[1]
        W = self.mu + torch.matmul(torch.sqrt(self.cov_diagonal()),Epsilon) #(D,num_samples)
        W1 = W[:self.d*self.h, :].T.reshape(num_samples, self.d, self.h)
        W2 = W[self.d*self.h:, :].T.reshape(num_samples, self.h,1)
        return W1, W2
    
    def kl_div(self, prior_mean, prior_cov):
        '''
        prior_mean: shape (total_num_weights,1)
        prior_cov: shape (total_num_weights, total_num_weights), diagonal matrix (all weights are independent)

        KL divergence between prior and approximate posterior KL(p||q)
        '''
        q_cov = self.cov_diagonal()
        p_det = torch.prod(torch.diagonal(prior_cov))
        q_det = torch.prod(torch.diagonal(q_cov))
        
        prior_cov_inv = torch.linalg.inv(prior_cov)
        trace = 0.5*(torch.sum(torch.diagonal(torch.matmul(prior_cov_inv, q_cov))))
        quadratic = 0.5*torch.matmul(torch.matmul((self.mu-prior_mean).T, prior_cov_inv), self.mu-prior_mean)
        
        return 0.5*torch.log(p_det)- 0.5*torch.log(q_det) - self.total_num_weights/2 + trace + quadratic
        
    def variational_lower_bound(self, X, y, prior_mean, prior_cov, num_epsilon_samples=NUM_POSTERIOR_SAMPLES):
        '''
        X: shape (N,d)
        y: shape (N,1)
        '''
        kl = self.kl_div(prior_mean, prior_cov)
        
        Epsilons = self.sample_epsilon(num_epsilon_samples).to(device) #(total_num_weights,num_samples)
        W1, W2 = self.sample_weights(Epsilons) #(num_samples, d, h), (num_samples, h,1)

        out_prob = self.forward(X, W1, W2) #(num_samples, N,1)
        y = y.unsqueeze(0) #(1,N,1)
        log_likelihoods = torch.sum(y*torch.log(out_prob+1e-6) + (1-y)*torch.log(1-out_prob+1e-6), dim=1) #(num_samples,1)

        expected_log_likelihood = torch.mean(log_likelihoods)

        print("___expected_log_likelihood", expected_log_likelihood.item())

        return expected_log_likelihood + kl

    
def test(test_data, model):
    X, y = test_data
    Epsilons = model.sample_epsilon(num_samples=NUM_POSTERIOR_SAMPLES//2).to(device)
    W1, W2 = model.sample_weights(Epsilons)
    pred_ll = model.predictive_likelihood(X, y, W1, W2)
    accuracy = model.accuracy(X, y, W1, W2)
    return torch.log(pred_ll), pred_ll, accuracy
    
def train(train_data, test_data, model, prior_params, num_epochs, lr):
    '''
    train_data and test_data: list of [X,y]
    model: BNN model
    prior_params: [prior_mean, prior_cov]
    num_epochs: int
    lr: float

    return: trained model
    '''
    X, y = train_data
    prior_mean, prior_cov = prior_params
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_accuracies = []
    train_log_pred_lls = []
    test_accuracies = []
    test_log_pred_lls = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # print("mu: ", model.mu)
        # print("p: ", model.p)
        loss = -model.variational_lower_bound(X, y, prior_mean, prior_cov)[0,0]
        # print("kk", loss.shape)

        log_pred_ll, pred_ll, accuracy = test(test_data, model)
        print("loss", loss)
        test_log_pred_lls.append(log_pred_ll)
        test_accuracies.append(accuracy)

        log_pred_ll, pred_ll, accuracy = test(train_data, model)
        train_log_pred_lls.append(log_pred_ll)
        train_accuracies.append(accuracy)

        loss.backward()
        optimizer.step()

        if epoch%500==0:
            print(f"epoch: {epoch}, loss: {loss}")
            
    return model, train_log_pred_lls, train_accuracies, test_log_pred_lls, test_accuracies





if __name__=="__main__":
    train_data = load_csv_data(data_path="data/bank-note/train.csv", preppend_one=True, remove_first_row=False)
    test_data = load_csv_data(data_path="data/bank-note/test.csv", preppend_one=True, remove_first_row=False)
    num = 100
    X_train = train_data[:num,:-1] #(N,5)
    y_train = train_data[:num,-1].reshape((-1,1)) #(N,1)
    X_test = test_data[:num,:-1] #(N,5)
    y_test = test_data[:num,-1].reshape((-1,1)) #(N,1)
    
    N,d = X_train.shape

    device="cpu"
    
    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

    data_train = [X_train, y_train]
    data_test = [X_test, y_test]
    # hidden_dims = [10,20,50]
    # activations = [torch.tanh, F.relu]
    hidden_dims = [50]
    activations = [F.relu]

    lr = 1e-3
    #lr = 0.5e-3
    #lr = 1e-4
    #lr = 1e-5
    epochs = [x for x in range(1000)]
    os.makedirs("./models", exist_ok=True)
    act_names = ["tanh", "relu"]
    for i, h in enumerate(hidden_dims):
        for j, activation in enumerate(activations):
            print(f"++++ hidden_dim: {h}, activation: {act_names[j]}")
            model = BNN(in_dim=d, hidden_dim=h, activation=activation).to(device)
            prior_mean = torch.zeros(model.total_num_weights,1).to(device)
            prior_cov = torch.eye(model.total_num_weights).to(device)
            trained_model, train_log_pred_lls, train_accuracies, test_log_pred_lls, test_accuracies = train(data_train, data_test, model, [prior_mean, prior_cov], num_epochs=1000, lr=lr)
            torch.save(trained_model.state_dict(), f"./models/bnn_{i}_{j}")
            train_log_pred_lls = [x.cpu().detach().numpy() for x in train_log_pred_lls]
            test_log_pred_lls = [x.cpu().detach().numpy() for x in test_log_pred_lls]
            train_accuracies = [x.cpu().detach().numpy() for x in train_accuracies]
            test_accuracies = [x.cpu().detach().numpy() for x in test_accuracies]
            
            print("max train accuracy: ", max(train_accuracies))
            print("max test accuracy: ", max(test_accuracies))
            
            every = 100
            epochs = epochs[0::every]
            train_accuracies, test_accuracies = train_accuracies[0::every], test_accuracies[0::every]
            train_log_pred_lls, test_log_pred_lls = train_log_pred_lls[0::every], test_log_pred_lls[0::every]
            
            plot_xy_curves(epochs, ys_list=[train_log_pred_lls, test_log_pred_lls], labels_list=["train", "test"], xlim=None, ylim=None, x_label="epoch", y_label="log predictive likelihood", title="log predictive likelihood", path=f"./figures/hw5/BNN", name=f"nll_bnn_{i}_{j}", vis=False)
            plot_xy_curves(epochs, ys_list=[train_accuracies, test_accuracies], labels_list=["train", "test"], xlim=None, ylim=None, x_label="epoch", y_label="accuracy", title="accuracy", path=f"./figures/hw5/BNN", name=f"acc_bnn_{i}_{j}", vis=False)
            
           