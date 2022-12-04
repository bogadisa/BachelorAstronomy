from random import random, seed
import autograd.numpy as np
from autograd import grad
# To do elementwise differentiation:
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from scaling.py import *

def create_X(x):
    return np.c_[np.ones((n, 1)), x, x**2]

def create_data(x):
    return 1 + 3*x + 5*x**2 #+ np.random.randn(n, 1)

def cost_func(beta):
    return (1.0/n)*np.sum((y - X @ beta)**2)

seed = np.random.seed(12345)

n = 100
x = np.random.rand(n,1)
y = create_data(x)

X = create_X(x)
XT_X = X.T @ X

beta = np.random.rand(3, 1)


def gradient_decent(X, y, beta, eta, derivative, n_iter, momentum=0):
    change = 0
    for i in range(n_iter):
        new_change = eta*derivative(beta) + momentum*change
        beta -= new_change
        change = new_change

    return beta

g_grad = egrad(cost_func)

H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

eta = 1.0/np.max(EigValues)
print(eta)
#eta = 0.001
n_iter = 1000
beta = random.sample(range(1, 100), 10)
beta = gradient_decent(X, y, beta, eta, g_grad, n_iter)
print(beta)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

print(beta_linreg)
