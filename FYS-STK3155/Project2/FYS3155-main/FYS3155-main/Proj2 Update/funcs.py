import numpy as np
from autograd import elementwise_grad as egrad

"""
def gradient_decent(X, y, beta, eta, derivative, n_iter, momentum=0):
    change = 0
    for i in range(n_iter):
        new_change = eta*derivative(beta) + momentum*change
        beta -= new_change
        change = new_change
    return beta

def gradient_decent_batch(X, y, beta, eta, derivative, n_iter, momentum=0, batch_size=1, epochs=1):
    change = 0
    for i in range(n_iter):
        new_change = eta*derivative(beta) + momentum*change
        beta -= new_change
        change = new_change
    return beta

def create_X(x):
    return np.c_[np.ones((n, 1)), x, x**2]

def create_data(x):
    return 1 + 5*x + 2*x**2 #+ np.random.randn(n, 1)

def cost_func(beta):
    return (1.0/n)*np.sum((y - X @ beta)**2)


seed = np.random.seed(12345)

n = 500
x = np.random.rand(n,1)
y = create_data(x)
X = create_X(x)

beta = np.random.rand(3, 1)

H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H)
lr = 1.0/np.max(EigValues)
n_iter = 1000
beta = np.random.rand(3, 1)

g_grad = egrad(cost_func) #derivate of cost func
beta = gradient_decent(X, y, beta, lr, g_grad, n_iter)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

print(beta_linreg)
"""

"""
Example 1:
#pandas.sample  -  Return a random sample of items from an axis of object.
def sgd_regressor(X, y, learning_rate=0.2, n_epochs=1000, k=40):

    w = np.random.randn(1,13)  # Randomly initializing weights
    b = np.random.randn(1,1)   # Random intercept value

    epoch=1

    while epoch <= n_epochs:

        temp = X.sample(k)

        X_tr = temp.iloc[:,0:13].values
        y_tr = temp.iloc[:,-1].values

        Lw = w
        Lb = b

        loss = 0
        y_pred = []
        sq_loss = []

        for i in range(k):

            Lw = (-2/k * X_tr[i]) * (y_tr[i] - np.dot(X_tr[i],w.T) - b)
            Lb = (-2/k) * (y_tr[i] - np.dot(X_tr[i],w.T) - b)
                        w = w - learning_rate * Lw
            b = b - learning_rate * Lb

            y_predicted = np.dot(X_tr[i],w.T)
            y_pred.append(y_predicted)

        loss = mean_squared_error(y_pred, y_tr)

        print("Epoch: %d, Loss: %.3f" %(epoch, loss))
        epoch+=1
        learning_rate = learning_rate/1.02

    return w,b

def predict(x,w,b):
    y_pred=[]
    for i in range(len(x)):
        temp_ = x
        X_test = temp_.iloc[:,0:13].values
        y = np.asscalar(np.dot(w,X_test[i])+b)
        y_pred.append(y)
    return np.array(y_pred)
w,b = sgd_regressor(X_train,y_train)
y_pred_customsgd = predict(X_test,w,b)
"""


#Example 2:

# Using Autograd to calculate gradients using SGD
# OLS example
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# Note change from previous example
def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)

# define the gradient
training_gradient = grad(CostOLS)

n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+2*x**2 #np.random.randn(n,1)
degree = 3

X = np.c_[np.ones((n,1)), x]
XT_X = X.T @ X
theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)
# Hessian matrix
H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

theta = np.random.randn(degree,1)
eta = 1.0/np.max(EigValues)
Niterations = 100

# Note that we request the derivative wrt third argument (theta, 2 here)
training_gradient = grad(CostOLS,2)

for iter in range(Niterations):
    gradients = (1.0/n)*training_gradient(y, X, theta)
    theta -= eta*gradients
print("theta from own gd")
print(theta)


n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(degree,1)

change = 0.0
delta_momentum = 0.3

for epoch in range(n_epochs):
    for i in range(m):
        random_index = M*np.random.randint(m)
        xi = X[random_index:random_index+M]
        yi = y[random_index:random_index+M]
        gradients = (1.0/M)*training_gradient(yi, xi, theta)
        eta = learning_schedule(epoch*m+i)
        # calculate update
        new_change = eta*gradients+delta_momentum*change
        # take a step
        theta -= new_change
        # save the change
        change = new_change
print("theta from own sdg with momentum")
print(theta)

"""
# Using Autograd to calculate gradients using AdaGrad and Stochastic Gradient descent
# OLS example
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# Note change from previous example
def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)

n = 10000
x = np.random.rand(n,1)
y = 2.0+3*x +4*x*x# +np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x, x*x]
XT_X = X.T @ X
theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)


# Note that we request the derivative wrt third argument (theta, 2 here)
training_gradient = grad(CostOLS,2)
# Define parameters for Stochastic Gradient Descent
n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
# Guess for unknown parameters theta
theta = np.random.randn(3,1)

# Value for learning rate
eta = 0.01
# Including AdaGrad parameter to avoid possible division by zero
delta  = 1e-8
for epoch in range(n_epochs):
    # The outer product is calculated from scratch for each epoch
    Giter = np.zeros(shape=(3,3))
    for i in range(m):
        random_index = M*np.random.randint(m)
        xi = X[random_index:random_index+M]
        yi = y[random_index:random_index+M]
        gradients = (1.0/M)*training_gradient(yi, xi, theta)
	# Calculate the outer product of the gradients
        Giter +=gradients @ gradients.T
	# Simpler algorithm with only diagonal elements
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]
        # compute update
        update = np.multiply(Ginverse,gradients)
        theta -= update
print("theta from own AdaGrad")
print(theta)
"""
