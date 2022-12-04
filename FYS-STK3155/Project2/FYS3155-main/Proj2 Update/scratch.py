from sklearn.linear_model import LinearRegression as OLS_reg
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autograd import grad
import numpy as np
import matplotlib.pyplot as plt


def CostOLS(y, X, beta):
    return (1/y.shape[0])*((y- X@beta).T)@(y- X@beta)

def CostRidge(y, X, beta, lambda_):
    return (1/y.shape[0])*((y- X@beta).T)@(y- X@beta) + lambda_*beta.T@beta

#Define the gradient with costfunction
gradient = grad(CostOLS, 2) #2 meaning beta

def gradient_decent(X, y, beta, lr, n_iter, momentum=0, batch_size=1, useAda=False, useRMS=False, useAdam=False, lambda_ = 0):
    MSE_list = [] #Store MSE scores every update to plot
    beta_list = [] #Store every time beta is updated
    change = 0
    m = int(n_iter/M) #number of minibatches
    y_pred = beta[0] + beta[1]*x + beta[2]*x*x
    MSE_list.append(mse(y, y_pred))
    beta_list.append(beta)
    if useAda:
        delta = 1e-8
        eta = 0.01
        for i in range(n_iter):
            Giter = np.zeros(shape=(3,3))
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]
                gradients = gradient(yi, xi, beta)

                # Calculate the outer product of the gradients
                Giter +=gradients @ gradients.T
                # Simpler algorithm with only diagonal elements
                Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]
                # compute update
                update = np.multiply(Ginverse,gradients)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)
                #Calculate MSE and store to list and plot.
                # momentum vs non-momentum
    elif useRMS:
        rho = 0.99
        eta = 0.01
        delta = 1e-8
        for i in range(n_iter):
            Giter = np.zeros(shape=(3,3))
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]
                gradients = gradient(yi, xi, beta)

                # Previous value for the outer product of gradients
                Previous = Giter
        	    # Accumulated gradient
                Giter +=gradients @ gradients.T
        	    # Scaling with rho the new and the previous results
                Gnew = (rho*Previous+(1-rho)*Giter)
        	    # Taking the diagonal only and inverting
                Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
        	    # Hadamard product
                update = np.multiply(Ginverse,gradients)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)

    elif useAdam:
        b1 = 0.9
        b2 = 0.999
        t = 0
        eps = 1e-8
        m_ = 0
        v = 0
        for i in range(n_iter):
            for i in range(m):
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                t = t + 1
                gradients = gradient(yi, xi, beta)
                m_ = b1*m_ + (1-b1)*gradients
                v = b2*v + (1-b2)*gradients**2
                m_hat = m_/(1-b1**t)
                v_hat = v/(1-b2**t)
                update = lr*m_hat/(np.sqrt(v_hat)+eps)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)

    else:
        for i in range(n_iter):
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]
                new_change = lr*gradient(yi, xi, beta) + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
                MSE_list.append(mse(y, y_pred))
                beta_list.append(beta)
                #Calculate MSE and store to list and plot.
                # momentum vs non-momentum
    #plt.plot(MSE_list)
    #plt.show()
    return beta, MSE_list, beta_list

n = 10000
x = np.random.rand(n,1)
#Analytical value. Static learning rate.
y = 2+3*x+4*x*x+0.1*np.random.rand(n,1)


X = np.c_[np.ones((n,1)), x, x**2] #design matrix
XT_X = X.T @ X
#theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H)
lr = 1.0/np.max(EigValues)

#X_train, X_test, y_train, y_test = train_test_split(X,z, test_size=0.2)
#Simple OLS fit
model = OLS_reg(fit_intercept=True)
model.fit(X, y)
y_pred = model.intercept_ + model.coef_[0][1]*x + model.coef_[0][2]*x*x

np.random.seed(200)
beta = np.random.randn(3,1)
np.random.seed(200)
beta_ada = np.random.randn(3,1)
np.random.seed(200)
beta_rms = np.random.randn(3,1)
np.random.seed(200)
beta_adam = np.random.randn(3,1)

#lr = 0.01
n_epochs = 120
M = 5   #size of each minibatch
beta, MSE_list, _ = gradient_decent(X, y, beta, lr, n_epochs, batch_size=M)
beta_ada, MSE_list_ada, _ = gradient_decent(X, y, beta_ada, lr, n_epochs, batch_size=M, useAda=True)
beta_rms, MSE_list_rms, _ = gradient_decent(X, y, beta_rms, lr, n_epochs, batch_size=M, useRMS=True)
beta_adam, MSE_list_adam, _ = gradient_decent(X, y, beta_adam, lr, n_epochs, batch_size=M, useAdam=True)
plt.plot(MSE_list, label="Default")
plt.plot(MSE_list_ada, label="Ada")
plt.plot(MSE_list_rms, label="RMS")
plt.plot(MSE_list_adam, label="Adam")
plt.legend()
plt.show()


"""
plt.plot(beta, label="Default")
plt.plot(beta_ada, label="Ada")
plt.plot(beta_rms, label="RMS")
plt.legend()
"""



print(beta)
print(beta_ada)
print(beta_rms)
print(beta_adam)
y_pred_grad = beta[0] + beta[1]*x + beta[2]*x*x
y_pred_grad_ada = beta_ada[0] + beta_ada[1]*x + beta_ada[2]*x*x
y_pred_grad_rms = beta_rms[0] + beta_rms[1]*x + beta_rms[2]*x*x

"""
plt.plot(x,y,".")
plt.plot(x, y_pred_grad, ".")
plt.plot(x, y_pred_grad_ada, ".")
plt.show()
"""

#NOTES:
# Opg a) juster hyperparametere slik at vi klart ser forskjell på performance.
# Prøv ulike learning rates, statisk.
# Adam : https://arxiv.org/abs/1412.6980
#Følg morten eks for nettverket.

#Lag nettverketet generalisert, e.g custom ant noder og lag.
#logistisk regresjon = nettverk med bare et output lag.
