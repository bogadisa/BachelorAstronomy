import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from copy import deepcopy

seed = 32455
#different activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    sig_x  = sigmoid(x)
    return sig_x*(1 - sig_x)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x)**2

def relu(x):
    return (np.maximum(0, x))

def relu_deriv(x):
    x_ = (x > 0) * 1
    return x_

def leaky_relu(x):
    x = np.where(x > 0, x, x * 0.01)
    return x

def leaky_relu_deriv(x):
    alpha = 0.1
    x = np.where(x > 0, alpha, 1)
    return x

def elu(x, alpha=0.01):
    xexp = np.exp(x)
    return np.where(x<0, alpha*(xexp - 1), x)

def elu_deriv(x, alpha=0.01):
    return np.where(x<0, alpha*np.exp(x), 1)

def linear(x):
    return x

def linear_deriv(x):
    return 1

def softmax(z):
    exp_term = np.exp(z)
    return exp_term/np.sum(exp_term, axis=1, keepdims=True)

#some functions used to generate data
def f(x):
    return 1 + 5*x + 3*x**2

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#In case we want to scale our data
def scale(X_train, X_test, Y_train, Y_test):
    """
    Scales the data for train and test using sklearn StandardScaler.

    Args:
        X_train (ndarray) : Array containing training data for features
        X_test (ndarray) : Array containing test data for features
        Y_test (ndarray) : Array containing training data for targets.
        Y_test (ndarray) : Array containing test data for targets.


    Returns:
        X_train_ (ndarray) : Scaled version of X_train
        X_test_ (ndarray) : Scaled version of X_test
        Y_test_ (ndarray) : Scaled version of Y_train
        Y_test_ (ndarray) : Scaled version of Y_test
    """
	#Scale data and return it
    X_train_ = deepcopy(X_train); X_test_ = deepcopy(X_test)
    Y_train = deepcopy(Y_train); Y_test = deepcopy(Y_test)
    scaler = StandardScaler()
    if len(X_train.shape) < 1:
        X_train_ = X_train_.reshape(-1,1)
        X_test_ = X_test_.reshape(-1,1)
    else:
        X_train_ = X_train
        X_test_ = X_test
    Y_train_ = Y_train.reshape(-1,1)
    Y_test_ = Y_test.reshape(-1,1)

    scaler.fit(X_train_)
    X_train_ = scaler.transform(X_train_)
    X_test_ = scaler.transform(X_test_)

    scaler.fit(Y_train_)
    Y_train_ = scaler.transform(Y_train_)
    Y_test_ = scaler.transform(Y_test_)

    return X_train_, X_test_, Y_train_, Y_test_

def MSE(y_data,y_model):
    """
    Returns Mean squared error for the target data vs model predictions. Lower meaning better
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    """
    Returns the R2 score for the target data vs model predictions. Zero to one, one meaning best.
    """
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data)) ** 2)
