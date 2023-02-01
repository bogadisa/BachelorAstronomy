import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error

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

def RELU(x):
    return (np.maximum(0, x))

def RELU_deriv(x):
    x_ = (x > 0) * 1
    return x_

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
	#Scale data and return it
    scaler = StandardScaler()
    if len(X_train.shape) < 1:
        X_train_ = X_train.reshape(-1,1)
        X_test_ = X_test.reshape(-1,1)
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
    n = np.size(y_model)
#     print(y_data.shape, y_model.shape)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data)) ** 2)

#taken from lecture notes, slightly modifed to work for regularization parameters
#instead of n_neurons
def plot_data(x,y,data,title=None, Type="Classification"):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar=fig.colorbar(cax)
    

    # put text on matrix elements
    if Type == "Classification":
        cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
        cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
        cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])
        for i, x_val in enumerate(np.arange(len(x))):
            for j, y_val in enumerate(np.arange(len(y))):
                c = "${0:.1f}\\%$".format( 100*data[j,i])  
                ax.text(x_val, y_val, c, va='center', ha='center')
    elif Type == "Regression":
        cbar.ax.set_ylabel('R2 score',rotation=90,fontsize=fontsize)
        cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
        cbar.set_ticklabels(['<0','0.2','0.4','0.6','0.8','1'])
        for i, x_val in enumerate(np.arange(len(x))):
            for j, y_val in enumerate(np.arange(len(y))):
                if data[j,i] < 0:
                    c = "$<0$"
                else:
                    c = "${0:.3f}$".format(data[j,i])  
                ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{Learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{Regularization\\ parameter}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector