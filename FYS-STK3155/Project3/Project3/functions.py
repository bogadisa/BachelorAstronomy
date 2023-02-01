import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.backend as K
import tensorflow as tf
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
    return K.mean((y_data-y_model)**2)

def R2(y_data, y_model):
    """
    Returns the R2 score for the target data vs model predictions. Zero to one, one meaning best.
    """
    return 1 - K.sum((y_data - y_model)**2) / K.sum((y_data - K.mean(y_data)) ** 2)

def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.div(residual, total))
  return r2

def keras_calculate_mcc_from_conf(confusion_m):
    """tensor version of MCC calculation from confusion matrix"""
    # as in Gorodkin (2004)
    N = K.sum(confusion_m)
    up = N * tf.linalg.trace(confusion_m) - K.sum(tf.matmul(confusion_m, confusion_m))
    down_left = K.sqrt(N ** 2 - K.sum(tf.matmul(confusion_m, K.transpose(confusion_m))))
    down_right = K.sqrt(N ** 2 - K.sum(tf.matmul(K.transpose(confusion_m), confusion_m)))
    mcc_val = up / (down_left * down_right + K.epsilon())
    return mcc_val


def keras_better_to_categorical(y_pred_in):
    """tensor version of to_categorical"""
    nclass = K.shape(y_pred_in)[1]
    y_pred_argmax = K.argmax(y_pred_in, axis=1)
    y_pred = tf.one_hot(tf.cast(y_pred_argmax, tf.int32), depth=nclass)
    y_pred = tf.cast(y_pred, tf.float32)
    return y_pred

def mcc(y_true, y_pred):
    """To calculate Matthew's correlation coefficient for multi-class classification"""
    # this is necessary to make y_pred values of 0 or 1 because
    # y_pred may contain other value (e.g., 0.6890)
    y_pred = keras_better_to_categorical(y_pred)

    # now it's straightforward to calculate confusion matrix and MCC
    confusion_m = tf.matmul(K.transpose(y_true), y_pred)
    return keras_calculate_mcc_from_conf(confusion_m)