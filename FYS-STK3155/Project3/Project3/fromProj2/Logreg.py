import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
from NeuralNetwork import Layer

seed = 32455
np.random.seed(seed)

class LogRegClass:
    """
    LogisticRegression class which is a NeuralNetwork network without hidden layers. Additionally
    a stepwise function is added to the output as the model will just output either 0 or 1.
    """
    def __init__(self, X_data, Y_data, sigma, sigma_d, epochs=100, batch_size=100, eta=0.1, lmbd=0):
        if len(X_data.shape) == 2:
            self.X_data_full = X_data
        else:
            self.X_data_full = X_data.reshape(-1, 1)
        if len(Y_data.shape) == 2:
            self.Y_data_full = Y_data
        else:
            self.Y_data_full = Y_data.reshape(-1, 1)

        np.random.seed(seed)
        self.n_inputs = self.X_data_full.shape[0]
        self.n_features = self.X_data_full.shape[1]
        self.n_outputs = self.Y_data_full.shape[1]

        self.layers = [Layer(self.n_features, self.n_outputs, sigma, sigma_d)]
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta #learning rate
        self.lmbd = lmbd
        self.accTest = []
        self.accTrain = []

        self.lossTest = []
        self.lossTrain = []

    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        z_ = layer1.sigma(z)
        layer1.get_a = z_
        #layer1.get_a = np.piecewise(z_, [z_ < 0.5, z_ >= 0.5], [0, 1])
        a = [layer1.get_a]
        self.output = a[-1]
        #self.output = z_

    def feedForwardOut(self, X):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias
        z = np.matmul(X, weights) + bias
        layer1.get_z = z
        z_ = layer1.sigma(z)
        layer1.get_a = np.piecewise(z_, [z_ < 0.5, z_ >= 0.5], [0, 1]) # 0 if activation is lower than .5, 1 if higher
        a = [layer1.get_a]
        return a

    def backProp(self):
        #Assumes just a input and output layer.
        Y_data = self.Y_data
        error_output = (self.output - Y_data) #Dervation of OLS
        error = [error_output]
        outLayer = self.layers[-1]

        ah = self.X_data
        w_grad = np.matmul(ah.T, error[-1])

        bias_grad_output = np.sum(error_output, axis=0)
        bias_grad = bias_grad_output

        weights_ = outLayer.get_weights
        bias_ = outLayer.get_bias

        outLayer.get_weights = weights_ - self.eta*(w_grad + self.lmbd*weights_*2)
        outLayer.get_bias = bias_ - self.eta*(bias_grad + self.lmbd*bias_*2)

    def train(self, X_test = None, Y_test = None, calcAcc = False):
        data_indices = np.arange(self.n_inputs)
        #Loop over epochs(i), with minibatches = batch_size, train network with backProp
        k = 0
        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_data_points = np.random.choice(data_indices, size=self.batch_size, replace=False)
                #chosen_data_points = np.random.choice(data_indices, size=self.X_data_full.shape[0], replace=False)

                self.X_data = self.X_data_full[chosen_data_points]
                self.Y_data = self.Y_data_full[chosen_data_points]

                self.feedForward()
                self.backProp()
            if calcAcc: #Per epoch calc MSE.
                predTrain = self.predict(self.X_data_full)
                predTrain = np.ravel(predTrain[0])
                y_data = np.ravel(self.Y_data_full)
                accTr = accuracy_score(y_data, predTrain)
                self.accTrain.append(accTr)

                pred = self.predict(X_test)
                pred = np.ravel(pred[0])
                accT = accuracy_score(Y_test, pred)
                self.accTest.append(accT)

                #y_tilde = self.output
                #y = Y_test # y_train
                #loss = -np.mean(y*np.log(y_tilde+1e-9) + (1-y)*np.log(1-y_tilde+1e-9))
                #self.lossTest.append(loss)
                """
                y_tilde = self.output
                y = Y_test
                loss = -np.mean(y*np.log(y_tilde) + (1-y)*np.log(1-y_tilde))
                """
        #print(self.accTest[-1])

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output


    def get_accTrain(self):
        return self.accTrain

    def get_accTest(self):
        return self.accTest
