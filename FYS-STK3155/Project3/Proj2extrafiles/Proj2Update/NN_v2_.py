import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score

seed = 32455
np.random.seed(seed)

class Layer:
    def __init__(self, prevLayer, n_nodes, sigma, simga_d, bias=0.1):
        self.n_nodes = n_nodes
        self.prevLayer = prevLayer

        if isinstance(prevLayer, Layer):
            self.n_weights = prevLayer.n_nodes
        else:
            self.n_weights = prevLayer

        self.init_weights()
        self.init_bias(bias)

        self.sigma = sigma
        self.sigma_d = simga_d

    def init_weights(self):
        self.weights = np.random.randn(self.n_weights, self.n_nodes)

    def init_bias(self, bias):
        self.bias = np.zeros(self.n_nodes) + bias

    def __str__(self):
        return f"{self.z.shape}, {self.weights.shape}, {self.bias.shape}"

    @property
    def get_bias(self):
        return self.bias

    @get_bias.setter
    def get_bias(self, bias):
        self.bias = bias

    @property
    def get_weights(self):
        return self.weights

    @get_weights.setter
    def get_weights(self, weights):
        self.weights = weights

    @property
    def get_z(self):
        return self.z

    @get_z.setter
    def get_z(self, z):
        self.z = z

    @property
    def get_a(self):
        return self.a

    @get_a.setter
    def get_a(self, a):
        self.a = a

class LinRegClass:
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
        self.accuracyTest = []

    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        z_ = layer1.sigma(z)
        layer1.get_a = np.piecewise(z_, [z_ < 0.5, z_ >= 0.5], [0, 1])
        a = [layer1.get_a]
        self.output = a[-1]

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
        error_output = (self.output - Y_data)*2 #Dervation of OLS
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



    def train(self, X_test = None, Y_test = None, calcAccuracy = False):
        data_indices = np.arange(self.n_inputs)
        #Loop over epochs(i), with minibatches = batch_size, train network with backProp
        k = 0
        for i in range(self.epochs):
            if i == self.epochs-1:
                print(f"Epochs {self.epochs}/{self.epochs}")
                print("Done.\n")
            if i>=k:
                print(f"Epochs {i}/{self.epochs}")
                k += int(self.epochs/5)

            for j in range(self.iterations):
                chosen_data_points = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[chosen_data_points]
                self.Y_data = self.Y_data_full[chosen_data_points]

                self.feedForward()
                self.backProp()
            if calcAccuracy: #Per epoch calc MSE.
                #output_outlayer = self.predict(self.X_data_full)
                #mseT = mean_squared_error(self.Y_data_full, output_outlayer)
                pred = self.predict(X_test)
                pred = np.ravel(pred[0])
                accT = accuracy_score(Y_test, pred)
                self.accuracyTest.append(accT)
                #print(f"Train: {mseT}.   Test: {mseTest} diff: {abs(mseT-mseTest)}")

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output

    def get_ACCtest(self):
        arr_outEpochs = np.array(self.accuracyTest)
        return arr_outEpochs

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    sig_x  = sigmoid(x)
    return sig_x*(1 - sig_x)

def relu(x):
    return (np.maximum(0, x))

def relu_deriv(x):
    x_ = (x > 0) * 1
    return x_

def leaky_relu(x):
    if x>0:
        return x
    else:
        return 0.01*x

def tanh_function(x):
    z = (2/(1 + np.exp(-2*x))) -1
    return z

def tanh_deriv(x):
    return 1 - (tanh_function(x))**2

def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_

def linear(x):
    return x

def linear_deriv(x):
    return 1

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def scale(X_train, X_test, Y_train, Y_test):
	#Scale data and return it + mean value from target train data.
    scaler = StandardScaler()
    X_train_ = X_train.reshape(-1,1)
    X_test_ = X_test.reshape(-1,1)
    Y_train_ = Y_train.reshape(-1,1)
    Y_test_ = Y_test.reshape(-1,1)

    scaler.fit(X_train_)
    X_train_ = scaler.transform(X_train_)
    X_test_ = scaler.transform(X_test_)

    scaler.fit(Y_train_)
    Y_train_ = scaler.transform(Y_train_)
    Y_test_ = scaler.transform(Y_test_)

    return X_train_, X_test_, Y_train_, Y_test_


from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=200,n_features=2,centers=2,random_state=6,cluster_std=1.3)
#plt.scatter(X, y)
#plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=1/4)

X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)


ep = 300

dnn = LinRegClass(X_train, Y_train, sigmoid, sigmoid_deriv, epochs = ep, eta = 0.00001, lmbd=0)
dnn.layers[-1].sigma = sigmoid
dnn.layers[-1].sigma_d = sigmoid_deriv
dnn.train(X_train, Y_train, calcAccuracy=True)
pred = dnn.predict(X_test)
pred = np.ravel(pred[0])

acc = dnn.get_ACCtest()
print(acc)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))

ax1.scatter(X_test[:,0], pred, label="predi")
ax1.scatter(X_test[:,0], Y_test, label="data", marker="x")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Value")
ax2.scatter(X_test[:,1], pred, label="predi")
ax2.scatter(X_test[:,1], Y_test, label="data", marker="x")
ax2.set_xlabel("Feature 2")
plt.legend()
plt.savefig("Logreg 2 features.png", dpi=300)
plt.show()


#dnn = NeuralNetwork(X_train_, Y_train_, 0, 0, sigmoid, sigmoid_deriv, epochs = ep, eta = 0.001)


"""
#Sigmoid
dnn = NeuralNetwork(X_train_, Y_train_, 2, 16, sigmoid, sigmoid_deriv, epochs = ep, eta = 0.001)
dnn.layers[-1].sigma = linear
dnn.layers[-1].sigma_d = linear_deriv
dnn.train(X_test_, Y_test_, calcMSE = True)
test_predict = dnn.predict(X_test_)



plt.scatter(X_test_, Y_test_, label="Actual", c="r")
plt.scatter(X_test_, test_predict, label="Model", alpha = 0.5)
plt.legend()
plt.savefig("25 ep sigm")
plt.show()
"""

"""
#RELU
dnn1 = NeuralNetwork(X_train_, Y_train_, 2, 16, relu, relu_deriv, epochs = ep, eta = 0.0001)
dnn1.layers[-1].sigma = linear
dnn1.layers[-1].sigma_d = linear_deriv
dnn1.train(X_test_, Y_test_, calcMSE = True)
test_predict = dnn1.predict(X_test_)


#Tanh
dnn2 = NeuralNetwork(X_train_, Y_train_, 2, 16, tanh_function, tanh_deriv, epochs = ep, eta = 0.0001)
dnn2.layers[-1].sigma = linear
dnn2.layers[-1].sigma_d = linear_deriv
dnn2.train(X_test_, Y_test_, calcMSE = True)
test_predict = dnn2.predict(X_test_)

#MSE vs epochs on training
mse = dnn.get_MSEtest()
mse1 = dnn1.get_MSEtest()
mse2 = dnn2.get_MSEtest()

plt.yscale("log")
plt.plot(np.arange(ep), mse, label = "Sigmoid lr: 0.001")
plt.plot(np.arange(ep), mse1, label = "RELU lr: 0.0001")
plt.plot(np.arange(ep), mse2, label = "Tanh lr: 0.0001")
plt.legend()
plt.title(f"Activation funcs : epochs {ep}")
#plt.savefig(f"Act funcs ep_{ep}", dpi=300)
plt.show()
"""
