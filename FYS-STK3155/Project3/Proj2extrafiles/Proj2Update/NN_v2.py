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

class NeuralNetwork:
    def __init__(self, X_data, Y_data, n_layers, n_nodes, sigma,
                sigma_d, epochs=100, batch_size=100, eta=0.1, lmbd=0):
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

        #initializing layers
        if isinstance(n_nodes, int):
            self.layers = [Layer(self.n_features, n_nodes, sigma, sigma_d)]
            for i in range(1, n_layers):
                self.layers.append(Layer(self.layers[i-1], n_nodes, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(Layer(self.layers[n_layers-1], self.n_outputs, sigma, sigma_d))
        else:
            self.layers = [Layer(self.n_features, n_nodes[0], sigma, sigma_d)]
            for i,n in enumerate(n_nodes[1:]):
                self.layers.append(Layer(self.layers[i-1], n, sigma, sigma_d))
            self.layers.append(Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d))

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta #learning rate
        self.lmbd = lmbd
        self.mseTest = [] #stores mse for each epoch on train data.

    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            z = np.matmul(a[-1], layer.get_weights) + layer.get_bias
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)

        self.output = a[-1] # (batch_size, nr_of_output_nodes=1), for regression

    def feedForwardOut(self, X):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        z = np.matmul(X, weights) + bias
        #print(z.shape, weights.shape, bias.shape)
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            weights = layer.get_weights
            bias = layer.get_bias
            z = np.matmul(a[-1], weights) + bias
            #print(z.shape, weights.shape, bias.shape)
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)
        return a[-1]
        #return z


    def backProp(self):
        Y_data = self.Y_data
        #print(self.output.shape)
        #print(Y_data.shape)
        error_output = (self.output - Y_data)*2 #Dervation of OLS
        #print(error_output.shape)
        #exit()

        error = [error_output]

        outLayer = self.layers[-1]
        w_grad_output = np.matmul(outLayer.prevLayer.get_a.T, error_output)
        w_grad = [w_grad_output]
        bias_grad_output = np.sum(error_output, axis=0)
        bias_grad = [bias_grad_output]

        weights_list = [outLayer.get_weights]
        bias_list = [outLayer.get_bias]

        #going through backwards
        for layer in reversed(self.layers[1:]):
            weights = layer.prevLayer.get_weights
            bias = layer.prevLayer.get_bias
            sigma_d = layer.prevLayer.sigma_d

            error.append(np.matmul(error[-1], layer.get_weights.T)*sigma_d(layer.prevLayer.get_z))

            ah = layer.prevLayer.get_a
            if isinstance(layer.prevLayer.prevLayer, int):
                ah = self.X_data
            w_grad.append(np.matmul(ah.T, error[-1]))

            bias_grad.append(np.sum(error[-1], axis=0))

            weights_list.append(weights)
            bias_list.append(bias)

        for i, layer in enumerate(reversed(self.layers)):
            #print(weights_list[i].shape)
            #print(f"b4 Layer wei shape {i}: {layer.get_weights.shape}")
            layer.get_weights = weights_list[i] - self.eta*w_grad[i]
            layer.get_bias = bias_list[i] - self.eta*bias_grad[i]
            #print(f"Layer wei shape {i}: {layer.get_weights.shape}")

    def train(self, X_test = None, Y_test = None, calcMSE = False):
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
            if calcMSE: #Per epoch calc MSE.
                #output_outlayer = self.predict(self.X_data_full)
                #mseT = mean_squared_error(self.Y_data_full, output_outlayer)
                output_outlayer = self.predict(X_test)
                mseTest_ = mean_squared_error(Y_test, output_outlayer)
                self.mseTest.append(mseTest_)
                #print(f"Train: {mseT}.   Test: {mseTest} diff: {abs(mseT-mseTest)}")

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output

    def get_MSEtest(self):
        arr_outEpochs = np.array(self.mseTest)
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
	#Scale data and return it
    scaler = StandardScaler()
    if len(X.data.shape) < 1:
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

n = 10000
x = np.linspace(0, 10, n*3)
z = np.linspace(0, 10, n*3)

def f(x):
    return 1 + 5*x + 3*x**2

#y = f(x)
y = FrankeFunction(x,z)
X = np.array([x,z]).T

X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=1/4)
X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)
ep = 100

#Sigmoid
dnn = NeuralNetwork(X_train_, Y_train_, 2, 16, relu, relu_deriv, epochs = ep, eta = 1e-5)
dnn.layers[-1].sigma = linear
dnn.layers[-1].sigma_d = linear_deriv
dnn.train(X_test_, Y_test_, calcMSE = True)
mse = dnn.get_MSEtest()
plt.plot(np.arange(ep), mse, label="mse")
plt.xlabel("epochs")
plt.ylabel("mse")
plt.legend()
plt.show()

test_predict = dnn.predict(X_test_)


"""
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


"""
Disussion algorithms:
speed
complexity
accuracy
tuneability
"""
