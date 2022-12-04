import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    def __init__(self, X_data, Y_data, n_layers, n_nodes, sigma, sigma_d, epochs=10, batch_size=100, eta=0.1, lmbd=0):
        if len(X_data.shape) == 2:
            self.X_data_full = X_data
        else:
            self.X_data_full = X_data.reshape(-1, 1)
        if len(Y_data.shape) == 2:
            self.Y_data_full = Y_data
        else:
            self.Y_data_full = Y_data.reshape(-1, 1)

        self.n_inputs = self.X_data_full.shape[0]
        self.n_features = self.X_data_full.shape[1]
        self.n_outputs = self.Y_data_full.shape[1]
        # if len(X_data.shape) == 2:
        #     self.n_features = X_data.shape[1]
        # else:
        #     self.n_features = 1

        # if len(X_data.shape) == 2:
        #     self.n_outputs = Y_data.shape[1]
        # else:
        #     self.n_outputs = 1
        



        #initializing layers
        if isinstance(n_nodes, int):
            self.layers = [Layer(self.n_features, n_nodes, sigma, sigma_d)]
            for i in range(1, n_layers+1):
                self.layers.append(Layer(self.layers[i-1], n_nodes, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d))
        else:
            self.layers = [Layer(self.n_features, n_nodes[0], sigma, sigma_d)]
            for i in n_nodes[1:]:
                self.layers.append(Layer(self.layers[i-1], i, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d))


        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd


    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        #print(np.shape(bias), np.shape(weights), np.shape(self.X_data))
        #z = np.matmul(weights, self.X_data) + bias
        # z = np.matmul(self.X_data, weights) + bias
        # a = [z]
        # layer1.get_a = z
        # for layer in self.layers[1:]:
        #     layer.get_z = z
        #     al = layer.sigma(z)
        #     layer.get_a = al
        #     a.append(al)
        #     z = al

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            z = np.matmul(a[-1], layer.get_weights) + layer.get_bias
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)

        self.output = a[-1]
        #self.a = np.array(a)

    def feedForwardOut(self, X):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        # print(np.shape(bias), np.shape(weights), np.shape(X.T))
        # z = np.matmul(X.T, weights) + bias
        # print(z.shape, np.matmul(X.T, weights).shape)
        # a = [z]
        # layer1.get_a = z
        # for layer in self.layers[1:]:
        #     layer.get_z = z
        #     al = layer.sigma(z)
        #     layer.get_a = al
        #     a.append(al)
        #     z = al

        z = np.matmul(X, weights) + bias
        print(z.shape, weights.shape, bias.shape)
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            weights = layer.get_weights
            bias = layer.get_bias
            z = np.matmul(a[-1], weights) + bias
            print(z.shape, weights.shape, bias.shape)
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)

        return z

    def backProp(self):
        Y_data = self.Y_data

        error_output = self.output - Y_data
        error = [error_output]

        outLayer = self.layers[-1]
        w_grad_output = np.matmul(outLayer.prevLayer.get_a, error[-1])
        w_grad = [w_grad_output]
        bias_grad_output = np.sum(error_output, axis=1)
        bias_grad = [bias_grad_output]

        weights_list = [outLayer.get_weights]
        bias_list = [outLayer.get_bias]

        #going through backwards
        for layer in reversed(self.layers[1:]):
            weights = layer.get_weights
            bias = layer.get_bias
            sigma_d = layer.prevLayer.sigma_d

            error.append(np.matmul(error[-1], weights.T)*sigma_d(layer.prevLayer.get_z))

            ah = layer.prevLayer.get_a
            w_grad.append(np.matmul(ah.T, error[-1]))

            bias_grad.append(np.sum(error[-1], axis=1))

            weights_list.append(weights)
            bias_list.append(bias)

        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias
        sigma_d = layer1.sigma_d
        error.append(np.matmul(error[-1], weights.T)*sigma_d(layer.prevLayer.get_z))

        ah = self.X_data
        w_grad.append(np.matmul(ah.T, error[-1]))
        bias_grad.append(np.sum(error[-1], axis=1))
        weights_list.append(weights)
        bias_list.append(bias)

        for i, layer in enumerate(reversed(self.layers)):
            layer.get_weights = weights_list[i] - self.eta*w_grad[i]
            layer.get_bias = bias_list[i] - self.eta*bias_grad[i]



        # for layer in reversed(self.layers[1:]):
        #     error.append(np.matmul(error[-1], layer.prevLayer.get_weights.T)*layer.sigma_d(layer.get_z))
        #     #error.append(np.sum(error[-1]*layer.prevLayer.get_weights*layer.sigma_d(layer.get_z))) #, axis=?
        #     #weights_grad = self.eta*np.matmul(error[-1].T, layer.prevLayer.get_a)
        #     #or
        #     weights_grad = self.eta*np.matmul(layer.prevLayer.get_a.T, error[-1])
        #     layer.get_weights = layer.get_weights - weights_grad
        #     layer.get_bias = layer.get_bias - self.eta*error[-1]

        
    def train(self):
        data_indices = np.arange(self.n_inputs)
        
        for i in range(self.epochs):
            for j in range(self.iterations):
                #(750,) 100
                #print(data_indices.shape, self.batch_size)
                chosen_data_points = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[chosen_data_points]
                self.Y_data = self.Y_data_full[chosen_data_points]

                self.feedForward()
                self.backProp()

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output


n = 1000
x = np.linspace(0, 10, n)

def f(x):
    return 1 + 5*x + 3*x**2 

y = f(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    sig_x  = sigmoid(x)
    return sig_x*(1 - sig_x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)

dnn = NeuralNetwork(X_train, Y_train, 2, 40, sigmoid, sigmoid_deriv)

dnn.train()

test_predict = dnn.predict(X_test)
print(test_predict.shape)
print(Y_test.shape)
plt.scatter(X_test, Y_test, label="Actual", c="r")
plt.scatter(X_test, test_predict, label="Model")
plt.legend()
plt.show()

"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(12)

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
#         print(self.weights)
        
    def init_bias(self, bias):
        self.bias = np.zeros(self.n_nodes) + bias

    @property
    def get_bias(self):
        return self.bias

    @get_bias.setter
    def get_bias(self, bias):
        if isinstance(bias, np.ndarray):
            self.bias = bias
        else:
            self.init_bias(bias)
    
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
    def __init__(self, X_data, Y_data, n_layers, n_nodes, sigma, sigma_d, epochs=10, batch_size=100, eta=0.1, lmbd=0):
        if len(X_data.shape) == 2:
            self.X_data_full = X_data
        else:
            self.X_data_full = X_data.reshape(-1, 1)
        if len(Y_data.shape) == 2:
            self.Y_data_full = Y_data
        else:
            self.Y_data_full = Y_data.reshape(-1, 1)

        self.n_inputs = self.X_data_full.shape[0]
        self.n_features = self.X_data_full.shape[1]
        self.n_outputs = self.Y_data_full.shape[1]
        # if len(X_data.shape) == 2:
        #     self.n_features = X_data.shape[1]
        # else:
        #     self.n_features = 1

        # if len(X_data.shape) == 2:
        #     self.n_outputs = Y_data.shape[1]
        # else:
        #     self.n_outputs = 1
        



        #initializing layers
        if isinstance(n_nodes, int):
            self.layers = [Layer(self.n_features, n_nodes, sigma, sigma_d)]
            for i in range(1, n_layers):
                self.layers.append(Layer(self.layers[i-1], n_nodes, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.outputLayer = Layer(self.layers[n_layers-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(self.outputLayer)
        else:
            self.layers = [Layer(self.n_features, n_nodes[0], sigma, sigma_d)]
            for i, n in enumerate(n_nodes[1:]):
                self.layers.append(Layer(self.layers[i-1], n, sigma, sigma_d))
            # self.output_layer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.outputLayer = Layer(self.layers[i-1], self.n_outputs, sigma, sigma_d)
            self.layers.append(self.outputLayer)
            
        self.layers = np.array(self.layers, dtype=Layer)

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd


    def change_bias(self, bias):
        for layer in self.layers:
            layer.get_bias = bias
                
        
    def feedForward(self):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        #print(np.shape(bias), np.shape(weights), np.shape(self.X_data))
        #z = np.matmul(weights, self.X_data) + bias
        # z = np.matmul(self.X_data, weights) + bias
        # a = [z]
        # layer1.get_a = z
        # for layer in self.layers[1:]:
        #     layer.get_z = z
        #     al = layer.sigma(z)
        #     layer.get_a = al
        #     a.append(al)
        #     z = al

        z = np.matmul(self.X_data, weights) + bias
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z, self.X_data_full.shape[1])
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            weights = layer.get_weights
            bias = layer.get_bias
            z = np.matmul(a[-1], weights) + bias
            #z = np.dot(weights, a[-1]) + bias
            #print(z.shape, weights.shape, bias.shape)
            layer.get_z = z
            layer.get_a = layer.sigma(z, self.X_data_full.shape[1])
            a.append(layer.get_a)

        self.output = a[-1]
        exit()
        #self.output = z
        #self.a = np.array(a)

    def feedForwardOut(self, X):
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        # print(np.shape(bias), np.shape(weights), np.shape(X.T))
        # z = np.matmul(X.T, weights) + bias
        # print(z.shape, np.matmul(X.T, weights).shape)
        # a = [z]
        # layer1.get_a = z
        # for layer in self.layers[1:]:
        #     layer.get_z = z
        #     al = layer.sigma(z)
        #     layer.get_a = al
        #     a.append(al)
        #     z = al

        z = np.matmul(X, weights) + bias
        #print(z.shape, weights.shape, bias.shape)
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z, self.X_data_full.shape[1])
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            weights = layer.get_weights
            bias = layer.get_bias
            z = np.matmul(a[-1], weights) + bias
            #print(z.shape, weights.shape, bias.shape)
            layer.get_z = z
            layer.get_a = layer.sigma(z, self.X_data_full.shape[1])
            a.append(layer.get_a)

        return z

    def backProp(self):
        
        Y_data = self.Y_data
        outLayer = self.layers[-1]

        error_output = (self.output - Y_data)*outLayer.sigma_d(outLayer.get_z, self.X_data_full.shape[1])
        error = [error_output]

        w_grad_output = np.matmul(outLayer.prevLayer.get_a.T, error[-1])
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

            error.append(np.matmul(error[-1], layer.get_weights.T)*sigma_d(layer.prevLayer.get_z, self.X_data_full.shape[1]))

            ah = layer.prevLayer.get_a
            if isinstance(layer.prevLayer.prevLayer, int):
                ah = self.X_data
            w_grad.append(np.matmul(ah.T, error[-1]))

            bias_grad.append(np.sum(error[-1], axis=0))

            weights_list.append(weights)
            bias_list.append(bias)

#         layer1 = self.layers[0]
#         weights = layer1.get_weights
#         bias = layer1.get_bias
#         sigma_d = layer1.sigma_d
#         error.append(np.matmul(error[-1], weights.T)*sigma_d(layer.prevLayer.get_z))

#         ah = self.X_data
#         w_grad.append(np.matmul(ah.T, error[-1]))
#         bias_grad.append(np.sum(error[-1], axis=0))
#         weights_list.append(weights)
#         bias_list.append(bias)
        #print(len(bias_list))
        for i, layer in enumerate(reversed(self.layers)):
            #print(f"Layer {i} weight shape={weights_list[i].shape}")
            layer.get_weights = weights_list[i] - self.eta*w_grad[i]
            layer.get_bias = bias_list[i] - self.eta*bias_grad[i]

        
    def train(self):
        data_indices = np.arange(self.n_inputs)
        
        for i in range(self.epochs):
            for j in range(self.iterations):
                #print(j)
                #(750,) 100
                #print(data_indices.shape, self.batch_size)
                chosen_data_points = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[chosen_data_points]
                self.Y_data = self.Y_data_full[chosen_data_points]

                self.feedForward()
                self.backProp()

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output
    
    def evaluate(self, X, y):
        return False


n = 1000
x = np.linspace(-1, 1, n)

def f(x):
    return 1+ 5*x - 3*x**2 

y = f(x)"""