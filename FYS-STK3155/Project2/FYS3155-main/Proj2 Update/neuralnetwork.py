#from scratch import gradient_decent
import numpy as np
from sklearn.model_selection import train_test_split

#Class neuralnetowrk, array of layers. Backprop function. Feed forward function.
#Class layer, stores weights in matrix. Stores index of layer, activation function.


#Backprop, computing the gradients.

#Two classes

#what is intrinsic to layer?
#wegihts and amount of nodes, bias
#so these are determined when a layer is initiated
#each node is also dependent on amount of features
#what about final layer?
#determined by categories
#how many categories?
#is it a layer in the traditional sense?





class Layer:
    def __init__(self, n_nodes, n_features, activation_func, func_derivative, bias=0.1, weights=None):
        self.n_nodes = n_nodes
        self.n_features = n_features
        if weights == None:
            self.initialize_weights()
        else:
            self.weights = weights
        if isinstance(bias, float):
            self.initialize_bias(bias)
        else:
            self.bias = bias

        self.activation_func = activation_func
        self.func_derivative = func_derivative

    def initialize_weights(self):
        self.weights = np.random.randn(self.n_features, self.n_nodes)

    def initialize_bias(self, bias):
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
    def get_n_nodes(self):
        return self.n_nodes

    @get_n_nodes.setter
    def get_n_nodes(self, n_nodes):
        self.n_nodes = n_nodes

    def get_z(self, a):
        weights = self.get_weights
        bias = self.get_bias
        #print(np.shape(bias), np.shape(weights), np.shape(a))
        z = np.matmul(a, weights) + bias
        return z

    def y(self, z):
        self.y = self.activation_func(z)
        return self.y

    @property
    def get_y(self):
        return self.y

    def get_u(self, y):
        weights = self.get_weights
        bias = self.get_bias

        u = np.matmul(y, weights) + bias

        return u

    @property
    def error(self):
        return self.error

    @error.setter
    def error(self, error):
        self.error = error



#NN?
#amount of hidden layers

class NeuralNetwork:
    def __init__(self, X_data, y_data, n_layers, n_nodes, n_categories, activation_funcs, funcs_derivative, epochs=10, batch_size=100, eta=0.1, lmbd=0):
        self.X_data_full = X_data
        self.y_data_full = y_data
        self.n_inputs = X_data.shape[0]
        if len(X_data.shape) == 2:
            self.n_features = X_data.shape[1]
        else:
            self.n_features = 1
        #print(self.n_features)

        self.layers = [Layer(n_nodes, self.n_features, activation_funcs, funcs_derivative)]
        if isinstance(n_nodes, int):
            for i in range(1, n_layers+1):
                self.layers.append(Layer(n_nodes, self.layers[i-1].get_n_nodes, activation_funcs, funcs_derivative))
        else:
            for i in n_nodes:
                self.layers.append(Layer(i, self.layers[i-1].get_n_nodes, activation_funcs, funcs_derivative))

        #output layer
        self.layers.append(Layer(n_categories, self.layers[i].get_n_nodes, activation_funcs, funcs_derivative))

        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_categories = n_categories
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

    def feed_forward(self, X_data):

        #input layer
        inputLayer = self.layers[0]
        z1 = inputLayer.get_z(X_data)

        #y = np.zeros((len(self.layers), z1.shape[0], z1.shape[1]))
        y = [inputLayer.y(z1)]

        for i, layer in enumerate(self.layers[1:]):
            u = layer.get_u(y[i])
            print(np.shape(u))
            
            y.append(layer.y(u))

        self.y = y

    def backProp(self, y_data):
        #(100, 10) (100,)
        #print(np.shape(self.y[-1]), np.shape(y_data))
        error_output = self.y[-1] - y_data
        self.layers[-1].error = error_output
        prevLayer = self.layers[-1]
        #-2 due to input and output layer, maybe it should be just -1 to include input layer
        for i in range(self.n_layers - 2): 
            layer = self.layers[-2-i]
            nextLayer = self.layers[-3-i]
            layer.error = np.matmul(prevLayer.error, prevLayer.get_weights)*layer.derivative(layer.get_z)
            layer.get_weights = layer.get_weights - self.eta*layer.error*nextLayer.get_y
            layer.get_bias = layer.get_bias - self.eta*layer.error
            prevLayer = layer

    def train(self):
        data_incides = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints = np.random.choice(data_incides, size=self.batch_size, replace=False)
                #print(np.shape(chosen_datapoints))
                self.X_data = self.X_data_full[chosen_datapoints]
                self.y_data = self.y_data_full[chosen_datapoints]

                self.feed_forward(self.X_data)
                self.backProp(self.y_data)

    def feed_forward_out(self, X):

        #input layer
        inputLayer = self.layers[0]
        z1 = inputLayer.get_z(X)

        #y = np.zeros((len(self.layers), z1.shape[0], z1.shape[1]))
        y = [inputLayer.y(z1)]

        for i, layer in enumerate(self.layers[1:]):
            u = layer.get(y[i])
            y.append(layer.y(u))

        return y

    def predict(self, X):
        return self.feed_forward_out(X)

