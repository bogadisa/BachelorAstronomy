from functions import *



class Layer:
    def __init__(self, prevLayer, n_nodes, sigma, simga_d, bias=0):
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
        self.weights = np.random.randn(self.n_weights, self.n_nodes) + 0.001

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
                sigma_d, epochs=1000, batch_size=100, eta=0.1, lmbd=0, Type="Regression"):
        np.random.seed(seed)
        #making sure the shape of our data is correct
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

        #initializing layers
        #checking if you want to have a different amount of nodes per layer
        if isinstance(n_nodes, int) or isinstance(n_nodes, np.int32):
            #first layer is initialized a bit differently
            self.layers = [Layer(self.n_features, n_nodes, sigma, sigma_d)]
            for i in range(1, n_layers):
                self.layers.append(Layer(self.layers[i-1], n_nodes, sigma, sigma_d))
            #output layer is also initialized differently
            self.layers.append(Layer(self.layers[n_layers-1], self.n_outputs, sigma, sigma_d))
        else:
            #same thing only with custom nodes per layer
            self.layers = [Layer(self.n_features, n_nodes[0], sigma, sigma_d)]
            for i,n in enumerate(n_nodes[1:]):
                self.layers.append(Layer(self.layers[i], n, sigma, sigma_d))

            self.layers.append(Layer(self.layers[i], self.n_outputs, sigma, sigma_d))

        #taken from lecture notes
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        #saving what kind of problem we have for later
        self.Type = Type

    #feeds the input data forward
    def feedForward(self):
        #1st layer works a bit differently
        layer1 = self.layers[0]
        weights = layer1.get_weights
        bias = layer1.get_bias

        z = np.matmul(self.X_data, weights) + bias
        #we want to be able to access these values at later points
        layer1.get_z = z
        layer1.get_a = layer1.sigma(z)
        a = [layer1.get_a]

        for layer in self.layers[1:]:
            z = np.matmul(a[-1], layer.get_weights) + layer.get_bias
            layer.get_z = z
            layer.get_a = layer.sigma(z)
            a.append(layer.get_a)

        self.output = a[-1] # (batch_size, nr_of_output_nodes=1), for regression

    #does the same as feedForward, only it takes a input and returns a value instead
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

    #performs the back propagation
    def backProp(self):
        Y_data = self.Y_data

        #cost function
        error_output = self.output - Y_data

        #want to be able to easily access all these data later
        error = [error_output]
        #again, the first layer is handled a bit differently
        #we also initialize our lists of values for later use
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
            #our first hidden layer is defined a bit differently
            if isinstance(layer.prevLayer.prevLayer, int):
                ah = self.X_data
            w_grad.append(np.matmul(ah.T, error[-1]))

            bias_grad.append(np.sum(error[-1], axis=0))

            weights_list.append(weights)
            bias_list.append(bias)


        #update our gradients, with learning rate and regularization parameter
        for i, layer in enumerate(reversed(self.layers)):
            layer.get_weights = weights_list[i] - self.eta*w_grad[i] + 2*self.lmbd*weights_list[i]
            layer.get_bias = bias_list[i] - self.eta*bias_grad[i] + 2*self.lmbd*bias_list[i]

    #calculates MSE throughout train. usefull for debugging
    def backProp_err(self):
        Y_data = self.Y_data
        error_output = self.output - Y_data

        outLayer = self.layers[-1]

        error_hidden = np.matmul(error_output, outLayer.get_weights.T) * outLayer.get_a * (1 - outLayer.get_a)

        output_weights_gradient = np.matmul(outLayer.prevLayer.get_a.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)

        outLayer.get_weights -= self.eta * output_weights_gradient
        outLayer.get_bias -= self.eta * output_bias_gradient
        self.layers[-2].get_weights -= self.eta * hidden_weights_gradient
        self.layers[-2].get_bias -= self.eta * hidden_bias_gradient

    #taken from lecture notes
    #traiins the network
    def train(self, X_test = None, Y_test = None, calcMSE = False):
        if calcMSE:
            self.mseTest = [] #stores mse for each epoch on train data.
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

    #predicts
    def predict(self, X):
        #again we have to reshape our data
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        output = self.feedForwardOut(X)
        return output

    #only used for clasification problems
    def evaluate(self, X, Y, Onehot=True):
        Y = np.copy(Y)
        if self.Type == "Regression":
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            output = self.feedForwardOut(X)
            output = output[~np.isnan(output)]
            Y = Y[~np.isnan(output)]
            return R2(Y, output)
        elif self.Type == "Classification":
            prediction = self.predict(X)

            #convert back from one-hot vectors
            if Onehot:
                Y = np.argmax(Y, axis=1)
                prediction = np.argmax(prediction, axis=1)

            prediction = prediction[~np.isnan(prediction)]
            Y = Y[~np.isnan(prediction)]
            return accuracy_score(Y, prediction)

    def get_MSEtest(self):
        arr_outEpochs = np.array(self.mseTest)
        return arr_outEpochs
