from scratch import gradient_decent
import numpy as np

#Class neuralnetowrk, array of layers. Backprop function. Feed forward function.
#Class layer, stores weights in matrix. Stores index of layer, activation function.


#Backprop, computing the gradients.


class NeuralNetwork:
    def __init__(self,
            X_data,
            Y_data,
            numLayers,
            numNodesLay ,
            nodesInput,
            nodesOut,

            epochs = 10,
            batch_size = 100,
            lr = 0.05,
            lmbd = 0.0):
        self.layers = np.zeros(0, dtype = Layer)
        self.X_data = X_data
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.Y_data = Y_data

        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lmbd = lmbd

        self.numLayers = numLayers
        self.numNodesHiddenLays = numNodesLay
        self.nodesInput = nodesInput
        self.nodesOut = nodesOut

        self.initializeLayers()

    def initializeLayers(self):
        for i in range(self.numLayers):
            if i == 0:
                layer(0, self.numNodesLay[i], sigmoid, self.n_features)
            else:
                layer(self.numNodesLay[i-1], self.numNodesLay[i], sigmoid, self.n_features)

        for layer in self.layers:
            layer.create_weightsandbias()

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    @staticmethod
    def relu(x):
        return max(0.0, x)
    @staticmethod
    def leaky_relu(x):
        if x>0::
            return x
        else:
            return 0.01*x
    @staticmethod
    def tanh_function(x):
        z = (2/(1 + np.exp(-2*x))) -1
        return z
    @staticmethod
    def softmax_function(x):
        z = np.exp(x)
        z_ = z/z.sum()
        return z_

class Layer:
    def __init__(self, prevNodes, nodes, activationFunc, n_features):
        self.prevNodes = prevNodes
        self.nodes = nodes
        self.actFunc = activationFunc
        self.weights = np.random.randn(self.n_features, nodes)
