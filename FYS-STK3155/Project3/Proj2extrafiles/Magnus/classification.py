from sklearn.datasets import load_breast_cancer, make_blobs
from classes import *

seed = 32455
np.random.seed(seed)
#loading data
cancer = load_breast_cancer()

inputs = cancer.data
targets = cancer.target
labels = cancer.feature_names[0:30]

#Converting to one-hot vectors
x = inputs
y = to_categorical_numpy(targets)

#Splitting into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)



n_neuron = 16
n_layers = 2

#learning rate and regularization parameters
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
# lmbd_vals = np.zeros(7)
Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for 

#Starting training, searching for best combination
def gridSearch(X_train, X_test, Y_train, Y_test, epochs, n_neuron, n_layers, OneHot=True):
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            dnn = NeuralNetwork(np.copy(X_train), np.copy(Y_train), n_layers, n_neuron, sigmoid, sigmoid_deriv, epochs = epochs, eta = eta, lmbd=lmbd, Type="Classification")
            #setting output activation to softmax
            dnn.layers[-1].sigma = softmax
            dnn.train()
            #finding the accuracy
            Train_accuracy[j, i] = dnn.evaluate(np.copy(X_train), np.copy(Y_train), OneHot)
            Test_accuracy[j, i] = dnn.evaluate(np.copy(X_test), np.copy(Y_test), OneHot)

    #plotting
    plot_data(eta_vals, lmbd_vals, Train_accuracy, 'sigmoid')
    plot_data(eta_vals, lmbd_vals, Test_accuracy, 'sigmoid')


n, m = 1000, 2
X, y = make_blobs(n_samples=n, centers=2, n_features=m, random_state=6)
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=1/4)

gridSearch(X_train, X_test, Y_train, Y_test, 25, n_neuron, n_layers, OneHot=False)