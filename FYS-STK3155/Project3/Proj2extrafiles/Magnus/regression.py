from classes import *
from sklearn import datasets

seed = 32455
np.random.seed(seed)
#our data
n = 1000
x = np.linspace(0, 10, n)

y = np.copy(x)
z = FrankeFunction(x, y)

#Used for franke function
X = np.array([x, y]).T
print(X.shape, z.shape)

# X = np.c_[np.ones((n,1)), x, x**2]
# y = f(x)
X_train, X_test, Y_train, Y_test = train_test_split(X, z,test_size=1/4)

X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

# ep = 100
# dnn = NeuralNetwork(X_train_, Y_train_, 2, 16, RELU, RELU_deriv, epochs = ep, eta = 0.00001)
# dnn.layers[-1].sigma = linear
# dnn.layers[-1].sigma_d = linear_deriv
# dnn.train(X_test_, Y_test_, calcMSE = True)
# mse = dnn.get_MSEtest()
# plt.plot(np.arange(ep), mse, label="mse")
# plt.xlabel("epochs")
# plt.ylabel("mse")
# plt.legend()
# plt.show()
# dnn = NeuralNetwork(X_train, Y_train, 3, 32, RELU, RELU_deriv, epochs = 100, eta = 0.00001)
# # changing the output-layer to have the correct activation function
# dnn.layers[-1].sigma = linear
# dnn.layers[-1].sigma_d = linear_deriv

# #for when we want to compare untrained data to trained data (debugging)
# test_predict_untrained = dnn.predict(X_test)
# dnn.train()

# test_predict = dnn.predict(X_test_)

# # # print(test_predict.shape)
# # # print(Y_test.shape)
# # plt.scatter(X_test[:, 1], Y_test, label="Target", c="r")
# plt.scatter(X_test_[:, 1], test_predict[:, 0], label="Model")
# plt.xlabel("x=y")
# plt.ylabel("f(x,y)")
# # # plt.scatter(X_test, test_predict_1[:, 0], label="Model")
# # # plt.scatter(X_test[:, 1], test_predict_1[:, 0], label="Model_untrained", marker="x", alpha=0.2)
# # # plt.scatter(X_test, Y_test, label="Actual", c="r")
# # # plt.scatter(X_test_, test_predict, label="Model", alpha = 0.5)
# # #plt.scatter(X_test, test_predict_untrained, label="Model_none")
# plt.legend()
# plt.show()

n_neuron = 16

#learning rate and regularization parameters
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
# lmbd_vals = np.zeros(7)

Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for 

#Starting training, searching for best combination
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = NeuralNetwork(np.copy(X_train), np.copy(Y_train), 2, n_neuron, RELU, RELU_deriv, epochs = 50, eta = eta, lmbd=lmbd)
        #setting output activation to softmax
        dnn.layers[-1].sigma = linear
        dnn.layers[-1].sigma_d = linear_deriv
        dnn.train()
        #finding the accuracy
        Train_accuracy[j, i] = dnn.evaluate(np.copy(X_train), np.copy(Y_train))
        Test_accuracy[j, i] = dnn.evaluate(np.copy(X_test), np.copy(Y_test))

# print(Train_accuracy)
#plotting
plot_data(eta_vals, lmbd_vals, Train_accuracy, Type="Regression", title="RELU")
plot_data(eta_vals, lmbd_vals, Test_accuracy, Type="Regression", title="RELU")
