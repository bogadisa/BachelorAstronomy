from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.utils import to_categorical
from NeuralNetwork import *
from Logreg import *
import seaborn as sns
import pandas as pd
#from functions import *
import pathlib
import warnings
warnings.filterwarnings("ignore")

colorpal = sns.color_palette("deep")
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


#Classification
def calcEtaLambda(X_train_, X_test_, Y_train, Y_test, epochs, act, actDeriv, title):
    """
    Calculates the best test accuracy using gridsearch of set values for eta and lambda parameter.
    Returns a pandas dataframe including all the values from Test_accuracy. Using the NeuralNetwork to train and predict.

    Args:
		X_train_ (ndarray) : Array containing training data, in this case scaled
		X_test_ (ndarray) : Array containing test data, in this case scaled
        Y_train (ndarray) : Array containing training data for targets
		Y_test (ndarray) : Array containing test data for targets
        epochs (int) : Number of epochs which the netowrk will train
        act (function) : Activation unction of which the hidden layers will use.
        act_deriv (function) : The derivative of the chosen activation function
        title (str) : Name of activation function which is tested


	Returns:
		df (pd.DataFrame) : Includes all the results for best accuracy found product of the gridsearch.
                            Col = eta values and rows = lambda values
        title ("str") : Name of activation function which is tested returned to identify dataframe.
    """
    eta_vals = np.logspace(-7, 1, 9)
    lmbd_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.insert(lmbd_vals, 0, 0)

    #Col = learning rate,  Rows = lambdas
    Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
    Test_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for

    for i, etaValue in enumerate(eta_vals):
        for j, lmbdValue in enumerate(lmbd_vals):
            dnn = NeuralNetwork(X_train_, Y_train, 2, 16, act, actDeriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
            dnn.train(X_test_, Y_test, calcAcc=True)
            accTr = dnn.get_accTrain()
            accTe = dnn.get_accTest()
            indexTrain = np.argmax(accTr)
            indexTest = np.argmax(accTe)
            accTra = accTr[indexTrain]
            accTes = accTe[indexTest]
            Train_accuracy[j, i] = accTra
            Test_accuracy[j, i] = accTes

    df = pd.DataFrame(Test_accuracy, columns= eta_vals, index = lmbd_vals)
    df.round(2)
    return df, title

def plotEtaLambda(epochs, data=None, savefig=True):
    """
    Runs gridsearch for NeuralNetwork using 3 activation functions: sigmoid, tanh and RELU.
    Uses the breast cancer dataset which is scaled and split into test and train. Search for eta and lambda.
    Uses calcEtaLambda() which returns the dataframes which is then plotted and saved in Plots/Classification/

    Args:
        epochs (int) : Number of epochs used for training.
        savefig (boolean) : Wether to save figure or not, deafult=True.
    """
    path = "./Plots/Classification"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    seed = 32455
    np.random.seed(seed)
    #loading data
    if not(isinstance(data, pd.DataFrame)):
        cancer = load_breast_cancer()

        inputs = cancer.data
        targets = cancer.target
        labels = cancer.feature_names[0:30]
    else:
        targets = data["HIGH_RISK"]
        inputs = data.loc[:, data.columns != "HIGH_RISK"]
        targets = to_categorical(targets)
        


    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dfSig, titleSig = calcEtaLambda(X_train_, X_test_, Y_train, Y_test,epochs, sigmoid, sigmoid_deriv, title="Sigmoid")
    dfTanh, titleTanh = calcEtaLambda(X_train_, X_test_, Y_train, Y_test,epochs, tanh, tanh_deriv, title="Tanh")
    dfRelu, titleRelu = calcEtaLambda(X_train_, X_test_, Y_train, Y_test,epochs, relu, relu_deriv, title="Relu")

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,6), sharey=True, tight_layout=True)
    #fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f"Accuracy Test data w/epochs={epochs}", fontsize = 20, y = 0.05)
    ax1.title.set_text(titleSig)
    ax2.title.set_text(titleTanh)
    ax3.title.set_text(titleRelu)

    ax1 = sns.heatmap(dfSig,  ax=ax1, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1%")
    ax2 = sns.heatmap(dfTanh,  ax=ax2, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1%")
    ax3 = sns.heatmap(dfRelu, ax=ax3, cbar=True, annot=True, annot_kws={"fontsize":11}, fmt=".1%")

    axs = [ax1, ax2, ax3]
    ax1.set(ylabel="Lambda")
    for ax in axs:
        ax.set(xlabel="Eta")
    fig.subplots_adjust(wspace=0.001)
    if savefig:
        plt.savefig(f"{path}/TestEtaLamdGrid_{epochs}.pdf", dpi=300)
    #plt.show()

def calcLayerNodes(X_train_, X_test_, Y_train, Y_test, epochs, act, actDeriv, title):
    """
    Calculates the best test accuracy using gridsearch of set values for layer and nodes parameter.
    Returns a pandas dataframe including all the values from Test_accuracy. Using the NeuralNetwork to train and predict.

    Args:
		X_train_ (ndarray) : Array containing training data, in this case scaled
		X_test_ (ndarray) : Array containing test data, in this case scaled
        Y_train (ndarray) : Array containing training data for targets
		Y_test (ndarray) : Array containing test data for targets
        epochs (int) : Number of epochs which the netowrk will train
        act (function) : Activation unction of which the hidden layers will use.
        act_deriv (function) : The derivative of the chosen activation function
        title (str) : Name of activation function which is tested


	Returns:
		df (pd.DataFrame) : Includes all the results for best accuracy found product of the gridsearch.
                            Col = layers and rows = nodes.
        title ("str") : Name of activation function which is tested returned to identify dataframe.
    """
    layer_vals = np.array([5,4,3,2,1])
    #layer_vals = np.array([2,1])
    nodes_vals = np.array([64,32,16,8,4,2])
    #nodes_vals = np.array([32,16])

    #For 300 epochs, manually read best values for eta and lambda for each activation function
    if title == "Sigmoid":
        etaValue = 0.1
        lmbdValue = 0
    elif title == "Tanh":
        etaValue = 0.001
        lmbdValue = 0
    elif title == "Relu":
        etaValue = 0.001
        lmbdValue = 0.0001
    else:
        etaValue = 1e-3
        lmbdValue = 0

    #Col = learning rate,  Rows = lambdas
    Train_accuracy=np.zeros((len(nodes_vals), len(layer_vals)))      #Define matrices to store accuracy scores as a function
    Test_accuracy=np.zeros((len(nodes_vals), len(layer_vals)))       #of learning rate and number of hidden neurons for

    for i, layerValue in enumerate(layer_vals):
        for j, nodeValue in enumerate(nodes_vals):
            dnn = NeuralNetwork(X_train_, Y_train, layerValue, nodeValue, act, actDeriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
            dnn.train(X_test_, Y_test, calcAcc=True)
            accTr = dnn.get_accTrain()
            accTe = dnn.get_accTest()
            indexTrain = np.argmax(accTr)
            indexTest = np.argmax(accTe)
            accTra = accTr[indexTrain]
            accTes = accTe[indexTest]
            Train_accuracy[j, i] = accTra
            Test_accuracy[j, i] = accTes

    df = pd.DataFrame(Test_accuracy, columns= layer_vals, index = nodes_vals)
    df.round(3)
    return df, title

def plotLayerNodes(epochs, savefig=True):
    """
    Runs gridsearch for NeuralNetwork using 3 activation functions: sigmoid, tanh and RELU.
    Uses the breast cancer dataset which is scaled and split into test and train. Search for layer and nodes.
    Uses calcLayerNodes() which returns the dataframes which is then plotted and saved in Plots/Classification/

    Args:
        epochs (int) : Number of epochs used for training.
        savefig (boolean) : Wether to save figure or not, deafult=True.
    """
    path = "./Plots/Classification"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data
    targets = cancer.target
    labels = cancer.feature_names[0:30]

    #Converting to one-hot vectors
    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dfSig, titleSig = calcLayerNodes(X_train_, X_test_, Y_train, Y_test,epochs, sigmoid, sigmoid_deriv, title="Sigmoid")
    dfTanh, titleTanh = calcLayerNodes(X_train_, X_test_, Y_train, Y_test,epochs, tanh, tanh_deriv, title="Tanh")
    dfRelu, titleRelu = calcLayerNodes(X_train_, X_test_, Y_train, Y_test,epochs, relu, relu_deriv, title="RELU")

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,6), sharey=True, tight_layout=True)
    #fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f"Accuracy Test data w/epochs={epochs}", fontsize = 20, y = 0.05)
    ax1.title.set_text(titleSig)
    ax2.title.set_text(titleTanh)
    ax3.title.set_text(titleRelu)

    ax1 = sns.heatmap(dfSig,  ax=ax1, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1%" )
    ax2 = sns.heatmap(dfTanh,  ax=ax2, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1%")
    ax3 = sns.heatmap(dfRelu, ax=ax3, cbar=True, annot=True, annot_kws={"fontsize":11}, fmt=".1%")

    axs = [ax1, ax2, ax3]
    ax1.set(ylabel="Nodes")
    ax1.set(xlabel="Layers")
    ax3.set(xlabel="Layers")
    fig.subplots_adjust(wspace=0.001)
    if savefig:
        plt.savefig(f"{path}/TestLayNodesGrid_{epochs}.pdf", dpi=300)
    #plt.show()

#Run accuracy vs epochs for sigmoid
def runAccTestTrain():
    """
    Plots test and training MSE using the breast cancer data set. Using the NeuralNetwork class
    with 2 layers with 16 nodes in each. Activation function set to sigmoid. Saves fig in /Plots/Classification
    """
    path = "./Plots/Classification"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data #30 features
    targets = cancer.target
    labels = cancer.feature_names[0:30]
    epochs = 250
    etaValue = 0.001
    lmbdValue = 0

    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dnn = NeuralNetwork(X_train_, Y_train, 2, 16, sigmoid, sigmoid_deriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
    dnn.train(X_test_, Y_test, calcAcc=True)
    accTr = dnn.get_accTrain()
    accTe = dnn.get_accTest()

    indexTrain = np.argmax(accTr)
    indexTest = np.argmax(accTe)
    #print(accTr[-1])
    #print(accTe[-1])
    plt.plot(np.arange(epochs), accTr, label = "Train")
    plt.plot(np.arange(epochs), accTe, label = "Test")
    plt.scatter(indexTrain, accTr[indexTrain], marker="x", color = "navy", s=35, label=f"Max acc train {100*accTr[indexTrain]:.1f}%")
    plt.scatter(indexTest, accTe[indexTest], marker="x", color="red", s=35, label=f"Max acc test {100*accTe[indexTest]:.1f}%")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (1=100%)")
    plt.title(f"Test vs Train Accuracy: lr={etaValue}")
    plt.savefig(f"{path}/TestvTrainepochs.pdf", dpi=300)
    #plt.show()


#LogisticRegression
def calcLogReg(X_train_, X_test_, Y_train, Y_test, epochs, act, actDeriv, title):
    """
    Calculates the best test accuracy using gridsearch of set values for eta and lambda parameter.
    Returns a pandas dataframe including all the values from Test_accuracy.
    Makes use of the simplified network structure with the LogisticRegression class.

    Args:
		X_train_ (ndarray) : Array containing training data, in this case scaled
		X_test_ (ndarray) : Array containing test data, in this case scaled
        Y_train (ndarray) : Array containing training data for targets
		Y_test (ndarray) : Array containing test data for targets
        epochs (int) : Number of epochs which the netowrk will train
        act (function) : Activation unction of which the hidden layers will use.
        act_deriv (function) : The derivative of the chosen activation function
        title (str) : Name of activation function which is tested


	Returns:
		df (pd.DataFrame) : Includes all the results for best accuracy found product of the gridsearch.
                            Col = eta values and rows = lambda values
        title ("str") : Name of activation function which is tested returned to identify dataframe.
    """

    bestVal = [0, 0, 0]
    for i, etaValue in enumerate(eta_vals):
        for j, lmbdValue in enumerate(lmbd_vals):
            dnn = LogRegClass(X_train_, Y_train, act, actDeriv, epochs = epochs, eta = etaValue, lmbd=lmbdValue)
            dnn.train(X_test_, Y_test, calcAcc=True)
            accTr = dnn.get_accTrain()
            accTe = dnn.get_accTest()
            indexTrain = np.argmax(accTr)
            indexTest = np.argmax(accTe)
            accTra = accTr[indexTrain]
            accTes = accTe[indexTest]

            if accTes > bestVal[0]:
                bestVal[0] = accTes
                bestVal[1] = etaValue
                bestVal[2] = lmbdValue
            Train_accuracy[j, i] = accTra
            Test_accuracy[j, i] = accTes

    #print(f"Acc: {bestVal[0]:.3f} Eta: {bestVal[1]} Lamb: {bestVal[1]} ")
    df = pd.DataFrame(Test_accuracy, columns= eta_vals, index = lmbd_vals)
    df.round(2)
    #print(df)
    #print(df)

    #sns.heatmap(df, annot=True, fmt=".1%")
    #plt.show()
    return df, title

def plotLogRegAct(epochs, savefig=True):
    """
    Runs gridsearch for LogisticRegression using 3 activation functions: sigmoid, tanh and RELU.
    Uses the breast cancer dataset which is scaled and split into test and train.
    Uses calcLogReg() which returns the dataframes which is then plotted and saved in Plots/LogReg/

    Args:
        epochs (int) : Number of epochs used for training.
        savefig (boolean) : Wether to save figure or not, deafult=True.
    """
    path = "./Plots/LogReg"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data
    targets = cancer.target
    labels = cancer.feature_names[0:30]

    #Converting to one-hot vectors
    x = inputs
    y = targets

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    dfSig, titleSig = calcLogReg(X_train_, X_test_, Y_train, Y_test,epochs, sigmoid, sigmoid_deriv, title="Sigmoid")
    dfTanh, titleTanh = calcLogReg(X_train_, X_test_, Y_train, Y_test,epochs, tanh, tanh_deriv, title="Tanh")
    dfRelu, titleRelu = calcLogReg(X_train_, X_test_, Y_train, Y_test,epochs, relu, relu_deriv, title="RELU")

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,6), sharey=True, tight_layout=True)
    #fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f"LogReg accuracy w/epochs={epochs}", fontsize = 20, y = 0.05)
    ax1.title.set_text(titleSig)
    ax2.title.set_text(titleTanh)
    ax3.title.set_text(titleRelu)

    ax1 = sns.heatmap(dfSig,  ax=ax1, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1%" )
    ax2 = sns.heatmap(dfTanh,  ax=ax2, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1%")
    ax3 = sns.heatmap(dfRelu, ax=ax3, cbar=True, annot=True, annot_kws={"fontsize":11}, fmt=".1%")

    axs = [ax1, ax2, ax3]
    ax1.set(ylabel="Lambda")
    ax1.set(xlabel="Eta")
    ax3.set(xlabel="Eta")
    fig.subplots_adjust(wspace=0.001)
    if savefig:
        plt.savefig(f"{path}/LogTest_EtaLambda_{epochs}.pdf", dpi=300)
    #plt.show()


#Regression
def calcRegression(X_train_, X_test_, Y_train_, Y_test_, epochs, act, actDeriv):
    """
    Calculates regression MSE in heatmap for different values of eta and lambda.
    If value == 0, its considered NaN value and unusuable. Set to 0, to not ruin
    color scale in heatmap. Returns the datafram containing results for best MSE

    Args:
		X_train_ (ndarray) : Array containing training data, in this case scaled
		X_test_ (ndarray) : Array containing test data, in this case scaled
        Y_train (ndarray) : Array containing training data for targets
		Y_test (ndarray) : Array containing test data for targets
        epochs (int) : Number of epochs which the netowrk will train
        act (function) : Activation unction of which the hidden layers will use.
        act_deriv (function) : The derivative of the chosen activation function

	Returns:
		df (pd.DataFrame) : Includes all the results for best accuracy found product of the gridsearch.
                            Col = eta values and rows = lambda values
    """
    eta_vals = np.logspace(-7, 1, 9)
    lmbd_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.insert(lmbd_vals, 0, 0)

    eta_vals = np.array([0.00001, 0.0001,0.001, 0.01, 0.1])
    lmbd_vals = np.array([0, 0.000001, 0.00001, 0.0001,0.001, 0.01])

    #Col = learning rate,  Rows = lambdas
    MSE_test =np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function

    for i, etaValue in enumerate(eta_vals):
        #print(f"{i}/{eta_vals.size}")
        for j, lmbdValue in enumerate(lmbd_vals):
            dnn = NeuralNetwork(X_train_, Y_train_, 2, 16, sigmoid, sigmoid_deriv, epochs = epochs, etaVal = etaValue, lmbd=lmbdValue)
            dnn.layers[-1].sigma = linear
            dnn.layers[-1].sigma_d = linear_deriv
            dnn.train(X_test_, Y_test_, calcMSE = True)
            mse = dnn.get_MSEtest()
            bestMSE = mse[np.argmin(mse)] #pulls the lowest value there is
            #mseValEnd = mse[-1]
            MSE_test[j, i] = bestMSE
            #print(bestMSE)

    df = pd.DataFrame(MSE_test, columns= eta_vals, index = lmbd_vals)
    df.round(2)
    return df

def runPlotEtaLambdaRegr(epochs, savefig=True):
    """
    Using FrankeFunction data is genereated and the data is scaled. Then using the calcRegression()
    the gridsearch is performed which returns dataframe. Only sigmoid is used as activation function.
    This is then plotted and saved in Plots/Regression/ folder.

    Args:
        epochs (int) : Number of epochs used for training.
    """
    path = "./Plots/Regression"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)

    n = 7000
    x = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)

    y = FrankeFunction(x,z)
    X = np.array([x,z]).T

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    df = calcRegression(X_train_, X_test_, Y_train_, Y_test_, epochs, sigmoid, sigmoid_deriv)

    plt.figure(figsize=(8,6))
    plt.tight_layout()
    plt.title(f"MSE gridsearch Sigmoid ep={epochs}")

    ax = sns.heatmap(df, annot=True, annot_kws={"fontsize":12}, fmt=".1e")
    ax.set(ylabel="Lambda")
    ax.set(xlabel="Eta")

    if savefig:
        plt.savefig(f"{path}/TestEtaLamdGrid_{epochs}.pdf", dpi=300)
    #plt.show()

def runPlotRegrAct(showruninfo=False):
    """
    Simple function plots MSE for 3 diff activation functions with the data genereated by
    f(x) = 1 + 5*x + 3*x**2 over 30 000 datapoints. Runs it through network with 2 Layers
    and 16 nodes in each. Scaled data is used to improve performance.
    Saves figure to /Plots/Regression
    """
    if showruninfo:
        print("Plots MSE for different activation functions for the NeuralNetwork (Sigmoid, RELU, Tanh)")
    seed = 32455
    np.random.seed(seed)

    path = "./Plots/Regression"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    n = 10000
    x = np.linspace(0, 10, n*3)

    def f(x):
        return 1 + 5*x + 3*x**2

    y = f(x)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)
    ep = 200

    #Sigmoid
    dnn = NeuralNetwork(X_train_, Y_train_, 2, 16, sigmoid, sigmoid_deriv, epochs = ep, etaVal = 0.001)
    dnn.layers[-1].sigma = linear
    dnn.layers[-1].sigma_d = linear_deriv
    dnn.train(X_test_, Y_test_, calcMSE = True)
    #test_predict = dnn.predict(X_test_)

    #RELU
    dnn1 = NeuralNetwork(X_train_, Y_train_, 2, 16, relu, relu_deriv, epochs = ep, etaVal = 0.001)
    dnn1.layers[-1].sigma = linear
    dnn1.layers[-1].sigma_d = linear_deriv
    dnn1.train(X_test_, Y_test_, calcMSE = True)
    #test_predict = dnn1.predict(X_test_)


    #Tanh
    dnn2 = NeuralNetwork(X_train_, Y_train_, 2, 16, tanh, tanh_deriv, epochs = ep, etaVal = 0.001)
    dnn2.layers[-1].sigma = linear
    dnn2.layers[-1].sigma_d = linear_deriv
    dnn2.train(X_test_, Y_test_, calcMSE = True)

    #LeakyRelu
    dnn3 = NeuralNetwork(X_train_, Y_train_, 2, 16, leaky_relu, leaky_relu_deriv, epochs = ep, etaVal = 0.0001)
    dnn3.layers[-1].sigma = linear
    dnn3.layers[-1].sigma_d = linear_deriv
    dnn3.train(X_test_, Y_test_, calcMSE = True)

    #MSE vs epochs on training
    mse = dnn.get_MSEtest()
    mse1 = dnn1.get_MSEtest()
    mse2 = dnn2.get_MSEtest()
    mse3 = dnn3.get_MSEtest()

    plt.figure(figsize=(7,5))
    plt.tight_layout()
    plt.yscale("log")
    plt.plot(np.arange(ep), mse, label = "Sigmoid lr: 0.01")
    plt.plot(np.arange(ep), mse1, label = "RELU lr: 0.001")
    plt.plot(np.arange(ep), mse2, label = "Tanh lr: 0.001")
    plt.plot(np.arange(ep), mse3, label = "LeakyRelu lr: 0.0001")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.title(f"Activation funcs MSE")
    plt.savefig(f"{path}/Act funcs ep_{ep}.pdf", dpi=300)
    #plt.show()

#SK learn logistic regression
def runSklearnLogreg():
    """
    Function which using the load_breast_cancer dataset calculates the accuracy using
    sk learn LogisticRegression module for a range of epochs, both for test and training data.
    Which to be used for comparison.
    """
    from sklearn.linear_model import LogisticRegression

    path = "./Plots/LogReg"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    seed = 32455
    np.random.seed(seed)
    #loading data
    cancer = load_breast_cancer()

    inputs = cancer.data
    targets = cancer.target
    labels = cancer.feature_names[0:30]
    x = inputs
    y = targets

    X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=1/4)
    X_train_, X_test_, Y_train_, Y_test_ = scale(X_train, X_test, Y_train, Y_test)

    iter = np.array([1, 3, 30, 100, 300, 1000, 10000])
    accUnscaled = np.zeros(iter.size)
    accScaled = np.zeros(iter.size)
    for i, iter_ in enumerate(iter):
        model = LogisticRegression(max_iter = iter_)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test_)
        accUn = accuracy_score(Y_test, Y_pred)
        accUnscaled[i] = accUn

        model.fit(X_train_, Y_train)
        Y_pred = model.predict(X_test_)
        accSc = accuracy_score(Y_test, Y_pred)
        accScaled[i] = accSc

    plt.figure(figsize=([6,5]))
    plt.tight_layout()
    arrCombined = np.array([accUnscaled, accScaled]).T
    df = pd.DataFrame(arrCombined, columns=np.array(["Unscaled data", "Scaled data"]), index=iter)
    ax = sns.heatmap(df, annot=True, annot_kws={"fontsize":13}, fmt=".1%")
    ax.set_ylabel("Epochs")
    ax.set_title("SKlearn regression Accuracy")
    plt.savefig(f"{path}/SKlearnAcc.pdf", dpi=300)
    #plt.show()




def runClassiAcc():
    """
    Runs all plots for Classification. Eta vs lambda. Layer vs nodes. Logistic regression for 3 diff epochs.
    """
    plotEtaLambda(epochs=30)
    plotLayerNodes(epochs=30)

    plotEtaLambda(epochs=300)
    plotLayerNodes(epochs=300)

    plotLogRegAct(3)
    plotLogRegAct(30)
    plotLogRegAct(300)

if __name__ == "__main__":
    msg = "In genResults"
    #Regression (task b)
    #runPlotEtaLambdaRegr(epochs=30) # 0.4 minutes
    #runPlotEtaLambdaRegr(epochs=300) # 3.8 minutes
    #runPlotEtaLambdaRegr(epochs=1000) # 12.6 minutes
    #runPlotEtaLambdaRegr(epochs=10000) # 124 minutes

    #Regression activation funcs (task c)
    #runPlotRegrAct()

    #Classification (task d+task e)
    #runAccTestTrain() #Accuracy vs epochs for Sigmoid.
    #runClassiAcc() #Network + logistic Gridsearch. EtaLambda and LayerNodes.
    #runSklearnLogreg() #comparison with sklearn
