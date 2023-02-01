import genResults
import GradDescent
import time


def main():
    """
    Imports the different functions to generate and plot all data.
    Seperated into 4 different blocks (GradDescent, Regression, Classification, LogReg Classification)
    #Note: best accuracy/mse is pulled and not just the last value in the list.
           E.g for 300 epochs, best value might be at 242 epochs instead.

    """

    """
    #1 GradDescent: Gridsearch for best combination of eta and lambda for the different optimizers: (Default SGD, Adagrad, RMSprop and ADAM)
    Uses the Frankiefunction data.
    eta_vals = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 10])
    lmbd_vals = np.array([0, 1e-5, 1e-4, 1e-3, 1e-2])
    Then plot the MSE vs epochs for the best found eta and lambda setting batchsize and epochs.
    Figures saved to ./Plots/GradDescent
    """
    startGrad = time.time()
    #Task a)
    GradDescent.runPlotsSGD(batchsize=70, epochs = 400, showruninfo=True)
    #GradDescent.runPlotsSGD(batchsize=40, epochs = 100, showruninfo=True)
    endGrad = time.time()
    print(f"GradDescent done: t={(endGrad-startGrad)/60:.2f} min\n")


    """
    #2 Regression: Gridsearch for best eta and lambda. If mse==0 in plot, is due to NaN value converted to 0.
    Tested with epochs =30, 300, 1000, 10 000. Using data from frankie func.
    Consistently seems to be best eta value=0.01 and lambda=0. Epochs beyond 300 seems redundant.

    Then plot how MSE vs epochs for diff activation funtions, with f(x)=1 + 5*x + 3*x**2.
    Figures saved to ./Plots/Regression.
    """
    startRegr = time.time()
    #Task b)
    genResults.runPlotEtaLambdaRegr(epochs=30) # 0.4 minutes
    genResults.runPlotEtaLambdaRegr(epochs=300) # 3.8 minutes
    #genResults.runPlotEtaLambdaRegr(epochs=1000) # 12.6 minutes, not much better
    #genResults.runPlotEtaLambdaRegr(epochs=10000) # 124 minutes

    #Task c)
    genResults.runPlotRegrAct()

    endRegr = time.time()
    print(f"Regression done: t={(endRegr-startRegr)/60:.2f} min\n")

    """
    #3 Classification: Gridsearch for best combination of eta and lambda for the different optimizers with 2 layers and 16 nodes.
    Using the sklearn breast_cancer dataset and using the 30 features to predict if positive or not. (binary classification)
    Then using the best combination of eta and lambda, do another gridsearch for layervs vs nodes.

    Then plot the MSE vs epochs for the best found eta and lambda setting batchsize and epochs.
    Figures saved to ./Plots/Classification
    """
    startClassi = time.time()
    #Task d)
    #Show result of overfitting by plotting MSE on test vs training data.
    genResults.runAccTestTrain() #Accuracy vs epochs for Sigmoid.

    #Search for eta, lambda, nodes and layers for 30 and 300 epochs.
    genResults.plotEtaLambda(epochs=30)
    genResults.plotLayerNodes(epochs=30)

    genResults.plotEtaLambda(epochs=300)
    genResults.plotLayerNodes(epochs=300)

    endClassi = time.time()
    print(f"Classification done: t={(endClassi-startClassi)/60:.2f} min\n")

    """
    #4 LogisticRegression on same dataset and predicting positive or not for breast cancer.
    Architecture is basically a netowrk with 0 hidden layers with a step function applied on the output.
    Plot a gridsearch for accuracy for 3 different activation functions: Sigmoid, Tanh and RELU.
    Figures saved to ./Plots/LogReg
    """
    startLogReg = time.time()
    #Task e)
    genResults.plotLogRegAct(3)
    genResults.plotLogRegAct(30)
    genResults.plotLogRegAct(300)

    #Plot accuracy for "sklearn.linear_model import LogisticRegression". Test on scaled and unscaled data.
    genResults.runSklearnLogreg() #comparison with sklearn

    endLogReg = time.time()
    print(f"LogisticRegression done: t={(endLogReg-startLogReg)/60:.2f} min\n")
    return

if __name__ == "__main__":
    main()
