from sklearn.linear_model import LinearRegression as OLS_reg
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import copy
import pandas as pd
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


def CostOLS(y, X, beta):
    return (1/y.shape[0])*((y- X@beta).T)@(y- X@beta)

def CostRidge(y, X, beta, lambda_):
    return (1/y.shape[0])*((y- X@beta).T)@(y- X@beta) + lambda_*beta.T@beta

#Define the gradient with costfunction
gradientOLS = grad(CostOLS, 2) #2 meaning beta
gradientRidge =  grad(CostRidge, 2) #2 meaning beta

def gradient_decent(X, x, y, beta, lr, n_iter, momentum=0, batch_size=20, useAda=False, useRMS=False, useAdam=False, lambda_ = 0):
    """
    Performs gradient_decent on x dataset with y being target data. Option of 3 different optimizers
    and uses Ridge if lambda value differs from 0. Returns MSE and estimated beta values.
    """
    beta = copy.deepcopy(beta) #avoid overwriting in original beta
    MSE_list = [] #Store MSE scores every update to plot
    beta_list = [] #Store every time beta is updated
    change = 0
    M = batch_size
    m = int(n_iter/M) #number of minibatches
    y_pred = beta[0] + beta[1]*x + beta[2]*x*x
    MSE_list.append(mse(y, y_pred))
    beta_list.append(beta)

    if useAda:
        delta = 1e-8
        for i in range(n_iter):
            #print(f"{i}/{len(n_iter)}")
            Giter = np.zeros(shape=(3,3))
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                # Calculate the outer product of the gradients
                Giter +=gradients @ gradients.T
                # Simpler algorithm with only diagonal elements
                Ginverse = np.c_[lr/(delta+np.sqrt(np.diagonal(Giter)))]
                # compute update
                update = np.multiply(Ginverse,gradients)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
            MSE_list.append(mse(y, y_pred))
            beta_list.append(beta)
                #Calculate MSE and store to list and plot.
                # momentum vs non-momentum
    elif useRMS:
        rho = 0.99
        lr = 0.01
        delta = 1e-8
        for i in range(n_iter):
            Giter = np.zeros(shape=(3,3))
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]
                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                # Previous value for the outer product of gradients
                Previous = Giter
        	    # Accumulated gradient
                Giter +=gradients @ gradients.T
        	    # Scaling with rho the new and the previous results
                Gnew = (rho*Previous+(1-rho)*Giter)
        	    # Taking the diagonal only and inverting
                Ginverse = np.c_[lr/(delta+np.sqrt(np.diagonal(Gnew)))]
        	    # Hadamard product
                update = np.multiply(Ginverse,gradients)

                new_change = update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
            MSE_list.append(mse(y, y_pred))
            beta_list.append(beta)
    elif useAdam:
        b1 = 0.9
        b2 = 0.999
        t = 0
        eps = 1e-4
        #eps = 0.001
        m_ = 0
        v = 0
        for i in range(n_iter):
            for i in range(m):
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                t = t + 1

                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                m_ = b1*m_ + (1-b1)*gradients
                v = b2*v + (1-b2)*gradients**2
                m_hat = m_/(1-b1**t)
                v_hat = v/(1-b2**t)
                update = lr*m_hat/(np.sqrt(v_hat)+eps)

                new_change = lr*update + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
            MSE_list.append(mse(y, y_pred))
            beta_list.append(beta)
    else:
        for i in range(n_iter):
            for i in range(m):
                #Split up X and y
                rand_ind = M*np.random.randint(m)
                xi = X[rand_ind:rand_ind+M]
                yi = y[rand_ind:rand_ind+M]

                if lambda_ != 0:
                    gradients = gradientRidge(yi, xi, beta, lambda_)
                else:
                    gradients = gradientOLS(yi, xi, beta)

                new_change = lr*gradients + momentum*change
                beta -= new_change
                change = new_change
                y_pred = beta[0] + beta[1]*x + beta[2]*x*x
            #Certain runs y_pred will be NaN/inifinity which stops run. If encountered put MSE = 10.
            try:
                MSE_list.append(mse(y, y_pred))
            except:
                MSE_list.append(10) #if NaN of infinity return value of 10
            beta_list.append(beta)
                #Calculate MSE and store to list and plot.
                # momentum vs non-momentum
    #plt.plot(MSE_list)
    #plt.show()
    return beta, MSE_list, beta_list

def calcSGDGridsearch(X, x, y, epochs, batch_size, optimizer=None):
    """
    Does the gridsearch for a set values of eta and lambda values. Optimizer is set
    in the argument.

    Args:
		X (ndarray) : 2D Feature matrix (e.g X = np.c_[np.ones((n,1)), x, x**2] #design matrix)
		x (ndarray) : 1-dimensional array including all the datapoints, used for continually calculating MSE for each epoch
        y (ndarray) : Targets which the model will try to reach.
        epochs (int) : Number of epochs which it will run.
        batchsize (int) : Size of batch which the model will train on. Large batch_size means more accurate gradients, but more computation needed.
        optimizer (str) : Which optimizer to use ,default=None
	Returns:
		df (pd.DataFrame) :
        bestMSE (float) : The best mean squared error found.
        bestBeta (ndarray) : A 1-dim array containing 3 values [B0, B1, B2] for the polynomial which are updated with SGD
        bestMSElist (ndarray) : For the best eta and lambda, we store the MSEvsEpochs list.
        bestEta (float) : Eta value linked with best MSE.
        bestLmbd (float) : Lambda value linked with best MSE.
    """
    M = batch_size
    eta_vals = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 10])
    lmbd_vals = np.array([0, 1e-5, 1e-4, 1e-3, 1e-2])

    useAda_ = False
    useRMS_ = False
    useAdam_ = False

    if optimizer=="Ada":
        useAda_ = True
    elif optimizer=="RMS":
        useRMS_ = True
    elif optimizer=="Adam":
        useAdam_ = True

    np.random.seed(200)
    beta = np.random.randn(3,1) #randomized initial values for beta

    #Col = learning rate,  Rows = lambdas
    Train_accuracy=np.zeros((len(lmbd_vals), len(eta_vals)))      #Define matrices to store accuracy scores as a function
    MSE_array=np.zeros((len(lmbd_vals), len(eta_vals)))       #of learning rate and number of hidden neurons for

    bestMSE = 100
    bestBeta = np.array([0])
    bestMSElist = np.array([0])
    bestEta = 1000
    bestLmbd = 1000
    for i, etaValue in enumerate(eta_vals):
        for j, lmbdValue in enumerate(lmbd_vals):
            beta_, MSE_list, _ = gradient_decent(X, x, y, beta, lr=etaValue, n_iter = epochs,
                                                batch_size=M, useAda=useAda_, useRMS=useRMS_,
                                                useAdam=useAdam_, lambda_=lmbdValue)
            #mse_val = MSE_list[-1]
            mse_val = MSE_list[np.argmin(MSE_list)]
            #Update when found lower value
            if mse_val < bestMSE:
                bestMSE = mse_val
                bestBeta = beta_
                bestMSElist = MSE_list
                bestEta = etaValue
                bestLmbd = lmbdValue
            MSE_array[j, i] = mse_val

    df = pd.DataFrame(MSE_array, columns= eta_vals, index = lmbd_vals)
    return df, bestMSE, bestBeta, bestMSElist, bestEta, bestLmbd

def runPlotsSGD(batchsize, epochs, showruninfo=False):
    """
    Runs the calculations for stochastic gradient descent with different optimizers.
    Searches for best eta and lambda value. This grid searchs and stored in dataframed.
    Which is then plotted with seaborn and saved to /Plotd/GradDescent/

    Methods is tested with a simple 2nd degree polynomial with 10 000 datapoints with
    some added noise. Additionally use sklearn OLS and calculate MSE for comparison.

    Args:
        batchsize (int) : Size of batch which the model will train on. Large batch_size
                          means more accurate gradients, but more computation needed.
        epochs (int) : Number of epochs which it will run. More epochs in most cases leads
                       better beta values and smaller MSE.
        showruninfo (boolean) : Wether to show some run info.

	Returns:

    """
    n_epochs = epochs #epochs
    M = batchsize #batchsize
    if showruninfo:
        print("Calculating and plotting gridsearch for SGD with optimizers. As well as MSE vs epochs for best found parameters")
    path = "./Plots/GradDescent"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    n = 10000
    x = np.random.rand(n,1)
    #Analytical value. Static learning rate.
    y = 2+3*x+4*x*x+0.1*np.random.rand(n,1)*0.2 #added noise

    X = np.c_[np.ones((n,1)), x, x**2] #design matrix
    XT_X = X.T @ X
    #theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
    H = (2.0/n)* XT_X
    EigValues, EigVectors = np.linalg.eig(H)
    lr = 1.0/np.max(EigValues)

    #Simple OLS fit for comparison
    model = OLS_reg(fit_intercept=True)
    model.fit(X, y)
    y_pred = model.intercept_ + model.coef_[0][1]*x + model.coef_[0][2]*x*x
    MSE_SKlearn = mse(y, y_pred)
    #print(f"SKlearn OLS: {MSE_SKlearn:.2e}")

    #Run gridsearch with default, ada, rms and adam optimizer.
    #Pulls the betavalues corresponding to the best MSE.
    df_Adam, bestMSE_Adam, bestBeta_Adam, bestMSE_list_Adam, bestEta_Adam, bestLmbd_Adam = calcSGDGridsearch(X, x, y, n_epochs, batch_size=M, optimizer = "Adam")
    #print(f"Adam done bestMSE={bestMSE_Adam:.2e}")
    df, bestMSE, bestBeta, bestMSE_list, bestEta, bestLmbd = calcSGDGridsearch(X, x, y, n_epochs,  batch_size=M )
    #print(f"Default done bestMSE={bestMSE:.2e}")
    df_Ada, bestMSE_Ada, bestBeta_Ada, bestMSE_list_Ada, bestEta_Ada, bestLmbd_Ada = calcSGDGridsearch(X, x, y, n_epochs, batch_size=M, optimizer = "Ada")
    #print(f"Ada done bestMSE={bestMSE_Ada:.2e}")
    df_RMS, bestMSE_RMS, bestBeta_RMS, bestMSE_list_RMS, bestEta_RMS, bestLmbd_RMS = calcSGDGridsearch(X, x, y, n_epochs, batch_size=M, optimizer = "RMS")
    #print(f"RMS done bestMSE={bestMSE_RMS:.2e}")

    if showruninfo:
        print(f"SKlearn OLS: {MSE_SKlearn:.2e}")
        print(f"Default done bestMSE={bestMSE:.2e}")
        print(f"Ada done bestMSE={bestMSE_Ada:.2e}")
        print(f"RMS done bestMSE={bestMSE_RMS:.2e}")
        print(f"Adam done bestMSE={bestMSE_Adam:.2e}")
        print(f"Done.\n")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16,8), sharey=True, tight_layout=True)
    #fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.rc('axes', titlesize=16)
    plt.subplots_adjust(hspace=0.1)
    plt.suptitle(f"SGD Opti - Gridsearch for eta & lambda", fontsize = 16, y = 0.04)
    ax1.title.set_text("Default")
    ax2.title.set_text("Ada")
    ax3.title.set_text("RMS")
    ax4.title.set_text("Adam")

    ax1 = sns.heatmap(df,  ax=ax1, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1e" )
    ax2 = sns.heatmap(df_Ada,  ax=ax2, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1e")
    ax3 = sns.heatmap(df_RMS, ax=ax3, cbar=False, annot=True, annot_kws={"fontsize":11}, fmt=".1e")
    ax4 = sns.heatmap(df_Adam, ax=ax4, cbar=True, annot=True, annot_kws={"fontsize":11}, fmt=".1e")

    axs = [ax1, ax2, ax3, ax4]
    ax1.set(ylabel="Lambda")
    ax1.set(xlabel="Eta")
    ax4.set(xlabel="Eta")
    fig.subplots_adjust(wspace=0.001)
    plt.savefig(f"{path}/SGD_gridsearch_opti_ep{n_epochs}_batch{M}.pdf", dpi=300)

    #Plot of best MSE from gridsearch for diff optimizer
    plt.figure(figsize=(8,6))
    plt.tight_layout()
    plt.title("SGD: MSE for different optimizers")
    plt.plot(np.arange(n_epochs+1), bestMSE_list_Ada, label=f"Ada {bestMSE_Ada:.2e} eta: {bestEta_Ada} lmbd: {bestLmbd_Ada}")
    plt.plot(np.arange(n_epochs+1), bestMSE_list_RMS, label=f"RMS {bestMSE_RMS:.2e} eta: {bestEta_RMS} lmbd: {bestLmbd_RMS}", linestyle="dashed")
    plt.plot(np.arange(n_epochs+1), bestMSE_list_Adam, label=f"Adam {bestMSE_Adam:.2e} eta: {bestEta_Adam} lmbd: {bestLmbd_Adam}")
    plt.plot(np.arange(n_epochs+1), bestMSE_list, label=f"Default {bestMSE:.2e} eta: {bestEta} lmbd: {bestLmbd}")
    plt.text(2.5, 4, f"SKLearnOLS: {MSE_SKlearn:.2e}", fontsize = 14)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{path}/Best_MSE_Opti_ep{n_epochs}_batch{M}.pdf", dpi=300)
    #plt.show()


if __name__ == "__main__":
    runPlotsSGD(batchsize=50, epochs = 400, showruninfo=True)
