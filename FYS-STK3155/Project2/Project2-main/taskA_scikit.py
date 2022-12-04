'''
Task A with Scikit Learn
'''
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd


seed = np.random.seed(1404)

def create_data(x):
    return 1 + 3*x + 5*x**2 #+ np.random.randn(n, 1)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, 0.1, x.shape)

def create_X(x):
    return np.c_[np.ones((n, 1)), x, x**2]

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

n = 1000
x = np.random.randint(1,10,n)#.reshape(1,-1)

y = create_data(x)#.ravel()
#x = np.arange(0, 1, 1/n)
#y = np.arange(0, 1, 1/n)
#y = FrankeFunction(x,y)
#print(np.max(y))
#plt.plot(x,y,'o')
#plt.show()
X = create_X(x)
#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) #Splitter dataene

XT_X = X.T @ X #Identity matrix?
H = (2.0/n)* XT_X
EigValues, EigVectors = np.linalg.eig(H) #To get eigenvalues
eta = 1.0/np.max(EigValues)

learn_rates = [eta] #np.geomspace(0.0001, 0.1, 10, endpoint = True) # this is ineffective,
epoch_range = np.arange(10,1000,100)#In the plot this is set to np.arange(10,1000,10)

lambdas = [1, 0.1,0.01,0.001,0.0001, 0]
plotting_dict = {'epoch_range':[],'learning_rate':[],'lambdas':[],'MSEtrain':[], 'MSEtest':[]}

for lamb in lambdas:
    #for rate in learn_rates: eta0=rate # looping through the learning rates I want to test
    # or instead of doing this add learning_rate = 'adaptive' to model instead of specifying eta0
        #print(f'Results for learning rate {rate}:\n')
    for ep in epoch_range: # for each learning rate, I test each epoch lenght
        model = SGDRegressor(max_iter=100000, eta0 = 0.1,learning_rate='adaptive',fit_intercept=False, alpha =0.0001) #Scikit learn Stocastic gradient descent regression
        #fit intercept is false because of the 1 in the feature matrix
        mse_train= [] # to be filled with train and test mses
        mse_test= [] #Planning to exchange with a matrix defined outside of loop for plotting all MSEs for each rate and epoch size
        epochs = np.arange(1, ep+1) #making a list for looping through
        plotting_dict['lambdas'].append(lamb)
        plotting_dict['epoch_range'].append(ep)
        plotting_dict['learning_rate'].append('adaptive') #or rate
        n_batch = 10 #I use 10 batches in each epoch
        batches = np.split(X_train, n_batch, axis=0) #Splitting the training data into the batches
        ybatches = np.split(y_train, n_batch, axis=0)
        batch_i= 0
        for epoch in epochs:
            for b, batch in enumerate(batches): #getting index to get the right y-batch
                y_batch = ybatches[b]
                try:
                    model.partial_fit(batch, y_batch)
                    batch_i += n_batch
                except ValueError: #An error that occured, perhaps not anymore
                    print(f'for some reason, there was a Value error{len(y_batch)}')
                    pass

                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)

                mse_train.append(MSE(y_train, y_pred_train))
                mse_test.append(MSE(y_test, y_pred_test))
        plotting_dict['MSEtrain'].append(MSE(y_train, y_pred_train))
        plotting_dict['MSEtest'].append(MSE(y_test, y_pred_test))
            #print('Coefficients: ', model.coef_) #printing coefficients for each epoch epoch_range
df = pd.DataFrame(plotting_dict)
#print(df)

#Subsetting data for plotting
lambda1 = df.loc[df['lambdas']==1]
lambda01 = df.loc[df['lambdas']==0.1]
lambda001 = df.loc[df['lambdas']==0.01]
lambda0 = df.loc[df['lambdas']==0]

#plotting
fig, ax = plt.subplots()
#ax.plot(lambda1['epoch_range'],lambda1['MSEtest'], label = 'MSE test 1 lambda')
ax.plot(lambda01['epoch_range'],lambda01['MSEtest'], label = 'MSE test 0.1 lambda')
ax.plot(lambda001['epoch_range'],lambda001['MSEtest'], label = 'MSE test 0.01 lambda')
ax.plot(lambda0['epoch_range'],lambda0['MSEtest'], label = 'MSE test OLS')
ax.set_xlabel('epochs')
ax.set_ylabel('MSE score')
plt.legend()
plt.savefig('MSE_epochs_lambdas_batch10.png', bbox_inches='tight')
plt.show()

#Prøver å plotte som heatmap, men burde nok heller ha lamda mot learning rate
import seaborn as sns
sns.set_theme()
fig2, ax2 = plt.subplots()
h_df = df[['epoch_range', 'lambdas', 'MSEtrain']]
h_df = h_df.pivot('epoch_range', 'lambdas', 'MSEtrain')
sns.heatmap(h_df,cmap="YlGnBu")


'''
Looking at the plot, MSE_epoch_lambda0and01, the lambda set to 0 represents an OLS,
while the 0.1 is a ridge regression with 0.1 weight. In this case, the two models
seem to have similar effect with the epochs used and a learning rate set to 1 over
the largest eigenvalue of the hessian matrix.
'''
