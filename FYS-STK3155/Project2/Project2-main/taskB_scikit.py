'''
Task B with Scikit Learn
'''
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from taskA_scikit import *
n = 1000
x = np.random.randint(1,10,n)#.reshape(1,-1)

x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
y = FrankeFunction(x,y)
#print(np.max(y))
#plt.plot(x,y,'o')
#plt.show()
X = create_X(x)
#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) #Splitter dataene



regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

y_preds = regr.predict(X_test)
print(MSE(y_test,y_preds))


print(regr.score(X_test, y_test))
#score(X, y[, sample_weight])Return the coefficient of determination of the prediction.
