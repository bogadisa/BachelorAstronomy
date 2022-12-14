import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)

maxdegree = 5

poly3 = PolynomialFeatures(degree=maxdegree)
X = poly3.fit_transform([x, y])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf3 = LinearRegression()
clf3.fit(X_train_scaled, y_train)
y_ = clf3.predict(X_test_scaled)


#x_ = np.linspace(-3, 3, int(0.2*n))
plt.scatter(x, y)
plt.scatter(X_test[:, 1], y_)

print(f"MSE={MSE(y_test, y_)}")
print(f"R2={R2(y_test, y_)}")