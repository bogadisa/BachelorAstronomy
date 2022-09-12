import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100, 1)
y = 2.0+5*x*x+0.1*np.random.randn(100, 1)

# # Y = a +b*x + c*x**2
# X = np.zeros((len(x), 3))
# X[:, 0] = 1
# X[:, 1] = x[:, 0]
# X[:, 2] = x[:, 0]**2

# XT = np.transpose(X)

# XTXinv = np.linalg.inv(np.matmul(XT, X))
# XTy = np.matmul(XT, y)
# beta = np.matmul(XTXinv, XTy)

# print(np.shape(beta), np.shape(X))
# # y_ = np.matmul(X, beta)

# x_ = np.linspace(0, 1, 1000)
# y_ = beta[0] + beta[1]*x_ + beta[2]*x_**2 

# plt.scatter(x, y)
# plt.plot(x_, y_, c="r")
# plt.show()

from sklearn.linear_model import LinearRegression