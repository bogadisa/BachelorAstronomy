import numpy as np
import sklearn

n = 200

x = np.linspace(0, 1, n)

X = np.c_[np.ones((n,1)), x]

A = np.array([3, 5])
B = np.array([1])

def create_data(x):
    return A[0]*x + A[1]*x**2 + B

y = create_data(x)


def x():
    return 1