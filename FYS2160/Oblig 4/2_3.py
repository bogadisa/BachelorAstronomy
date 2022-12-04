import numpy as np
import matplotlib.pyplot as plt


c = 0.1
wb = 6e-26
rho = 0.05
beta = (1+2*c)/(1-c)

def sigma(T, epsilon):
    return T/(wb*(1-c))*np.log((1+beta*epsilon)/(1-beta*epsilon/2))

def epsilon_func(r, r0):
    return (r/r0)**2-1

epsilon = np.linspace(0, 2, 100)

plt.plot(epsilon, sigma(1, epsilon))
plt.show()