import numpy as np
import matplotlib.pyplot as plt

N = 100

q = range(N)

def factorial(x):
    if x!=0 and x!=1:
        return x*factorial(x-1)
    else:
        return 1

def omega(N, q):
    l = len(q)
    tmp = np.zeros(l)
    for i in range(l):
        tmp[i] = factorial(N - 1 + q[i])/factorial(q[i])/factorial(N-1)
    return tmp

#plt.plot(np.array([i for i in range(4)]), np.array([factorial(i) for i in range(4)]))

plt.plot(q, omega(N, q)**2)
plt.xlabel(r"Total energy $q=q_A+q_B$, $q_A=q_B$")
plt.ylabel(r"Multiplicity $\Omega_{tot} (N, q)$")
plt.yscale("log")
plt.show()