from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt

lmbda = A**2

def psi2(x):
    return lmbda * np.exp(-2 * lmbda*abs(x))


expected = 0
expectedsqrd = 0.5*lmbda**-2

def stddev(expected, expectedsqrd):
    return np.sqrt(expectedsqrd - expected**2)


a = -10
b = -a
N = 1000

x = np.linspace(a, b, N)

zeros = np.zeros(N)

psi2max = max(psi2(x))
psi2min = min(psi2(x))

boundries = np.linspace(psi2min, psi2max, N)

plt.plot(x, psi2(x))
plt.plot(zeros + stddev(expected, expectedsqrd), boundries, linestyle="--", c="black")
plt.plot(zeros - stddev(expected, expectedsqrd), boundries, linestyle="--", c="black")
plt.xlabel("x")
plt.ylabel(r"$|\Psi(x,t)|^2$")
plt.grid(1)
plt.show()