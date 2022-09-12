import numpy as np
import matplotlib.pyplot as plt
#from scipy.constants import hbar

hbar = 6.582119569e-16
a = 0.614e-9
A = np.sqrt(1/a)
B = A

img = 0 + 1j

def E(n):
    return 0.25 * n**2

def psi(n, x):
    if n%2 == 0:
        return A*np.sin(np.pi*n/(2*a)*x) 
    else:
        return B*np.cos(np.pi*n/(2*a)*x)

c1 = 0.4
c2 = 0.3
c3 = 0.2
c4 = 0.1

c = [c1, c2, c3, c4]

N = 10000
x = np.linspace(-a, a, N)

fm = 1e-15

def absPsi2(t, x):
    s = 0

    for i, c_ in enumerate(c):
        s += np.sqrt(c_) * psi(i+1, x) * np.exp(-img/hbar * E(i+1)*t)
    return np.abs(s)**2


plt.plot(x, absPsi2(5*fm, x), label=r"$|\Psi|^2$")
plt.plot(x, np.abs(np.sqrt(1)*psi(2, x))**2, label=r"$|\psi_2|^2$")
plt.title(r"$|\Psi(x,t)|^2$ ved $t=5fs$ og $|\psi_2(x)|$")
plt.xlabel("x[nm/10]")
plt.ylabel(r"$|\Psi|^2$ og $|\psi_2|^2$")
plt.legend()
plt.show()