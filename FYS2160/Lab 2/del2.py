from tkinter.tix import TList
import numpy as np
import matplotlib.pyplot as plt

n = 1000
V = np.linspace(0.4, 6, n)
G = np.zeros(n)
P = np.copy(G)

def trapezoidal(x, b, a):
    dx = (b-a)/len(x)
    return dx/2*(x[0]+x[-1]+2*x[1:-1])

#T = 0.89
Tlist = np.linspace(0.4, 1, 5)
for T in Tlist:
    T = 0.89
    P  = 8*T/(3*V-1)-3/V**2
    G = -8/3*T*np.log(3*V-1)-3/V+P*V
    for i in range(n):
        i= 0

    px = 0.6166
    gy = -4.1

plt.plot(P, V)
plt.xlabel("P [*]")
plt.ylabel("G [*]")
plt.show()

# Tlist = [0.5, 0.8]

# V = np.linspace(0.4, 3, n)
# for T in Tlist:
#     P  = 8*T/(3*V-1)-3/V**2
#     G = -8/3*T*np.log(3*V-1)-3/V+P*V
#     plt.plot(V, P)
# plt.show()

