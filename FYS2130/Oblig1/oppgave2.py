import numpy as np
import matplotlib.pyplot as plt

k = 8
m = 2
A = 1.9
phi = 1.08

omega = np.sqrt(k/m)

t = np.linspace(0, 2*np.pi, 100)

x = A*np.cos(omega*t + phi)
v = -A*omega*np.sin(omega*t + phi)

plt.plot(x, m*v)
plt.axis("equal")
plt.xlabel("x(t)[m]")
plt.ylabel("Bevegelsesmengde [kg m/s]")
plt.show()


T = 2*np.pi/omega
L = A
M = m
V = L/T

plt.plot(x/L, m*v/(M*V))
plt.axis("equal")
plt.show()