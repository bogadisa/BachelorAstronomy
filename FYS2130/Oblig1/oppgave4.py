import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)

A = 1
omega = np.pi/2
phi = 0

x = A*np.cos(omega*t + phi)
x = np.where(x<0, -x, x)

plt.plot(t, x)
plt.ylim(0, 1.25)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

m = 1
v = -A*omega*np.sin(omega*t + phi)
v = np.where(x<0, -v, v)

plt.plot(x, m*v)
plt.xlim(0, 1)
plt.xlabel("x(t)")
plt.ylabel("Bevegelsesmengde m*v")
plt.show()