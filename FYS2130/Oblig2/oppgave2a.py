import numpy as np
import matplotlib.pyplot as plt

N = 10000

k0 = 1e-6
kL = 1e-1

k = np.linspace(k0, kL, N)

cells = 3000
L = 30 / 1000 # mm to m
cellWidth = L / cells

dimensions0 = np.array([0.1, 0.3]) / 1000
dimensionsL = np.array([0.3, 0.1]) / 1000

height = np.linspace(dimensions0[0], dimensionsL[0], N)
width = np.linspace(dimensions0[1], dimensionsL[1], N)

rho0 = 1500 #kg/m^3
rhoL = 2500

rho = np.linspace(rho0, rhoL, N)

x = np.linspace(0, L, N)


def f(x, height, width, rho, cellWidth, k):
    m = rho * (height * width *cellWidth)

    omega = np.sqrt(k/m)

    return omega/(2*np.pi)

klog = np.logspace(-6, -1, N)


plt.plot(x*1000, f(x, height, width, rho, cellWidth, k), label="Lineær")
plt.ylabel("y[hz]")
plt.plot(x*1000, f(x, height, width, rho, cellWidth, klog), label="Logaritmisk")
plt.legend()
plt.xlabel("x[mm]")
plt.ylabel("y[hz]")
plt.grid(1)
plt.show()