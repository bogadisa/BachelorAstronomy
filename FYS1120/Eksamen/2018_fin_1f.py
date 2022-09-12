import numpy as np
import matplotlib.pyplot as plt


L = 100
N = 10000
dx = L/N

IL = 200
I = np.linspace(-50, L+50, IL)
x, y = np.meshgrid(I, I, indexing="ij")

def rho(x):
    return 1

def E(q, r):
    V = np.zeros(np.shape(x))
    for i in range(len(x.flat)):
        R = np.array([x.flat[i] - r, y.flat[i]])
        V.flat[i] += q/(4*np.pi*np.linalg.norm(R))
    return -np.array(np.gradient(V))

Efield = np.zeros((2, IL, IL))
for i in range(N):
    xi = (i + 1/2)*dx
    drho = rho(x)*dx
    Efield += E(drho, xi)

plt.imshow(Efield)