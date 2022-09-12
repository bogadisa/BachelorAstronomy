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

C4 = 261.63
C4sharp = 277.18

def A(F, m, omega0, omegaF, b):
    return (F/m)/np.sqrt((omega0**2 - omegaF**2)**2 + (b*omegaF/m)**2)


frequencies = np.array([C4, C4sharp]) * (2*np.pi)
m = rho * (height * width *cellWidth)

F = 100

b = 1e-7

label = ["C4", "C4#"]
omega0 = f(x, height, width, rho, cellWidth, k) * 2*np.pi
for i, omegaF in enumerate(frequencies):
    Amp = A(F, m, omega0, omegaF, b)
    plt.plot(x, Amp, label=f"{label[i]}, b = {b}")
plt.legend()
plt.grid(1)
plt.show()

#c)
def Q(m, k, b, A):
    i = np.argmax(A)
    return np.sqrt(m[i]*k[i]/b**2)


print(f"I denne sistuasjonen er Q={Q(m, k, b, Amp)}")

#d)
def dt(Q):
    return Q/(2*np.pi)


print(f"Membranen svinger i t={dt(Q(m, k, b, Amp))}sekunder")