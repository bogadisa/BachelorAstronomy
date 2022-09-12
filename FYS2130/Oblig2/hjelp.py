import numpy as np
import matplotlib.pyplot as plt

N = 10000

Ncells = 3000
length_tot = 0.03 #30mm i m

height = np.linspace(0.0003, 0.0001, N)
width = np.linspace(0.0001, 0.0003, N)
density = np.linspace(1500, 2500, N)

cell = np.linspace (0, length_tot, N)
k = np.linspace(1e-6, 1e-1,N)

L = 30 / 1000
CellWidth = L/Ncells
m = height*width*density*CellWidth

C4 = 261.63
C4s = 277.18
C4w = 2*np.pi*C4
C4sw = 2*np.pi*C4s
F = 100
b = 10**(-9)


omega0 = np.sqrt(k/m)
AC4 = (F/m)/np.sqrt((omega0**2 - C4w**2)**2 + (b*C4w/m)**2)
AC4s = (F/m)/np.sqrt((omega0**2 - C4sw**2)**2 + (b*C4sw/m)**2)

plt.plot(cell, AC4)
plt.plot(cell, AC4s)
plt.xlabel("posisjon(m)")
plt.ylabel("Amplitude")
plt.show()