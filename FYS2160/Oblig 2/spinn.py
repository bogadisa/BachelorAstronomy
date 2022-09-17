import numpy as np
import matplotlib.pyplot as plt
import random

M = 10000
N = 50

E = np.zeros(M)
sm = np.copy(E)
sp = np.copy(E)

for i in range(M):
    s = [random.randint(0, 1) for j in range(N)]
    sm[i] = s.count(0)
    sp[i] = s.count(1)

    E[i] = sp[i] - sm[i]

plt.hist(E, 50)
plt.xlabel(r"Energi $[\mu B]$")
plt.ylabel("Antall microstates")
plt.show()