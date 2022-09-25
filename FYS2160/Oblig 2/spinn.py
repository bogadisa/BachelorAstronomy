import numpy as np
import matplotlib.pyplot as plt
import random

M = 10000
N = 50

E = np.zeros(M)
sm = np.copy(E)
sp = np.copy(E)

for i in range(M):
    #coin-flip
    s = [random.randint(0, 1) for j in range(N)]
    #antiparallel spinn
    sm[i] = s.count(0)
    #parallel spinn
    sp[i] = s.count(1)
    #regner energi
    E[i] = sp[i] - sm[i]

s = np.linspace(-N, N, M)/2

#finner fordelingen analytisk
def Omega(N, s):
    return 2**N*np.exp(-2*s**2/N)


bins = 50

#gjør det mulig å se begge to samtidig
distribution = Omega(N, s)
normalized_distribution = distribution/max(distribution)
scaled_distribution =  normalized_distribution*max(np.histogram(E, bins)[0])

plt.hist(E,bins, rwidth=2)
plt.plot(s, scaled_distribution)
plt.xlabel(r"Energi $[\mu B]$")
plt.ylabel("Antall microstates")
plt.show()