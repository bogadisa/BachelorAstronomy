import numpy as np
import matplotlib.pyplot as plt

#Oppgave 1
def egensirkel(M):
    n = len(M)

    d = []
    r = []
    j = 0
    for i in range(n):
        d.append(M[i, j])
        r.append(np.sum(abs(M[i, j+1:])) + np.sum(abs(M[i, :j])))
        j +=1
    #Oppgave 2
    w, v = np.linalg.eig(M)
    return d, r, w

def sirkel(x, y, r):
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(x + r*np.cos(theta), y + r*np.sin(theta))