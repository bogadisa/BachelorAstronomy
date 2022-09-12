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


#Oppgave 3a
A1 = np.array([-2, 0, 1/2, 1])
A2 = np.array([-1/4, 1, 1/4, 0])
A3 = np.array([0, 0, 3, -1])
A4 = np.array([1/8, 1/8, 1/4, 2])

A = np.array([A1, A2, A3, A4])

d, r, lmda = egensirkel(A)

for d_, r_ in zip(d, r):
    sirkel(d_.real, d_.imag, r_)
plt.axis("equal")
plt.show()

#Oppgave 3b
B1 = np.array([3, 1, 1])
B2 = np.array([1, 0, 1])
B3 = np.array([-1, 1, -2])

B = np.array([B1, B2, B3])

d, r, lmda = egensirkel(B)

for d_, r_ in zip(d, r):
    sirkel(d_.real, d_.imag, r_)

plt.scatter(lmda.real, lmda.imag)
plt.axis("equal")
plt.show()

#Oppgave 3c
C = np.ones((5, 5))
C -= np.diag(range(5))

d, r, lmda = egensirkel(C)

for d_, r_ in zip(d, r):
    sirkel(d_.real, d_.imag, r_)

plt.scatter(lmda.real, lmda.imag)
plt.axis("equal")
plt.show()

#Oppgave 3d
D = np.diag(range(1, 5))

d, r, lmda = egensirkel(D)

for d_, r_ in zip(d, r):
    sirkel(d_.real, d_.imag, r_)

plt.scatter(lmda.real, lmda.imag)
plt.axis("equal")
plt.show()