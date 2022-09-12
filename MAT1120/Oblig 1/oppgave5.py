import sympy as sp
from sympy.printing.pretty.pretty import pretty_print
from scipy.linalg import null_space
import numpy as np

row1 = [0, 0, 0, 0, 1]
row2 = [0, 0, 1/2, 1, 0]
row3 = [0, 0, 0, 0, 0]
row4 = [0, 1, 1/2, 0, 0]
row5 = [1, 0, 0, 0, 0]
A = np.array([row1, row2, row3, row4, row5])

p = 0.85

K = np.ones((5, 5)) / 5

I = np.eye(5)

G = p*A + (1 - p)*K

GI = G - I

W ,V = np.linalg.eig(G)

score_vector = np.zeros(np.shape(V[0]))
for i in range(len(V)):
    if np.isclose(W[i], 1):
        pretty_print(W[i])
        s = sum(V_ for V_ in V[i])
        score_vector == V[i] / s

pretty_print(score_vector)