import numpy as np
import torch
import time

N, P, D = 5000, 200, 300

X = torch.full((N, D), 1)
T = torch.full((P, D), 0)



# print(X)
# print(T)

Xnp = np.zeros((N, D))
Tnp = np.ones((P, D))

d = np.zeros((N, P))
t0 = time.time()
for i in range(Xnp.shape[0]):
 for j in range(Tnp.shape[0]):
    d[i, j] = np.dot((Xnp[i]-Tnp[j])**2, ((Xnp[i]-Tnp[j])**2).T)


# print(d)
dt = time.time() - t0

print(dt)

# X = torch.unsqueeze(X, dim=1)
# T = torch.unsqueeze(T, dim=0)
# d = (X - T)**2
# d = X**2 -2XT.T+T**2

t0 = time.time()
d = -2*torch.mm(X, T.t())
d += torch.sum(torch.pow(X, 2), dim=1).unsqueeze(1)
d += torch.sum(torch.pow(T, 2), dim=1).unsqueeze(0)
dt = time.time() - t0

print(dt)