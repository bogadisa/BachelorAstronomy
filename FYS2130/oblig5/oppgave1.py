import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x[0] < -np.pi or np.pi < x[-1]:
        print("Not defined")
    else:
        return np.where((np.pi > x) & (x > 0.5*np.pi), 1, 0)


N_points = 10000
x = np.linspace(-np.pi, np.pi, N_points)

def f_k(x, N):
    a0 = 1/2
    omega1 = 1
    
    T = 2*np.pi/omega1
    k = np.linspace(0, N_points-1, N_points, dtype=int)
    

    y = a0/2

    for k_ in range(1, N):
        ak = 2/T*(np.sin(k_*omega1*np.pi) - np.sin(k_*omega1*np.pi/2))/(k_*omega1)
        bk = -2/T*(np.cos(k_*omega1*np.pi) - np.cos(k_*omega1*np.pi/2))/(k_*omega1)
        y += ak*np.cos(k_*omega1*x) + bk*np.sin(k_*omega1*x)

    return y


for i, N in enumerate([5, 10, 100, 1000]):
    plt.subplot(2, 2, i+1)
    plt.plot(x, f(x))
    plt.ylim(-0.2, 1.2)
    plt.plot(x, f_k(x, N))
    
plt.show()