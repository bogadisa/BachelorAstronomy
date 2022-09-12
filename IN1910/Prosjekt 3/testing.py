import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import tri


class ChaosGame:
    def __init__(self, n, r):
        self.n, self.r = n, r
        self._generate_ngon()
    
    def _generate_ngon(self):
        n = self.n

        theta = np.linspace(0, 2*np.pi, n+1)

        c = np.array([np.sin(theta), np.cos(theta)])

        self.c = c

    @property
    def n_gon(self):
        try:
            return self.c
        except AttributeError:
            self._generate_ngon()
            return self.c
    
    def plot_ngon(self):
        c = self.n_gon
        plt.plot(c[0], c[1])
        plt.scatter(c[0], c[1])
        plt.axis("equal")
        plt.show()

    def _starting_point(self):
        """Kommer senere"""
    
    def iterate(self, steps, discard=5):
        n, r = self.n, self.r
        c = self.n_gon

        X = np.zeros((steps, 2))
        X[0] = self._starting_point()
        colors = np.zeros(steps)
        for i in range(steps-1):
            j = np.random.randint(0, n)
            X[i+1] = (X[i]+c[:, j])/r
            colors[i] = j

        self._X = X[discard:]
        self._colors = colors[discard:]

    @property
    def X(self):
        try:
            return self._X
        except AttributeError:
            self.iterate()
            return self._X
        
    @property
    def colors(self):
        try:
            return self._colors
        except AttributeError:
            self.iterate()
            return self._colors

    def plot(self, color=False, cmap="jet"):
        X = self.X
        if not(color):
            plt.scatter(X[:, 0], X[:, 1], c="black")
        
        elif color:
            colors = self.colors
            plt.scatter(X[:, 0], X[:, 1], color=colors, cmap=cmap)

    def show(self, color=False, cmap="jet"):
        self.plot(color, cmap)
        plt.axis("equal")
        plt.show()

    def testing(self, N):
        n = self.n


        X = np.zeros((N, 2, n))
        c = self.n_gon
        w = np.random.rand(N, n)
        for i in range(N):
            w[i] = w[i]/np.sum(w[i])
            X[i] = w[i]*c[:,:-1]


        plt.plot(c[0], c[1])
        plt.scatter(c[0], c[1])
        plt.axis("equal")
        plt.scatter(X[:, 0], X[:, 1], c="r")
        plt.show()


ngons = [3, 4, 5, 6, 7, 8]
for n in ngons:
    ngon = ChaosGame(n)
    ngon.plot_ngon()
