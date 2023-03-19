import numpy as np

def test_disjointed(Xs):
    for i, X1 in enumerate(Xs):
        for X2 in Xs[i+1:]:
            if np.intersect1d(X1, X2, assume_unique=True):
                raise Exception("The sets are not disjoint")