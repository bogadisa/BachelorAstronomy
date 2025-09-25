import numpy as np

def rk4(f:callable, x0:np.ndarray, tspan:list, condition:np.ndarray, p:float, names:list) -> tuple:
    t0, t1 = tspan
    x = [x0]
    t = [t0]
    # for limiting runtime
    j = 0

    #will equal 0 if any of the conditions fail
    not_there_yet = np.prod([not(x0[i] - condition[i] < 0) for i in range(len(x0))] + [not(t0 - t1 < 0)])
    print([name for name in names], ":\n", [not(x0[i] - condition[i] < 0) for i in range(len(x0))] + [not(t0 - t1 < 0)])
    dms = np.zeros(len(condition) + 1)
    dms[:len(x[-1])] = p*x[-1]/f(t[-1], x[-1])
    dms[-1] = p*t[-1]
    h = -abs(dms[np.argmin(abs(dms))])
    while not_there_yet and ~np.isnan(h) and j < 200000:
        k1 = h * f(t[-1], x[-1])
        k2 = h * f(t[-1] + 0.5*h, x[-1] + 0.5*k1)
        k3 = h * f(t[-1] + 0.5*h, x[-1] + 0.5*k2)
        k4 = h * f(t[-1] + h, x[-1] + k3)
        
        x.append(x[-1] + (k1 + 2*(k2 + k3) + k4) / 6)
        t.append(t[-1] + h)

        not_there_yet = np.prod([not(x[-1][i] - condition[i] < 0) for i in range(len(x[-1]))] + [not(t[-1] - t1 < 0)])
        
        dms[:len(x[-1])] = p*x[-1]/f(t[-1], x[-1])
        dms[-1] = p*t[-1]
        h = -abs(dms[np.argmin(abs(dms))])
        j += 1

    x, t = np.array(x), np.array(t)

    #format : [L, R, P, T, M]
    print([name for name in names], ":\n", [not(x[-1][i] - condition[i] < 0) for i in range(len(x[-1]))] + [not(t[-1] - t1 < 0)])
    print(f"Finished after {j=} iterations")
    print(f"{x0=}, {t0=}")
    print(f"{condition=}, {t1=}")
    print(f"min(x)={np.min(x[~np.isnan(x).any(axis=1)], axis=0)}, {min(t)=}")
    print(f"max(x)={np.max(x[~np.isnan(x).any(axis=1)], axis=0)}, {max(t)=})")

    return x, t