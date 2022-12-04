import numpy as np
from scipy import stats

R = 8.3144598

def T(Rt):
    r = Rt/1e5
    return (25 - 24*np.log(r))

def freedom(c, Mmol, T):
    return 2/(c**2*Mmol/R/T-1)

def find_a(x, y):
    return stats.linregress(x, y)[0]

def find_c(a, L):
    return 2*L*a

gases = np.array(["Argon", "CO2", "Air20", "Air50", "Air70"])
n = len(gases)

#T_gas = np.zeros()
T_mean_gas = np.zeros(n)
c_gas = np.zeros(n)
f_gas = np.zeros(n)
for i, gas in enumerate(gases):
    T_mean_gas = np.mean(T(Rt[i]))
    