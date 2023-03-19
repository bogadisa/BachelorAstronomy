import numpy as np
from scipy.constants import G


kappa = np.sqrt(8*np.pi*G) #N m^2 kg^-2

def N_to_z(N):
    return np.exp(-N) - 1

def z_to_N(z):
    return -np.log(1 + z)

def equation_of_motions_inv(N, x):
    x1, x2, x3, lamda = x

    # lamda = lamda_inverse(x)
    d_lamda = -np.sqrt(6)*lamda**2*x1

    dx1 = -3*x1 + 0.5*np.sqrt(6)*lamda*x2**2 + 0.5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = -0.5*np.sqrt(6)*lamda*x1*x2 + 0.5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = -2*x3 + 0.5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    
    return np.array([dx1, dx2, dx3, d_lamda])

def omega_to_rho(omega, z):
    H = 1/(1 + z)
    return omega*3*H**2/kappa**2

def rho_to_P(rho, V):
    return rho - 2*V

def EoS_param(rho, P):
    return P/rho