import numpy as np
from scipy.constants import G, c
from scipy import integrate, interpolate, optimize
import matplotlib.pyplot as plt
# from scipy.stats import chisquare


kappa = np.sqrt(8*np.pi*G) #N m^2 kg^-2

def N_to_z(N):
    return np.exp(-N) - 1

def z_to_N(z):
    return -np.log(1 + z)

def equation_of_motions_pow(N, x):
    x1, x2, x3, lamda = x

    # lamda = lamda_inverse(x)
    d_lamda = -np.sqrt(6)*lamda**2*x1

    dx1 = -3*x1 + 0.5*np.sqrt(6)*lamda*x2**2 + 0.5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = -0.5*np.sqrt(6)*lamda*x1*x2 + 0.5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = -2*x3 + 0.5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    
    return np.array([dx1, dx2, dx3, d_lamda])

def equation_of_motions_exp(N, x):
    x1, x2, x3= x

    lamda = 3/2

    dx1 = -3*x1 + 0.5*np.sqrt(6)*lamda*x2**2 + 0.5*x1*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2 = -0.5*np.sqrt(6)*lamda*x1*x2 + 0.5*x2*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3 = -2*x3 + 0.5*x3*(3 + 3*x1**2 - 3*x2**2 + x3**2)
    
    return np.array([dx1, dx2, dx3])

def densities(x):
    x1, x2, x3 = x
    omega_phi = x1**2 + x2**2
    omega_r = x3**2
    omega_m = 1 - omega_phi - omega_r

    return np.array([omega_phi, omega_r, omega_m])

def omega_to_rho(omega, z):
    H = 1/(1 + z)
    return omega*3*H**2/kappa**2

def EoS_param(x, rho_phi, z):
    x1, x2, x3 = x
    H = 1/(1 + z)
    V = 3*H**2*x2**2/kappa**2

    return 1 - 2*V/rho_phi


def solution_pow(pow_variables_init, N0, N1, N):
    #solve the equations
    pow_solution = integrate.solve_ivp(equation_of_motions_pow, t_span=[N0, N1], y0=pow_variables_init, t_eval=N, rtol=1e-8, atol=1e-8)
    #get the variables that the solution generates
    N = pow_solution.t
    z = N_to_z(N)
    pow_variables = pow_solution.y

    #divide the solution
    pow_x = pow_variables[:3]
    pow_lamda = pow_variables[-1]

    #convert to densities
    pow_omega = densities(pow_x)
    
    pow_omega_phi = pow_omega[0]

    #we need rho to find EoS
    pow_rho_phi = omega_to_rho(pow_omega_phi, z)
    #find EoS
    pow_EoS_phi = EoS_param(pow_x, pow_rho_phi, z)

    return pow_omega, pow_EoS_phi, z

#same almost as the one above, but takes a different equation as input
def solution_exp(exp_variables_init, N0, N1, N):
    exp_solution = integrate.solve_ivp(equation_of_motions_exp, t_span=[N0, N1], y0=exp_variables_init, t_eval=N, rtol=1e-8, atol=1e-8)
    N = exp_solution.t
    z = N_to_z(N)
    exp_variables = exp_solution.y

    exp_x = exp_variables[:3]

    exp_omega = densities(exp_x)
    
    exp_omega_phi = exp_omega[0]

    exp_rho_phi = omega_to_rho(exp_omega_phi, z)

    exp_EoS_phi = EoS_param(exp_x, exp_rho_phi, z)

    return exp_omega, exp_EoS_phi, z

#just a function that tidies up the code
def pretty_plot(ax, i, param):
    if param == "hubble" or param == "age" or "dL" in param:
        ax.invert_xaxis()
        ax.set_xlabel("z")
        if not("dL" in param):
            ax.set_xscale("log")
        ax.legend()
        if param == "hubble":
            ax.set_ylabel(r"Hubble parameter $H/H_0$ [*]")
            ax.set_yscale("log")
        elif param == "age":
            ax.set_ylabel(r"Age of the universe $H_0 t_0$ [*]")
        elif param == "dL":
            ax.set_ylabel(r"Luminosity distance $H_0 d_L/c$")
        else:
            ax.set_ylabel(r"Luminosity distance $d_L$ [Gpc]")

    else:
        ax[i].invert_xaxis()
        ax[i].set_xscale("log")

        if param == "EoS":
            if not(i):
                ax[i].set_ylabel("EoS")
            ax[i].set_xlabel("z")

        elif param == "omega":
            if not(i):
                ax[i].set_ylabel("Dominance")
            ax[i].set_ylim(-0.1, 1.1)

        ax[i].legend()

def hubble(omegas, N, EoS):
    omega_phi, omega_r, omega_m = omegas
    z = N_to_z(N)

    #we need to flip to make everything work as inteded
    phi_int = np.flip(integrate.cumtrapz(3*(1 + np.flip(EoS)), N, initial=0))

    H = np.sqrt(omega_m[-1]*(1+z)**3 + omega_r[-1]*(1+z)**4 + omega_phi[-1]*np.exp(phi_int))

    return H


def age(N, H):
    #flip again
    return np.flip(integrate.cumtrapz(np.flip(1/H), N, initial=0))
    # return integrate.cumtrapz(-1/H, N, initial=N[0])

def d_L(N, H):
    #flip again
    z = np.flip(N_to_z(N))
    # print(z)
    # # print(np.exp(-N))
    # # print(z)

    # return (1+z)*np.flip(integrate.cumtrapz(np.flip(-N*np.exp(-N)/H), N, initial=0))
    # return (1+z)*np.flip(integrate.cumtrapz(np.flip(1/H), z, initial=0))
    #Since we are using only low values of z, I dont convert to N
    return np.flip((1+z)*integrate.cumtrapz(1/H, z, initial=0))

def dL_with_units(dL, h):
    # print(dL)
    # print(dL*3/h)
    #3/h comes from H0 = 100h
    #we need to multiply by c t00, which introduces the 3
    #after canceling units and converting to the right units
    #we are left with 3/h
    return dL*3/h #c/H0*1e-3

def chisquare(obs, exp, sigma):
    #could not get this version to give any sane numbers
    # return np.sum((obs - exp)**2/sigma**2)
    
    evaluation = (obs - exp)**2/exp
    return np.sum(evaluation[:-1])

def lcdm_model(omega_m0, z):
    return np.sqrt(omega_m0*(1 + z)**3 + (1 - omega_m0))

def lcdm_model_fitting(h):
    def fit(z, omega_m0):
        N = z_to_N(z)
        H0 = lcdm_model(omega_m0, z)
        dL = d_L(N, H0)
        dL_w = dL_with_units(dL, h)
        fig, ax = plt.subplots()
        ax.plot(z, dL_w)
        pretty_plot(ax, 0, "dL*")
        plt.show()

        return dL_w
    return fit