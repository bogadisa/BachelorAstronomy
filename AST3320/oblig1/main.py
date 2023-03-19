import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from functions import *

omega_names = ["\Omega_\phi", "\Omega_r", "\Omega_m"]

if __name__ == "__main__":
    n = 10000
    z0 = 2e7
    z1 = 0
    z = np.logspace(7, 0, n)

    N0 = z_to_N(z0)
    N1 = z_to_N(z1)

    print(N0, N1)

    N = np.linspace(N0, N1, n)

    #power law
    pow_x1 = 5e-5
    pow_x2 = 1e-8
    pow_x3 = 0.9999
    pow_lamda = 1e9

    pow_variables_init = np.array([pow_x1, pow_x2, pow_x3, pow_lamda], dtype=np.float64)
    # print(pow_x)


    #exponential law
    exp_x1 = 0
    exp_x2 = 5e-13
    exp_x3 = 0.9999


    # pow_omega = np.zeros((n, pow_x.shape[0]), dtype=np.float64)


    pow_solution = integrate.solve_ivp(equation_of_motions_inv, t_span=[N0, N1], y0=pow_variables_init, t_eval=N, rtol=1e-8, atol=1e-8)
    N = pow_solution.t
    z = N_to_z(N)
    pow_variables = pow_solution.y

    pow_x = pow_variables[:3]
    pow_lamda = pow_variables[-1]

    pow_omega = densities(pow_x)
    
    pow_omega_phi = pow_omega[0]

    pow_rho_phi = omega_to_rho(pow_omega_phi, z)
    pow_P_phi = rho_to_P(pow_rho_phi, phi)

    pow_EoS_phi = EoS_param(pow_rho_phi, P)


    figure, [ax1, ax2] = plt.subplots(2, 2)

    for omega, omega_name in zip(pow_omega, omega_names):
        ax1[0].plot(z, omega, label=rf"${omega_name}$")

    ax1[0].set_ylim(-0.1, 1.1)
    ax1[0].invert_xaxis()
    ax1[0].set_xscale("log")
    ax1[0].set_xlabel("z")
    ax1[0].set_ylabel("Dominance")

    plt.legend()
    plt.show()









    # for i, z_ in enumerate(z):
        # print(pow_x)

        # pow_omega[i] = densities(pow_x)
        # print(densities(pow_x))


        # pow_rho = rho(pow_omega, z_)

        # pow_x = equation_of_motion(pow_x, pow_lamda)
        # pow_x1, pow_x2, pow_x3 = pow_x
        
        # pow_lamda += -np.sqrt(6)*pow_lamda**2*pow_x1
    














        # pow_V = inverse_power(pow_phi)
        # pow_gamma = get_gamma(pow_V, pow_phi)

        # pow_dx = equation_of_motion(pow_x, pow_lamda)
        # pow_d_lamda = d_lamda(x, lamda, gamma)


        # pow_x += pow_dx

        # omega = densities(pow_x)

