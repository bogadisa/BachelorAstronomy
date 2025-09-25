import numpy as np
from functions import *

omega_names = ["\Omega_\phi", "\Omega_r", "\Omega_m"]

#I hope the variables and functions tell you what is happening
#I did not get as much time as I wanted in order to properly comment my code.abs(
# appologies in advance

if __name__ == "__main__":
    n = 10000
    z0 = 2e7
    z1 = 0
    data = np.loadtxt("sndata.txt", skiprows=5)
    data_z = data[:, 0]
    data_dL = data[:, 1]
    data_err = data[:, 2]
    z = np.logspace(7, 0, n)
    # z0 = data_z[-1]
    # z1 = data_z[0]
    # z = data_z

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

    exp_variables_init = np.array([exp_x1, exp_x2, exp_x3, ], dtype=np.float64)

    #solving for the different models
    pow_omega, pow_EoS_phi, z = solution_pow(pow_variables_init, N0, N1, N)
    exp_omega, exp_EoS_phi, z = solution_exp(exp_variables_init, N0, N1, N)

    figure, [ax1, ax2] = plt.subplots(2, 2)

    for omega, omega_name in zip(pow_omega, omega_names):
        ax1[0].plot(z, omega, label=rf"${omega_name}$")
    pretty_plot(ax1, 0, "omega")

    for omega, omega_name in zip(exp_omega, omega_names):
        ax1[1].plot(z, omega, label=rf"${omega_name}$")
    pretty_plot(ax1, 1, "omega")

    
    ax2[0].plot(z, pow_EoS_phi, label=rf"$\omega_\phi$")
    pretty_plot(ax2, 0, "EoS")

    ax2[1].plot(z, exp_EoS_phi, label=rf"$\omega_\phi$")
    pretty_plot(ax2, 1, "EoS")
    plt.show()

    pow_H0 = hubble(pow_omega, N, pow_EoS_phi)
    exp_H0 = hubble(exp_omega, N, exp_EoS_phi)

    flat_omega_m0 = 0.3
    flat_H0 = lcdm_model(flat_omega_m0, z)

    fig, ax = plt.subplots()
    ax.plot(z, pow_H0, label="Inverse power")
    ax.plot(z, exp_H0, label="Exponential")
    ax.plot(z, flat_H0, label=r"$\Lambda$CDM")
    pretty_plot(ax, 0, "hubble")
    plt.show()

    
    pow_age = age(N, pow_H0)
    exp_age = age(N, exp_H0)
    flat_age = age(N, flat_H0)

    fig, ax = plt.subplots()
    ax.plot(z, pow_age, label="Inverse power")
    ax.plot(z, exp_age, label="Exponential")
    ax.plot(z, flat_age, label=r"$\Lambda$CDM")
    pretty_plot(ax, 0, "age")
    plt.show()
    #the current age of the universe (dimensionless)
    print("Inv-pow law:", pow_age[0], "Exp pot", exp_age[0], "lambda cdm", flat_age[0])

    zmin = 2
    
    pow_dL = d_L(N[z<=zmin], pow_H0[z<=zmin])
    exp_dL = d_L(N[z<=zmin], exp_H0[z<=zmin])

    
    
    fig, ax = plt.subplots()
    ax.plot(z[z<=zmin], pow_dL, label="Inverse power")
    ax.plot(z[z<=zmin], exp_dL, label="Exponential")
    pretty_plot(ax, 0, "dL")
    plt.show()


    

    ip_dL = interpolate.interp1d(data_z, data_dL, fill_value="extrapolate") # bounds_error=data_err,
    ip_dL = ip_dL(z[z<=zmin])

    ip_err = interpolate.interp1d(data_z, data_err, fill_value="extrapolate")
    ip_err = ip_err(z[z<=zmin])




    h = 0.7
    pow_dL_w = dL_with_units(pow_dL, h)
    exp_dL_w = dL_with_units(exp_dL, h)
    # print(np.sum(pow_dL_w), pow_dL_w.shape)

    popt, pcov = optimize.curve_fit(lcdm_model_fitting(h), data_z, data_dL, p0=0.3, sigma=data_err, bounds=(0, 1))
    # print(popt)
    print(popt[0])
    # optimized_H0 = lcdm_model(popt[0], z)
    # print(optimized_H0)
    # optimized_dL = d_L(N[z<=zmin], optimized_H0[z<=zmin])
    # optimized_dL_w = dL_with_units(optimized_dL, h)
    

    fig, ax = plt.subplots()
    ax.plot(z[z<=zmin], pow_dL_w, label="Inverse power")
    ax.plot(z[z<=zmin], exp_dL_w, label="Exponential")
    # ax.plot(z[z<=zmin], optimized_dL_w, label=r"Optimized $\lambda$CDM")
    # ax.scatter(data_z, data_dL, label="From data", c="black")
    ax.errorbar(data_z, data_dL, yerr=data_err, c="black")
    ax.plot(z[z<=zmin], ip_dL, label="Extrapolated data", c="black")
    pretty_plot(ax, 0, "dL*")
    plt.show()

    pow_chi = chisquare(ip_dL, pow_dL_w, ip_err)
    exp_chi = chisquare(ip_dL, exp_dL_w, ip_err)

    print(f"{pow_chi=}, {exp_chi=}")

    
    

