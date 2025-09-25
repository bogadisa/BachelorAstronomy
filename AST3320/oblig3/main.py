import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, c, G, pi
from scipy.integrate import odeint, solve_ivp, cumtrapz


class Inflation:
    def __init__(self, init_val : dict, potential : str = "Default"):
        self.init_val = init_val
        self.potential = potential

    #a solver function that can be ussed for any of the tasks
    def __solver(self, psi : list, tau : float, v_func : callable) -> list:
        psi_i = self.init_val["psi_i"]

        #dv = dv/dpsi
        v, dv = v_func(psi)
        h = self.find_h(psi, v)

        #psi[0] = psi, psi[1] = dpsi/dtau
        return [psi[1], -3*h*psi[1] - dv, h]

    #from the porject description
    def find_h(self, psi : list, v : float) -> float:
        h = np.sqrt(8*pi/3*(0.5*psi[1]**2 + v))
        return h

    #the phi^2 potential converted to unitless variables and its differential
    def default_v(self, psi : list) -> list:
        psi_i = self.init_val["psi_i"]
        
        #psi[0] = psi, psi[1] = dpsi/dtau
        v = 3/(8*pi*psi_i**2)*psi[0]**2
        return [v, 2*v/psi[0]]

    #the starobinsky potential and its derivative
    def starobinsky(self, psi : list) -> list:
        psi_i = self.init_val["psi_i"]
        c = np.sqrt(16*pi/3)

        #the top after differential
        top = 6*c*np.exp(-c*psi[0])*(1 - np.exp(-c*psi[0]))
        #the top for the regular potential
        top_2 = 3*(1 - np.exp(-c*psi[0]))**2

        #both expressions share the same bottom
        bot = 8*pi*(1 - np.exp(-c*psi_i))**2

        return (top_2/bot, top/bot)

    #solves the differential equations numerically, takes the tau range and task as arguments,
    #this makes running code quick and easy
    def solve(self, tau_range : float, task : str) -> None:
        psi_i = self.init_val["psi_i"]
        d_psi_i = self.init_val["d_psi_i"]
        h_int_i = 0 # ln(ai/ai)=0, also refered to as h_int
        y0 = [psi_i, d_psi_i, h_int_i]

        # sets the temporal resolution
        n = 10000
        t = np.linspace(0, tau_range, n)

        #chooses the right potential depending on the task
        if task in ["e", "f", "g", "i", "j", "k"]:
            v_func = self.default_v
        else:
            v_func = self.starobinsky

        y = odeint(self.__solver, y0, t, args=(v_func,))

        [self.psi, self.d_psi, self.h_int], self.tau = y.T, t
        self.v, self.dv = v_func(y.T)

    #finds the total remaining e-folds
    def find_N(self, h_int, epsilon):
        N_tot = h_int[epsilon >= 1][0] #n_tot = ln(af/ai), af=a(epsilon=1)
        print(f"Inflation lasted for {N_tot=:.3f} e-folds.")
        
        N = N_tot - h_int
        return N

    #epsilon varies depending on the potentials
    def find_epsilon(self, psi):
        if self.potential == "Default":
            return 1/(4*pi*psi**2)  
        elif self.potential == "starobinsky":
            exp_y = np.exp(-np.sqrt(16*pi/3)*psi)
            epsilon = 4/3 * (exp_y/(1 - exp_y))**2
            return epsilon
        #in case something is wrong
        else:
            print(f"{self.potential} potential has not been implemented")

    def find_eta(self, psi):
        if self.potential == "Default":
            return self.find_epsilon(psi)
        elif self.potential == "starobinsky":
            exp_y = np.exp(-np.sqrt(16*pi/3)*psi)
            eta = 4/3 * (2*exp_y**2 - exp_y)/(1 - exp_y)**2
            return eta
        else:
            print(f"{self.potential} potential has not been implemented")

    #analytical expression from the lecture notes, after being converted to unitless variables
    def SRA_notes(self, tau):
        psi_i = self.init_val["psi_i"]

        pi4 = 1/4/pi

        psi = psi_i - pi4/psi_i*tau
        d_psi = -pi4/psi

        return psi, d_psi

    #the equation of state parameter w_phi
    def EoS(self, psi, d_psi):
        psi_i = self.init_val["psi_i"]

        #needs to be in list form
        psi = [psi]
        v, dv = self.default_v(psi)

        p = 0.5*d_psi**2 - v
        rho = 0.5*d_psi**2 + v

        w = p/rho
        return w

    #plots all information, depending on what task
    def plot(self, task = "e"):
        #all potentially useful information
        psi, d_psi, h_int, tau = self.psi, self.d_psi, self.h_int, self.tau
        v = self.v
        psi_i = self.init_val["psi_i"]

        #some tasks require subplots, this generalizes the code used for all tasks
        fig, ax = plt.subplots(1, 1)
        ax = np.array([ax], dtype=type(ax))

        #finds analytical solution
        psi_a, d_psi_a = self.SRA_notes(tau)

        if task == "e" or task == "f":
            plt.plot(tau, psi, label="Numerical", color="tab:blue")
            #plots the first derivative, usefull for debugging
            # plt.plot(tau, d_psi, label=r"$\frac{\partial \psi}{\partial \tau}$", color="tab:orange")

            plt.ylim((-1, 10))
            if task == "f":

                plt.plot(tau, psi_a, label="SRA from notes", color="tab:blue", linestyle="--")
                #plots the first derivative, usefull for debugging
                # plt.plot(tau, d_psi_a, label=r"$\dot \psi$ (notes)", color="tab:orange", linestyle="--")
                plt.ylim((-10, 10))

            plt.ylabel(r"$\psi$")
            plt.legend()

        if task == "g":

            #epsilon for the numerical and analytical solution
            epsilon = self.find_epsilon(psi)
            epsilon_a = self.find_epsilon(psi_a)

            N = self.find_N(h_int, epsilon)

            plt.plot(tau[epsilon<=1], epsilon[epsilon<=1], label="Numerical")
            plt.plot(tau[epsilon_a<=1], epsilon_a[epsilon_a<=1], label="SRA from notes", color="tab:blue", linestyle="--")
            plt.axvline(tau[epsilon>=1][0], color="black", linestyle="--", label="N=0")
            plt.ylim((-1.1, 1.1))
            plt.ylabel(r"$\epsilon$")
            plt.legend()


        if task == "i":
            w = self.EoS(psi, d_psi)

            plt.plot(tau, w)
            plt.ylabel("w")
            plt.legend()

        if task == "j" or task == "n-j":
            plt.close()
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
            fig.suptitle(r"$\epsilon$ and $\eta$ as function N")

            # N = self.N(tau)
            # N = self.find_N(psi)

            epsilon = self.find_epsilon(psi)
            eta = self.find_eta(psi)
            N = self.find_N(h_int, epsilon)


            # ax[0].plot(N[epsilon<=1], epsilon[epsilon<=1], label=r"$\epsilon$")
            ax[0].plot(N, epsilon, label=r"$\epsilon$")
            ax[0].set_ylim((-1.1, 1.1))
            # ax[0].axvline(0, color="black", linestyle="--", label="N=0")
            ax[0].legend(loc="upper left")


            ax[1].plot(N, eta, label=r"$\eta$")
            ax[1].set_ylim((-1.1, 1.1))
            # ax[1].axvline(0, color="black", linestyle="--", label="N=0")
            # if task == "n-j":
            #     ax[1].set_xscale("log")
            ax[1].legend(loc="upper left")

        if task == "k" or task == "n-k":
            
            #finds epsilon and eta, no need to specify the correct task
            #it has already been handled
            epsilon = self.find_epsilon(psi)
            eta = self.find_eta(psi)
            N = self.find_N(h_int, epsilon)

            #from the project description
            n = 1 - 6*epsilon + 2*eta
            r = 16*epsilon

            #finds the indecies for all N within the range 50<=N<=60
            N_range = np.argwhere((N <= 60) & (N >= 50))

            plt.plot(n[N_range], r[N_range])

        
        if task == "m":
            #closes the figure created for all tasks
            plt.close()
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
            ax[0].plot(tau, psi, label=r"$\psi$", color="tab:blue")
            ax[0].set_ylim((-1, 2.5))
            ax[0].legend(loc="lower right")

            ax[1].plot(tau, h_int, label=r"$\ln{\frac{a}{a_i}}$")
            ax[1].legend(loc="lower right")


            


        

        #some general plotting stuff
        if (task != "j" and task != "n-j") and (task != "k" and task != "n-k"):
            ax[-1].set_xlim(0, tau[-1])
            ax[-1].set_xlabel(r"$\tau$")
        elif task == "j" or task == "n-j":
            N_tot = max(N)
            ax[-1].set_xlim(-5*N_tot/100, N_tot) #5% extra after 0
            ax[-1].invert_xaxis()
            ax[-1].set_xlabel(r"$N$")
        else:
            ax[-1].set_xlabel("n")
            ax[-1].set_ylabel("r")


        plt.show()


def main():
    #defines
    init_val = {
        "psi_i" : 8.9251,
        "d_psi_i" : 0,
    }
    model = Inflation(init_val)

    #asks you in the terminal for what task you wish to run
    task = input("\n What task do you wish to run? [e, f, g, i, j, k, m, n-j, n-k] ")
    while task not in "e, f, g, i, j, k, m, n-j, n-k":
        print(f"{task=} has not been implemented yet or does not exist, please try again with a task from the list")
        task = input("\n What task do you wish to run? [e, f, g, i, j, k, m, n-j, n-k] ")

    #all the tasks that use the phi^2 potential
    if task in ["e", "f", "g", "i", "j", "k"]:
        model.solve(2000, task=task)
        model.plot(task=task)
    #all the tasks that use the starobinsky potential
    else:
        init_val["psi_i"] = 2
        model = Inflation(init_val, potential="starobinsky")
        model.solve(3000, task=task)
        model.plot(task=task)

if __name__ == "__main__":
    main()