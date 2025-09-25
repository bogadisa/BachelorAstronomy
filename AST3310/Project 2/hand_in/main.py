import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.constants import G, sigma, k
from scipy.optimize import fsolve
from scipy import integrate
from cross_section import cross_section
from ode_solver import rk4
import pandas as pd

class DataLoader:
    def __init__(self, filename : str, n=1000) -> None:
        #loading the data from opacity, we will interpolate from this, so I reserve the names logX for later
        self.name = filename.split(".")[0]
        if self.name == "epsilon_branch":
            #epsilon_branch.txt contains the percentage of energy production per branch for a given temperature log_10(T)
            self.dlogT, self.dx = self.load_data(filename)
        else:
            self.dlogT, self.dlogR, self.dlogx = self.load_data(filename)
        self.n = n

        #for checking if the bounds were ever exceeded
        self.x, self.y = [], []

    @property
    def data(self) -> list:
        if self.name == "epsilon_branch":
            return [self.dlogT, self.dx]
        else:
            return [self.dlogT, self.dlogR, self.dlogx]
            
    
    #i for interpolated
    @property
    def data_interp(self) -> list:
        try:
            if self.name == "epsilon_branch":
                return [self.logT, self.x]
            else:
                return [self.logT, self.logR, self.logx]
        except AttributeError:
            n = self.n
            if self.name == "epsilon_branch":
                dlogT, dx = self.data
                self.logR = np.ones(n)
                self.logT = np.linspace(dlogT[0], dlogT[-1]+1, n)
                self.logx = np.log(self(self.logT, self.logR))
            else:
                dlogT, dlogR, dlogx = self.data
                self.logR = np.linspace(dlogR[0], dlogR[-1]+1, n)
                self.logT = np.linspace(dlogT[0], dlogT[-1]+1, n)
                self.logx = self(self.logT, self.logR)
            


            return [self.logT, self.logR, self.logx]
    
    def interpolate(self) -> callable:
        try:
            return self.func

        except AttributeError:
            if self.name == "epsilon_branch":
                dlogT, dlogx = self.data
                dlogR = np.ones(len(dlogT))
                def func(logT, dlog, dlogT=dlogT, dlogx=dlogx):
                    return dlogx[np.argmin(abs(dlogT - logT))]

                self.func = func

                return self.func

            else:
                dlogT, dlogR, dlogx = self.data 
                self.func = RectBivariateSpline(dlogT, dlogR, dlogx)

            return self.func
        
    def __call__(self, x, y) -> np.ndarray:
        func = self.interpolate()
        self.x.append(x)
        self.y.append(y)
        return func(x, y)

    def load_data(self, filename:str) -> tuple:
        if self.name == "epsilon_branch":
            data = np.loadtxt(filename, skiprows=1)
            logT = data[:, 0]
            x = data[:, 1:]
            return logT, x
        else:
            logT = np.loadtxt(filename, skiprows=2, usecols=0)
            with open(filename, "r") as f:
                lines = f.readlines()

            ncols = len(lines[0].rstrip().split())

            data = np.loadtxt(filename, usecols=range(1, ncols-1))
            logR = data[0]
            logx = data[1:]
        
        return logT, logR, logx

    def check_out_of_bounds(self) -> None:
        print(f"Out of bounds check for {self.name}")
        bounds = ["Upper", "Lower"]

        x, y = np.array(self.x), np.array(self.y)
        if self.name == "epsilon_branch":
            dlogT, dx = self.data
            bounds_exceeded_y = [False, False]
        else:
            dlogT, dlogR, dlogx = self.data 
            bounds_exceeded_y = [max(y) > max(dlogR), min(y) < min(dlogR)]
        bounds_exceeded_x = [max(x) > max(dlogT), min(x) < min(dlogT)]
        for i, bound in enumerate(bounds):
            if bounds_exceeded_x[i]:
                print(bound, "bound exceeded for T")
            if bounds_exceeded_y[i]:
                print(bound, "bound exceeded for R")

        if sum(bounds_exceeded_x) == 0 and sum(bounds_exceeded_y) == 0:
            print("No bounds were exceeded")

    def plot_data(self) -> None:
        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes
        dlogT, dlogR, dlogx = self.data

        xx, yy = np.meshgrid(dlogT, dlogR)

        ax1.pcolormesh(xx.T, yy.T, dlogx)

        logT, logR, logx = self.data_interp

        xx, yy = np.meshgrid(logT, logR)

        cont = ax2.pcolormesh(xx.T, yy.T, logx)
        cbar = fig.colorbar(cont, ax=axes.ravel().tolist())
        ax1.set_xlabel(r"$\log_10$ R [g $cm^{-3}$]") ; ax2.set_xlabel(r"$\log_10$ R [g $cm^{-3}$]")
        ax1.set_ylabel(r"$\log_10$ T [K]")
        cbar.set_label(r"$\log_10$ $\kappa$ [$cm^2$ $g^{-1}$]")
        plt.show()

    def sanity_check(self, x, y, target) -> None:
        func = self.interpolate()

        #func(x, y) doesnt work correctly, as there are duplicate values in the arrays
        prediction = np.array([func(x[i], y[i], grid=False) for i in range(len(x))])

        rel_err = abs((target - prediction)/prediction)*100

        #formated for latex
        data = {r"$\log_{10} T$" : x, r"$\log_{10} R$ (cgs)" : y, rf"$\log_{10}$ {self.name} target (cgs)" : target, 
                rf"$\log_{10}$ {self.name} prediction (cgs)" : prediction, r"$error_{rel}$ (\%)" : rel_err}
        df = pd.DataFrame(data=data)
        print("-"*len("  ".join(df.columns))+"--")
        print(df.to_string(index=False))
        print("-"*len("  ".join(df.columns))+"--")
        print()
        #saves the scores so that it can be copy-pasted into latex
        df.to_csv(f"{self.name}_sanity", sep = "&", index=False, float_format="%.3f")




class Sun:
    m_u = 1.6605e-27
    ad = 2/5
    #sun params
    Ls = 3.846e26 # W
    rs = 6.96e8 # m
    Ms = 1.989e30 #kg
    rhos = 1.408e3 #kg m^-3
    Ps = 3.45e16 #Pa
    def __init__(self, fractions : list, L0=1.0, R0=1.0, M0=1.0, rho0=1.42e-7, T0=5770.0, P0=1, sanity=None) -> None:
        self.X, self.Y_32, self.Y, self.Z_73, self.Z_74, self.Z_147 = fractions
        #so that I get closer values to example 1
        if sanity is not None and sanity == 1:
            self.mu = 0.6
        else:
            self.mu = 1/(2*self.X + 3/4*(self.Y + self.Y_32) + 4/7*self.Z_73 + 5/7*self.Z_74 + 8/14*self.Z_147)

        self.sanity = sanity

        self.L0 = L0*self.Ls
        self.R0 = R0*self.rs
        self.M0 = M0*self.Ms
        #in the example, rho is not given as a fraction of density in the solar core
        if sanity is not None:
            self.rho0 = rho0
        else:
            self.rho0 = rho0*self.rhos
        self.T0 = T0
        self.P0 = self.find_P0(self.rho0, self.T0)*P0

        #will be used in the solver
        self.initial = np.array([self.L0, self.R0, self.P0, self.T0])

        #a way to fetch data
        self.OpacityClass = DataLoader("opacity.txt")
        self.EpsilonClass = DataLoader("epsilon.txt")
        self.EpsilonBranchClass = DataLoader("epsilon_branch.txt")


    def __call__(self, m : float, X : np.ndarray) -> np.ndarray:
        L, r, P, T = X
        # all the variables we need to solve the diff equations
        rho = self.find_rho(P, T)
        kappa = self.find_kappa(T, rho)
        # kappa = 3.98
        epsilon = self.find_epsilon(T, rho)
        epsilon_branch = self.find_epsilon_branch(T)
        H_P = self.find_H_P(r, T, m)
        stable = self.find_stable(kappa, H_P, L, rho, T, r)
        U = self.find_U(T, rho, r, kappa, H_P, m, P)
        xi = self.find_xi(stable, U, H_P)
        F_C = self.find_F_C(rho, T, H_P, r, xi, m)
        F_R = self.find_F_R(r, L, F_C)
        
        #the diff equations
        dr = 1/(4*np.pi*r**2*rho)
        dP = -G*m/(4*np.pi*r**4)
        dL = epsilon

        if stable <= self.ad:
            dT = - 3*kappa*L/(256*np.pi**2*sigma*r**4*T**3)
            star = stable
            F_C = 0
        else:
            star = self.find_star(xi, stable, U, H_P)
            dT = star * T/P * dP

        #nabla_p
        p = star - xi**2
        if self.sanity is not None and self.sanity == 1:
            print("\n Initial conditions:")
            print(f"{T=:.3E}, {rho=}, {r/self.rs=}, {m/self.Ms=}, {kappa=}, {self.mu=}")
            F_c_percent = F_C/(L/(4*np.pi*r**2))

            print("\n Results:")
            print(f"{H_P=:.3E}, {U=:.3E}, {xi=:.3E}, {F_C/(F_C + F_R)=}")

            print(f"\n ad < p < * < stable")
            print(f"{self.ad} < {p} < {star} < {stable:.3E}")
            quit()

        #only stores data once for every iteration of rk
        if (self.i)%5 == 0:
            self.F_C.append(F_C) 
            self.F_R.append(F_R)
            self.nablas.append(np.array([stable, star, self.ad]))
            self.epsilon_branch.append(epsilon_branch)

        self.i += 1
            
        return np.array([dL, dr, dP, dT])

    #not radius
    def find_R(self, rho, T) -> float:
        return rho/(T*1e-6)**3

    def find_rho(self, P, T) -> float:
        a = 7.6e-16 #4*simga/c
        P_gas = P - (a*T**4)/3
        
        rho = P_gas*self.mu*self.m_u/T/k
        return rho

    def find_P0(self, rho, T) -> float:
        a = 7.6e-16 #4*simga/c
        return rho*k*T/(self.mu*self.m_u) + (a*T**4)/3

    def find_kappa(self, T, rho) -> float:
        rho_cgs = rho/1000 #converts to cgs units
        R = self.find_R(rho_cgs, T)
        
        logR = np.log10(R)
        logT = np.log10(T)
        
        #convert back to SI units
        kappa = 10**self.OpacityClass(logT, logR)[0][0] / 10
        return kappa

    def find_epsilon(self, T, rho) -> float:
        rho_cgs = rho/1000 #converts to cgs units
        R = self.find_R(rho_cgs, T)
        
        logR = np.log10(R)
        logT = np.log10(T)
        
        #convert back to SI units
        epsilon = 10**self.EpsilonClass(logT, logR)[0][0] / 10000
        return epsilon
    
    def find_epsilon_branch(self, T) -> np.ndarray:
        logT = np.log10(T)
        #already comes as a percentage
        epsilon_branch_precent = self.EpsilonBranchClass(logT, None)
        return epsilon_branch_precent


    def find_stable(self, kappa, H_P, L, rho, T, r) -> float:
        #easier to read
        top = 3*L*kappa*rho*H_P
        bot = 64*sigma*T**4*np.pi*r**2

        return top/bot

    def find_xi(self, stable, U, H_P) -> float:
        lm = H_P
        a = 1 ; b = U/lm**2 ; c = b**2*lm*4/lm; d = b*(stable - self.ad)

        coeff = [a, b, c, d]

        #the only real solution according to wolfram alpha
        # Z is a term that shows up multiple times
        Z = np.cbrt(np.sqrt((27*a*a*d + 9*a*b*c - 2*b*b*b)**2 + 4*(3*a*c - b*b)**3) + 27*a*a*d + 9*a*b*c - 2*b*b*b)
        # the actual solution
        xi = Z/(3*2**(1/3)*a) - 2**(1/3)*(3*a*c - b*b)/(3*a*Z) - b/(3*a)

        return xi

    def find_H_P(self, r, T, m) -> float:
        g = G*m/r**2

        H_P = k*T/(g*self.mu*self.m_u)

        return H_P
        
    def find_star(self, xi, stable, U, H_P) -> float:
        lm = H_P 
        star = stable - lm**2/U*xi**3
        return star

    def find_U(self, T, rho, r, kappa, H_P, m, P) -> float:
        c_P = 5/2*k/(self.mu*self.m_u)
        g = G*m/r**2

        delta = 1

        #easier to read
        top = 64*sigma*T**3
        bot = 3*kappa*rho**2*c_P

        U = top/bot*np.sqrt(H_P/(g*delta))
        return U

    def find_F_C(self, rho, T, H_P, r, xi, m) -> float:
        c_P = 5/2*k/(self.mu*self.m_u)
        g = G*m/r**2
        
        delta = 1
        lm = H_P

        F_C = rho*c_P*T*np.sqrt(g*delta)*H_P**(-3/2)*(lm/2)**2*xi**3    
        return F_C

    def find_F_R(self, r, L, F_C):
        F_R = L/(4*np.pi*r*r) - F_C

        return F_R

    def solve(self, initial=None, Mspan=None) -> None:
        if initial is None:
            initial = self.initial
        if Mspan is None:
            Mspan = [self.M0, 0.00005*self.M0]
        # for storing values throughout
        self.r, self.L, self.F_C, self.F_R, self.epsilon_branch, self.nablas = [], [], [], [], [], []
        # for keeping track of when to store values
        self.i = 0


        names = ["L", "R", "P", "T", "M"]
        condition = np.array([0.0005*self.L0, 0, 0, 0])
        y, t = rk4(f=self, x0=initial, tspan=Mspan, condition=condition, p=0.005, names=names)
        L = y[:, 0] ; r = y[:, 1] ; P = y[:, 2] ; T = y[:, 3]
        M = t

        self.L, self.r, self.P, self.T = L, r, P, T
        self.M = M

        self.EpsilonClass.check_out_of_bounds() ; self.OpacityClass.check_out_of_bounds() ; self.EpsilonBranchClass.check_out_of_bounds()
       
    def plot_data(self, data_to_plot="cross_section") -> None:
        if data_to_plot == "cross_section":
            params = np.array([self.r, self.L, np.array(self.F_C)])

            #sometimes the program crashes due to taking the square root of a negative number
            #this gets rid of the last few iterations that are affected
            finite_params = params[:, ~np.isnan(params).any(axis=0)]

            cross_section(finite_params[0], finite_params[1], finite_params[2], show_every=int(20*self.i*0.5e-4+1))

        elif data_to_plot == "nablas":
            nablas = np.array(self.nablas)
            
            #allows you to plot when program crashes
            nablas_ = nablas[~np.isnan(nablas).any(axis=1)]
            r_ = np.ones(nablas_.shape).T*self.r[~np.isnan(nablas).any(axis=1)]/self.rs

            nabla_labels = [r"$\nabla_{stable}$", r"$\nabla_{*}$", r"$\nabla_{ad}$"]
            
            plt.plot(r_.T, nablas_, label=nabla_labels)
            plt.legend()
            plt.yscale("log")
            plt.ylabel(r"$\nabla$")
            plt.xlabel(r"$R/R_\odot$")
            plt.show()

        elif data_to_plot == "flux":
            r, F_C, F_R = self.r, np.array(self.F_C), np.array(self.F_R)

            F_tot = F_C + F_R
            plt.plot(r/self.rs, F_C/F_tot, label=r"$F_{con}$")
            plt.plot(r/self.rs, F_R/F_tot, label=r"$F_{rad}$")
            plt.ylabel(r"$F_x/F_{tot}$")
            plt.xlabel(r"$R/R_\odot$")
            plt.legend()
            plt.show()

        elif data_to_plot == "param":
            L, r, P, T= self.L, self.r, self.P, self.T
            M = self.M
            rho = self.find_rho(P, T)
            fig, axs = plt.subplots(2, 3)
            for ax in axs.flatten():
                ax.set_xlabel(r"$R/R_\odot$ [*]")

            axs[0, 0].set_ylabel(r"$L/L_\odot [*]$")
            axs[0, 0].plot(r/self.rs, L/self.Ls, label="L")
            axs[0, 0].legend()
            axs[0, 0].invert_xaxis()

            axs[0, 1].set_ylabel(r"$P [Pa]$")
            axs[0, 1].plot(r/self.rs, P, label="P")
            axs[0, 1].legend()
            axs[0, 1].invert_xaxis()
            axs[0, 1].set_yscale("log")

            axs[0, 2].set_ylabel(r"$\rho/\rho_\odot [*]$")
            axs[0, 2].plot(r/self.rs, rho/self.rhos, label=r"$\rho$")
            axs[0, 2].legend()
            axs[0, 2].invert_xaxis()
            axs[0, 2].set_yscale("log")

            axs[1, 0].set_ylabel(r"$T [K]$")
            axs[1, 0].plot(r/self.rs, T, label="T")
            axs[1, 0].legend()
            axs[1, 0].invert_xaxis()

            axs[1, 1].set_ylabel(r"$M/M_\odot [*]$")
            axs[1, 1].plot(r/self.rs, M/self.Ms, label="M")
            axs[1, 1].legend()
            axs[1, 1].invert_xaxis()

            fig.delaxes(axs[1, 2])
            plt.show()

        if data_to_plot == "branches":
            fig, ax = plt.subplots(1, 1)
            epsilon_branch, r, T = np.array(self.epsilon_branch), self.r, self.T
            
            chain_names = ["PP1", "PP2", "PP3", "CNO"]
            for i in range(len(chain_names)):
                ax.plot(r/self.rs, epsilon_branch[:, i], label=chain_names[i])

            ax.invert_xaxis()
            ax.legend()
            ax.set_ylabel(r"$\epsilon/\epsilon_{tot} [*]$")
            ax.set_xlabel(r"$R/R_\odot [*]$")
            plt.show()

if __name__ == "__main__":
    #from the sanity checks
    logT_kappa = np.array([3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 3.770, 3.775, 3.780, 3.795, 3.800])
    logR_kappa = np.array([-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95, -5.80, -5.75, -5.70, -5.55, -5.50])
    target_kappa = np.array([-1.55, -1.51, -1.57, -1.61, -1.67, -1.33, -1.20, -1.02, -1.39, -1.35, -1.31, -1.16, -1.11])

    logT_epsilon = np.array([3.750, 3.755])
    logR_epsilon = np.array([-6.00, -5.95])
    target_epsilon = np.array([-87.995, -87.267])

    OpacityClass = DataLoader("opacity.txt")
    OpacityClass.sanity_check(logT_kappa, logR_kappa, target_kappa) #comment this line to remove Opacity sanity check
    EpsilonClass = DataLoader("epsilon.txt")
    EpsilonClass.sanity_check(logT_epsilon, logR_epsilon, target_epsilon) #comment this line to remove Epsilon sanity check
    # quit()
    # OpacityClass.plot_data()

    fractions = [0.7, 1e-10, 0.29, 1e-7, 1e-7, 1e-11]
    
    SunClass = Sun(fractions, P0=100, R0=0.9, M0=0.9)

    #sanity=1 -> Example 1
    #uncomment below line to run sanity check for example 1
    # SunClass = Sun(fractions, R0=0.84, M0=0.99, T0=0.9e6, rho0=55.9, sanity=1)
    SunClass.solve()

    # L, r, P, T = SunClass.optimize()
    # print(L, r, P, T)

    SunClass.plot_data(data_to_plot="flux")
    SunClass.plot_data(data_to_plot="nablas")
    SunClass.plot_data(data_to_plot="param")
    SunClass.plot_data(data_to_plot="branches")
    SunClass.plot_data()