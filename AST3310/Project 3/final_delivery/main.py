# visualiser
import FVis3 as FVis

import numpy as np
from scipy.constants import k, m_u, G
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class convection2D:
    #sun values
    R0 = 6.96e8 #m
    m = 1.989e30 #kg

    #photosphere values
    T0 = 5778 #K
    P0 = 1.8e4 #Pa

    #as defined in text
    g = G*m/R0**2 # note: it is here defined positive, meaning the - needs to be added elsewhere
    mu = 0.61
    gamma = 5/3
    nabla = 2/5 + 1e-5

    def __init__(self, pertubation=False, save_snapshots=False):

        """
        define variables
        """ 
        self.pertubation = pertubation
        self.save_snapshots = save_snapshots

        Nx = 300
        Ny = 100
        self.Nx, self.Ny = Nx, Ny

        x0 = 0 ; x1 = 12e6 #m
        # y0 = 0 ; y1 = -4e6 #m
        y0 = -4e6 ; y1 = 0 #m
        R0 = self.R0

        x = np.linspace(x0, x1, Nx)
        y = np.linspace(y0, y1, Ny) + R0
        self.dx = (x1 - x0)/Nx
        self.dy = (y1 - y0)/Ny

        self.rx, self.ry = np.meshgrid(x, y)

        self.initialise()


    def initialise(self) -> None:

        """
        initialise temperature, pressure, density and internal energy
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        rx, ry = self.rx, self.ry

        mu, nabla, g, R0 = self.mu, self.nabla, self.g, self.R0

        T0, P0 = self.T0, self.P0



        if self.pertubation:
            left_corner, centre, right_corner = [np.zeros((Ny, Nx))]*3
            
            #2D gaussian
            def gaussian_2D(pos, mu, sigma):
                gaussian = np.exp(-np.sum(0.5*(pos - mu)**2/sigma**2, axis=2))
                return gaussian/np.max(gaussian)

            pos = np.empty(rx.shape + (2,))
            pos[:, :, 0] = rx
            pos[:, :, 1] = ry

            #centre of the box
            mu_x = (rx[0, -1] - rx[0, 0])/2
            # mu_y = (ry[-1, 0] - ry[0, 0])/2 + R0

            #bottom of the box
            mu_y = ry[0, 0] #R0
            # mu_y = ry[-1, 0]
            
            Sigma = np.array([ 5e6 ,  1e6])

            # -----------------------------
            # centre
            # mu_y = ry[0, 0] + (ry[-1, 0] - ry[0, 0])/2
            Mu = np.array([mu_x , mu_y])

            Sigma = np.array([ 3e5 , 3e6])

            centre = gaussian_2D(pos, Mu, Sigma)

            # -----------------------------
            #sides

            # bottom corner
            # Mu = np.array([mu_x + mu_x, mu_y])
            # left_corner = gaussian_2D(pos, Mu, Sigma)
            # Mu = np.array([mu_x - mu_x, mu_y])
            # right_corner = gaussian_2D(pos, Mu, Sigma)
            

            # middle sides
            # Sigma = np.array([ 2e5, 2e6])
            Sigma = np.array([ 3e5 , 3e6])
            # mu_y = ry[0, 0] + (ry[-1, 0] - ry[0, 0])/2
            Mu = np.array([mu_x + 2*mu_x/3, mu_y])
            # left_corner = gaussian_2D(pos, Mu, Sigma)
            # Sigma = np.array([ 2e5,  1e6])
            Sigma = np.array([ 3e5 , 3e6])
            Mu = np.array([mu_x - 2*mu_x/3, mu_y])
            # right_corner = gaussian_2D(pos, Mu, Sigma)

            #similiar poles
            T = left_corner + centre + right_corner

            #opposite poles
            # T = -left_corner + right_corner   #Doesnt make physical sense, since it means the spatial temperature gradient isnt continious

            # -----------------------------

            #positive pertubations
            T_pertubation = 14*T0*T
            self.T_pert = T_pertubation

            #negative pertubations
            # T = -2*T0*T/np.max(T)
            



        T = T0 - nabla*mu*m_u*g*(ry - R0)/k

        # P = np.exp(mu*m_u*G*self.m/(ry*k*T)) + P0
        # P = P0*np.exp(-mu*m_u*g/(k*T)*(ry - R0))
        P = P0*(T/T0)**(1/nabla)
        if self.pertubation:
            T += T_pertubation
            print("\n Pertubations added \n")

        else:
            print("\n Proceeding with no pertubations")
        # P = P0*(T/T0)**(1/nabla)

        #from ideal gas law
        rho = self.find_rho(P, T)
        # if self.pertubation:
        #     T += T_pertubation
        #     print("\n Pertubations added \n")

        # else:
        #     print("\n Proceeding with no pertubations")

        self.T, self.rho, self.P = T, rho, P

        u = np.zeros((Nx, Ny)).T
        w = np.zeros((Nx, Ny)).T

        self.u, self.w = u, w

        e = self.find_e(rho, T)

        self.e = e

        self.i = 0
        self.dt = 0

    def timestep(self, phi, dphi, v, dv) -> float:

        """
        calculate timestep
        """

        p = 0.1

        rel_phi = np.abs(dphi/phi)
        # gets of all the places where phi was 0 and caused the program to divide by 0
        rel_phi[rel_phi >= float("+inf")] = 0
        
        # finds the max once
        delta_phi = np.max(rel_phi, axis=1)
        # finds the max second, but this time excluding NaN
        delta_phi = np.max(delta_phi[~np.isnan(delta_phi).any(axis=1)], axis=1)
        
        #transposes to get right shape
        #dv can never be zero, so no need to worry about overflow and such
        rel_xy = np.abs(v.T/dv).T
        delta_xy = np.max(np.max(rel_xy, axis=1), axis=1)


        delta = np.max(np.concatenate((delta_phi, delta_xy)))
        # delta will always equal 0 on the first iteration
        if delta == 0:
            dt = 0.01
        else:
            dt = p/delta

        #so that the program isnt too slow
        if dt < 0.1:
            dt = 0.1

        # this means the program crashed, so there is no point in continuing simulation
        if np.isnan(dt):
            dt = 1000
        
        return dt

    def boundary_conditions(self) ->  None:

        """
        boundary conditions for energy, density and velocity
        """
        # vertical velocity condition
        w = self.w
        w[[0, -1], :] = 0

        # horizontal velocity condition
        u = self.u
        u[ 0] = (-u[ 2] + 4*u[ 1])/3
        u[-1] = (-u[-3] + 4*u[-2])/3

        # density and energy conditions
        e, T, rho = self.e, self.T, self.rho
        gamma, mu, g, dy = self.gamma, self.mu, self.g, self.dy
        
        dP_dy = -g*rho

        de_dy_upper = 1/(gamma - 1)*dP_dy[0]
        de_dy_lower = 1/(gamma - 1)*dP_dy[-1]

        # the following two ways of defining e works, neither seem to do anything different
        # I choose to do the first as it seems more readable
        # e[ 0] = (-2*dy*de_dy_upper - e[2]  + 4*e[1] )/3
        # e[-1] = ( 2*dy*de_dy_lower - e[-3] + 4*e[-2])/3

        e[0]  = (4*e[ 1] - e[ 2])/(-2*dy*g*mu*m_u/k/T[ 0] + 3)
        e[-1] = (4*e[-2] - e[-3])/( 2*dy*g*mu*m_u/k/T[-1] + 3)
        
        rho[ 0] = (gamma - 1)*mu*m_u/(k*T[0] )*e[ 0]
        rho[-1] = (gamma - 1)*mu*m_u/(k*T[-1])*e[-1]

        self.u[:], self.w[:], self.e[:], self.rho[:] = u, w, e, rho

    def central_x(self, var : np.ndarray) -> np.ndarray:

        """
        central difference scheme in x-direction
        """
        dx = self.dx

        d_var = 0.5*(np.roll(var, -1, axis=1) - np.roll(var, 1, axis=1))/dx
        return d_var

    def central_y(self, var : np.ndarray) -> np.ndarray:

        """
        central difference scheme in y-direction
        """
        dy = self.dy

        inner_sol = 0.5*(var[2:] - var[:-2])/dy
        
        #from the boundary condition hint
        upper_sol = np.array([0.5*( -var[ 2] + 4*var[ 1] - 3*var[ 0])/dy])
        lower_sol = np.array([0.5*(3*var[-1] - 4*var[-2] +   var[-3])/dy])

        #combine into one matrix
        d_var = np.concatenate((upper_sol, inner_sol, lower_sol))
        return d_var

    def upwind_x(self, var : np.ndarray, v : np.ndarray) -> np.ndarray:

        """
        upwind difference scheme in x-direction
        """
        dx = self.dx
        #uses roll and where to easily compute the differential
        d_var = np.where(v >= 0, (var - np.roll(var, 1, axis=1))/dx, (np.roll(var, -1, axis=1) - var)/dx)
        return d_var

    def upwind_y(self, var : np.ndarray, v : np.ndarray) -> np.ndarray:

        """
        upwind difference scheme in y-direction
        """
        dy = self.dy

        _var = var[1:-1]
        _v = v[1:-1]

        #as defined in assignmet, does not do anything about boundaries
        inner_sol = np.where(_v >= 0, (_var - var[:-2])/dy, (var[2:] - _var)/dy)

        #This prioritises using upwind when possible, otherwise resorting to 3-point
        upper_sol = np.where(v[ 0] >= 0, 0.5*( -var[2]  + 4*var[1]  - 3*var[0])/dy, (var[1] - var[0])/dy)
        lower_sol = np.where(v[-1] >= 0, (var[-1] - var[-2])/dy, 0.5*(3*var[-1] - 4*var[-2] +   var[-3])/dy)
        # upper_sol = 0.5*( -var[2]  + 4*var[1]  - 3*var[0])/dy
        # lower_sol = 0.5*(3*var[-1] - 4*var[-2] +   var[-3])/dy

        #making sure they have the right shape
        upper_sol = np.array([upper_sol])
        lower_sol = np.array([lower_sol])

        #combine into one matrix
        d_var = np.concatenate((upper_sol, inner_sol, lower_sol))
        return d_var

    def hydro_solver(self) -> float:

        """
        hydrodynamic equations solver
        """
        rho, u, w, e, P, T = self.rho, self.u, self.w, self.e, self.P, self.T
        y = self.ry
        g = self.g

        # Finding d_rho
        du_dx = self.central_x(u)
        dw_dy = self.central_y(w)

        d_rho_dx = self.upwind_x(rho, u)
        d_rho_dy = self.upwind_y(rho, w)

        d_rho = -rho*(du_dx + dw_dy) - u*d_rho_dx - w*d_rho_dy

        # Finding d_rho_u
        du_dx = self.upwind_x(u, u)

        d_rho_u_dx = self.upwind_x(rho*u, u)
        d_rho_u_dy = self.upwind_y(rho*u, w)

        dP_dx = self.central_x(P)
        
        d_rho_u = -rho*u*(du_dx + dw_dy) - u*d_rho_u_dx - w*d_rho_u_dy - dP_dx

        # Finding d_rho_w
        du_dx = self.central_x(u)
        dw_dy = self.upwind_y(w, w)

        d_rho_w_dx = self.upwind_x(rho*w, u)
        d_rho_w_dy = self.upwind_y(rho*w, w)

        dP_dy = self.central_y(P)
        self.dP_dy = dP_dy #saves it for debugging

        d_rho_w = -rho*w*(du_dx + dw_dy) - u*d_rho_w_dx - w*d_rho_w_dy - dP_dy + rho*(-g) #-g because it is defined positive and not negative
        self.d_rho_w = d_rho_w

        # Finding de
        dw_dy = self.central_y(w)

        de_dx = self.upwind_x(e, u)
        de_dy = self.upwind_y(e, w)
        
        de = -(e + P)*(du_dx + dw_dy) - u*de_dx - w*de_dy

        # faster, but potentially less stable
        # self.u[:], self.w[:], self.e[:], self.rho[:] = u, w, e, rho
        # self.boundary_conditions()
        # u, w, e, rho = self.u, self.w, self.e, self.rho

        #progressing
        phi = np.array([rho, rho*u, rho*w, e])
        dphi = np.array([d_rho, d_rho_u, d_rho_w, de])

        dx, dy = self.dx, self.dy
        v = np.array([u, w])
        dv = np.array([dx, dy])

        dt = self.timestep(phi, dphi, v, dv)

        
        # slower, but potentially more stable
        # self.u[:], self.w[:], self.e[:], self.rho[:] = u, w, e, rho
        # self.boundary_conditions()
        # u, w, e, rho = self.u, self.w, self.e, self.rho

        #calculate the new updated variables
        rho_new = rho + d_rho*dt

        u_new = (rho*u + d_rho_u*dt)/rho_new
        w_new = (rho*w + d_rho_w*dt)/rho_new

        e_new = e + de*dt  

        P_new = self.find_P(e_new)
        T_new = self.find_T(rho_new, e=e_new)

        #store all new varaible
        self.rho[:], self.u[:], self.w[:], self.e[:], self.P[:], self.T[:] = rho_new, u_new, w_new, e_new, P_new, T_new
        #sets the boundary condtions
        self.boundary_conditions()

        #tracks the current iteration
        self.i += 1
        #tracks the current time in the simulation
        self.dt += dt
        #keep you updated, also usefull to find when in the simulation something goes wrong
        if self.i%50 == 0:
            print("\n", self.i,f": {dt:.3f}",  f" ({self.dt:5.2f}s)\n" if self.dt<60 else f" ({int(self.dt/60):2}min {self.dt%60:3.1f}s)")

        #Was not happy with the snapshots generated by the FVis module
        #this gives me consistent sized images
        if self.save_snapshots:
            if abs(self.dt-(4*60)) < 0.1:
                # self.show("T")
                # self.show("rho")
                # self.show("w")
                self.show("e")
            if abs(self.dt-(9*60+55)) < 0.1:
                # self.show("T")
                # self.show("rho")
                # self.show("w")
                self.show("e")

        #ctrl + c doesnt always stop the program, this is a safeguard
        if self.i == 100000:
            print("Exiting as planned")
            quit()

        return dt

    #some formulas to keep lines of code easier to debug

    #from equation 29
    def find_e(self, rho, T):
        gamma, mu = self.gamma, self.mu

        return 1/(gamma - 1)*rho/(mu*m_u)*k*T

    #from equation 30
    def find_P(self, e):
        gamma = self.gamma
        return (gamma-1)*e

    def find_T(self, rho, P=None, e=None):
        mu = self.mu
        # overloaded function to do different things different times,
        # not really necessary
        if e is None:
            return P*mu*m_u/(k*rho)
        if P is None:
            gamma = self.gamma
            return (gamma - 1)*mu*m_u/(k*rho)*e


    #finds rho for ideal gas
    def find_rho(self, P, T) -> float:
        mu = self.mu

        return P/(k*T)*mu*m_u

    #for plotting initial conditions or after a few iterations
    #the plot looks the same to the figure in the animation for
    #easier comparison
    def show(self, param = "T") -> None:
        if param == "T":
            Z = self.T
            param += " [K]"
        elif param == "rho":
            Z = self.rho
            param += " [kg/m^3]"
        elif param == "P":
            Z = self.P
            param += " [Pa]"
        elif param == "e":
            Z = self.e

            param += " [J/m^3]"

        elif param == "w":
            while self.i <20:
                self.hydro_solver()  
            Z = self.w
            param += " [m/s]"

        elif param == "dP_dy":

            if self.i < 1:
                while self.i <10:
                    self.hydro_solver()  
            Z = self.dP_dy
            param += " [Pa/m]"
        else:
            print(f"{param} is not supported, try `T`, `rho`, `P`, `e`, `w` or `dP_dy` instead")
            print("\nNow exiting the program...")
            quit()
        
        fig, ax = plt.subplots(1, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.1)

        rx, ry = self.rx*1e-6, self.ry*1e-6

        plot = ax.contourf(rx, ry - np.min(ry), Z, 500, cmap="jet", label=param)
        ax.set_aspect("equal")

        N_arrows = 20
        min_N = np.min([self.Nx, self.Ny])

        step_q = min_N//N_arrows
        cs_approx = np.mean(np.sqrt(1.67*self.P))
        
        qx, qy = rx[::step_q, ::step_q], ry[::step_q, ::step_q]
        qy = qy - np.min(qy)

        quiverscale = 0.02
        arrscale = 0.2*N_arrows*cs_approx/quiverscale
        print(f"{arrscale=:.3f}")

        quiver = ax.quiver(qx, qy, qx, qy, units='height', scale=arrscale, width=0.003, color='k')

        quiver.set_UVC(self.u[::step_q, ::step_q], self.w[::step_q, ::step_q])

        ticks = np.linspace(np.max(Z), np.min(Z), 6)

        cbar = fig.colorbar(plot, cax=cax, ticks=ticks)
        cbar.set_label(param)

        ax.set_xlabel("Horizontal distance [Mm]")
        ax.set_ylabel("Vetical distance [Mm]")

        plt.legend()
        plt.show()

    #finds the extent, for use in the FVis.animate_2D method call
    #it finds the physical size of the box
    def extent(self) -> list:
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        R0 = self.R0
        R0 = Ny*dy

        x0 = 0
        x1 = Nx*dx

        y0 = R0 - Ny*dy
        y1 = R0

        extent = [x0, x1, y0, y1]
        #converts from [m] to [Mm]
        return [x/1e6 for x in extent]


def main():
    #asks if you want pertubations
    pertubation = bool(input("\n With pertubations? [Y/n] ").lower() == "y")
    model = convection2D(pertubation=pertubation)

    #asks if you wish to simulate or check the initial conditions
    if bool(input("\n Do you wish to simulate? If no, initial conditions will show instead. [Y/n] ").lower() == "n"):
        print("\n Plotting initial conditions")
        #model.show(param="...") allows you to plot the selected variable (if supported)
        #useful for debugging and checking initial conditions
        # model.show(param="T")
        model.show(param="w")
        # model.show(param="rho")

    #if you dont want initial conditions, the simulation starts
    else:
        print("\n Starting simulation")

        model.save_snapshots = bool(input("\n Do you wish to save snapshots? [Y/n]").lower() == "y")

        vis = FVis.FluidVisualiser()

        #sim time = x min + y sec
        sim_time = 10*60 + 0
        vis.save_data(sim_time, model.hydro_solver, rho=model.rho, u=model.u, w=model.w, P=model.P, T=model.T, e=model.e, sim_fps=1.0)

        #finds the physical size of the box
        extent = model.extent()
        #defines the units of the different axes
        units = {"Lx" : "Mm", "Lz" : "Mm"}

        #asks if you wish to save the animation, if no the animation is played but not saved
        save = bool(input("\n Do you wish to save the animation? [Y/n] ").lower() == "y")

        vis.animate_2D("T", 
                       save=save, 
                       video_fps=22, 
                       extent=extent, 
                       units=units, 
                       quiverscale=0.2)
        vis.animate_2D("rho", 
                       save=save, 
                       video_fps=22, 
                       extent=extent, 
                       units=units, 
                       quiverscale=0.2)

        vis.animate_2D("w", 
                       save=save, 
                       video_fps=22, 
                       extent=extent, 
                       units=units, 
                       quiverscale=0.2)
                    #    snapshots=[0, 4*60, 9*60+55])
        #asks if you wish to delete the simulated data
        vis.delete_current_data()


# Run your code here
if __name__ == '__main__':
    main()