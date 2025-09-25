# visualiser
import FVis3 as FVis

import numpy as np
from scipy.constants import k, m_u, G
import matplotlib.pyplot as plt

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

    def __init__(self):

        """
        define variables
        """ 

        Nx = 300
        Ny = 100
        self.Nx, self.Ny = Nx, Ny

        x0 = 0 ; x1 = 12e6 #m
        y0 = 0 ; y1 = -4e6 #m
        R0 = self.R0

        x = np.linspace(x0, x1, Nx)
        y = np.linspace(y0, y1, Ny) + R0
        self.dx = (x1 - x0)/Nx
        self.dy = (y1 - y0)/Ny

        self.rx, self.ry = np.meshgrid(x, y)

        self.initialise()


    def initialise(self):

        """
        initialise temperature, pressure, density and internal energy
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        rx, ry = self.rx, self.ry

        mu, nabla, g, R0 = self.mu, self.nabla, self.g, self.R0

        T0, P0 = self.T0, self.P0

        #as derived in the paper
        # T = -mu*m_u*G*self.m/(ry*k)*self.nabla + T0
        T = T0 - nabla*mu*m_u*g*(ry - R0)/k


        # P = np.exp(mu*m_u*G*self.m/(ry*k*T)) + P0
        P = P0*np.exp(-mu*m_u*g/(k*T)*(ry - R0))
        #from ideal gas law
        rho = self.find_rho(P, T)

        self.T, self.rho, self.P = T, rho, P

        u = np.zeros((Nx, Ny)).T
        w = np.zeros((Nx, Ny)).T

        self.u, self.w = u, w

        e = self.find_e(rho, T)

        self.e = e

        self.i = 0
        self.dt = 0

    def timestep(self, phi, dphi, v, dv):

        """
        calculate timestep
        """

        p = 0.1

        # phi_copy = phi
        # print(phi[~np.nonzero(phi).any(axis=1)].shape)
        # quit()
        # new_phi = phi[:, ~np.nonzero(phi).any(axis=1)]

        # dphi = phi[phi <= 1e-12]
        # phi = phi[phi <= 1e-12]

        # print(phi)
        rel_phi = np.abs(dphi/phi)
        # print(rel_phi.shape)
        # rel_phi = rel_phi[:, ~np.isnan(rel_phi).any(axis=0)]
        # rel_phi = rel_phi[:, ~np.isnan(rel_phi).any(axis=1)]
        # print(rel_phi.shape)
        rel_phi[rel_phi >= float("+inf")] = 0
        # print(rel_phi.shape)
        # quit()
        # delta_phi = np.max(np.max(rel_phi, axis=1), axis=1)
        delta_phi = np.max(rel_phi, axis=1)
        delta_phi = np.max(delta_phi[~np.isnan(delta_phi).any(axis=1)], axis=1)
        # print(np.min(np.abs(phi)))
        # print(delta_phi.shape)
        # print(f"{delta_phi=}")
        # quit()

        rel_xy = np.abs(v.T/dv).T
        delta_xy = np.max(np.max(rel_xy, axis=1), axis=1)
        # print(f"{delta_xy=}")


        delta = np.max(np.concatenate((delta_phi, delta_xy)))
        if delta == 0:
            dt = 0.01
        else:
            dt = p/delta

        
        # print(np.sum(np.where(np.isnan(dphi), 1, 0)))

        # dt_phi = np.min(p/delta_phi)

        # dt_xy = np.min(0.99/delta_phi)

        # dt = np.min([dt_phi, dt_xy])

        if dt < 0.01:
            dt = 0.01

        if np.isnan(dt):
            dt = 1000
        
        return dt

    def boundary_conditions(self):

        """
        boundary conditions for energy, density and velocity
        """
        # vertical velocity
        w = self.w
        w[[0, -1], :] = 0

        # horizontal velocity
        u = self.u
        u[0] = (-u[2] + 4*u[1])/3
        u[-1] = (-u[-3] + 4*u[-2])/3

        # density and energy
        dP_dy, e, de_dy, T, rho, d_rho_dy = self.dP_dy, self.e, self.de_dy, self.T, self.rho, self.d_rho_dy
        gamma, mu, g = self.gamma, self.mu, self.g
        
        dP_dy = -g*rho

        # need to explain these in the text
        # de_dy[[0, -1], :] = 1/(gamma - 1)*dP_dy[[0, -1], :]
        # d_rho_dy[[0, -1], :] = de_dy[[0, -1], :]*(gamma - 1)*mu*m_u/(k*T[[0, -1], :])

        de_dy_upper = 1/(gamma - 1)*dP_dy[0]
        de_dy_lower = 1/(gamma - 1)*dP_dy[-1]

        # de_dy[0] = de_dy_upper
        # de_dy[-1] = de_dy_lower

        dy = self.dy
        e[0] = (-2*dy*de_dy_upper - e[2] + 4*e[1])/3
        e[-1] = (2*dy*de_dy_lower - e[-3] + 4*e[-2])/3

        # d_rho_dy_upper = mu*m_u/(k*T[0])*dP_dy[0]
        # d_rho_dy_lower = mu*m_u/(k*T[-1])*dP_dy[0]

        # d_rho_dy[0] = d_rho_dy_upper*(gamma - 1)*mu*m_u/(k*T[0])
        # d_rho_dy[-1] = d_rho_dy_lower*(gamma - 1)*mu*m_u/(k*T[-1])

        # rho[0] = (2*dy*de_dy_upper - rho[2] + 4*rho[1])/3
        # rho[-1] = (2*dy*de_dy_lower - rho[-3] + 4*rho[-2])/3

        rho[0]  = (gamma - 1)*mu*m_u/(k*T[0])*e[0]
        rho[-1] = (gamma - 1)*mu*m_u/(k*T[-1])*e[-1]


        self.u[:], self.w[:], self.e[:], self.rho[:] = u, w, e, rho

    def central_x(self, var):

        """
        central difference scheme in x-direction
        """
        dx = self.dx

        d_var = 0.5*(np.roll(var, 1, axis=1) - np.roll(var, -1, axis=1))/dx
        # d_var = 0.5*(np.roll(var, 1, axis=0) - np.roll(var, -1, axis=0))/dx
        return d_var

    def central_y(self, var):

        """
        central difference scheme in y-direction
        """
        dy = self.dy

        _var = var[1:-1]

        # inner_sol = 0.5*(np.roll(_var, 1, axis=1) - np.roll(_var, -1, axis=1))/dy
        inner_sol = 0.5*(var[2:] - var[:-2])/dy
        
        upper_sol = np.array([0.5*( -var[2]  + 4*var[1]  - 3*var[0])/dy])
        lower_sol = np.array([0.5*(3*var[-1] - 4*var[-2] +   var[-3])/dy])

        d_var = np.concatenate((upper_sol, inner_sol, lower_sol))
        return d_var

    def upwind_x(self, var, v):

        """
        upwind difference scheme in x-direction
        """
        dx = self.dx

        d_var = np.where(v >= 0, (var - np.roll(var, -1, axis=1))/dx, (np.roll(var, 1, axis=1) - var)/dx)
        # d_var = np.where(v >= 0, (var - np.roll(var, -1, axis=0))/dx, (np.roll(var, 1, axis=0) - var)/dx)
        return d_var

    def upwind_y(self, var, v):

        """
        upwind difference scheme in y-direction
        """
        dy = self.dy

        _var = var[1:-1]
        _v = v[1:-1]

        inner_sol = np.where(_v >= 0, (_var - var[:-2])/dy, (var[2:] - _var)/dy)
        upper_sol = np.array([0.5*( -var[2]  + 4*var[1]  - 3*var[0])/dy])
        lower_sol = np.array([0.5*(3*var[-1] - 4*var[-2] +   var[-3])/dy])

        d_var = np.concatenate((upper_sol, inner_sol, lower_sol))
        return d_var

    def hydro_solver(self):

        """
        hydrodynamic equations solver
        """
        rho, u, w, e, P, T = self.rho, self.u, self.w, self.e, self.P, self.T
        y = self.ry
        g = self.g

        

        # rho_u, rho_w = rho*u, rho*w
        rho_u = rho*u
        rho_w = rho*w

        # du_dx = self.upwind_x(u, u)
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
        # dP_dy = self.upwind_y(P, w)
        # dP_dy = -g*rho

        d_rho_w = -rho*w*(du_dx + dw_dy) - u*d_rho_w_dx - w*d_rho_w_dy - dP_dy + rho*(-g)

        # Finding de
        dw_dy = self.central_y(w)

        de_dx = self.upwind_x(e, u)
        de_dy = self.upwind_y(e, w)
        
        de = -(e + P)*(du_dx + dw_dy) - u*de_dx - w*de_dy


        # self.u[:], self.w[:], self.e[:], self.de_dy, self.rho[:], self.d_rho_dy, self.dP_dy = u, w, e, de_dy, rho, d_rho_dy, dP_dy
        # self.boundary_conditions()
        # u, w, e, rho = self.u, self.w, self.e, self.rho
        # rho_u = rho*u
        # rho_w = rho*w

        #finding time derivatives
        # d_rho = -rho*(du_dx + dw_dy) - u*d_rho_dx - w*d_rho_dy

        # d_rho_u = -rho*u*(du_dx + dw_dy) - u*d_rho_u_dx - w*d_rho_u_dy - dP_dx
        # d_rho_w = -rho*w*(du_dx + dw_dy) - u*d_rho_w_dx - w*d_rho_w_dy - dP_dy + rho*(-g) #try changing the direction of g

        # de = -(e + P)*(du_dx + dw_dy) - u*de_dx - w*de_dy

        #lasts 30.69s (much faster though)
        # self.u[:], self.w[:], self.e[:], self.de_dy, self.rho[:], self.d_rho_dy, self.dP_dy = u, w, e, de_dy, rho, d_rho_dy, dP_dy
        # self.boundary_conditions()
        # u, w, e, rho = self.u, self.w, self.e, self.rho

        #progressing
        phi = np.array([rho, rho*u, rho*w, e])
        dphi = np.array([d_rho, d_rho_u, d_rho_w, de])

        dx, dy = self.dx, self.dy
        v = np.array([u, w])
        dv = np.array([dx, dy])

        dt = self.timestep(phi, dphi, v, dv)
        # print(dt)
        # quit()

        
        #lasts 31.8s (a bit slower)
        self.u[:], self.w[:], self.e[:], self.de_dy, self.rho[:], self.d_rho_dy, self.dP_dy = u, w, e, de_dy, rho, d_rho_dy, dP_dy
        self.boundary_conditions()
        u, w, e, rho = self.u, self.w, self.e, self.rho

        rho_new = rho + d_rho*dt

        u_new = (rho*u + d_rho_u*dt)/rho_new
        w_new = (rho*w + d_rho_w*dt)/rho_new

        e_new = e + de*dt  

        P_new = self.find_P(e_new)
        # T_new = self.find_T(rho_new, P_new)
        T_new = self.find_T(rho, e=e_new)

        self.rho[:], self.u[:], self.w[:], self.e[:], self.P[:], self.T[:] = rho_new, u_new, w_new, e_new, P_new, T_new
        
        # self.de_dy, self.d_rho_dy, self.dP_dy = de_dy, d_rho_dy, dP_dy
        # self.boundary_conditions()
        # u, w, e, rho = self.u, self.w, self.e, self.rho

        
        self.i += 1
        self.dt += dt
        print(self.i,f": {dt:.3f} ({self.dt:5.2f}s)")
        if self.i == 10000:
            print("Exiting as planned")
            quit()

        return dt

    def find_e(self, rho, T):
        gamma, mu = self.gamma, self.mu

        return 1/(gamma - 1)*rho/(mu*m_u)*k*T

    def find_P(self, e):
        gamma = self.gamma
        return (gamma-1)*e

    def find_T(self, rho, P=None, e=None):
        mu = self.mu

        if e is None:
            return P*mu*m_u/(k*rho)
        if P is None:
            gamma = self.gamma
            return (gamma - 1)*mu*m_u/(k*rho)*e


    def find_rho(self, P, T):
        mu = self.mu

        return P/(k*T)*mu*m_u

    def show(self):
        # plt.imshow(self.T)
        plt.contourf(self.rx, self.ry, self.T, 1000)
        plt.colorbar()
        plt.show()

def main():
    model = convection2D()
    # model.show()
    # model.hydro_solver()
    # model.show()

    # print(type(model.hydro_solver))

    vis = FVis.FluidVisualiser()

    #31.8
    vis.save_data(31.8, model.hydro_solver, rho=model.rho, u=model.u, w=model.w, P=model.P, T=model.T, e=model.e, sim_fps=1.0)
    vis.animate_2D("T")
    vis.delete_current_data()


if __name__ == '__main__':
    # Run your code here
    
    main()
    