from scipy import integrate, interpolate
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c, G, k_B, hbar

#converting to cgs
c = c.cgs.value
G = G.cgs.value
k_B = k_B.cgs.value
hbar = hbar.cgs.value

class BBN:
    #some mass constants in cgs
    mn = 1.6749e-24
    mp = 1.6726e-24
    me = 9.1094e-28

    #as defined in the project
    T0 = 2.725 #K
    h = 0.7
    H0 = 100*h*1e5/(3.09e22*1e2)
    Neff = 3
    omega_b0 = 0.05
    rho_c0 = 3*H0*H0/(8*np.pi*G)

    def __init__(self, Tspan : list):
        self.Ti = Tspan[0]
        self.Tf = Tspan[1]

        #initializes Yn and Yp from the equilibrium values, assum ing p and n make up all baryonic mass
        Yn, Yp = self.Y_init(self.Ti)
        self.Yn, self.Yp = Yn, Yp

        #initializes the radiation density today, using already stored values
        omega_r0 = self.omega_r0_init()
        self.omega_r0 = omega_r0

    #as defined in equation 16 andn 17
    def Y_init(self, Ti) -> tuple:
        mn = self.mn ; mp = self.mp

        Yn = 1/(1 + np.exp((mn - mp)*c*c/(k_B*Ti)))
        Yp = 1 - Yn

        return Yn, Yp

    #as defined in equation 15
    def omega_r0_init(self):
        #later self.Neff will be changed before calling the function again
        H0, T0, Neff = self.H0, self.T0, self.Neff

        top = 8*np.pi**3*G*(k_B*T0)**4
        bot = 45*H0**2*hbar**3*c**5

        return top/bot*(1 + Neff*7/8*np.cbrt(4/11)**4)

    #solution to task d
    def find_t(self, T) -> float:
        H0, omega_r0, T0 = self.H0, self.omega_r0, self.T0
        t_inv = 2*H0*np.sqrt(omega_r0)*(T/T0)**2
        return 1/t_inv

    #solution to task d
    def find_T(self, t) -> float:
        H0, omega_r0, T0 = self.H0, self.omega_r0, self.T0
        return T0/np.sqrt(2*H0*np.sqrt(omega_r0)*t)

    #solution to task d
    def find_a(self, t) -> float:
        H0, omega_r0 = self.H0, self.omega_r0
        return np.sqrt(2*H0*np.sqrt(omega_r0)*t)

    #as defined in equation 14
    def find_H(self, a) -> float:
        H0, omega_r0 = self.H0, self.omega_r0
        return H0*np.sqrt(omega_r0)/a**2

    #from lecture notes
    def find_rho_b(self, a) -> float:
        rho_b0 = self.omega_b0*self.rho_c0

        rho_b = rho_b0/a**3

        return rho_b

    #as defined in equation 10
    def dYn(self, H, Y : list, gamma : list) -> float:
        Yn, Yp = Y
        gamma_n, gamma_p = gamma

        dYn = Yp*gamma_p - Yn*gamma_n
        return dYn

    #as defined in equation 11
    def dYp(self, H, Y : list, gamma : list) -> float:
        dYp = -self.dYn(H, Y, gamma)
        return dYp

    # all following equation, until otherwise specified are taken from the paper
    # Y and gamma are already selected when sent into the function calls

    # technically Y and gamma are np.ndarrays, but functionally the same as lists

    # gamma contains reaction rates. 
    # the _x tells you what equation in table 2 part b) it comes from
    def dYD_1(self, H, Y : list, gamma : list) -> float:
        Yn, Yp, YD = Y
        gamma_D, pn = gamma

        dYD = Yn*Yp*pn - YD*gamma_D
        return dYD

    def dYHe3_2(self, H, Y : list, gamma : list) -> float:
        Yp, YD, YHe3 = Y
        gamma_He3, pD = gamma

        dYHe3 = Yp*YD*pD - YHe3*gamma_He3
        return dYHe3

    def dYT_3(self, H, Y : list, gamma : list) -> float:
        Yn, YD, YT = Y
        gamma_T, nD = gamma

        dYT = Yn*YD*nD - YT*gamma_T
        return dYT

    def dYT_4(self, H, Y : list, gamma : list) -> float:
        Yn, Yp, YT, YHe3 = Y
        nHe3_p, pT_n = gamma

        dYT = Yn*YHe3*nHe3_p - Yp*YT*pT_n
        return dYT

    def dYHe4_5(self, H, Y : list, gamma : list) -> float:
        Yp, YT, YHe4 = Y
        gamma_He4_p, pT_gamma = gamma

        dYHe4 = Yp*YT*pT_gamma - YHe4*gamma_He4_p
        return dYHe4

    def dYHe4_6(self, H, Y : list, gamma : list) -> float:
        Yn, YHe3, YHe4 = Y
        gamma_He4_n, nHe3_gamma = gamma

        dYHe4 = Yn*YHe3*nHe3_gamma - YHe4*gamma_He4_n
        return dYHe4

    def dYHe3_7(self, H, Y : list, gamma : list) -> float:
        Yn, YD, YHe3 = Y
        DD_n, nHe3_D = gamma

        dYHe3 = 0.5*YD*YD*DD_n - Yn*YHe3*nHe3_D
        return dYHe3

    def dYT_8(self, H, Y : list, gamma : list) -> float:
        Yp, YD, YT = Y
        DD_p, pT_D = gamma

        dYT = 0.5*YD*YD*DD_p - Yp*YT*pT_D
        return dYT

    def dYHe4_9(self, H, Y : list, gamma : list) -> float:
        YD, YHe4 = Y
        gamma_He4_D, DD_gamma = gamma

        dYHe4 = 0.5*YD*YD*DD_gamma - YHe4*gamma_He4_D
        return dYHe4
    
    def dYHe4_10(self, H, Y : list, gamma : list) -> float:
        Yp, YD, YHe3, YHe4 = Y
        DHe3, He4p = gamma

        dYHe4 = YD*YHe3*DHe3 - YHe4*Yp*He4p
        return dYHe4

    def dYHe4_11(self, H, Y : list, gamma : list) -> float:
        Yn, YD, YT, YHe4 = Y
        DT, He4n = gamma

        dYHe4 = YD*YT*DT - YHe4*Yn*He4n
        return dYHe4

    def dYHe4_15(self, H, Y : list, gamma : list) -> float:
        YD, YT, YHe3, YHe4 = Y
        He3T_D, He4D = gamma

        dYHe4 = YHe3*YT*He3T_D - YHe4*YD*He4D
        return dYHe4

    def dYBe7_16(self, H, Y : list, gamma : list) -> float:
        YHe3, YHe4, YBe7 = Y
        gamma_Be7, He3He4 = gamma

        dYBe7_16 = YHe3*YHe4*He3He4 - YBe7*gamma_Be7
        return dYBe7_16

    def dYLi7_17(self, H, Y : list, gamma : list) -> float:
        YT, YHe4, YLi7 = Y
        gamma_Li7, THe4 = gamma

        dYLi7_17 = YT*YHe4*THe4 - YLi7*gamma_Li7
        return dYLi7_17

    def dYLi7_18(self, H, Y : list, gamma : list) -> float:
        Yn, Yp, YLi7, YBe7 = Y
        nBe7_p, pLi7_n = gamma

        dYLi7_18 = Yn*YBe7*nBe7_p - Yp*YLi7*pLi7_n
        return dYLi7_18 

    def dYHe4_20(self, H, Y : list, gamma : list) -> float:
        Yp, YHe4, YLi7 = Y
        pLi7_He4, He4He4_p = gamma

        dYHe4_20 = Yp*YLi7*pLi7_He4 - 0.5*YHe4*YHe4*He4He4_p
        return dYHe4_20
        
    def dYHe4_21(self, H, Y : list, gamma : list) -> float:
        Yn, YHe4, YBe7 = Y
        nBe7_He4, He4He4_n = gamma

        dYHe4_21 = Yn*YBe7*nBe7_He4 - 0.5*YHe4*YHe4*He4He4_n
        return dYHe4_21

    #The next equations are all reaction rates function

    #as defined in equation 12
    def gamma_n(self, T, q) -> float:
        tau = 1700 #s
        
        T_nu = np.cbrt(4/11)*T

        T9 = T*1e-9
        T_nu9 = T_nu*1e-9

        #from project
        Z = 5.93/T9
        Z_nu = 5.93/T_nu9

        int1 = lambda x: (x + q)**2*np.sqrt(x**2 - 1)*x/((1 + np.exp( x*Z))*(1 + np.exp(-(x + q)*Z_nu)))
        int2 = lambda x: (x - q)**2*np.sqrt(x**2 - 1)*x/((1 + np.exp(-x*Z))*(1 + np.exp( (x - q)*Z_nu)))

        int1_solved = integrate.quad(int1, 1, 250)[0]
        int2_solved = integrate.quad(int2, 1, 250)[0]

        return (int1_solved + int2_solved)/tau

    #as defined in equation 13
    def gamma_p(self, T, q) -> float:
        return self.gamma_n(T, -q)


    # the rest of the functions take q as an argument for no reason at all
    def pn(self, T, q, rho_b) -> float:
        return 2.5e4*rho_b

    def gamma_D(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        pn = self.pn(T, q, rho_b)
        
        lamda_D = 4.68e9*pn/rho_b*np.sqrt(T9**3)*np.exp(-25.82/T9)
        return lamda_D

    def pD(self, T, q, rho_b) -> float:
        T9 = T*1e-9

        pd = 2.24e3*rho_b/np.cbrt(T9**2)*np.exp(-3.72/np.cbrt(T9))*(1 + 0.112*np.cbrt(T9) + 3.38*np.cbrt(T9**2) + 2.65*T9)
        return pd

    def gamma_He3(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        pd = self.pD(T, q, rho_b)

        return 1.63e10*pd/rho_b*np.sqrt(T9**3)*np.exp(-63.75/T9)

    def nD(self, T, q, rho_b) -> float:
        T9 = T*1e-9

        nD = rho_b*(75.5 + 1250*T9)
        return nD
 
    def gamma_T(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        nD = self.nD(T, q, rho_b)

        gamma_T = 1.63e10*nD/rho_b*np.sqrt(T9**3)*np.exp(-72.62/T9)
        return gamma_T

    def nHe3_p(self, T, q, rho_b) -> float:
        return 7.06e8*rho_b

    def pT_n(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        nHe3_p = self.nHe3_p(T, q, rho_b)

        pT_n = nHe3_p*np.exp(-8.864/T9)
        return pT_n

    def pT_gamma(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        pT_gamma = 2.87e4*rho_b/crT9**2*np.exp(-3.87/crT9)*(1 + 0.108*crT9 + 0.466*crT9**2 + 0.352*T9 + 0.3*crT9*T9+ 0.576*crT9**2*crT9)
        return pT_gamma

    def gamma_He4_p(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        pT_gamma = self.pT_gamma(T, q, rho_b)

        gamma_He4_p = 2.59e10*pT_gamma/rho_b*np.sqrt(T9**3)*np.exp(-229.9/T9)
        return gamma_He4_p

    def nHe3_gamma(self, T, q, rho_b) -> float:
        T9 = T*1e-9

        return 6e3*rho_b*T9

    def gamma_He4_n(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        nHe3_gamma = self.nHe3_gamma(T, q, rho_b)

        gamma_He4_n = 2.6e10*nHe3_gamma/rho_b*np.sqrt(T9**3)*np.exp(-238.8/T9)
        return gamma_He4_n
    
    def DD_n(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        DD_n = 3.9e8*rho_b/crT9**2*np.exp(-4.26/crT9)*(1 + 0.0979*crT9 + 0.642*crT9**2 + 0.440*T9)
        return DD_n

    def nHe3_D(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        DD_n = self.DD_n(T, q, rho_b)

        nHe3_D = 1.73*DD_n*np.exp(-37.94/T9)
        return nHe3_D

    def DD_p(self, T, q, rho_b) -> float:
        return self.DD_n(T, q, rho_b)

    def pT_D(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        DD_p = self.DD_p(T, q, rho_b)

        pT_D = 1.73*DD_p*np.exp(-46.8/T9)
        return pT_D

    def DD_gamma(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        DD_gamma = 24.1*rho_b/crT9**2*np.exp(-4.26/crT9)*(crT9**2 + 0.685*T9 + 0.152*T9*crT9 + 0.265*T9*crT9**2)
        return DD_gamma

    def gamma_He4_D(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        DD_gamma = self.DD_gamma(T, q, rho_b)

        gamma_He4_D = 4.5e10*DD_gamma/rho_b*np.sqrt(T9**3)*np.exp(-276.7/T9)
        return gamma_He4_D

    def DHe3(self, T, q, rho_b) -> float:
        T9 = T*1e-9

        DHe3 = 2.6e9*rho_b/np.sqrt(T9**3)*np.exp(-2.99/T9)
        return DHe3

    def He4p(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        DHe3 = self.DHe3(T, q, rho_b)

        He4p = 5.5*DHe3*np.exp(-213/T9)
        return He4p
 
    def DT(self, T, q, rho_b) -> float:
        T9 = T*1e-9

        DT = 1.38e9*rho_b/np.sqrt(T9**3)*np.exp(-0.745/T9)
        return DT
 
    def He4n(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        DT = self.DT(T, q, rho_b)

        He4n = 5.5*DT*np.exp(-204.1/T9)
        return He4n

    def He3T_D(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        He3T_D = 3.88e9*rho_b/crT9**2*np.exp(-7.72/crT9)*(1 + 0.054*crT9)
        return He3T_D
  
    def He4D(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        He3T_D = self.He3T_D(T, q, rho_b)

        He4D = 1.59*He3T_D*np.exp(-166.2/T9)
        return He4D
 
    def He3He4(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        He3He4 = 4.8e6*rho_b/crT9**2*np.exp(-12.8/crT9)*(1 + 0.0326*crT9 - 0.219*crT9**2 - 0.0499*T9 + 0.0258*crT9*T9 + 0.0150*crT9**2*T9)
        return He3He4
      
    def gamma_Be7(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        He3He4 = self.He3He4(T, q, rho_b)

        gamma_Be7 = 1.12e10*He3He4/rho_b*np.sqrt(T9**3)*np.exp(-18.42/T9)
        return gamma_Be7
    
    def THe4(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        THe4 = 5.28e5*rho_b/crT9**2*np.exp(-8.08/crT9)*(1 + 0.0516*crT9)
        return THe4

    def gamma_Li7(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        THe4 = self.THe4(T, q, rho_b)

        gamma_Li7 = 1.12e10*THe4/rho_b*np.sqrt(T9**3)*np.exp(-28.63/T9)
        return gamma_Li7

    def nBe7_p(self, T, q, rho_b) -> float:
        nBe7_p = 6.74e9*rho_b
        return nBe7_p

    def pLi7_n(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        nBe7_p = self.nBe7_p(T, q, rho_b)

        pLi7_n = nBe7_p*np.exp(-19.07/T9)
        return pLi7_n

    def pLi7_He4(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        crT9 = np.cbrt(T9)

        pLi7_He4 = 1.42e9*rho_b/crT9**2*np.exp(-8.47/crT9)*(1 + 0.0493*crT9)
        return pLi7_He4

    def He4He4_p(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        pLi7_He4 = self.pLi7_He4(T, q, rho_b)

        He4He4_p = 4.64*pLi7_He4*np.exp(-201.3/T9)
        return He4He4_p

    def nBe7_He4(self, T, q, rho_b) -> float:
        T9 = T*1e-9

        nBe7_He4 = 1.2e7*rho_b*T9
        return nBe7_He4

    def He4He4_n(self, T, q, rho_b) -> float:
        T9 = T*1e-9
        nBe7_He4 = self.nBe7_He4(T, q, rho_b)

        He4He4_n = 4.64*nBe7_He4*np.exp(-220.4/T9)
        return He4He4_n

    # appends all the different gammas into a single list
    # if I had the energy I would rearrange it into a dictionary
    def find_gamma(self, T, Y, q) -> np.ndarray:
        n_elements = len(Y)
        
        gamma = []
        gamma.append(self.gamma_n(T, q))
        gamma.append(self.gamma_p(T, q))

        #D
        if n_elements >= 3:
            t = self.find_t(T)
            a = self.find_a(t)

            rho_b = self.find_rho_b(a)

            gamma.append(self.gamma_D(T, q, rho_b))
            gamma.append(self.pn(T, q, rho_b))

        #T
        if n_elements >= 4:
            gamma.append(self.gamma_T(T, q, rho_b))
            gamma.append(self.nD(T, q, rho_b))
            gamma.append(self.DD_p(T, q, rho_b))
            gamma.append(self.pT_D(T, q, rho_b))
        
        #He3
        if n_elements >= 5:
            gamma.append(self.gamma_He3(T, q, rho_b))
            gamma.append(self.pD(T, q, rho_b))
            gamma.append(self.nHe3_p(T, q, rho_b))
            gamma.append(self.pT_n(T, q, rho_b))
            gamma.append(self.DD_n(T, q, rho_b))
            gamma.append(self.nHe3_D(T, q, rho_b))
            
        #He4
        if n_elements >= 6:
            gamma.append(self.gamma_He4_p(T, q, rho_b))
            gamma.append(self.pT_gamma(T, q, rho_b))
            gamma.append(self.gamma_He4_n(T, q, rho_b))
            gamma.append(self.nHe3_gamma(T, q, rho_b))
            gamma.append(self.gamma_He4_D(T, q, rho_b))
            gamma.append(self.DD_gamma(T, q, rho_b))
            gamma.append(self.DHe3(T, q, rho_b))
            gamma.append(self.He4p(T, q, rho_b))
            gamma.append(self.DT(T, q, rho_b))
            gamma.append(self.He4n(T, q, rho_b))
            gamma.append(self.He3T_D(T, q, rho_b))
            gamma.append(self.He4D(T, q, rho_b))

        #Li7
        if n_elements >= 7:
            gamma.append(self.gamma_Li7(T, q, rho_b))
            gamma.append(self.THe4(T, q, rho_b))
            gamma.append(self.pLi7_He4(T, q, rho_b))
            gamma.append(self.He4He4_p(T, q, rho_b))

        #Be7
        if n_elements >= 8:
            gamma.append(self.gamma_Be7(T, q, rho_b))
            gamma.append(self.He3He4(T, q, rho_b))
            gamma.append(self.nBe7_p(T, q, rho_b))
            gamma.append(self.pLi7_n(T, q, rho_b))
            gamma.append(self.nBe7_He4(T, q, rho_b))
            gamma.append(self.He4He4_n(T, q, rho_b))

        return np.array(gamma)

    #calculates all changes to dY_i
    def dY(self, H, Y, gamma) -> np.ndarray:
        n_elements = len(Y)

        #with all elements (n_elements = 8) included, it takes the form:
        #[n, p, D, T, He3, He4, Li7, Be7]
        dY = np.zeros(n_elements)

        dYn = self.dYn(H, Y[:2], gamma[:2])
        dYp = -dYn

        #D
        if n_elements >= 3:
            #as you can see, the gammas become very unintuitive
            dYD = self.dYD_1(H, Y[[0, 1, 2]], gamma[[2, 3]])

            dYn -= dYD
            dYp -= dYD

            # dYn and dYp do not need to be saved all the time,
            # as they are the only two elements that are always used
            dY[2] = dYD
        
        #T
        if n_elements >= 4:
            dYT = self.dYT_3(H, Y[[0, 2, 3]], gamma[[4, 5]])
            dYT_8 = self.dYT_8(H, Y[[1, 2, 3]], gamma[[6, 7]])

            dYn -= dYT
            dYD -= dYT

            dYp += dYT_8
            dYD -= 2*dYT_8
            dYT += dYT_8

            dY[2] = dYD
            dY[3] = dYT
        
        #He3
        if n_elements >= 5:
            dYHe3 = self.dYHe3_2(H, Y[[1, 2, 4]], gamma[[8, 9]])
            dYT_4 = self.dYT_4(H, Y[[0, 1, 3, 4]], gamma[[10, 11]])
            #maybe some issues
            dYHe3_7 = self.dYHe3_7(H, Y[[0, 2, 4]], gamma[[12, 13]])

            dYp -= dYHe3
            dYD -= dYHe3

            dYn -= dYT_4
            dYp += dYT_4
            dYHe3 -= dYT_4
            dYT += dYT_4

            #maybe some issues
            dYn += dYHe3_7
            dYD -= 2*dYHe3_7
            dYHe3 += dYHe3_7

            dY[2] = dYD
            dY[3] = dYT
            dY[4] = dYHe3

        #He4
        if n_elements >= 6:
            dYHe4 = self.dYHe4_5(H, Y[[1, 3, 5]], gamma[[14, 15]])
            dYHe4_6 = self.dYHe4_6(H, Y[[0, 4, 5]], gamma[[16, 17]])
            dYHe4_9 = self.dYHe4_9(H, Y[[2, 5]], gamma[[18, 19]])
            dYHe4_10 = self.dYHe4_10(H, Y[[1, 2, 4, 5]], gamma[[20, 21]])
            dYHe4_11 = self.dYHe4_11(H, Y[[0, 2, 3, 5]], gamma[[22, 23]])
            dYHe4_15 = self.dYHe4_15(H, Y[[2, 3, 4, 5]], gamma[[24, 25]])

            dYp -= dYHe4
            dYT -= dYHe4

            dYn -= dYHe4_6
            dYHe3 -= dYHe4_6
            dYHe4 += dYHe4_6

            dYD -= 2*dYHe4_9
            dYHe4 += dYHe4_9

            dYp += dYHe4_10
            dYD -= dYHe4_10
            dYHe3 -= dYHe4_10
            dYHe4 += dYHe4_10

            dYn += dYHe4_11
            dYD -= dYHe4_11
            dYT -= dYHe4_11
            dYHe4 += dYHe4_11

            dYD += dYHe4_15
            dYT -= dYHe4_15
            dYHe3 -= dYHe4_15
            dYHe4 += dYHe4_15

            dY[2] = dYD
            dY[3] = dYT
            dY[4] = dYHe3
            dY[5] = dYHe4

        #Li7
        if n_elements >= 7:
            dYLi7 = self.dYLi7_17(H, Y[[3, 5, 6]], gamma[[26, 27]])
            dYHe4_20 = self.dYHe4_20(H, Y[[1, 5, 6]], gamma[[28, 29]])

            dYT -= dYLi7
            dYHe4 -= dYLi7


            dYp -= dYHe4_20
            dYHe4 += 2*dYHe4_20
            dYLi7 -= dYHe4_20
            
            dY[3] = dYT
            dY[4] = dYHe3
            dY[5] = dYHe4
            dY[6] = dYLi7

        #Be7
        if n_elements >= 8:
            dYBe7 = self.dYBe7_16(H, Y[[4, 5, 7]], gamma[[30, 31]])
            dYLi7_18 = self.dYLi7_18(H, Y[[0, 1, 6, 7]], gamma[[32, 33]])
            dYHe4_21 = self.dYHe4_21(H, Y[[0, 5, 7]], gamma[[34, 35]])

            dYHe3 -= dYBe7
            dYHe4 -= dYBe7

            dYn -= dYLi7_18
            dYp += dYLi7_18
            dYBe7 -= dYLi7_18
            dYLi7 += dYLi7_18

            dYn -= dYHe4_21
            dYHe4 += 2*dYHe4_21
            dYBe7 -= dYHe4_21
            
            dY[4] = dYHe3
            dY[5] = dYHe4
            dY[6] = dYLi7
            dY[7] = dYBe7

        #saving dYn and dYp
        dY[0] = dYn ; dY[1] = dYp

        return -dY/H

    def diff_eqs(self, T, Y) -> np.ndarray:
        #from the project
        q = 2.53

        t = self.find_t(T)
        a = self.find_a(t)
        H = self.find_H(a)

        rho_b = self.find_rho_b(a)

        gamma = self.find_gamma(T, Y, q)

        dY = self.dY(H, Y, gamma)

        # divide by T, because the T parameter isnt passed as ln(T)
        # Theoretically dY/T is the same, but due to float arithmatic it might make the system more unstable
        # or more stable, I dont know, but it works
        return dY/T

    #call this to solve task f
    def solve_task_f(self, plot=False) -> None: 
        Ti, Tf = self.Ti, self.Tf
        Yn, Yp = self.Yn, self.Yp

        #solves the diff equations
        solution = integrate.solve_ivp(self.diff_eqs, t_span=[Ti, Tf], y0=[Yn, Yp], method="Radau", rtol=1e-12, atol=1e-12)

        #plots it
        if plot:
            fig, ax = self.create_fig(xlabel="T [K]", ylabel=r"A$_i$Y$_i$", xlim=[Ti, Tf], ylim=[1e-3, 1.5])
            self.plot_task(ax, solution)
            self.show(ax)

    #call this to solve task h
    def solve_task_h(self, plot=False) -> None:
        Ti, Tf = self.Ti, self.Tf
        Yn, Yp = self.Yn, self.Yp

        #defined as described in task
        Y0 = np.zeros(3)
        Y0[0] = Yn ; Y0[1] = Yp

        solution = integrate.solve_ivp(self.diff_eqs, t_span=[Ti, Tf], y0=Y0, method="Radau", rtol=1e-12, atol=1e-12)

        if plot:
            fig, ax = self.create_fig(xlabel="T [K]", ylabel=r"A$_i$Y$_i$", xlim=[Ti, Tf],  ylim=[1e-3, 1.5])
            self.plot_task(ax, solution)
            self.show(ax)

    #call this to solve task i
    def solve_task_i(self, plot=False) -> None:
        Ti, Tf = self.Ti, self.Tf
        Yn, Yp = self.Yn, self.Yp

        #defined as described in task
        # np.zeros(x), where 2<=x<=8 is the amount of species you want to include
        Y0 = np.zeros(8)
        Y0[0] = Yn ; Y0[1] = Yp

        solution = integrate.solve_ivp(self.diff_eqs, t_span=[Ti, Tf], y0=Y0, method="Radau", rtol=1e-12, atol=1e-12)

        if plot:
            fig, ax = self.create_fig(xlabel="T [K]", ylabel=r"A$_i$Y$_i$", xlim=[Ti, Tf], ylim=[1e-11, 1e1])
            self.plot_task(ax, solution)
            self.show(ax)

    #call this to solve task j
    def solve_task_j(self, n) -> None:
        """
            n: the number of different values of omgea_b0 you want to use as basis for interpolation
        """
        if n<4:
            print("Requires at least 4 points due to cubic interpolation")
            quit()
        Ti, Tf = self.Ti, self.Tf
        Yn, Yp = self.Yn, self.Yp
        Y0 = np.zeros(8)
        Y0[0] = Yn ; Y0[1] = Yp

        omega_b0 = np.logspace(np.log10(0.01), np.log10(1), n)
        solutions = []

        for i in range(len(omega_b0)):
            # this changes how rho_b is calculated
            self.omega_b0 = omega_b0[i]
            # to keep you updated and not doubt if the program is running
            print(f"\n Working on iteration {i+1} out of {n}... \n")
            solution = integrate.solve_ivp(self.diff_eqs, t_span=[Ti, Tf], y0=Y0, method="Radau", rtol=1e-12, atol=1e-12)
            solutions.append(solution)

        chi_s, best_omega_b0 = self.optimize(solutions, omega_b0, r"$\Omega_{b0}$", task="j")
    
    #call this to solve task k
    def solve_task_k(self, n) -> None:
        """
            n: the number of different values of Neff you want to use as basis for interpolation
        """
        if n<4:
            print("Requires at least 4 points due to cubic interpolation")
            quit()
        Ti, Tf = self.Ti, self.Tf
        Yn, Yp = self.Yn, self.Yp
        Y0 = np.zeros(8)
        Y0[0] = Yn ; Y0[1] = Yp

        Neff = np.linspace(1, 5, n)
        solutions = []

        for i in range(len(Neff)):
            #it is not enough to just set self.Neff
            #also self.omega_r0_init() needs to be called
            self.Neff = Neff[i]
            self.omega_r0 = self.omega_r0_init()
            # to keep you updated and not doubt if the program is running
            print(f"\n Working on iteration {i+1} out of {n}... \n")
            solution = integrate.solve_ivp(self.diff_eqs, t_span=[Ti, Tf], y0=Y0, method="Radau", rtol=1e-12, atol=1e-12)
            solutions.append(solution)

        chi_s, best_omega_b0 = self.optimize(solutions, Neff, r"N$_{eff}$", task="k")

    def optimize(self, solutions : list, x : np.ndarray, x_name : str, task : str) -> tuple:
        """
            x: an array of the variable we wish to interpolate over
        """
        #observations
        #format : [value, std]
        YD_Yp = np.array([2.5, 0.03])*1e-5
        YHe4 = np.array([0.254, 0.003])/4
        YLi7_Yp = np.array([1.6, 0.3])*1e-10

        obs_values = np.array([YD_Yp, 4*YHe4, YLi7_Yp])

        #extracts the data from the solutions
        Y0 = []
        for solution in solutions:
            Y = solution.y
            Y0.append(Y[:, -1])

        #the array is transposed into the correct shape
        Y0 = np.abs(np.array(Y0).T)

        #
        n = 1000
        if task == "j":
            #np.log10(x) because of logspace
            #np.log(Y0) because it is smoother
            f = interpolate.interp1d(np.log10(x), np.log(Y0), kind="cubic")

            x_new = np.logspace(np.log10(x[0]), np.log10(x[-1]), n)

            Y = np.exp(f(np.log10(x_new)))
        
        else:
            #no longer logspace
            # f = interpolate.interp1d(x, np.log(Y0), kind="cubic")
            f = interpolate.interp1d(x, Y0, kind="cubic")

            x_new = np.linspace(x[0], x[-1], n)

            # Y = np.exp(f(x_new))
            Y = f(x_new)


        #extract values
        Yp = Y[1]
        YD = Y[2]
        pred_YHe4 = Y[5]

        # as mentioned in the project description
        YHe3 = Y[4] + Y[3]
        YLi7 = Y[6] + Y[7]

        pred_values = np.array([YD/Yp, 4*pred_YHe4, YLi7/Yp])

        #caluclates chi^2
        chi = np.sum((pred_values.T - obs_values[:, 0])**2/obs_values[:, 1]**2, axis=1)

        #the probability curve
        P = 1/np.sqrt(2*np.prod(obs_values[:, 1]))*np.exp(-chi)
            
        #select the best fit values and store for later
        best_i = np.argmin(chi)
        best_chi = chi[best_i] ; best_x = x_new[best_i]


        #the rest is plotting
        if task == "k":
            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 10), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        else:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 10), gridspec_kw={'height_ratios': [1, 4, 1]})


        ax[0].plot(x_new, 4*pred_YHe4, label=r"$He^4$", c="tab:green")
        ax[0].fill_between(x_new, 4*(YHe4[0] + YHe4[1]), 4*(YHe4[0] - YHe4[1]), alpha=0.5, facecolor="tab:green")
        ax[0].set_ylim(0.2, 0.3)
        ax[0].axvline(x=best_x, c="black", linestyle="--")
        ax[0].set_ylabel(r"4Y$_{He^4}$")
        ax[0].legend()

        ax[1].plot(x_new, YD/Yp, label=r"D", c="tab:blue")
        ax[1].fill_between(x_new, YD_Yp[0] + YD_Yp[1], YD_Yp[0] - YD_Yp[1], alpha=0.5, facecolor="tab:blue")

        ax[1].plot(x_new, YHe3/Yp, label=r"$He^3$", c="tab:orange")
        if task == "j":
            ax[1].plot(x_new, YLi7/Yp, label=r"$Li^7$", c="tab:red")
            ax[1].fill_between(x_new, YLi7_Yp[0] + YLi7_Yp[1], YLi7_Yp[0] - YLi7_Yp[1], alpha=0.5, facecolor="tab:red")

            ax[1].set_ylim(1e-11, 1e-3)
            ax[1].set_yscale("log")
        else:
            ax[1].set_ylim(1e-5, 4e-5)

        ax[1].axvline(x=best_x, c="black", linestyle="--")
        ax[1].set_ylabel(r"Y$_i$/Y$_p$")
        ax[1].legend()

        if task == "k":
            ax[2].plot(x_new, YLi7/Yp, label=r"$Li^7$", c="tab:red")
            ax[2].fill_between(x_new, YLi7_Yp[0] + YLi7_Yp[1], YLi7_Yp[0] - YLi7_Yp[1], alpha=0.5, facecolor="tab:red")
            ax[2].axvline(x=best_x, c="black", linestyle="--")
            ax[2].set_ylim(1e-10, 5e-10)
            ax[2].set_ylabel(r"Y$_i$/Y$_p$")
            ax[2].legend()


        ax[-1].plot(x_new, P/np.max(P), c="black")
        ax[-1].axvline(x=best_x, c="black", linestyle="--", label=x_name + f"={best_x:.3f}" + r",  $\chi^2$=" + f"{best_chi:.3f}")
        ax[-1].set_ylabel("Normalized probability")
        
        ax[-1].set_xlabel(x_name)
        ax[-1].set_xlim((x[0], x[-1]))
        ax[-1].legend()

        
        if task == "j":
            ax[-1].set_xscale("log")
        
        plt.show()

        print(best_chi, best_x)
        return best_chi, best_x

    #creates figures for task f->i
    def create_fig(self, xlabel, ylabel, xlim=None, ylim=None) -> tuple:
        fig, ax = plt.subplots(1, 1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.invert_xaxis()

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    #plots individual graphs for task f->i
    def plot(self, ax, x, y, Y, label, style="-"):
        n_elements = len(Y)

        if n_elements < 4:
            if "(eq)" in label:
                style = "--"

            if label.split()[0] == "n":
                color = "tab:blue"
            elif label.split()[0] == "p":
                color = "tab:orange"
            elif label.split()[0] == "D":
                color = "tab:green"

            ax.plot(x, y, label=label, linestyle=style, c=color)
        else:
            ax.plot(x, y, label=label, linestyle=style)

    
    #plots all relevant graphs for task f->i
    def plot_task(self, ax, solution):
        T = solution.t
        Y = solution.y
        n_elements = len(Y)
        
        #[n, p, D, T, He3, He4, Li7, Be7]
        Ai = [1, 1, 2, 3, 3, 4, 7, 7]

        Yn, Yp = Y[:2]

        self.plot(ax, T, Yp*Ai[1], Y, label="p")
        self.plot(ax, T, Yn*Ai[0], Y, label="n")
        if n_elements < 4:
            Yn_eq, Yp_eq = self.Y_init(T)
            self.plot(ax, T, Yn_eq*Ai[0], Y, label="n (eq)")
            self.plot(ax, T, Yp_eq*Ai[0], Y, label="p (eq)")

        if n_elements >= 3:
            YD = Y[2]
            self.plot(ax, T, YD*Ai[2], Y, label="D")

        if n_elements >= 4:
            YT = Y[3]
            self.plot(ax, T, YT*Ai[3], Y, label=r"T")

        if n_elements >= 5:
            YHe3 = Y[4]
            self.plot(ax, T, YHe3*Ai[4], Y, label=r"He$^3$")

        if n_elements >= 6:
            YHe4 = Y[5]
            self.plot(ax, T, YHe4*Ai[5], Y, label=r"He$^4$")

        if n_elements >= 7:
            YLi7 = Y[6]
            self.plot(ax, T, YLi7*Ai[6], Y, label=r"Li$^7$")

        if n_elements >= 8:
            YBe7 = Y[7]
            self.plot(ax, T, YBe7*Ai[7], Y, label=r"Be$^7$")

    #for showing the graphs
    def show(self, ax):
        ax.legend()
        plt.show()



if __name__ == "__main__":
    #used in task f
    # Tspan = [100e9, 0.1e9]

    #used in task h->k
    Tspan = [100e9, 0.01e9]

    BBN_class = BBN(Tspan = Tspan)

    # uncomment the one you wish to run

    # BBN_class.solve_task_f(plot=True)
    # BBN_class.solve_task_h(plot=True)
    # BBN_class.solve_task_i(plot=True)
    # BBN_class.solve_task_j(n=20)
    BBN_class.solve_task_k(n=20)