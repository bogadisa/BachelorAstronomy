#ikke kodemal
import ast2000tools.solar_system as astSS
import ast2000tools.constants as astC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ast2000tools.utils as utils
seed = utils.get_seed("bjornhod")

class Lander:
    def __init__(self, seed, mass, A, vel):
        #lagrer mesteparten av all informasjon jeg trenger til senere
        self.SS = astSS.SolarSystem(seed)
        self.pmass = self.SS.masses[0]*astC.m_sun #oppgitt i solar masse
        self.radii = self.SS.radii[0] #opgitt i km
        self.mass = mass #oppgit i kg
        self.A=A #oppgit i m**2

        print(f"Massen til planeten er {self.pmass:g}kg med radius {self.radii:g}km")

        self.pos = np.array([0, 40000+self.radii])*1000 #oppgit i meter
        self.vel = vel
    
    #gjør det enklere å hente ofte brukte variabler
    def __call__(self):
        gamma = astC.G
        pmass = self.pmass
        return gamma, pmass

    def find_g(self, x):
        gamma, pmass = self()
        vecR = x
        r = abs(vecR)
        #finner tyngdekraften ved avstand r fra jordas sentrum
        G = gamma*pmass/r**2*vecR/r
        return G

    #finner trykket slik oppgitt i oppgaven
    def pressure(self, x):
        h = x[1]-self.radii*1000
        rho_0 = self.SS.atmospheric_densities[0]

        #printer ut nødvendig informasjon
        #print(f"{rho_0=:g}")
        g = self.find_g(self.radii*1000)
        #print(f"Tyngdekraften på overflaten til planeten er {g=:g}")
        h_scale = 75200/g
        #print(f"Som gir oss en {h_scale=:g}")
        rho = rho_0*np.exp(-h/h_scale)
        return rho

    #finner friksjon i atmosfæren
    def find_drag(self, pos, vel):
        A = self.A
        v = np.linalg.norm(vel)
        rho = self.pressure(pos)
        F_D = -1/2*rho*A*v*vel
        return F_D/self.mass

    def find_F_tot(self, pos, vel, arg="scalar"):
        F_D = self.find_drag(pos, vel)
        if arg == "scalar":
            F_g = np.array([0, self.find_g(pos[1])])
        #legger til støtte for array input
        #endte opp med å ikke bruke noe særlig
        elif arg == "array":
            F_g = np.transpose(np.array([np.zeros(len(F_D)), self.find_g(pos[:, 1])]))
        return F_D - F_g

    #bruker euler cromer til å finne neste steg
    def advance(self, x, v, a, dt, n=0):
        F = self.find_F_tot(x, v)
        anew = F
        vnew = v + a*dt
        xnew = x + vnew*dt
        return xnew, vnew, anew
    
    #selve while-løkken
    def progress(self, x, v, a, dt):
        ground = self.radii*1000
        xold, vold, aold = x[1], v[1], a[1]

        n=1
        drag_max = np.linalg.norm(self.find_drag(x[-1], v[-1])*self.mass)
        #sjekker om posisjonen er over bakken og friksjonen ikke er for stor
        while x[-1][1] >= ground: # and np.linalg.norm(self.find_drag(x[-1], v[-1])*self.mass)<25000:
            xnew, vnew, anew = self.advance(xold, vold, aold, dt, n)
            x.append(xnew) ; v.append(vnew) ; a.append(anew)
            xold, vold, aold = xnew, vnew, anew
            n+=1
            if np.linalg.norm(self.find_drag(x[-1], v[-1])*self.mass) > drag_max:
                drag_max = np.linalg.norm(self.find_drag(x[-1], v[-1])*self.mass)

        #gir informasjon om hva som skjedde og interesante stats som kan brukes videre
        else:
            distance = abs(x[-1][1]-ground)
            drag = self.find_drag(x[-1], v[-1])*self.mass
            print(f"Luftmotstanden ved landing er {np.linalg.norm(drag):g}N, og den maksimale luftmotstanden er {drag_max:g}N")
            print(f"Landet roveren? {not(x[-1][1]>ground)}.")
            print(f"Brant roveren opp? {not(np.linalg.norm(drag)<25000)}.")
            print(f"Hvor lang tid tok det? {n*dt:g}s")
            print(f"Hastigheten mot bakken er {v[-1][1]:g}m/s, Høyden er {distance:g}m")
        
        #lager en tidsarray
        t = np.linspace(0, dt*n, n+1)
        
        return x, v, a, t

    #starter simuleringen
    def simulate(self, dt):
        #første tidsteg
        x = [self.pos]
        v = [self.vel]
        a = [np.zeros(np.shape(self.pos))]
        #andre tidssteg
        xnew, vnew, anew = self.advance(x[0], v[0], a[0], dt)
        x.append(xnew) ; v.append(vnew) ; a.append(anew)

        #starter løkken
        x, v, a, t = self.progress(x, v, a, dt)

        return x, v, a, t
    
    def plot(self, dt):
        x, v, a, t = self.simulate(dt)
        x = np.array(x) #-self.radii*1000
        x[:, 1] = x[:, 1] - self.radii*1000
        v = np.array(v)
        a = np.array(a)

        #Finner kraften roveren erfarer
        F = a * self.mass

        #plotter høyde, fart mot overflate, akselerasjon i y-retning og luftmotstand+tyngdekraft
        fig, axs=plt.subplots(4,1)
        axs[0].plot(t, x[:, 1], label = "Høyde")
        axs[1].plot(t, v[:, 1], label = "Fart mot overlate")    
        axs[2].plot(t[1:], a[1:, 1], label = "Akselerasjon")
        axs[3].plot(t, F, label = "F_drag")
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()
        axs[3].set_xlabel("t[s]")
        axs[0].set_ylabel("h[m]")
        axs[1].set_ylabel("v[m/s]")
        axs[2].set_ylabel("a[m/s^2]")
        axs[3].set_ylabel("F[N]")
        plt.show()

    #lager en video
    def show_video(self, dt):
        x, v, a, t = self.simulate(dt)
        x = np.array(x)
        x = np.transpose(x)
        
        x[1] = x[1] - self.radii*1000
        
        self.SS.generate_landing_video(t, x, 0)


#setter konstanter vi trenger     
mass = 100 #kg
r = 1 #m
A = np.pi*r**2 #m**2
vel = np.array([100, -100])  #m/s

#lager et instanse av klassen
rover = Lander(seed, mass, A, vel)

#flere konstanter
dt = 0.01 #sekunder

rover.plot(dt)
#lager video
#rover.show_video(dt)