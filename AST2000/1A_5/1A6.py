#ikke kodemal
import ast2000tools.solar_system as astSS
import ast2000tools.constants as astC
import numpy as np
import matplotlib.pyplot as plt
import random as r


class Planet:
    def __init__(self, seed):
        #lager solsystemet og henter ut relevante variable
        self.SS=astSS.SolarSystem(seed)
        mass=self.SS.masses[0]*astC.m_sun #gitt i solar masse, så må gange konvertere
        radii=self.SS.radii[0]*1000 #gitt i km, må konvertere til meter
        self.P=[mass, radii]
        print(f"Massen til planeten er {self.P[0]:5.4}kg og med en radius {self.P[1]:5.4}meter")
    def v_esc(self):
        #funksjon for å finne unnslipningshastighet
        M=self.P[0]
        R=self.P[1]
        G=astC.G
        v=np.sqrt(2*G*M/R)
        return v

class Particles:
    def __init__(self, T, N):
        self.T=T #gitt i Kelvin
        self.N=N #antall partikler
        self.m=astC.m_H2 #massen til et hydrogen atom
        self.k=astC.k_B  #Boltzman konstanten
    def generator(self):
        pos=[]
        vel=[]
        mu=0
        k=self.k
        m=self.m
        #finner sigma analytisk
        sigma=np.sqrt(k*T/m)
        for i in range(N):
            #bruker universiell fordeling for å generere partikler
            pos.append(np.array([r.random()*L for j in range(3)]))
            #bruker gaussisk randomiserer for å generere hastigheten tiil enhver partikkel
            v_temp=[r.gauss(mu, sigma) for j in range(3)]
            #for å gjøre regningen senere sjekker jeg om tallet vi fikk var uendelig
            #om det er det så generer vi det på nytt helt til alle tallene ikke er uendelig
            while float("inf") in v_temp:
                v_temp=[r.gauss(mu, sigma) for j in range(3)]
            vel.append(np.array(v_temp))
        #lagrer de for senere bruk
        self.pos=np.array(pos)
        self.vel=np.array(vel)
    def P(self, v):
        m=astC.m_H2
        T=self.T
        k=astC.k_B
        #bruker Maxwell-Boltzmann likningen til å finne sannsynlighetstettheten
        return np.sqrt(m/(2*np.pi*k*T)*np.exp(1/2*m*v**2/k*T))*4*np.pi*v**2
    def K_energy(self):
        vel=self.vel
        pos=self.pos
        m, k, T, N=self.m, self.k, self.T, self.N
        v=np.array([np.linalg.norm(vel[i,:]) for i in range(N)])
        #finner kinetisk energi numerisk
        K_num=sum(1/2*m*v[i]**2 for i in range(N))/N
        #finner kinetisk energi analytisk
        K_ana=3/2*k*T
        print(f"The numerically derived kinetic energy in the box is {K_num:g}, while the analytic formula says it should be {K_ana:g}")
        #finner relativ effekt
        e_rel=abs(K_num-K_ana)/K_ana
        print(f"Giving us a relative error of {e_rel*100:.2f}%")
    def plot(self):
        vel=self.vel
        pos=self.pos
        #plotter egenskapene til partiklene for å dobbelsjekke at det var riktig
        #finner sannsynligheten for at en partikkel har en gitt hastighet
        P_array=np.zeros((self.N, 3))
        for i in range(self.N):
            for j in range(3):
                P_array[i,j]=self.P(vel[i,j])
        #gjør meg klar for å plotte
        figs, (ax1, ax2, ax3)=plt.subplots(3)
        figs.tight_layout()
        ax=[ax1, ax2, ax3]
        fig=plt.figure()
        axx=fig.add_subplot(projection="3d")
        komponenter=["x","y","z"]
        #plotter histogrammene
        for i in range(3):
            ax[i].set_title(f"Fart i {komponenter[i]}-retning")
            ax[i].set_ylabel("Sannsynlighetstetthet")
            ax[i].set_xlabel("[m/s]")
            ax[i].hist([vel[:,i], P_array[:,i]], 100, range=(-20000,20000))
        #plotter posisjonen til partiklene
        axx.scatter(pos[:,0], pos[:,1], pos[:,2])
        plt.show()


class Rocket(Planet):
    def __init__(self, seed, T, mass):
        self.T=T #gitt i Kelvin
        self.m=astC.m_H2 #massen til et hydrogen atom
        self.k=astC.k_B  #Boltzman konstanten
        self.mass=mass #massen til rakketen
        super().__init__(seed)
    def box(self, N, L):
        self.L=L
        self.N=N
        #genererer alle partiklene i boksen
        self.particles=Particles(self.T, N)
        self.particles.generator()
    def simulate(self, dt, n):
        vel=self.particles.vel  #.flatten()
        pos=self.particles.pos  #.flatten()
        L, N, m=self.L, self.N, self.m
        self.dt, self.n=dt, n
        #lager en liste for å samle all informasjonen vi trenger fra
        #når en partikel kolliderer med en vegg
        #første plass er antall kollisjoner, andre er etterlatt momentum
        #tredje er antallet som unslipper igjennom hullet
        #siste er hastighetvektoren i x-retning til partiklene som unslipper
        walls=[0,0,0,0]
        #går igjennom hvert tidssteg
        for i in range(n):
            #oppdaterer possisjon med eulers metode
            pos=pos+vel*dt
            #sjekker om noe spesielt skjer
            #først om veggen treffes
            hit=np.where(pos<=0, 1, 0)
            #om partiklen er innenfor hullet i boksen
            hole1=np.where(pos[:,1:]<=3*L/4, 1, 0)
            hole2=np.where(pos[:,1:]>=L/4, 1, 0)
            hole=hole1+hole2
            #oppdaterer hastigheter og posisjon for hvor det var kollisjon
            #for å slippe å generere nye partikler og tenke på bevarelse av bevegelsesmengde
            #fortsetter alle partikler i banene sine, uavhengig om de egentlig unnslapp
            vel=np.where(pos<=0, -vel, vel)
            pos=np.where(pos<=0, -pos, pos)
            vel=np.where(pos>=L, -vel, vel)
            pos=np.where(pos>=L, L-pos, pos)
            #sjekker alle partikler
            for j in range(N):
                #om partikelen traff veggen
                if hit[j, 0]==1:
                    #registrer treffet
                    walls[0]+=1
                    #registrerer etterlatt momentum
                    walls[1]+=vel[j, 0]*m
                    #sjekker om partikelen som traff veggen er innenfor hullet
                    if hole[j, 0]==2 and hole[j, 1]==2:
                        #registrerer at partikelen traff hullet
                        walls[2]+=1
                        #registrerer momentum de skulle ha etterlatt
                        walls[3]-=abs(vel[j, 0])*m
        self.walls=walls
        
    def pressure(self):
        walls=self.walls
        dt, n, k, T, L, N=self.dt, self.n, self.k, self.T, self.L, self.N
        m=astC.m_H2
        dT=dt*n
        V=L**3
        A=L**2
        #regner ut trykk numerisk
        #siden hastighetene allerede er oppdaterte når jeg samler de
        #trenger jeg ikke å gange med -1
        dp=walls[1]
        F=dp/dT
        self.P_num=F/A
        #regner det analytisk
        self.P_ana=(N/V)*k*T
        #finner relativ feil
        error=abs(self.P_num-self.P_ana)/self.P_ana
        print(error, self.P_num, self.P_ana)
        print(f"Forskjellen mellom trykket beregnet analytisk og numerisk gir en relativ feil på {error*100:.2f}%")
    def vel(self):
        walls=self.walls
        dt, n, L, m, N, mass=self.dt, self.n, self.L, self.m, self.N, self.mass
        dT=dt*n
        dp=walls[3]
        F=-dp/dT
        dv=F/mass
        
        t=self.v_esc()/dv
        #deler med 20*60=1200sekunder som tilsvarer 20min
        self.n_boxes=round(t/(20*60))
        print(f"Hver boks bidro Delta v, {dv:5.4}m/s^2, i en periode Delta T på et nanosekund")
        print(f"Det kreves {self.n_boxes:g}bokser for å nå unnslipningshastigheten under 20min")



    def fuel_loss(self):
        walls=self.walls
        n_boxes=self.n_boxes
        dt, n, L, m, N, mass=self.dt, self.n, self.L, self.m, self.N, self.mass
        dT=dt*n
        loss=n_boxes*m*walls[2]*(20*60)/dT
        print(f"Raketten krever {loss:.0f}kg drivstoff for å nå {self.v_esc():.0f}m/s")










#lagrer alle konstantene brukt
#lager instanser av alle klassene
#kaller på alle funksjonene vi trenger       
#seed har jeg bare tilfeldigvalgt selv, ettersom jeg ikke var klar over at man fikk et utdelt
#12321 er tilfeldig valgt og bassert på 111^2  
seed=12321    
T=10000 #Kelvin
N=10**5 #antall partikler
L=10**-6 #lengde på boks oppgitt i meter
mass=1000 #massen til satelitten
model1=Rocket(seed, T, mass)
model1.box(N, L)

par=Particles(T, N)
#lager alle partiklene
par.generator()
#viser hvordan fordelingen av posisjon- og hastighetsvektor er fordelt
#par.plot()
#beregner midlere kinetisk energi og gir oss relativ feil
par.K_energy()

#flere variabler
dt=10**-12 #tidssteg
dT=10**-9 #tidsperiode
n=int(dT/dt) #antall tidssteg
#simulerer bevegelsen av partiklene
model1.simulate(dt, n)
#finner relativ feil for trykket.
model1.pressure()
#beregner antall bokser vi trenger for å unslippe
model1.vel()
#finner hvor mye drivstoff vi bruker
model1.fuel_loss()