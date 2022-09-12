#ikke kodemal
import ast2000tools.solar_system as astSS
import ast2000tools.constants as astC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ast2000tools.utils as utils
seed = utils.get_seed("bjornhod")


class System:
    def __init__(self, seed):
        #lagrer all nødvendig informasjon som trengs senere
        self.SS=astSS.SolarSystem(seed)
        self.pmass=self.SS.masses   #gitt i solar masse, så må gange konvertere
        self.smass=self.SS.star_mass
        self.ppos=self.SS.initial_positions
        self.pvel=self.SS.initial_velocities

    #lager en metode som lar meg hente ofte brukte variabler fortere
    def __call__(self):
        gamma=astC.G_sol
        pmass=self.pmass
        smass=self.smass
        n_p=self.SS.number_of_planets
        return gamma, pmass, smass, n_p
    
    #finner den planeten lengst unna ettersom denne bruker lengst tid
    def find_furthest_planet(self):
        rmax=np.linalg.norm(np.array([0, 0]))
        gamma, pmass, smass, n_p=self()
        planets=self.ppos
        for i in range(n_p):
            r=np.linalg.norm(planets[:,i])
            if r>rmax:
                rmax=r
                pf=i
        return pf
    
    #en metode for å finne kraften F fra stjernen på alle planeter
    def findF(self, x):
        gamma, pmass, smass, n_p=self()
        F=np.zeros(np.shape(x))
        for i in range(n_p):
            #finner posisjonsvektoren til planeten
            vecR=-x[:, i]
            #finner avstanden
            r=np.linalg.norm(vecR)
            #setter det inn i uttrykket for kraften og finner retningen
            G=gamma*pmass[i]*smass/r**2*vecR/r
            #setter kraften inn for riktig planet
            F[:, i]+=G
        return F

    #progresjonen igjennom alle tidstegene det tar for at alle planeter kommer rundt
    def progress(self, x, v, a, pf, dt):
        n=1
        #finner hvor langt planeten lengst unna beveger seg i første tidsteget
        tolx=abs(x[1][0 ,pf]-x[0][0,pf])
        toly=abs(x[1][1 ,pf]-x[0][1,pf])
        
        xold, vold, aold = x[1], v[1], a[0]

        #sjekker om avstanden fra den siste posisjonen til planeten lengst unna
        #i det siste tidsteget er fra hvor den starta
        while (abs(x[n][0, pf]-x[0][0, pf])>tolx or abs(x[n][1, pf]-x[0][1, pf])>toly) or n<1e4:
            anew, vnew, xnew=self.advance(xold, vold, aold, dt)
            a.append(anew) ; v.append(vnew) ; x.append(xnew)
            xold, vold, aold = xnew, vnew, anew
            n+=1
        #lager x og t slik at de kan brukes til plot senere
        t=np.linspace(0, dt*(n-1), n+1)
        x=np.array(x)
        return x, t

    #euler-cromer
    def advance(self, x, v, a, dt):
        anew=self.findF(x)/self.pmass
        vnew=v+anew*dt
        xnew=x+vnew*dt
        return anew, vnew, xnew

    #starter simuleringen
    def simulate(self, dt):
        #get all info
        gamma, pmass, smass, n_p=self()
        
        #forbereder for euler-cromer
        x=[self.ppos]
        v=[self.pvel]
        a=[np.zeros(np.shape(x))]

        #vi trenger to tidssteg for å sette igang while løkken (tolx og toly)
        anew, vnew, xnew=self.advance(x[0], v[0], a[0], dt)
        a[0]=anew ; v.append(vnew) ; x.append(xnew)
        
        #finner planeten lengst unna
        pf=self.find_furthest_planet()
        #fortsetter simuleringen og retunerer det vi får ut av den
        return self.progress(x, v, a, pf, dt)

    #lager en figur av posisjonen til 
    def figure_2(self):
        gamma, pmass, smass, n_p=self()

        pos=np.array([np.linalg.norm(self.ppos[:,i]) for i in range(n_p)])
        x=np.zeros(n_p+1)
        y=x.copy()
        x[:n_p]=pos
        x[-1]=0
        for i in range(n_p+1):
            if x[i]<0:
                x[i]=-x[i]
        plt.rcParams['axes.facecolor'] = 'black'
        plt.scatter(x[:-1], y[:-1], label="Planeter")
        plt.scatter(0, 0, c="yellow", label="Sol")
        plt.xlabel("r: [AU]")
        plt.legend(labelcolor="white")
        plt.show()
            
    
    #plotter banene til alle planetene for å forsikre oss om at de har gått rundt
    def plot(self, dt):
        x, t=self.simulate(dt)
        gamma, pmass, smass, n_p=self()
        
        #fig=plt.figure(figsize=(8, 8))
        pf=self.find_furthest_planet()
        x_=x[:, :, pf]
        max_val=0
        for i in range(len(x[:, 1])):
            if abs(x_[i, 0])>max_val:
                max_val=abs(x_[i, 0])
            if abs(x_[i, 1])>max_val:
                max_val=abs(x_[i, 1])

        plt.xlim(-max_val, max_val) ; plt.ylim(-max_val, max_val)
        plt.xlabel("x[AU]") ; plt.ylabel("y[AU]")
        #plotter hver av banene
        for i in range(n_p):
            plt.plot(x[:, 0, i], x[:, 1, i])
        #plt.axis("equal")
        plt.show()
        plt.plot(x[:, 0, 0], x[:, 1, 0])
        plt.xlabel("x[AU]") ; plt.ylabel("y[AU]")
        plt.show()
    
    #generer videoen
    def show_video(self, dt):
        x, t=self.simulate(dt)
        gamma, pmass, smass, n_p=self()
        #x har feil form og må derfor restruktureres
        #tidligere form:
        #np.shape(x)=(len(t), 2, n_p)
        x_new=np.zeros((2, n_p, len(t)))
        for i in range(2):
            for j in range(n_p):
                x_new[i, j]=x[:, i, j]
        self.SS.generate_orbit_video(t, x_new)
    
    #en metode som tester ut de første par tidstegene for feilsøking
    #animerer de første tidstegene for å kunne visualisere hvor det
    #kan ha gått galt
    def testing(self, dt):
        gamma, pmass, smass, n_p=self()

        #printer ut viktige verdier de for de første tidstegene
        pos=self.ppos
        vel=self.pvel
        x, y=pos[0,0], pos[1, 0]
        vx, vy=vel[0, 0], vel[1, 0]
        print(f"{x=:g} , {y=:g}")
        print(f"{vx=:g} , {vy=:g}")

        vecR=-pos[:, 0]
        r=np.linalg.norm(vecR)
        print(f"{r=:g}, {pmass[0]=:g}")
        G=gamma*pmass[0]*smass/r**3*vecR
        print(f"{G[0]/pmass[0]=:g} , {G[1]=:g}") #/pmass[0]
        Fx=G[0] ; Fy=G[1]

        fig=plt.figure(figsize=(8, 8))
        
        #plotter en planet, med sol og vektor pilen til
        #kraften fra sola på planeten
        plt.scatter(0, 0)
        plt.scatter(x, y)
        plt.quiver(x, y, Fx, Fy)
        plt.show()
        
        xinit, yinit=x, y
        vxinit, vyinit=vx, vy
        ax=Fx/pmass[0] ; ay=Fy/pmass[0]
        vx=vxinit+ax*dt ; vy=vyinit+ay*dt
        x=x+vx*dt ; y=y+vy*dt
        pos=[[xinit, x], [yinit, y]]
        vel=[[vxinit, vx], [vyinit, vy]]
        print(f"{x=:g} , {y=:g}")
        print(f"{vx=:g} , {vy=:g}")

        
        plt.style.use('seaborn-pastel')
        def animate(i):
            posi=np.array(pos)

            vecR=-posi[: ,i]
            r=np.linalg.norm(vecR)

            G=gamma*(pmass[0]*smass)/r**2*(vecR/r)
            Fx=G[0] ; Fy=G[1]

            vx, vy=vel[0][-1], vel[1][-1]
            x, y=pos[0][-1], pos[1][-1]

            ax=Fx/pmass[0] ; ay=Fy/pmass[0]
            vx1=vx+ax*dt ; vy1=vy+ay*dt
            x1=x+vx1*dt ; y1=y+vy1*dt
            
            vel[0].append(vx1) ; vel[1].append(vy1)
            pos[0].append(x1) ; pos[1].append(y1)

            scat=plt.scatter(x1, y1)
            return scat, 
        
        anim = animation.FuncAnimation(fig, animate, frames=200)
        plt.show()


#en klasse for tilfelle hvor det er to stjerner
#ihvertfall et forsøk på det
#virker som at kraften blir anvendt på feil legemer
#også mulig med noe enhet feil
class Binary(System):
    def __init__(self, pos, vel, mass):
        #noen metoder er fortsatt avhengig av variabler 
        #som krever et solsystem
        super().__init__(12321)

        #lagrer all informasjon slik jeg ønsker det å være
        #for bruk i denne klassen
        pos, vel, mass=self.flip([pos, vel, mass])
        self.ppos = pos
        self.pvel = self.unit_converter(vel, "vel")
        self.pmass = mass
        self.pmass[0] = self.unit_converter(mass[0], "mass")
    
    #en metode som reorganiserer en liste med matriser
    def flip(self, M):
        for i in range(len(M)):
            M[i]=np.transpose(M[i])
        return M

    #konverterer ulike enheter til ønsket enheter
    def unit_converter(self, unit, type):
        if type=="vel":
            vel=unit
            yr=astC.yr ; AU=astC.AU
            for i in range(len(vel)):
                vel[i]=vel[i]*AU/yr
            return vel

        if type=="mass":
            mass=unit
            solar_mass=astC.m_sun
            mass=mass/solar_mass
            return mass

        if type=="time":
            dt=unit
            yr=astC.yr
            dt=dt/yr
            return dt

    #samme metode som tidligere, men nå med flere/annereledes variabler   
    def __call__(self):
        gamma, pmass, smass, n_p=super().__call__()
        n_p=len(self.ppos[0])
        return gamma, pmass, smass, n_p

    #finner F, men nå begge veier
    #ikke bare sol på planet, men også planet på sol
    def findF(self, x):
        gamma, pmass, smass, n_p=self()
        F=np.zeros(np.shape(x))
        for i in range(n_p):
            for j in range(i+1, n_p):
                vecR=x[:, j] - x[:, i]
                r=np.linalg.norm(vecR)
                G=gamma*pmass[i]*pmass[j]/r**3*vecR
                F[:, i]+=G
        return F

    #modifisert metoden fra superklassen til å nå gå et viss antall tidssteg
    def progress(self, x, v, a, pf, dt):
        n=1

        xold, vold, aold = x[1], v[1], a[0]
        while n<self.N:
            anew, vnew, xnew=self.advance(xold, vold, aold, dt)
            a.append(anew) ; v.append(vnew) ; x.append(xnew)
            xold, vold, aold = xnew, vnew, anew
            n+=1
        t=np.linspace(0, dt*(n-1), n+1)
        x=np.array(x)
        return x, t

    def plot(self, dt, N):
        self.N=N
        return super().plot(dt)
    #tar nå et til element N for antall iterasjoner
    def show_video(self, dt, N):
        self.N=N
        x, t=self.simulate(dt)
        gamma, pmass, smass, n_p=self()
        x_new=np.zeros((2, n_p, len(t)))
        for i in range(2):
            for j in range(n_p):
                x_new[i, j]=x[:, i, j]
        self.SS.generate_binary_star_orbit_video(t, x_new[:, 0], x_new[:, 1], x_new[:, 2])

    #plotter posisjonene til planeten og stjernene
    #de første 10 tidsstegene med piler for retningen
    #til kraften. Dette hjelper med feilsøking
    def testing(self, dt):
        gamma, pmass, smass, n_p=self()
        pos=self.ppos
        vel=self.pvel

        print(pmass)

        F=self.findF(pos)
        plt.xlabel("x[AU]") ; plt.ylabel("y[AU]")
        plt.scatter(pos[0], pos[1])
        plt.quiver(pos[0], pos[1], F[0], F[1])
        for n in range(10):
            F=self.findF(pos)
            vel=vel+F/pmass*dt
            pos=pos+vel*dt
            plt.scatter(pos[0], pos[1])
            plt.quiver(pos[0], pos[1], F[0], F[1])
        plt.show()

M46NU5=System(seed)

#lager en figur brukt i artiklen
#M46NU5.figure_2()

#tidshopp
dt=1e-3
#M46NU5.testing(dt)

M46NU5.plot(dt)
#M46NU5.show_video(dt)

#lagrer informasjonen til binær systemet
planet_pos=np.array([-1.5, 0])
star1_pos=np.array([0, 0])
star2_pos=np.array([3, 0])

planet_vel=np.array([0, -1])
star1_vel=np.array([0, 30])
star2_vel=np.array([0, -7.5])

planet_mass=6.39e23
star1_mass=1
star2_mass=4

pos=np.array([planet_pos, star1_pos, star2_pos])
vel=np.array([planet_vel, star1_vel, star2_vel])
mass=np.array([planet_mass, star1_mass, star2_mass])

binary=Binary(pos, vel, mass)

#finner tidssteg per år
dt=400/astC.yr
#antall tidssteg ønsket
N=1e6
#verken plot og show_video gir riktige resultater
#binary.plot(dt, N)
#binary.show_video(dt, N)

#dette funksjonkallet viser hvorfor det er
#binary.testing(dt)