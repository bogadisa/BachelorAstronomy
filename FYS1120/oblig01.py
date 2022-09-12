from matplotlib.colors import Normalize
import numpy as np
from scipy.constants import epsilon_0, elementary_charge
import matplotlib.pyplot as plt


#funksjonen for å finne E og V i et punkt
#tar en liste r med matriser for å regne E og V i disse punktene
#Q er en liste med ladningstørrelser Q[i] med tilhørende posisjon R[i]
def findEfieldinapoint(r, Q, R):
    #sjekker dimensjonen til r
    #koden kan forenkles ved å gjøre det om til en klasse
    #å kunne finne potensial og elektrisk felt i ulike dimensjoner er nyttig
    #da kan du enkelt sammenlikne om koden gjør det samme i alle dimensjoner
    if len(r)==3:
        x, y, z=r
        #lager V slik at den har samme form som x, y og z
        V=np.zeros(np.shape(x))
        #går igjennom alle elementene i x, y og z matrisene
        for i in range(len(x.flat)):
            #lager lille r (posisjonen til test ladningen)
            r=np.array([x.flat[i], y.flat[i], z.flat[i]])
            #regner ut V for alle tre dimensjonene
            V.flat[i]+=1/(4*np.pi*epsilon_0)*sum(Q[j]/np.linalg.norm(r-R[j]) for j in range(len(Q)))
        #finner E som den negative gradienten til V
        E=-np.array(np.gradient(V))
        #retunerer E og V
        return E, V
    #gjør akkurat det samme bare for to dimensjoner
    elif len(r)==2:
        x, y=r
        V=np.zeros(np.shape(x))
        for i in range(len(x.flat)):
            r=np.array([x.flat[i], y.flat[i]])
            V.flat[i]+=sum(Q[j]/np.linalg.norm(r-R[j]) for j in range(len(Q)))
        E=-np.array(np.gradient(V))
        return E, V
    #gjør akkurat det samme bare for en dimensjon
    else:
        x=r
        V=np.zeros(np.shape(x))
        for i in range(len(x)):
            r=np.array([x[i]])
            V.flat[i]+=sum(Q[j]/np.linalg.norm(r-R[j]) for j in range(len(Q)))
        E=-np.array(np.gradient(V))
        return E, V

    

N=20    #antall intervaller-1
a=1     #verdien til a
#lager x, y og z matrisene i tre dimensjoner
L=np.linspace(-5*a,5*a,N+1)
x, y, z=np.meshgrid(L ,L, L, indexing="xy", sparse=False)

q=elementary_charge     #elementærladning

#lager strukturen til CO2 molekylet
Q1=2*q ; r1=np.array([0, 0, 0])
Q2=-q  ; r2=np.array([a, 0, 0])
Q3=-q  ; r3=np.array([-a, 0, 0])
#lagrer informasjonen i en liste
Q=[Q1, Q2, Q3] 
R=[r1, r2, r3]

#lager et scatterplot for å illustrere hvordan molekylet er bygget opp
fig=plt.figure()
ax=fig.add_subplot(projection="3d")
R_=np.array(R)
s=[10**3, 10**2, 10**2] ; c=np.array(["#000000", "#FFFFFF", "#FFFFFF"])
edgecolors=np.array(["#000000", "#000000", "#000000"]) ; alpha=[1, 1, 1]
ax.set_xlabel("[x]")
ax.scatter(R_[:, 0], R_[:, 1], R_[:, 2], s=s, c=c, edgecolors=edgecolors, alpha=alpha)

#kaller på funksjonen fra tidligere
E, V=findEfieldinapoint([x, y, z], Q, R)

#importerer det jeg trenger for å animere
from matplotlib import animation
plt.style.use('seaborn-pastel')

#lager figuren
fig = plt.figure()
#ax=plt.axes(xlim=(-5*a, 5*a), ylim=(-5*a, 5*a))

#animerer potensialet og det elektriske feltet for xy-planet for hver verdi av z
def animate(i):
    #fjerner forrige streamplot
    ax.collections = []
    ax.patches=[]
    #hadde problemer med hvite hull i contourf, på denne måten blir de borte
    cont= plt.contourf(x[:,:,0], y[:,:,0], V[:, :, i-1], 25)
    #[:,:, i] henter ut de riktige verdiene ved z_i.
    #[:, :, 0] henter ut xy-planet som er det samme uavhengig av i og er derfor i=0
    cont= plt.contourf(x[:,:,0], y[:,:,0], V[:, :, i], 25)
    #om du vil heller plotte quiver istedenfor streamplot kan det endres her
    #quiv= plt.quiver(x[:, :, 0], y[:, :, 0], E[0][:, :, i], E[1][:, :, i])
    stre= plt.streamplot(x[:, :, 0], y[:, :, 0], E[1][:, :, i], E[0][:, :, i])
    return cont , stre

#lager animasjonen
#plt.ylabel("[y]") ; plt.xlabel("[x]")
#anim = animation.FuncAnimation(fig, animate, frames=N)
#plt.show()

#lagrer animasjonen som en gif, slik at den kjører glattere (animasjonen blir fort hakkete)
#anim.save('CO2_molecule_2.gif', writer='imagemagick')

#gjør det samme som over bare i en dimensjon for å finne potensialet langs x-aksen
x = np.linspace(a+0.1,1000,10000)
Q1=2*q ; r1=np.array([0])
Q2=-q  ; r2=np.array([a])
Q3=-q  ; r3=np.array([-a])
Q=[Q1, Q2, Q3]
R=[r1, r2, r3]

E, V=findEfieldinapoint(x, Q, R)

def findEanalyticallyfaraway(x):
    V=-q/(4*np.pi*epsilon_0)*2*a**2/(x*(x-a)*(x+a))
    E=np.gradient(-V)
    return V

#gjør plotet enklere å tolke ved å se på log10 av verdiene
xlog = np.log10(x)
Elog_1 = np.log10(abs(E))
E_2=findEanalyticallyfaraway(x)
Elog_2= np.log(abs(E_2))
#plt.axis([0, 3.1, -30, -17])
plt.ylabel("log10 av det elektriske feltet E") ; plt.xlabel("log10 av x")
plt.plot(xlog, Elog_1, label="Numerisk")
plt.plot(xlog, Elog_2, label="Analytisk")
plt.legend()
plt.show()