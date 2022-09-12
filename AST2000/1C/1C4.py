#ikke kodemal
import ast2000tools.constants as astC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ast2000tools.utils as utils
seed = utils.get_seed("bjornhod")


data = []
data.append(np.loadtxt("star0_5.99.txt")) ; data.append(np.loadtxt("star1_1.11.txt"))
data.append(np.loadtxt("star2_1.63.txt")) ; data.append(np.loadtxt("star3_1.77.txt"))
data.append(np.loadtxt("star4_3.20.txt"))


class DataExtractor:
    lmda_0 = 656.28

    def __init__(self, data, n_planets=5):
        """Lagrer all data for senere bruk"""
        self.data = data
        self.n_planets = n_planets

    @property
    def observations(self):
        """Sender tilbake hvor mange observasjoner det er 
        gjort for hver planet og den største observasjonperioden"""
        data = self.data
        n_planets = self.n_planets

        observations = []
        max_observations = 0
        for i in range(n_planets):
            observations.append(len(data[i][:, 0]))
            if len(data[i][:, 0]) > max_observations:
                max_observations = len(data[i][:, 0])
        
        return np.array(observations) , max_observations

    @property
    def t(self):
        """Sender tilbake observasjonstider for alle stjerner"""
        data = self.data
        n_planets = self.n_planets
        observations, max_observations = self.observations

        
        t = np.zeros((5, max_observations))

        for i in range(n_planets):
            t[i, :observations[i]] = data[i][:, 0]
            t[i, observations[i]:] = np.float('NaN')

        return t
    
    @property
    def lmda(self):
        """Sender tilbake hastigheter"""
        data = self.data
        n_planets = self.n_planets
        observations, max_observations = self.observations

        
        lmda = np.zeros((5, max_observations))

        for i in range(n_planets):
            lmda[i, :observations[i]] = data[i][:, 1]
            lmda[i, observations[i]:] = np.float('NaN')

        return lmda

    @property
    def flux(self):
        data = self.data
        n_planets = self.n_planets
        observations, max_observations = self.observations

        
        flux = np.zeros((5, max_observations))

        for i in range(n_planets):
            flux[i, :observations[i]] = data[i][:, 2]
            flux[i, observations[i]:] = np.float('NaN')

        return flux

    @property
    def hastighet(self):
        """Finner hastighetene til alle stjerner"""
        data = self.data
        n_planets = self.n_planets

        c = astC.c
        lmda_0 = self.lmda_0
        lmda = self.lmda

        delta_lmda = lmda - lmda_0

        v_r = c * delta_lmda / lmda_0
    
        return v_r

    def midlere_hastighet_setup(self):
        """"Finner midlere hastigheter gitt for alle stjerner"""
        n_planets = self.n_planets

        hastighet = self.hastighet
        observations, max_observations = self.observations

        v_r =np.zeros(n_planets)
        for i in range(n_planets):
            v_r[i] = sum(hastighet[i, j] for j in range(observations[i]))

        v_r_mean = v_r / observations

        return v_r_mean
        
    @property
    def midlere_hastighet(self):
        """Lagrer midlere hastigheten slik at vi ikke trenger å regne den ut for hver gang"""
        try:
            return self.midlere_hastighet_

        #om variablen ikke er definert ennå vil den bli definert
        except AttributeError:
            self.midlere_hastighet_ = self.midlere_hastighet_setup()
            return self.midlere_hastighet_

    def GetHastigheterPlot(self, i, ploting=True):
        """Finner observert hastigheter for hver av planetene,
        dette kan brukes til å plotte dem alle sammen om ploting=False, 
        eller plotte individuelt dersom ploting=True (default)"""
        hastighet = self.hastighet[i]
        midlere_hastighet = self.midlere_hastighet[i]
        t = self.t[i]

        hastighet_obs = hastighet - midlere_hastighet
        if not(ploting):
            return t, hastighet_obs
        else:
            plt.plot(t, hastighet_obs)
            plt.xlabel('t[dager]')
            plt.ylabel('v[m/s]')
            plt.show()
    
    def PlotHastighet(self):
        """Plotter alle hastighetskurvene"""
        n_planets = self.n_planets
        
        fig, axs = plt.subplots(n_planets, 1)
        for i in range(n_planets):
            t, hastighet_obs = self.GetHastigheterPlot(i, ploting=False)
            axs[i].plot(t, hastighet_obs)
            axs[i].set_ylabel("v[m/dag]")
        axs[-1].set_xlabel("t[dager]")
        plt.show()

    def GetLyskurvePlot(self, i, ploting=True):
        """Finner lyskurven til hver av stjernene og om
        ploting=False retuneres disse sammen med tider t,
        ellers plottes det individuelt (default)"""
        flux = self.flux[i]
        t = self.t[i]

        if not(ploting):
            return t, flux
        else:
            plt.plot(t, flux)
            plt.xlabel("t[dager]")
            plt.ylabel("Fluks[J/m^2]")
            plt.show()

    def PlotLyskurve(self):
        """Plotter alle lyskruvene"""
        n_planets = self.n_planets
        
        fig, axs = plt.subplots(n_planets, 1)
        for i in range(n_planets):
            t, flux = self.GetLyskurvePlot(i, ploting=False)
            axs[i].plot(t, flux)
            axs[i].set_ylabel("Flux[J/m^2*s]")
        axs[-1].set_xlabel("t[dager]")
        plt.show()

    def HastighetRadielModel(self, t, t0, P, vr, i):
        """Finner radiel hastighet model"""
        return vr * np.cos(2*np.pi/P * (t - t0))

    def LSF(self, t, t0, P, vr, i):
        """Minste kvadraters metode"""
        hastighet = self.hastighet[i]
        midlere_hastighet = self.midlere_hastighet[i]

        hastighet_obs = hastighet - midlere_hastighet

        hastighet_ = (hastighet_obs - self.HastighetRadielModel(t, t0, P, vr, i))**2
        hastighet_ = np.where(np.isnan(hastighet_), 0, hastighet_)
        
        s = np.sum(hastighet_)
        return s
    
    def LSF_setup(self, n, i=2):
        """Finner variabel grenser vi bruker i minste kvadraters metode.
        Fungerer kun for i=2 (eller stjerne 2)"""
        t = self.t[i]

        t0min = 2000
        t0max = 2400

        t0 = np.linspace(t0min, t0max, n)

        vrmin = 30
        vrmax = 41

        vr = np.linspace(vrmin, vrmax, n)

        Pmin = 4000
        Pmax = 5000

        P = np.linspace(Pmin, Pmax, n)

        return t, t0, P, vr

    def LSF_FinnKombo(self, n, i):
        """Finner den beste/laveste Delta og 
        hvilken kombinasjo det krever"""
        t, t0, P, vr = self.LSF_setup(n, i)

        smin = self.LSF(t, t0[0], P[0], vr[0], i)
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    s = self.LSF(t, t0[j], P[k], vr[l], i)
                    if s < smin:
                        smin = s
                        combo = [t0[j], P[k], vr[l]]
        
        return combo, smin

    def GetHastigheterPlotLFS(self, n, i, ploting=True):
        """Plotter hastighetplot med estimert ekte hastighet.
        Fungerer ikke akkurat nå p.g.a LSF_setup ikke fungerer
         for alle stjerner"""
        hastighet = self.hastighet[i]
        midlere_hastighet = self.midlere_hastighet[i]
        t = self.t[i]

        hastighet_obs = hastighet - midlere_hastighet

        combo, smin = self.LSF_FinnKombo(n, i)
        t0, P, vr = combo
        
        hastighet_model = vr * np.cos(2*np.pi/P * (t - t0))

        if not(ploting):
            return t, hastighet_obs, hastighet_model
        else:
            print(f"Vi får {t0=:.0f}, {P=:.0f} og vs={vr:.3f}, resulterende i et kvadrat {smin:g}")
            plt.plot(t, hastighet_obs)
            plt.plot(t, hastighet_model)
            plt.xlabel('t[dager]')
            plt.ylabel('v[m/s]')
            plt.show()

    def PlotHastigheterLFS(self, n):
        n_planets = self.n_planets
        
        fig, axs = plt.subplots(n_planets, 1)
        for i in range(n_planets):
            t, hastighet_obs, hastighet_model = self.GetHastigheterPlot(n, i, ploting=False)
            axs[i].plot(t, hastighet_obs)
            axs[i].plot(t, hastighet_model)
            axs[i].set_title(f"Hastighetkurve til stjerne {i}")
        plt.show()


if __name__ == "__main__":
    #seedet mitt slutter på 74
    seed74 = DataExtractor(data)
    seed74.GetHastigheterPlot(2)
    #seed74.PlotHastighet()
    seed74.GetLyskurvePlot(2)
    #seed74.PlotLyskurve()

    n = 20

    seed74.GetHastigheterPlotLFS(n, 2)

    G = astC.G
    sm = astC.m_sun
    P = 4300
    def day_to_sec(days):
        sec_per_day = 24*60*60
        return days*sec_per_day
    m_s = 1.63 * sm
    v_s = 38

    m_p = (day_to_sec(P)/(2*np.pi*G))**(1/3)*m_s**(2/3)*v_s

    #print(f"Massen til planeten, før minste kvadraters metode, er ca. {m_p:g}")
    
    P = 4263
    v_s = 33.474

    m_p = (day_to_sec(P)/(2*np.pi*G))**(1/3)*m_s**(2/3)*v_s

    print(f"Massen til planeten, etter minste kvadraters metode, er ca. {m_p:g}")

    def vs(t, vr, P, t0):
        return vr*np.cos((2*np.pi) / P * (t - t0))

    vp = 33.474 * m_s / m_p

    print(f"Hastigheten til planeten er {vp=:g}")

    t0 = 3339.4
    t1 = 3340.4

    r = vp * day_to_sec(t1 - t0) / 2

    print(f"Radiusen til planeten er {r=:g}")

    volum_p = 4/3 * np.pi * r**3

    density_p = m_p / volum_p

    print(f"Tettheten til planeten er {density_p:g}, med volum {volum_p:g}m^3")

    a_s = 22711.4

    a_p = a_s * m_s / m_p

    print(f"Planeten har en halvakse {a_p=:g}")