#ikke kodemal
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
import ast2000tools.constants as astC
seed = utils.get_seed("bjornhod")


data = []

days = [0, 2, 4, 6, 8, 11, 13, 15, 17, 19]

for i in days:
    data.append(np.loadtxt(f"spectrum_day{i}.txt"))

class DataExtractor:
    def __init__(self, data, days):
        """Lagrer all data for senere bruk"""
        self.data = data
        self.days = days

    @property
    def wave_length(self):
        """Gir deg bølgelengden til alle dager"""
        data = self.data
        wave_length = []

        for i in range(len(data)):
            wave_length.append(data[i][:, 1])
        
        return np.array(wave_length)

    @property
    def lmda(self):
        """Gir deg fluks til alle dager"""
        data = self.data
        lmda = []

        for i in range(len(data)):
            lmda.append(data[i][:, 0])
        
        return np.array(lmda)

    @staticmethod
    def days_to_sec(days):
        """Konverterer dager til sekunder"""
        return days*24*60*60

    def plot_spectra(self, n, show="True"):
        """Plotter fluks som en funksjon av bølgelengde"""
        wave_length = self.wave_length[n]
        lmda = self.lmda[n]
        days = self.days[n]

        plt.plot(lmda, wave_length)
        plt.title(f"Dag {days}")
        plt.ylabel("Normalisert fluks")
        plt.xlabel("Bølgelengde [nm]")
        
        #gjør at jeg kan bruke metoden senere uten å vise det med engang
        if show == "True":
            plt.show()

    def plot_spectra_all(self):
        wave_length = self.wave_length
        lmda = self.lmda
        days = self.days

        n = len(days)
        fig, ax = plt.subplots(int(n/2), 2)
        for i in range(int(n/2)):
            ax[i, 0].plot(lmda[i], wave_length[i])
            ax[i, 0].set_title(f"Dag {days[i]}")
            ax[i, 1].plot(lmda[i], wave_length[i])
            ax[i, 1].set_title(f"Dag {days[i+1]}")
        ax[2, 0].set_ylabel("Normalisert fluks")
        ax[-1, 0].set_xlabel("Bølgelengde [nm]")
        ax[-1, 1].set_xlabel("Bølgelengde [nm]")
        plt.show()

    def F_model(self, lmda, F_min, sigma, lmda_center):
        """Uttrykket for F_model"""
        F_max = 1
        return F_max + (F_min - F_max) * np.exp(-(lmda - lmda_center)**2 / (2*sigma**2))

    def LSF(self, lmda, F_min, sigma, lmda_center, n):
        """Finner observert og model, deretter finner Delta"""
        F_obs = self.wave_length[n]
        F_model = self.F_model(lmda, F_min, sigma, lmda_center)

        Delta = np.sum((F_obs - F_model)**2)
        return Delta

    def LSF_finn_kombo(self, F_min, sigma, lmda_center, n):
        """Finner komboen som gir lavest Delta"""
        lmda = self.lmda[n]
        #setter standard
        Delta_min = self.LSF(lmda, F_min[0], sigma[0], lmda_center[0], n)
        Delta_min_kombo = [F_min[0], sigma[0], lmda_center[0]]

        #tester ut alle kombinasjoner
        for F_min_ in F_min:
            for sigma_ in sigma:
                for lmda_center_ in lmda_center:
                    Delta_new = self.LSF(lmda, F_min_, sigma_, lmda_center_, n)
                    #lagrer kun om den er bedre enn tidligere beste
                    if Delta_new < Delta_min:
                        Delta_min = Delta_new
                        Delta_min_kombo = [F_min_, sigma_, lmda_center_]
        print(Delta_min)
        return Delta_min_kombo

    def plot_spectra_LSF(self, Delta_min_kombo, n):
        """Plotter fluks som en funksjon av bølgelengde, med den beste approksimasjonen"""
        F_min, sigma, lmda_center = Delta_min_kombo
        lmda = self.lmda[n] #<====================0

        #Bruker F_model med beste kombinasjon
        F_model = self.F_model(lmda, F_min, sigma, lmda_center)

        self.plot_spectra(n, show="False")
        plt.plot(lmda, F_model, label = "Minste kvadraters metode")
        plt.legend()
        plt.show()


    
        
        





if __name__ == "__main__":
    seed74 = DataExtractor(data, days)
    
    #for i in range(len(days)):
    #    seed74.plot_spectra(i)

    #seed74.plot_spectra(0)

    #Finner intervaller F_min, sigma og lmda_center befinner seg innenfor ved øyet
    N = 30

    F_min_min = 0.9
    F_min_max = 0.7

    lmda_center_min = 656.38
    lmda_center_max = 656.391

    sigma_min = 656.391 - 656.386
    sigma_max = 656.397 - 656.38

    lmda_center = np.linspace(lmda_center_min, lmda_center_max, N)
    sigma = np.linspace(sigma_min, sigma_max, N)
    F_min = np.linspace(F_min_min, F_min_max, N)

    n = 0
    Delta_min_kombo = seed74.LSF_finn_kombo(F_min, sigma, lmda_center, n)

    #print(Delta_min_kombo)

    seed74.plot_spectra_LSF(Delta_min_kombo, n)

    #seed74.plot_spectra_all()

    #regner og printer ut relativ hastighet
    lmda_0 = 656.3
    lmda = np.array([656.3873, 656.3862, 656.3887, 656.3902, 656.3913, 656.3899, 656.3876, 656.3860, 656.3885, 656.3914])
    v_rel = astC.c * (lmda_0 - lmda) / lmda_0
    
    #print(v_rel)

    #minste kvadraters metode -> relativ hastighet
    v_rel_LSF = astC.c * (lmda_0 - 656.3879) / lmda_0
    print(v_rel_LSF)

    relativ_feil = (v_rel_LSF - v_rel[0]) / v_rel_LSF

    print(relativ_feil*100)