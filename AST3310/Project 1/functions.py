import numpy as np
from scipy.constants import k, e, h, epsilon_0


#all masses are assumed to be given in atomic mass units
u = 931.49410242 #*3e-5**2 #MeV/c^2
mu = 1.6605402E-27 #kg
N_A = 6.0221408e+23 #Moles

#energy made per positron
e_p = 2 * 0.511 #MeV per e+

#Solar core parameters
rho = 1.62e5 #*6.022e26 #kg m^-3   (pay attention to units!!!!!)
T = 1.57e7 #K
n_e = 0.5*rho/mu*(1 + 0.7)


def energy_out(mass_loss):
    energy_mass = mass_loss * u
    return energy_mass


def energy_loss_percent(energy_out, neutrino_energy):
    return np.sum(neutrino_energy)/energy_out

def number_density(fraction, mass):
    #electron density is calculated a bit differently
    #int(mass) will always be 0 for electrons and only electrons
    if int(mass) == 0:
        return 0.5*rho/mu*(1 + 0.7)
    return rho*fraction/(int(mass)*mu)

def cm_volume_to_m(volume):
    return volume/1e6

def lamda_to_r(lamda, ni, nk):
    conditional = 1/(1 + int(ni==nk))
    return lamda*ni*nk/rho*conditional

def MeV_to_J(energy):
    return energy*1.60218e-13

def get_reaction_rates(T):
    #pre calculate all the variants of T9 that we will need to calculate the reaction rates
    T9 = T/1e9 #K
    T9_ = T9/(1+4.95e-2*T9)
    T9__ = T9/(1+0.759*T9)

    T9_12_ = 1/np.sqrt(T9)
    T9_23 = T9**(2/3) ; T9_23_ = 1/T9_23
    T9_13 = T9**(1/3) ; T9_13_ = 1/T9_13 ; T9__13_ = T9_**(-1/3) ; T9___13_ = T9__**(-1/3)
    T9_43 = T9**(4/3)
    T9_53 = T9**(5/3)
    T9__56 = T9_**(5/6) ; T9___56 =T9__**(5/6)
    T9_32_ = T9**(-3/2)

    #store the reaction rates in the form of a dictionary
    #each chain contains all steps, meaning PP0 appears in each PP chain key
    reaction_rates = {
        "PP0" : np.array([4.01e-15*T9_23_*np.exp(-3.380*T9_13_)*(1 + 0.123*T9_13 + 1.09*T9_23 + 0.938*T9)]), #3H11 -> He^3_2 + e+
        "PP1" : np.array([4.01e-15*T9_23_*np.exp(-3.380*T9_13_)*(1 + 0.123*T9_13 + 1.09*T9_23 + 0.938*T9), #3H11 -> He^3_2 + e+
                          6.04e10*T9_23_*np.exp(-12.276*T9_13_)*(1 + 0.034*T9_13 - 0.522*T9_23 - 0.124*T9 + 0.353*T9_43 + 0.213*T9_53)]), #2He32 -> He32 + 2H11
        "PP2" : np.array([4.01e-15*T9_23_*np.exp(-3.380*T9_13_)*(1 + 0.123*T9_13 + 1.09*T9_23 + 0.938*T9), #3H11 -> He^3_2 + e+
                          5.61e6*T9__56*T9_32_*np.exp(-12.826*T9__13_), #He32 + He42 -> Be74
                          1.34e-10*T9_12_*(1 - 0.537*T9_13 + 3.86*T9_23 + 0.0027/T9*np.exp(2.515e-3/T9)), #Be74 + e- -> Li73
                          1.096e9*T9_23_*np.exp(-8.472*T9_13_) - 4.830e8*T9___56*T9_32_*np.exp(-8.472*T9___13_) + 1.06e10*T9_32_*np.exp(-30.422/T9)]), #Li73 + H11 -> 2He42s
        "PP3" : np.array([4.01e-15*T9_23_*np.exp(-3.380*T9_13_)*(1 + 0.123*T9_13 + 1.09*T9_23 + 0.938*T9), #3H11 -> He^3_2 + e+
                          5.61e6*T9__56*T9_32_*np.exp(-12.826*T9__13_), #He32 + He42 -> Be74
                          3.11e5*T9_23_*np.exp(-10.262*T9_13_) + 2.53e3*T9_32_*np.exp(-7.306/T9)]), #Be74 + H11 -> 2He42
        #These are all dependent on how fast N14 + H1 is (its a bottleneck)
        "CNO" : np.array([4.90e7*T9_23_*np.exp(-15.288*T9_13_ - 0.092*T9*T9)*(1 + 0.027*T9_13 - 0.788*T9_23 - 0.149*T9 + 0.261*T9_43 + 0.127*T9_53) + 2.37e3*T9_32_*np.exp(-3.011/T9) + 2.19e4*np.exp(-12.53/T9)])
    }
    # next we divide all by Avogadro`s constant
    # we also implement the upper limit for electron capture for the [Be74 + e-] reaction
    for key in reaction_rates:
        for i, r in enumerate(reaction_rates[key]):
            if (key == "PP2") and i == 2:
                reaction_rates[key][i] = np.where((T< 1e6) & (r >= 1.57e-7/n_e), 1.57e-7/n_e, r)
            reaction_rates[key][i] = r/N_A
    return reaction_rates

def MB(E, T):
    return np.exp(-E/k/T)

def sigma(E, mi, mk, zi, zk):
    m = (mi + mk)*mu
    return np.exp(-np.sqrt(m/2/E)*zi*zk*e*e*np.pi/h/epsilon_0)