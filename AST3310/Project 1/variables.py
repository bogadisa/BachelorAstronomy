import numpy as np
from functions import cm_volume_to_m, lamda_to_r, number_density, energy_out, MeV_to_J

#given in units of atomic mass m_u
e =  5.4858e-4
proton = 1.0078 
He_32  = 3.0160
He_42  = 4.0026
Li_73  = 7.016004
Be_74  = 7.0169287
Be_84  = 8.0053
B_85   = 8.0246073
C_126  = 12.0110
C_136  = 13.0034
N_137  = 13.0057
N_147  = 14.0031
N_157  = 15.0001
O_158  = 15.0031

#given in units of MeV
neutrinos = {
    "PP0": 0.265,
    "PP1": 0,
    "PP2": 0.813,
    "PP3": 6.711,
    "CNO": [0.707, 0.997]
}

#used to calculate energy release as a function of mass loss, keep track of neutrinos and how many times a reaction is repeated
# key : [[m_reactant1, m_reactant2, neutrino energy, positrons annihilated], times reapeted for next step, [...
branches_steps = {
    "PP0" : [[3*proton, He_32, neutrinos["PP0"], 1], 1],
    "PP1" : [[3*proton, He_32, neutrinos["PP0"], 1], 2, 
             [2*He_32, He_42 + 2*proton, neutrinos["PP1"], 0], 1],
    "PP2" : [[3*proton, He_32, neutrinos["PP0"], 1], 1,
             [He_32 + He_42, Be_74, 0, 0], 1,
             [Be_74, Li_73, neutrinos["PP2"], 1], 1,
             [Li_73 + proton, 2*He_42, 0, 0], 1],
    "PP3" : [[3*proton, He_32, neutrinos["PP0"], 1], 1,
             [He_32 + He_42, Be_74, 0, 0], 1,
             [Be_74 + proton, B_85, neutrinos["PP3"], 0], 1],
    "CNO" : [[4*proton, He_42, sum(neutrinos["CNO"]), 0], 1] #N_147 + proton -> O_158
}

branches_steps_energy_calc = {
    "PP0" : [[3*proton, He_32, neutrinos["PP0"], 1], 1],
    "PP1" : [[3*proton, He_32, neutrinos["PP0"], 1], 2, 
             [2*He_32, He_42 + 2*proton, neutrinos["PP1"], 0], 1],
    "PP2" : [[3*proton, He_32, neutrinos["PP0"], 1], 1,
             [He_32 + He_42, Be_74, 0, 0], 1,
             [Be_74, Li_73, neutrinos["PP2"], 1], 1,
             [Li_73 + proton, 2*He_42, 0, 0], 1],
    "PP3" : [[3*proton, He_32, neutrinos["PP0"], 1], 1,
             [He_32 + He_42, Be_74, 0, 0], 1,
             [Be_74 + proton, B_85, 0, 0], 1,
             [B_85, Be_84, neutrinos["PP3"], 1], 1,
             [Be_84, 2*He_42, 0, 0], 1],
    "CNO" : [[C_126 + proton, N_137, 0, 0], 1,
             [N_137, C_136, neutrinos["CNO"][0], 1], 1,
             [C_136 + proton, N_147, 0, 0], 1,
             [N_147 + proton, O_158, 0, 0], 1,
             [O_158, N_157, neutrinos["CNO"][1], 1], 1,
             [N_157 + proton, C_126 + He_42, 0, 0], 1]
}

#different densities taken from the assignment
X = 0.7
Y_3 = 1e-10
Y_4 = 0.29
Z_Li = 1e-7
Z_Be = 1e-7
Z_N = 1e-11


#sun parameters
rho = 1.62e5 #*6.022e26 #u m^-3

# T = 1e8

u = 931.494 #MeV c^-2
N_A = 6.0221408e+23 

#As taken from the lecture notes
n_e = 0.5*rho/u*(1 + X)

# [fraction, mass, atomic number]
fractions = {
    "proton" : [X, proton, 1],
    "He32" : [Y_3, He_32, 2],
    "He42" : [Y_4, He_42, 2],
    "Li73" : [Z_Li, Li_73, 3],
    "Be74" : [Z_Be, Be_74, 4],
    "N147" : [Z_N, N_147, 7],
    #electrons` number density is calculated independently of fraction
    #electrons` atomic number is 1
    "e" : [0, e, 1] 
}



# key : [[reactant1, reactant 2, if dependent on other reaction then name of dependent reaction else false]
reactants = {
    "PP0" : [["proton", "proton", False]],
    "PP1" : [["proton", "proton", False],
             ["He32", "He32", "PP0"]],
    "PP2" : [["proton", "proton", False],
             ["He32", "He42", "PP0"],
             ["Be74", "e", "PP23"],
             ["Li73", "proton", False]],
    "PP3" : [["proton", "proton", False],
             ["He32", "He42", "PP0"],
             ["Be74", "proton", "PP23"]],
    "CNO" : [["N147", "proton", False]]
}

#energies taken from lecture notes, used to debug
Q = {
        "PP0" : [1.177 + 0.265 + 5.494],
        "PP1" : [1.177 + 0.265 + 5.494,
                 12.860],
        "PP2" : [1.177 + 0.265 + 5.494,
                 1.586,
                 0.049 + 0.813,
                 17.346],
        "PP3" : [1.177 + 0.265 + 5.494,
                 1.586,
                 0.137 + 8.367 + 6.711 + 2.995],
        "CNO" : [1.944 + 1.513 + 0.707 + 7.551 + 7.297 + 1.757 + 0.997 + 4.966]
    }