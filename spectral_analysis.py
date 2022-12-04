import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

idata = np.loadtxt("IRAS13120_spectrum_1.txt")

v_data = idata[:, 0]
TA_data = idata[:, 1]

# From CLASS:
# b = 81.644
# FWHM = 323.103
# a = 6.18183e-2
b =  81.677 ; b_std = 2.221
FWHM = 323.272 ; FWHM_std = 4.488
a = 6.8128e-2 ; a_std = 0

#FWHM rule:
c = FWHM/(2*np.sqrt(2*np.log(2))) ; c_std = c/FWHM*FWHM_std
print(c, c_std)
def gaussian(x, a, b, c):
    return a*np.exp(-(x - b)**2/(2*c**2))

#num data points
n = 10000

vmin, vmax = np.min(v_data), np.max(v_data)
# vmin, vmax = -700, 1000
v = np.linspace(vmin, vmax, n)

TA_gauss = gaussian(v, a, b, c)

plt.plot(v_data, TA_data, label="Extracted data")
#plt.plot(v, TA_gauss, label="Gaussian fit")
plt.xlabel("v [km/s]")
plt.ylabel(r"$T_A [K]$")
plt.legend()
plt.show()

#conversion to flux
#assumes a point source
#Estimates around 230Ghz, which is what our measurements are centered around
#Gamma^-1 ~ 35-40Jy/K
gamma_inv = 37.5 ; gamma_inv_pm = 2.5

#from lecture:
# gamma^-1 = 24.415/nu_A

Sv_data = gamma_inv*TA_data
Sv_gauss = gamma_inv*TA_gauss
print(a*gamma_inv, a_std*gamma_inv)

Sv_data_pm = gamma_inv_pm*TA_data
Sv_data_upper = Sv_data + Sv_data_pm
Sv_data_lower = Sv_data - Sv_data_pm

Sv_gauss_std = gaussian(v, a_std, b_std, c_std)*gamma_inv

Sv_gauss_pm = gamma_inv_pm*TA_gauss
Sv_gauss_upper = Sv_gauss + Sv_gauss_std
Sv_gauss_lower = Sv_gauss - Sv_gauss_std

Sv_gauss_upper = gaussian(v, a+a_std, b+b_std, c+c_std)*gamma_inv
Sv_gauss_lower = gaussian(v, a-a_std, b-b_std, c-c_std)*gamma_inv

def std_e(f, b, std_A):
    return (f*b*std_A)

def std_AB(f, A, B, std_A, std_B):
    return f*np.sqrt((std_A/A)**2+(std_B/B)**2)

def std(x, a, b, c, b_std, c_std):
    term1 = (a-4)/(2*c**2)**2*(x-b)*c**2*np.exp((x-b)**2/(2*c**2))*b_std
    term2 = -(4/(2*c**2)**2*a*(x-b)**2*c*np.exp((x-b)**2/(2*c**2)))*c_std
    return np.sqrt(term1**2+term2**2)

bc_std = std_AB((v-b)**2/(2*c**2), b, c, b_std, c_std)
abc_std = std_e(gaussian(v, a, b, c), 1, bc_std)

Sv_gauss_std = std(v, a*gamma_inv, b, c, b_std, c_std)
Sv_gauss_upper = Sv_gauss + Sv_gauss_std
Sv_gauss_lower = Sv_gauss - Sv_gauss_std

Sv_gauss_lower_cut = np.where(v<FWHM/2+b, Sv_gauss_lower, 0)
Sv_gauss_lower_cut = np.where(-FWHM/2-b<v, Sv_gauss_lower_cut, 0)
Sv_gauss_upper_cut = np.where(v<FWHM/2+b, Sv_gauss_upper, 0)
Sv_gauss_upper_cut = np.where(-FWHM/2-b<v, Sv_gauss_upper_cut, 0)

Sv_gauss_pm = gamma_inv_pm*TA_gauss
Sv_gauss_upper = Sv_gauss + Sv_gauss_pm
Sv_gauss_lower = Sv_gauss - Sv_gauss_pm

plt.plot(v_data, Sv_data_upper, label="Upper limit extracted data", alpha=.3)
plt.plot(v_data, Sv_data_lower, label="Lower limit extracted data", alpha=.3)
plt.plot(v_data, Sv_data, label="Extracted data")
plt.fill_between(v_data, Sv_data_lower, Sv_data_upper, alpha=.3)

# plt.plot(v, Sv_gauss_upper, label=r"$+\sigma$ gaussian fit", alpha=.3)
# plt.plot(v, Sv_gauss_lower, label=r"$-\sigma$ gaussian fit", alpha=.3)
# plt.plot(v, Sv_gauss, label="Gaussian fit")
# plt.fill_between(v, Sv_gauss_lower, Sv_gauss_upper, alpha=.3)

# plt.plot(v, Sv_gauss_upper, label=r"Upper gaussian fit", alpha=.3)
# plt.plot(v, Sv_gauss_lower, label=r"Lower gaussian fit", alpha=.3)
# plt.plot(v, Sv_gauss, label="Gaussian fit")
# plt.fill_between(v, Sv_gauss_lower, Sv_gauss_upper, alpha=.3)

# plt.ylim(-0.5, a*gamma_inv+1)
plt.xlabel("v [km/s]")
plt.ylabel(r"$F_\nu [Jy]$")
plt.legend()
plt.show()

#from assignment

D_L = 139.4 #Mpc
z = 0.0308
H0 = 67.8 #km/s/Mpc
Omega_M = 0.307
Omega_A = 0.693
k = 0 #flat geometry

def L_CO21(DL, v, z, Sv):
    #intSv = integrate.simpson(Sv, v)
    intSv = np.trapz(Sv, v)
    print(intSv)
    return 3.25e7*DL**2/(223.500**2*(1+z)**3)*intSv

def L_CO_density(DL, v, z, Sv):
    return 3.25e7*DL**2/(223.500**2*(1+z)**3)*Sv

#from assignment
alpha_CO = 1.7 #+-0.4 Solar masses [K km/s pc^2]^-1
alpha_CO_std = 0.4

L = L_CO21(D_L, v, z, Sv_gauss)
tot_mass_gauss = alpha_CO*L
tot_mass_gauss_m = tot_mass_gauss - alpha_CO_std*L
tot_mass_gauss_p = tot_mass_gauss + alpha_CO_std*L

print(f"Sv_peak={np.max(Sv_gauss)}")
print(f"{L:g}")
print(f"The total molecular gass of Galaxy IRAS 13120-5453 is {tot_mass_gauss:.3e} +- {alpha_CO_std*L:.3e}") #and ranges from {tot_mass_gauss_m:.3e} to {tot_mass_gauss_p:.3e} solar masses according to our gaussian fit")

L = L_CO_density(D_L, v, z, Sv_gauss)
tot_mass_gauss = alpha_CO*L
tot_mass_gauss_m = tot_mass_gauss - alpha_CO_std*L
tot_mass_gauss_p = tot_mass_gauss + alpha_CO_std*L

L = L_CO_density(D_L, v_data, z, Sv_data)
tot_mass_data = alpha_CO*L
tot_mass_data_m = tot_mass_data - alpha_CO_std*L
tot_mass_data_p = tot_mass_data + alpha_CO_std*L

#print(f"The total molecular gass of Galaxy IRAS 13120-5453 is {tot_mass_gauss:.3e} solar masses according to our gaussian fit")

plt.plot(v_data-b, tot_mass_data_p, label="Upper limit mass density", alpha=.3)
plt.plot(v_data-b, tot_mass_data_m, label="Lower limit mass density", alpha=.3)
plt.plot(v_data-b, tot_mass_data, label="mass density")
plt.fill_between(v_data-b, tot_mass_data_m, tot_mass_data_p, alpha=.3)

# plt.plot(v, tot_mass_gauss_p, label="Upper limit mass density", alpha=.3)
# plt.plot(v, tot_mass_gauss_m, label="Lower limit mass density", alpha=.3)
# plt.plot(v, tot_mass_gauss, label="mass density")
# plt.fill_between(v, tot_mass_gauss_m, tot_mass_gauss_p, alpha=.3)
plt.xlabel("v [km/s]")
plt.ylabel(r"$\rho_{H_2+He}[M_\odot]$")
plt.legend()
plt.show()

#calculate the noise


"""
AS> set type line
LAS> set unit v f
LAS> set align v c
LAS> set weight s
LAS> set bad and
LAS> set match 2
LAS> set plot histo
LAS> set mode x tot
LAS> set mode y tot
LAS> pl
LAS> average /resample
Consistency checks:
  Checking Data type and regular x-axis sampling
  Checking Source Name
  Checking Position information
  Checking Offset position
  Checking Line Name
  Checking Spectroscopic information
  Checking Calibration information
  Checking Switching information
Reference spectrum:
  Source Name       : IRAS13120
  Coordinate System : EQUATORIAL  2000.0
  Proj. Center (rad): lambda     3.469304, beta    -0.962667, tolerance 4.8E-08
  Line Name         : CO(2-1)I1312
  Frequency (MHz)   : rest    2.237E+05, resol    7.630E-02
  Velocity (km/s)   : resol   -1.023E-01, offset    0.000E+00
  Alignment (chan)  : tolerance    10.0%
  Calibration       : beeff  1.000,gain  0.100
Velocity alignment, automatic resampling:
- Input axes:
    Doppler:   0.0000
    Resolution: from -.102274 to .102274 km/s
- Output axis:
    Frequency range: from 221657.850 to 225658.211 MHz
    Velocity range: from 2681.012 to -2681.094 km/s
    Nchan: 52429
    Rchan: 26214.600
    Restf: 223658.000 MHz
    Image: 235659.036 MHz
    Fres: .076301 MHz
    Doppler:   0.0000
    Voff: .000 km/s
    Vres: -.102274 km/s
LAS> sm box 20
LAS> sm box 7
LAS> pl"""