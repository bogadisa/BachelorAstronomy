import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.signal import argrelextrema

palette = iter(['#9b59b6', '#4c72b0', '#55a868', '#c44e52', '#dbc256'])

# Critical pressure, volume and temperature
# These values are for the van der Waals equation of state for CO2:
# (p - a/V^2)(V-b) = RT. Units: p is in Pa, Vc in m3/mol and T in K.
pc = 7.404e6
Vc = 1.28e-4
Tc = 304

def vdw(Tr, Vr):
    """Van der Waals equation of state.

    Return the reduced pressure from the reduced temperature and volume.

    """

    pr = 8*Tr/(3*Vr-1) - 3/Vr**2
    return pr


def vdw_maxwell(Tr, Vr):
    """Van der Waals equation of state with Maxwell construction.

    Return the reduced pressure from the reduced temperature and volume,
    applying the Maxwell construction correction to the unphysical region
    if necessary.

    """

    pr = vdw(Tr, Vr)
    if Tr >= 1:
        # No unphysical region above the critical temperature.
        return pr

    if min(pr) < 0:
         raise ValueError('Negative pressure results from van der Waals'
                         ' equation of state with Tr = {} K.'.format(Tr))

    # Initial guess for the position of the Maxwell construction line:
    # the volume corresponding to the mean pressure between the minimum and
    # maximum in reduced pressure, pr.
    iprmin = argrelextrema(pr, np.less)
    iprmax = argrelextrema(pr, np.greater)
    Vr0 = np.mean([Vr[iprmin], Vr[iprmax]])

    def get_Vlims(pr0):
        """Solve the inverted van der Waals equation for reduced volume.

        Return the lowest and highest reduced volumes such that the reduced
        pressure is pr0. It only makes sense to call this function for
        T<Tc, ie below the critical temperature where there are three roots.

        """

        eos = np.poly1d( (3*pr0, -(pr0+8*Tr), 9, -3) )
        roots = eos.r
        roots.sort()
        Vrmin, _, Vrmax = roots
        return Vrmin, Vrmax

    def get_area_difference(Vr0):
        """Return the difference in areas of the van der Waals loops.

        Return the difference between the areas of the loops from Vr0 to Vrmax
        and from Vrmin to Vr0 where the reduced pressure from the van der Waals
        equation is the same at Vrmin, Vr0 and Vrmax. This difference is zero
        when the straight line joining Vrmin and Vrmax at pr0 is the Maxwell
        construction.

        """

        pr0 = vdw(Tr, Vr0)
        Vrmin, Vrmax = get_Vlims(pr0)
        return quad(lambda vr: vdw(Tr, vr) - pr0, Vrmin, Vrmax)[0]

    # Root finding by Newton's method determines Vr0 corresponding to
    # equal loop areas for the Maxwell construction.
    Vr0 = newton(get_area_difference, Vr0)
    pr0 = vdw(Tr, Vr0)
    Vrmin, Vrmax = get_Vlims(pr0)

    # Set the pressure in the Maxwell construction region to constant pr0.
    pr[(Vr >= Vrmin) & (Vr <= Vrmax)] = pr0
    return pr0

n = 500
Vr = np.linspace(0.5, 3, n)

# def plot_pV(T):
#     Tr = T / Tc
#     c = next(palette)
#     #ax.plot(Vr, vdw(Tr, Vr), lw=2, alpha=0.3, color=c)
#     ax.plot(Vr, vdw_maxwell(Tr, Vr), lw=2, color=c, label='{:.2f}'.format(Tr))



# fig, ax = plt.subplots()

Tlist = np.linspace(0.85, 0.99999, n)
pr0 = np.zeros(n)
for i, Tr in enumerate(Tlist):
    pr0[i] = vdw_maxwell(Tr, Vr)

#zeros = np.zeros((n,)) + pr0
plt.plot(Tlist, pr0)
plt.xlabel(r"$T_b [*]$")
plt.ylabel(r"$P_b(T_B) [*]$")
plt.show()
# for T in range(270, 320, 10):
#     plot_pV(T)
dPdT = (pr0[-1]-pr0[1])/(Tlist[-1]-Tlist[0])
print(f"dP/dT = {dPdT}")
H =dPdT*(Vr[-1]-Vr[0])/(Tlist[-1]-Tlist[0])
print(f"H = {H}")

R = 8.31446
plt.plot(Tlist, H/Tlist)
plt.xlabel(r"$T_b [*]$")
plt.ylabel(r"$H_v/T_b [*]$")
plt.show()

Tc = 647.096
Pc = 22.064

Tlist_new = np.linspace(0, 1, n) * Tc
pr0_new = dPdT*Tlist_new

Tw = 373
Pw = 1
T_hat = Tw/Tc
P_hat = Pw/Pc
print(f"T_hat={T_hat}")
print(f"P_hat={P_hat}")


print(H/R/T_hat)

plt.plot(Tlist_new, pr0_new)
plt.xlabel(r"$T_b [K]$")
plt.ylabel(r"$P_b(T_B) [MPa]$")
plt.show()


# ax.set_xlim(0.4, 3)
# ax.set_xlabel('Reduced volume')
# ax.set_ylim(0, 1.6)
# ax.set_ylabel('Reduced pressure')
# ax.legend(title='Reduced temperature')

# plt.show()