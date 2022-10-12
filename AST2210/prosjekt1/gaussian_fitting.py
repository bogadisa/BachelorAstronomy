import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "viridis"
   

idata = np.load("idata_square.npy")
spect_pos = np.load("spect_pos.npy")

points = ["A", "B", "C", "D"]
A = np.array([49, 197]) ; 
B = np.array([238, 443]) ; 
C = np.array([397, 213]) ; 
D = np.array([466, 52])

ABCD = np.array([A, B, C, D])
xi = np.arange(0, 8)

wavelength_spectrum = np.zeros((4, len(idata[0, 0, :])))
wavelength_spectrum_avg = np.zeros(8)

for i, [x, y] in enumerate(ABCD):
    wavelength_spectrum[i] = idata[y-1,x-1,:]
    
for i in range(8):
    wavelength_spectrum_avg[i] = np.mean(idata[:, :, i])

wavelength_func = interp1d(xi, spect_pos)
# wavelength_func = spect_pos


#choosing out the point of the peak for each of the points
Ai_b = 1
Bi_b = 4
Ci_b = 4
Di_b = 2
i_b = [Ai_b, Bi_b, Ci_b, Di_b]

#choosing points that are on the baseline for each of the points
Ai_d = [5, 6, 7]
Bi_d = [0, 7]
Ci_d = [0, 7]
Di_d = [5, 6, 7]

i_d = [Ai_d, Bi_d, Ci_d, Di_d]

def find_mean(data, points):
    n = len(points)
    s = 0
    for i in points:
        s += data[i]
    
    return s/n

#finding the constant term
d = np.array([find_mean(wavelength_spectrum[i], i_d[i]) for i in range(len(i_d))])

#finding the peak value and mean of the gaussian
peak = np.array([wavelength_spectrum[i][i_b[i]] for i in range(len(i_b))])

b = np.array([spect_pos[i_b[i]] for i in range(len(i_b))])

#fidning the amplitude of the gaussian
a = peak - d

#finding the standard deviation by using FWHM

def find_stddiv(x, xmean):
    N = len(x)
    return np.array([np.sqrt(np.sum((x - xmean[i])**2)/N) for i in range(len(xmean))])

xii = np.linspace(0, 7, 100)
c = np.array([np.std(wavelength_func(xii) - b[i]) for i in range(len(ABCD))])
#c = find_stddiv(wavelength_func(xi), b)



def g(x, a, b, c, d):
    return a*np.exp(-(x - b)**2/(2*c**2)) + d

lmbda = np.linspace(spect_pos[0], spect_pos[7], 100)
fig, ax = plt.subplots(2, 2)
fig.tight_layout()
fig.suptitle("Spectral analysis")
for i, [x, y] in enumerate(ABCD):
    wavelength_spectrum[i] = idata[y-1,x-1,:]
    popt, pcov = curve_fit(g, spect_pos, wavelength_spectrum[i], [a[i], b[i], c[i], d[i]])
    ax.flatten()[i].plot(spect_pos, wavelength_spectrum[i], ls="--", lw=1, marker="x", label="Spectra at point")
    ax.flatten()[i].plot(lmbda, g(lmbda, *popt), ls="-", lw=1, label="Model at point")
    ax.flatten()[i].set_xlabel(r"Wavelength $\lambda$ [Å]")
    ax.flatten()[i].set_ylabel("Intensity")
    ax.flatten()[i].set_title(f"{points[i]}: (x, y) = ({x}, {y})")

ax.flatten()[i].legend()
plt.show()

_i_b = 3
_i_d = [0, 7]

d_ = find_mean(wavelength_spectrum_avg, _i_d)
peak_ = wavelength_spectrum_avg[_i_b]
b_ = wavelength_func(_i_b)
a_ = peak_ - d_
c_ = np.std(wavelength_func(xii) - b_)

global_values = [a_, b_, c_, d_]
popt, pcov = curve_fit(g, spect_pos, wavelength_spectrum_avg, global_values)
plt.plot(spect_pos, wavelength_spectrum_avg, ls="--", lw=1, marker="x", label="Global average spectra")
plt.plot(lmbda, g(lmbda, *popt), ls="-", lw=1, label="Model for global average")
plt.xlabel(r"Wavelength $\lambda$ [Å]")
plt.ylabel("Intensity")
plt.legend()
plt.show()

plt.scatter(np.arange(0, len(spect_pos)), spect_pos, label=r"Wavelength $\lambda_i$")
plt.xlabel("i [idx]")
plt.ylabel(r"$\lambda$ [Å]")
plt.legend()
plt.show()