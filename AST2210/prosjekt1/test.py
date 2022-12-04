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
A = np.array([49, 197]) ; B = np.array([238, 443]) ; C = np.array([397, 213]) ; D = np.array([466, 52])
ABCD = np.array([A, B, C, D])
xi = np.arange(0, 8)

wavelength_spectrum = np.zeros((4, len(idata[0, 0, :])))
wavelength_spectrum_avg = np.zeros(8)
for i, [x, y] in enumerate(ABCD):
    wavelength_spectrum[i] = idata[y-1,x-1,:]
for i in range(8):
    wavelength_spectrum_avg[i] = np.mean(idata[:, :, i])

wavelength_func = interp1d(xi, spect_pos)

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

b = np.array([wavelength_func(i_b[i]) for i in range(len(i_b))])

#fidning the amplitude of the gaussian
a = peak - d

#finding the standard deviation by using FWHM

def find_stddiv(x, xmean):
    N = len(x)
    return np.array([np.sqrt(np.sum((x - xmean[i])**2)/N) for i in range(len(xmean))])
    #return np.sqrt(sum((x - xmean)**2)/len(x))
#c = abs(a)/(np.sqrt(8*np.log(2)))
#c = 1/(np.sqrt(2*np.pi)*a)
#c = [0.06, 0.06, 0.06, 0.06]
#c = find_stddiv(wavelength_func(xi), b)
# #c = np.std(wavelength_spectrum) #
# c = np.copy(b)
# for i in range(4):
#     c[i] = find_stddiv(wavelength_spectrum[i], d[i])
#c = np.std(wavelength_spectrum-d.reshape(-1, 1), axis=0)

#c = np.array([np.std(xi - i_b[i]) for i in range(len(ABCD))])
xii = np.linspace(0, 7, 100)
c = np.array([np.std(wavelength_func(xii) - b[i]) for i in range(len(ABCD))])
#c = find_stddiv(wavelength_func(xi), b)

#print(wavelength_spectrum-d.reshape(-1, 1))
def g(x, a, b, c, d):
    #a, b, c, d = P
    return a*np.exp(-(x - b)**2/(2*c**2)) + d

lmbda = np.linspace(wavelength_func(0), wavelength_func(7), 100)

fig, ax = plt.subplots(2, 2)
fig.tight_layout()
fig.suptitle("Spectral analysis")
for i, [x, y] in enumerate(ABCD):
    wavelength_spectrum[i] = idata[y-1,x-1,:]
    popt, pcov = curve_fit(g, spect_pos, wavelength_spectrum[i], [a[i], b[i], c[i], d[i]])
    #ax.flatten()[i].plot(wavelength_spectrum_avg, ls="--", lw=1, marker="x", label="Average spectra")
    ax.flatten()[i].plot(wavelength_func(xi), wavelength_spectrum[i], ls="--", lw=1, marker="x", label="Spectra at point")
    ax.flatten()[i].plot(lmbda, g(lmbda, *popt), ls="-", lw=1, label="Model at point")
    ax.flatten()[i].set_xlabel(r"Wavelength $\lambda_i$")
    ax.flatten()[i].set_ylabel("Intensity")
    ax.flatten()[i].set_title(f"{points[i]}: (x, y) = ({x}, {y})")

ax.flatten()[i].legend()
plt.show()