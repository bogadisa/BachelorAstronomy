import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.constants import c as c_vacuum

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "viridis"
   

idata = np.load("idata_square.npy")
spect_pos = np.load("spect_pos.npy")

xi = np.arange(0, 8)
xii = np.linspace(0, 7, 100)
wavelength_func = interp1d(xi, spect_pos)
lmbda = wavelength_func(xii)

FE_I = 6173 #Å

# def Å_to_m(x):
#     return x*1e-10

points = ["A", "B", "C", "D"]
A = np.array([49, 197]) ; 
B = np.array([238, 443]) ; 
C = np.array([397, 213]) ; 
D = np.array([466, 52])

ABCD = np.array([A, B, C, D])

def doppler(lmbda_obs, lmbda_EM):
    delta_lmbda = lmbda_obs - lmbda_EM

    return c_vacuum*delta_lmbda/lmbda_EM

def g(x, a, b, c, d):
    return a*np.exp(-(x - b)**2/(2*c**2)) + d

def find_parameters():
    peak = np.min(idata, axis=2)
    i_b = np.argmin(idata, axis=2)
    b = wavelength_func(i_b)
    lmbda_ = np.zeros((np.shape(b)[0], np.shape(b)[1], len(lmbda))) + lmbda
    c = np.std(lmbda_ - b.reshape(550, 750, 1), axis=2)
    d = np.max(idata, axis=2)
    a = peak - d
    return a, b, c, d


a, b, c, d = find_parameters()
popt = np.zeros((np.shape(a)[0], np.shape(a)[1], 4))
for i in range(550):
    for j in range(750):
        popt[i, j], pcov = curve_fit(g, spect_pos, idata[i, j], [a[i, j], b[i, j], c[i, j], d[i, j]])
    # print(i)

def doppler_field(lmbda_obs, lmbda_EM):
    data = doppler(lmbda_obs, lmbda_EM)
    return data

b_sun = np.mean(popt[:, :, 1])

print(f"The average spectra line of the enitre FOV is {b_sun}Å, meaning the FOV appears to move towards us at {doppler(b_sun, FE_I)}m/s")
doppler_data = doppler_field(popt[:, :, 1], FE_I) - doppler(b_sun, FE_I)

h = 100
w = 150
x1, y1 = 525, 325

#point E

E = np.array([681, 507])
ABCD = np.array([A, B, C, D, E])
points = ["A", "B", "C", "D", "E"]

fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(doppler_data)
rect = Rectangle((x1, y1), w, h, linewidth=1, edgecolor="r", facecolor="none")
ax.add_patch(rect)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Velocity")
ax.set_title("Velocity from Doppler effect marking A, B, C and D")
ax.set_xlabel("x [idx]")
ax.set_ylabel("y [idx]")
fig.tight_layout()
markers = ["$A$", "$B$", "$C$", "$D$", "$E$"]
for i, [x, y] in enumerate(ABCD):
    ax.scatter(ABCD[i, 0], ABCD[i, 1], s=60, marker=markers[i], c="red")
    print(f"Point {points[i]} has a wavelength {popt[y-1, x-1, 1]} resulting in a velocity {doppler_data[y-1, x-1]}m/s")
plt.show()

x1, y1 = rect.get_xy()
x2 = x1 + rect.get_width()
y2 = y1 + rect.get_height()
slice_x = slice(x1,x2)
slice_y = slice(y1,y2)
doppler_data_cut = doppler_data[slice_y, slice_x]
doppler_data_subfield = doppler_data_cut

fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(doppler_data_subfield, extent=(x1, x2, y1, y2))
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Velocity")
ax.set_title("Velocity from Doppler effect of subregion")
ax.set_xlabel("x [idx]")
ax.set_ylabel("y [idx]")
fig.tight_layout()
plt.show()

plt.scatter(idata[:,:,-1].flatten()[::17], doppler_data.flatten()[::17])
plt.grid(False)
plt.title(r"Scatter plot of intensity vs doppler velocity")
plt.xlabel(r"Intensity [*]")
plt.ylabel(r"Doppler velocity [m/s]")
plt.show()