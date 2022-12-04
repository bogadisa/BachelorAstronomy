import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "viridis"
   

idata = np.load("idata_square.npy")
spect_pos = np.load("spect_pos.npy")


points = ["A", "B", "C", "D"]
A = np.array([49, 197]) ; B = np.array([238, 443]) ; C = np.array([397, 213]) ; D = np.array([466, 52])
ABCD = np.array([A, B, C, D])


wavelength_spectrum = np.zeros((4, len(idata[0, 0, :])))
wavelength_spectrum_avg = np.zeros(8)
for i in range(8):
    wavelength_spectrum_avg[i] = np.mean(idata[:, :, i])



fig, ax = plt.subplots(2, 2)
fig.tight_layout()
fig.suptitle("Spectral analysis")
for i, [x, y] in enumerate(ABCD):
    wavelength_spectrum[i] = idata[y-1,x-1,:]

    ax.flatten()[i].plot(wavelength_spectrum_avg, ls="--", lw=1, marker="x", label="Average spectra")
    ax.flatten()[i].plot(wavelength_spectrum[i], ls="--", lw=1, marker="x", label="Spectra at point")
    ax.flatten()[i].set_xlabel(r"Wavelength $\lambda_i$ [idx]")
    ax.flatten()[i].set_ylabel("Intensity [*]")
    ax.flatten()[i].set_title(f"{points[i]}: (x, y) = ({x}, {y})")

ax.flatten()[i].legend()
plt.show()

plt.plot(wavelength_spectrum_avg, ls="--", lw=1, marker="x", label="Average spectra")
plt.plot(wavelength_spectrum[1], ls="--", lw=1, marker="x", label=f"[x, y] = {ABCD[1]}")
plt.plot(wavelength_spectrum[2], ls="--", lw=1, marker="x", label=f"[x, y] = {ABCD[2]}")
plt.xlabel(r"Wavelength $\lambda_i$ [idx]")
plt.ylabel("Intensity [*]")
plt.legend()
plt.show()
plt.plot(wavelength_spectrum_avg, ls="--", lw=1, marker="x", label="Average spectra")
plt.plot(wavelength_spectrum[0], ls="--", lw=1, marker="x", label=f"[x, y] = {ABCD[0]}")
plt.plot(wavelength_spectrum[3], ls="--", lw=1, marker="x", label=f"[x, y] = {ABCD[3]}")
plt.xlabel(r"Wavelength $\lambda_i$ [idx]")
plt.ylabel("Intensity [*]")
plt.legend()
plt.show()