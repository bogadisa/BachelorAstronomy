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

h = 100
w = 150
x1, y1 = 525, 325

wav_idx = 4 #or any other index in the range, here between 0 and 7
intensity_data = idata[:,:,wav_idx]

fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(intensity_data)
rect = Rectangle((x1, y1), w, h, linewidth=1, edgecolor="r", facecolor="none")
ax.add_patch(rect)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Intensity")
ax.set_title("Granulation pattern marking A, B, C and D")
ax.set_xlabel("x [idx]")
ax.set_ylabel("y [idx]")
fig.tight_layout()
markers = ["$A$", "$B$", "$C$", "$D$"]
for i, [x, y] in enumerate(ABCD):
    ax.scatter(x, y, s=60, marker=markers[i], c="red")
plt.show()




# ABCDi = np.copy(ABCD)
# scalars_xy = np.shape(intensity_data_subfield)
# ABCDi[:, 0] = ABCDi[:, 0] / scalars_xy[0]
# ABCDi[:, 1] = ABCDi[:, 1] / scalars_xy[1]
x1, y1 = rect.get_xy()
x2 = x1 + rect.get_width()
y2 = y1 + rect.get_height()
slice_x = slice(x1,x2)
slice_y = slice(y1,y2)
idata_cut = idata[slice_y, slice_x, :]
intensity_data_subfield = idata_cut[:,:,wav_idx]

fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(intensity_data_subfield, extent=(x1, x2, y1, y2))
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Intensity")
ax.set_title("Granulation pattern of subregion")
ax.set_xlabel("x [idx]")
ax.set_ylabel("y [idx]")
fig.tight_layout()
plt.show()
