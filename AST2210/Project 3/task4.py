from functions import *
from wavelength_conversion import get_shifted_wavelengths

filename = "ADP.2017-03-27T12_08_50.541.fits"

hdu = fits.open(filename)
hdu.info()

data = hdu[1].data
hdr = hdu[1].header


print(data.shape)

lambda0 = hdr["CRVAL3"]
dlambda = hdr["CD3_3"]
len_wave = hdr["NAXIS3"]
wavelengths = np.linspace(lambda0, lambda0 + (len_wave-1)*dlambda, len_wave)


spots = np.array([0, 0]) + np.array([1, 1])*np.array([[82, 239], [184, 188], [162, 154]])
markers = ["$A$", "$B$", "$C$", "$D$"]
emission_lines, line_names = get_shifted_wavelengths()

# plot_flux(np.nanmean(data, axis=0), markers=[markers, spots])
line_colors = ["red","green", "violet", "blue", "orange", "purple", "yellow", "black", "grey", "brown"]
for marker, [x, y] in zip(markers, spots):
    indx = aperture(5, x, y)
    
    for j, line in enumerate(emission_lines):
        for i in range(len(line)):
            if not(line[i] < wavelengths[0] or wavelengths[-1] < line[i]):
                plt.axvline(x=line[i], c=line_colors[j], alpha=0.4, label=line_names[j])

    plot_spectra(data, wavelengths, indx, marker=marker)
    plt.legend()
    plt.show()