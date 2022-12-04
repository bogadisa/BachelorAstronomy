from functions import *
from wavelength_conversion import get_shifted_wavelengths

OIII_file = "O3.fits"

hdu_OIII = fits.open(OIII_file)
hdu_OIII.info()

data_OIII = hdu_OIII[0].data
hdr_OIII = hdu_OIII[0].header

Halpha_file = "Halpha.fits"

hdu_Halpha = fits.open(Halpha_file)
hdu_Halpha.info()

data_Halpha = hdu_Halpha[0].data
hdr_Halpha = hdu_Halpha[0].header

# plot_flux(data_OIII, hdr=hdr_OIII)
# plot_flux(data_Halpha)#, hdr=hdr_Halpha)

print(np.std(data_Halpha))
print(np.std(data_OIII))

filename = "ADP.2017-03-27T12_08_50.541.fits"

hdu = fits.open(filename)
hdu.info()

data = hdu[1].data
hdr = hdu[1].header

lambda0 = hdr["CRVAL3"]
dlambda = hdr["CD3_3"]
len_wave = hdr["NAXIS3"]
wavelengths = np.linspace(lambda0, lambda0 + (len_wave-1)*dlambda, len_wave)

boundaries = [6850, 7100]
lower_boundary, upper_boundary = boundaries
lower_indx = np.array(np.where(wavelengths>=lower_boundary))[0, 0]
upper_indx = np.array(np.where(wavelengths<=upper_boundary))[-1, -1]

extracted_data, extracted_wavelengths = extract_spectral_data(lower_indx, upper_indx, data, wavelengths)

mean = np.nanmean(extracted_data, axis=0)

plot_std_flux(data_Halpha, hdr_Halpha, use_mean=True, mean=mean)
plot_std_flux(data_OIII, hdr_OIII, use_mean=True, mean=mean)