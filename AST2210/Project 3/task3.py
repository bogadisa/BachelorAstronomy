from functions import *

filename = "ADP.2017-03-27T12_08_50.541.fits"

hdu = fits.open(filename)
hdu.info()

data = hdu[1].data
hdr = hdu[1].header


print(data.shape)
#[Spectral, spatial1, spatial2]

flux_mean = np.nanmean(data, 0)
# print(flux_mean.shape)
# plot_flux(flux_mean)#, hdr=hdr)


lambda0 = hdr["CRVAL3"]
dlambda = hdr["CD3_3"]
len_wave = hdr["NAXIS3"]
wavelengths = np.linspace(lambda0, lambda0 + (len_wave-1)*dlambda, len_wave)

boundaries = [6850, 7100]
lower_boundary, upper_boundary = boundaries
lower_indx = np.array(np.where(wavelengths>=lower_boundary))[0, 0]
upper_indx = np.array(np.where(wavelengths<=upper_boundary))[-1, -1]

extracted_data, extracted_wavelengths = extract_spectral_data(lower_indx, upper_indx, data, wavelengths)

flux_extracted = np.nansum(extracted_data, 0)
plot_continuum(flux_extracted, boundaries=boundaries, hdr=hdr)


#spots of interest

    # aperture_data.append(collapsed[indx[0]:indx[1], indx[2]:indx[3]])
    # aperture_data_mean.append(np.mean(aperture_data[-1], keepdims=True))
    # aperture_data_rms.append(np.sum(aperture_data - aperture_data_mean[-1]))
    
# aperture_data = np.array(aperture_data)
# aperture_data_mean = np.array(aperture_data_mean)
# aperture_data_rms = np.array(aperture_data_rms)
# # plot_flux(flux_mean_extracted, boundaries=boundaries, hdr=hdr, markers=[markers, spots])
# # rms_noise = np.sqrt(np.sum((aperture_data - aperture_data_mean)**2, axis=1))
# # aperture_flux_mean = np.mean(aperture_data)
# # aperture_flux_sum = np.sum(aperture_data)
# # aperture_std_dev = np.std(aperture_data)


# for i, [marker, [x, y]] in enumerate(zip(markers, spots)):
#     plt.plot(aperture_data[i])