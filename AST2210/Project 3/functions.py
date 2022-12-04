from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import matplotlib.colors as colors

def plot_flux(flux_mean, hdr=False, boundaries=[4750, 9351], markers=None, show=True):
    fig = plt.figure(figsize=(10,10))
    if hdr:
        wcs = WCS(hdr)[0,:,:] # indexing[spectral, vertical, horizontal]
        ax = plt.subplot(1,1,1, projection=wcs)
        if markers != None:
            markers, spots = markers
            for marker, [x, y] in zip(markers, spots):
                ax.scatter(x, y, s=60, marker=marker, c="black")
        im = ax.imshow(flux_mean, cmap="Spectral", vmin=0, vmax=2137)
    else:
        if markers != None:
            markers, spots = markers
            for marker, [x, y] in zip(markers, spots):
                plt.scatter(x, 317-y, s=60, marker=marker, c="black")
        im = plt.imshow(np.flip(flux_mean, 0), cmap="Spectral", vmin=0, vmax=2137)
        plt.xlabel("px [0.2 arcsec]")
        plt.ylabel("px [0.2 arcsec]")

        
    plt.title(rf"Average flux density in the range $\lambda \in {boundaries}\AA$")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Flux density [$10^{-20}$ergs $s^{-1}cm^{-2}\AA{}^{-1}$]")
    if show:
        plt.show()
    else:
        return im

def plot_continuum(flux_mean, hdr=False, boundaries=[4750, 9351], markers=None, show=True):
    fig = plt.figure(figsize=(10,10))
    if hdr:
        wcs = WCS(hdr)[0,:,:] # indexing[spectral, vertical, horizontal]
        ax = plt.subplot(1,1,1, projection=wcs)
        if markers != None:
            markers, spots = markers
            for marker, [x, y] in zip(markers, spots):
                ax.scatter(x, y, s=60, marker=marker, c="black")
        im = ax.imshow(flux_mean, cmap="Spectral", norm=colors.LogNorm())
    else:
        im = plt.imshow(np.flip(flux_mean, 0), cmap="Spectral", norm=colors.LogNorm())

        
    plt.title(rf"Continuum map derived from flux densities in the range $\lambda \in {boundaries}\AA$")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Integrated Flux [$10^{-20}$ergs $s^{-1}cm^{-2}$]")
    if show:
        plt.show()
    else:
        return im

#choose values for the emission lines you wish to study
def extract_spectral_data(lower_indx, upper_indx, data, wavelengths):
    extracted_data = data[lower_indx:upper_indx]
    extracted_wavelengths = wavelengths[lower_indx:upper_indx]
    return extracted_data, extracted_wavelengths

def write_file(new_filename, lower_wavelength, new_data, hdr):
    hdr["CRVAL3"] = lower_wavelength
    hdu = fits.PrimaryHDU(new_data, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.info()
    hdul.writeto(new_filename+".fits")

def aperture(r, center_x, center_y):
    indx = [center_y-r, center_y+r, center_x-r, center_x+r]
    return indx


def plot_spectra(flux_data, wavelength_data, aperture=None, method=None, show=False, marker=None):
    if aperture != None:
        indx = aperture
        flux_data = np.copy(flux_data[:, indx[0]:indx[1], indx[2]:indx[3]])

    if method != None:
        flux_integrated = method(flux_data, axis=0)
    else:
        print(np.sum(flux_data, axis=1).shape)
        flux_integrated = np.sum(np.sum(flux_data, axis=1), axis=1)

    plt.plot(wavelength_data, flux_integrated)
    plt.xlabel(r"Wavelength [$\AA$]")
    plt.ylabel(r"Flux density [$10^{-20}$ergs $s^{-1}cm^{-2}\AA{}^{-1}$]")
    if marker != None:
        plt.title(f"Spectra at location {marker}")

    if show:
        plt.show()

def plot_std_flux(flux_data, hdr=None, use_mean=False, mean=0):
    fig = plt.figure(figsize=(10, 10))
    if not(use_mean):
        sigma = np.std(flux_data)
    else:
        sigma = np.sqrt((np.nanmean(flux_data - mean))**2)


    if hdr != None:
        wcs = WCS(hdr)[0,:,:]
        ax = plt.subplot(1,1,1, projection=wcs)
        ax.contour(flux_data, levels=[3*sigma, 10*sigma, 20*sigma], cmap="copper")
        im = ax.imshow(flux_data, cmap="gist_rainbow", norm=colors.LogNorm(vmin=1))
    else:
        ax = plt.subplot(1,1,1)
        ax.contour(np.flip(flux_data, 0), levels=[3*sigma, 10*sigma, 20*sigma])
        ax.set_xlabel("px [0.2 arcsec]")
        ax.set_ylabel("px [0.2 arcsec]")
        im = ax.imshow(np.flip(flux_data, 0), cmap="gist_rainbow", norm=colors.LogNorm(vmin=1))# vmin=0, vmax=10*sigma)

    print(sigma)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Integrated Flux [$10^{-20}$ergs $s^{-1}cm^{-2}$]")
    plt.show()