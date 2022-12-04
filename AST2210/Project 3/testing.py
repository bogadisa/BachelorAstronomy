import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
filename = "ADP.2017-03-27T12_08_50.541.fits"

from astropy.wcs import WCS
hdu=fits.open(filename) #reads the file

plt.rcParams["font.size"] = 14
hdr=hdu[1].header # extracts the FITS header which containts important information about the data contained in the files.
data=hdu[1].data#[:,5:-8,5:-8] #extracts the image data from
#extracting the spatial WCS from the header
wcs = WCS(hdr)[0,:,:] # indexing[spectral, vertical, horizontal]
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(1,1,1, projection=wcs)
ax.set_title(r"Average flux density in the range $\lambda \in [4750, 9351]\AA{}$")
im = ax.imshow(np.nanmean(data,0),cmap="gray", vmin=0,vmax=2137) # no longer needs np.flip because of WCS
plt.xlabel("RA")
plt.ylabel("Dec")
plt.colorbar(im,fraction=0.046, pad=0.04, label="Flux density [$10^{-20}$ergs$^{-1}$cm$^{-2}\AA{}^{-1}$]")
plt.show()