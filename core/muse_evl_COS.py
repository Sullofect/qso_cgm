import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from muse_compare_z import compare_z


path_cos = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_FUV.fits')
data_cos = fits.getdata(path_cos, 1, ignore_missing_end=True)
wave = data_cos['wave']
flux = data_cos['flux']
z = 0.6282144177077355
limit = 912 * (1 + z)

#
mask = np.where((wave < limit  + 100) * (wave > limit - 100))
wave = wave[mask]
flux = flux[mask]

conti = np.median(flux)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.luminosity_distance(z).to(u.cm).value
nuL = np.log10(4 * np.pi * conti * limit * d_l ** 2)
print(nuL)

distance = 10 * u.kpc
print(distance.to(u.cm).value)

# Plot
# plt.plot(wave, flux)
# plt.xlim(limit - 100, limit + 100)
# plt.ylim(1000, 2000)
# plt.show()

