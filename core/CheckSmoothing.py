import os
import aplpy
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


# Check smoothing OII nebulae
# UseSeg = (1.5, 'gauss', 1.5, 'gauss')
UseDataSeg = (1.5, 'gauss', None, None)
path_SB_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'\
    .format('HE0435-5304', 'OII', *UseDataSeg)
path_3DSeg = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'\
    .format('HE0435-5304', 'OII', *UseDataSeg)
path_cube = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_{}_{}_{}_{}.fits'\
    .format('HE0435-5304', 'OII', *UseDataSeg)

data_SB_OII = fits.open(path_SB_OII)[1].data
data_3DSeg = fits.open(path_3DSeg)[0].data
data_Seg = fits.open(path_3DSeg)[1].data
data_cube = fits.open(path_cube)[1].data

data_SB_OII_smoothed = np.nansum(np.where(data_3DSeg !=0, data_cube, np.nan), axis=0) * 1.25 / 0.2 / 0.2 * 1e-3

kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)
# kernel = Box2DKernel(width=2)
data_SB_OII = np.where(data_Seg != 0, data_SB_OII, np.nan)
data_SB_OII_smoothed_after = convolve(data_SB_OII, kernel)

plt.figure()
plt.imshow(data_SB_OII_smoothed - data_SB_OII_smoothed_after, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=5)
# plt.imshow(data_SB_OII, origin='lower', cmap='gray', vmin=-0.1, vmax=0.1)
plt.colorbar(label='Difference')
plt.show()

# Lyalpha nebulae
# path_SB = '../../MUSEQuBES+CUBS/SB_Lya/Maps_unsmoothed_gsm0/J012403+004432.fits'
# path_SB_smoothed = '../../MUSEQuBES+CUBS/SB_Lya/J012403+004432.fits'
#
#
# data_SB = fits.open(path_SB)[0].data
# data_SB_smoothed = fits.open(path_SB_smoothed)[0].data
# data_SB_smoothed_after = convolve(data_SB, kernel)
#
# plt.figure()
# plt.imshow(data_SB_smoothed - data_SB_smoothed_after, origin='lower', cmap='gist_heat_r')
# plt.colorbar(label='Difference')
# plt.show()