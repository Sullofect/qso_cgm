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
from astropy.table import Table, join
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

# Testing
# Construct HI catalogs
# path_table_gals = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_galaxies.dat'
# path_table_properties = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/HI_Properties.dat'
# path_HI_table = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/HI_Table.dat'
# t_g = Table.read(path_table_gals, format='ascii')
# t_p = Table.read(path_table_properties, format='ascii')
# merged = join(t_g, t_p, keys='Galaxy', join_type='inner')
#
# # Select with detected HI class
# merged = merged[merged['HI_class'] != '...']
# merged = merged[merged['HI_class'] != '-']
# merged.write(path_HI_table, format='ascii.fixed_width', overwrite=True)
raise ValueError()



# Lyalpha nebulae
path_SB = '../../MUSEQuBES+CUBS/SB_Lya/Maps_unsmoothed_gsm0/J111113-080401.fits'
path_SB_smoothed = '../../MUSEQuBES+CUBS/SB_Lya/Maps_smoothed_gsm1/J111113-080401.fits'
path_seg_Lya = '../../MUSEQuBES+CUBS/SB_Lya/Maps_smoothed_gsm1/J111113-080401_mask.fits'

data_SB = fits.open(path_SB)[0].data * 0.1
# data_SB_smoothed = fits.open(path_SB_smoothed)[0].data
# data_SB_smoothed_after = convolve(data_SB, kernel)


seg_Lya = fits.open(path_seg_Lya)[0].data
seg_Lya = np.where(seg_Lya == 0, seg_Lya, 1)
# data_SB_smoothed = np.where(seg_Lya, data_SB_smoothed, np.nan)
# data_SB_smoothed_after = np.where(seg_Lya, data_SB_smoothed_after, np.nan)
data_SB = np.where(seg_Lya, data_SB, np.nan)
#
# plt.figure()
# plt.imshow(data_SB_smoothed - data_SB_smoothed_after, origin='lower', cmap='gist_heat_r')
# plt.colorbar(label='Difference')
# plt.show()


# Path to Lyalpha data
object = 'CUBE_PSFSubMask_250_CSub_80_8_Lya_Sel1003_1243'
path_Lya = '../../MUSEQuBES+CUBS/SB_Lya/Cubes/{}.fits'.format(object)
path_Lya_seg = '../../MUSEQuBES+CUBS/SB_Lya/Cubes/{}.Objects_Id1.fits'.format(object)
# path_seg = '../../MUSEQuBES+CUBS/SB_Lya/Maps_smoothed_gsm1/{}_mask.fits'.format('J123055-113909')
# seg = fits.open(path_seg)[0].data

# seg_3D = fits.open(path_Lya_seg)[0].data
# any_true = seg_3D.any(axis=0)
# z_indices = np.arange(seg_3D.shape[0])[:, None, None]
# zmin = np.where(any_true, np.argmax(seg_3D, axis=0), -1)
# zmax = np.where(any_true, seg_3D.shape[0] - 1 - np.argmax(seg_3D[::-1], axis=0), -1)
# seg_3D_filled = np.where((z_indices < zmin[np.newaxis, :, :]) | (z_indices > zmax[np.newaxis, :, :]) , seg_3D, 1)

#
kernel = Gaussian2DKernel(x_stddev=2.0, y_stddev=2.0)
cube_Lya = fits.open(path_Lya)[0].data
cube_Lya_seg = fits.open(path_Lya_seg)[0].data
SB_Lya = np.nansum(np.where(cube_Lya_seg, cube_Lya, np.nan), axis=0) * 1.25 / 0.2 / 0.2 * 1e-3
cube_Lya = convolve(cube_Lya, kernel.array[np.newaxis, :, :])
SB_Lya_smoothed = np.nansum(np.where(cube_Lya_seg, cube_Lya, np.nan), axis=0) * 1.25 / 0.2 / 0.2 * 1e-3


def ComputeGini(image, segmap):
    image = image.flatten()
    segmap = np.asarray(segmap.flatten(), dtype=bool)
    sorted_pixelvals = np.sort(np.abs(image[segmap])) # note the absolute value
    # sorted_pixelvals = np.sort(image[segmap])
    n = len(sorted_pixelvals)
    indices = np.arange(1, n+1)  # start at i=1
    gini = (np.sum((2*indices-n-1) * sorted_pixelvals) /
            (float(n-1) * np.sum(sorted_pixelvals)))
    return gini

Gini = ComputeGini(SB_Lya, seg_Lya)
Gini_smoothed = ComputeGini(SB_Lya_smoothed, seg_Lya)

segmap = np.asarray(seg_Lya.flatten(), dtype=bool)
plt.figure()
# plt.plot(seg_3D[:, 150, 150], '-k')
# plt.plot(seg_3D_filled[:, 150, 150], '--r')
# plt.imshow(SB_Lya, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=1)
# plt.imshow(np.abs(data_SB), origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=1)
bins = np.linspace(0, SB_Lya.max(), 100)
plt.hist(np.abs(SB_Lya).ravel()[segmap], bins=bins, alpha=0.5, label='original={}'.format(np.round(Gini, 3)))
plt.hist(SB_Lya_smoothed.ravel()[segmap], bins=bins, alpha=0.5, label='Smoothed={}'.format(np.round(Gini_smoothed, 3)))
plt.yscale('log')
plt.legend()
plt.show()
raise ValueError('Check smoothing of Lyalpha nebulae')


# QSO information
# path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
# data_qso = ascii.read(path_qso, format='fixed_width')
# cubename, z_qso, seeing = data_qso['name'], data_qso['redshift'], data_qso['seeing']
# # data_qso = data_qso[data_qso['name'] == cubename]
# # ra_qso, dec_qso, z_qso, seeing = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], \
# #                                  data_qso['redshift'][0], data_qso['seeing'][0]
#
#
# # Physical scale
# cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
# scales = d_A_kpc * seeing / 206265
# print(np.max(scales), cubename[np.argmax(scales)])
# print(scales) #
#
# size_max = 7.3 # kpc
# smooth_pixel = size_max / (cosmo.angular_diameter_distance(z_qso).value * 1e3) * 206265 / 0.2 / 2.3548
# print(np.round(smooth_pixel, 1))


# Check smoothing OII nebulae
# UseSeg = (1.5, 'gauss', 1.5, 'gauss')
# UseDataSeg = (1.5, 'gauss', None, None)
# path_SB_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'\
#     .format('HE0435-5304', 'OII', *UseDataSeg)
# path_3DSeg = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'\
#     .format('HE0435-5304', 'OII', *UseDataSeg)
# path_cube = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_{}_{}_{}_{}.fits'\
#     .format('HE0435-5304', 'OII', *UseDataSeg)
#
# data_SB_OII = fits.open(path_SB_OII)[1].data
# data_3DSeg = fits.open(path_3DSeg)[0].data
# data_Seg = fits.open(path_3DSeg)[1].data
# data_cube = fits.open(path_cube)[1].data
#
# data_SB_OII_smoothed = np.nansum(np.where(data_3DSeg !=0, data_cube, np.nan), axis=0) * 1.25 / 0.2 / 0.2 * 1e-3
#
kernel = Gaussian2DKernel(x_stddev=1.0, y_stddev=1.0)
# print(kernel.array - Gaussian2DKernel(x_stddev=2.0, y_stddev=2.0).array)
# raise SystemExit
# # kernel = Box2DKernel(width=2)
# data_SB_OII = np.where(data_Seg != 0, data_SB_OII, np.nan)
# data_SB_OII_smoothed_after = convolve(data_SB_OII, kernel)
#
# plt.figure()
# plt.imshow(data_SB_OII_smoothed - data_SB_OII_smoothed_after, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=5)
# # plt.imshow(data_SB_OII, origin='lower', cmap='gray', vmin=-0.1, vmax=0.1)
# plt.colorbar(label='Difference')
# plt.show()
