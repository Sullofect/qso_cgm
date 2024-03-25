import os
import aplpy
# import lmfit
import numpy as np
import matplotlib as mpl
import gala.potential as gp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import stats
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import PixCoord
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian2DKernel
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
import biconical_outflow_model_3d as bicone
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from gala.units import galactic, solarsystem, dimensionless
from photutils.isophote import EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.isophote import Ellipse
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None):
    gc.recenter(ra_qso, dec_qso, width=1500 / 3600, height=1500 / 3600)
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=600, zorder=100)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)
    if type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$6'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(35)
        gc.add_label(0.98, 0.94, cubename, size=35, relative=True, horizontalalignment='right')
        gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        gc.colorbar.set_ticks([-300, -200, -100, 0, 100, 200, 300])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        gc.add_label(0.98, 0.94, 'NGC3945', size=35, relative=True, horizontalalignment='right')
        # gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'GasMap_sigma':
        # gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
    else:
        gc.colorbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        gc.colorbar.set_axis_label_text(r'$\rm log([O \, III]/[O \, II])$')

    # Scale bar
    # gc.add_scalebar(length=3 * u.arcsecond)
    gc.add_scalebar(length=450 * u.arcsecond)
    gc.scalebar.set_corner('top left')
    gc.scalebar.set_label(r"$450'' \approx 51 \mathrm{\; pkpc}$")
    # gc.scalebar.set_label(r"$3'' \approx 20 \mathrm{\; pkpc}$")
    gc.scalebar.set_font_size(20)

    # Hide
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.ticks.set_length(30)

    # Label
    # xw, yw = gc.pixel2world(146, 140)  # original figure
    xw, yw = gc.pixel2world(140, 140)
    # gc.show_arrows(xw, yw, -0.000035 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000035 * yw, color='k')
    # xw, yw = 40.1333130960119, -18.864847747328896
    # gc.show_arrows(xw, yw, -0.000020 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000020 * yw, color='k')
    # gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
    # gc.add_label(0.88, 0.70, r'E', size=20, relative=True)

# NGC3945
path_ETG = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/NGC3945_mom1.fits'
path_ETG_new = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/NGC3945_mom1_new.fits'
path_ETG_mom2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/NGC3945_mom2.fits'
path_ETG_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/allcubes/NGC3945_cube.fits'
path_figure_mom1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/NGC3945_mom1.png'
path_figure_mom2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/NGC3945_mom2.png'
path_figure_spec = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/NGC3945_spec.png'

# Load the kinematic map
hdul_ETG = fits.open(path_ETG)
hdr_ETG = hdul_ETG[0].header
hdr_ETG['NAXIS'] = 2
hdr_ETG.remove('NAXIS3')
hdr_ETG.remove('CTYPE3')
hdr_ETG.remove('CDELT3')
hdr_ETG.remove('CRPIX3')
hdr_ETG.remove('CRVAL3')
v_ETG = hdul_ETG[0].data[0, :, :] - 1281
hdul_ETG_new = fits.ImageHDU(v_ETG, header=hdr_ETG)
hdul_ETG_new.writeto(path_ETG_new, overwrite=True)
ra_gal, dec_gal = 178.307208, 60.675556

# Load the cube
hdul_ETG_cube = fits.open(path_ETG_cube)
hdr_ETG_cube = hdul_ETG_cube[0].header
flux = hdul_ETG_cube[0].data
flux = np.where(~np.isnan(v_ETG)[np.newaxis, :, :], flux, np.nan)


v_array = np.arange(hdr_ETG_cube['CRVAL3'], hdr_ETG_cube['CRVAL3'] + flux.shape[0] * hdr_ETG_cube['CDELT3'],
                    hdr_ETG_cube['CDELT3']) / 1e3  - 1281  # Convert from m/s to km/s, # and then shift wrt v_gal
print(v_array)
mask_v1 = (v_array < 0) * (v_array > -350)
mask_v2 = (v_array < 250) * (v_array > 0)
mask_v3 = (v_array < 100) * (v_array > -150)

v_1 = v_array[mask_v1]
v_2 = v_array[mask_v2]
v_3 = v_array[mask_v3]
flux_1 = flux[mask_v1, :, :]
flux_2 = flux[mask_v2, :, :]
flux_3 = flux[mask_v3, :, :]
print(np.shape(flux_1), np.shape(v_2))

# Moments
flux_cumsum_1 = np.cumsum(flux_1, axis=0)
flux_cumsum_1 /= flux_cumsum_1.max(axis=0)
v_1_array = np.zeros_like(flux_1)
v_1_array[:] = v_1[:, np.newaxis, np.newaxis]

v_10_1 = np.take_along_axis(v_1_array, np.argmin(np.abs(flux_cumsum_1 - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
v_50_1 = np.take_along_axis(v_1_array, np.argmin(np.abs(flux_cumsum_1 - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
v_90_1 = np.take_along_axis(v_1_array, np.argmin(np.abs(flux_cumsum_1 - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
sigma_1 = v_90_1 - v_10_1
sigma_1 /= 2.563  # W_80 = 2.563sigma

#
flux_cumsum_2 = np.cumsum(flux_2, axis=0)
flux_cumsum_2 /= flux_cumsum_2.max(axis=0)
v_2_array = np.zeros_like(flux_2)
v_2_array[:] = v_2[:, np.newaxis, np.newaxis]

v_10_2 = np.take_along_axis(v_2_array, np.argmin(np.abs(flux_cumsum_2 - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
v_50_2 = np.take_along_axis(v_2_array, np.argmin(np.abs(flux_cumsum_2 - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
v_90_2 = np.take_along_axis(v_2_array, np.argmin(np.abs(flux_cumsum_2 - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
sigma_2 = v_90_2 - v_10_2
sigma_2 /= 2.563  # W_80 = 2.563sigma

flux_cumsum_3 = np.cumsum(flux_3, axis=0)
flux_cumsum_3 /= flux_cumsum_3.max(axis=0)
v_3_array = np.zeros_like(flux_3)
v_3_array[:] = v_3[:, np.newaxis, np.newaxis]

v_10_3 = np.take_along_axis(v_3_array, np.argmin(np.abs(flux_cumsum_3 - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
v_50_3 = np.take_along_axis(v_3_array, np.argmin(np.abs(flux_cumsum_3 - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
v_90_3 = np.take_along_axis(v_3_array, np.argmin(np.abs(flux_cumsum_3 - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
sigma_3 = v_90_3 - v_10_3
sigma_3 /= 2.563  # W_80 = 2.563sigma



#
sigma = np.where(v_ETG <= 0, sigma_1, sigma_2)
sigma = np.where((v_ETG > 50) | (v_ETG < -50), sigma, sigma_3)
sigma = np.where(~np.isnan(v_ETG), sigma, np.nan)

# Guess the velocity field and compare with the original one
v_guess = np.where(v_ETG <= 0, v_50_1, v_50_2)
v_guess = np.where((v_ETG > 50) | (v_ETG < -50), v_guess, v_50_3)
v_guess = np.where(~np.isnan(v_ETG), v_guess, np.nan)

hdul_ETG_sigma = fits.ImageHDU(sigma, header=hdr_ETG)
hdul_ETG_sigma.writeto(path_ETG_mom2, overwrite=True)

# testing
plt.figure(figsize=(5, 5))
plt.imshow(v_guess, origin='lower', cmap='coolwarm', vmin=-350, vmax=350)
plt.colorbar()
plt.show()

#
fig = plt.figure(figsize=(10, 5), dpi=300)
plt.plot(v_array, flux[:, 183, 174], 'k')
plt.xlabel('Velocity [km/s]')
plt.ylabel('Flux [arbitrary unit]')
fig.savefig(path_figure_spec, bbox_inches='tight')


# Plot the kinematic map
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_ETG_new, figure=fig, hdu=1)
gc.show_colorscale(vmin=-350, vmax=350, cmap='coolwarm')
APLpyStyle(gc, type='GasMap', ra_qso=ra_gal, dec_qso=dec_gal)
fig.savefig(path_figure_mom1, bbox_inches='tight')

# Plot the sigma map
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_ETG_mom2, figure=fig, hdu=1)
gc.show_colorscale(vmin=0, vmax=300, cmap=Dense_20_r.mpl_colormap)
APLpyStyle(gc, type='GasMap_sigma', ra_qso=ra_gal, dec_qso=dec_gal)
fig.savefig(path_figure_mom2, bbox_inches='tight')

#
# plt.figure(figsize=(8, 8))
# plt.imshow(data[0, :, :], origin='lower', cmap='RdBu_r', vmin=-200, vmax=200)
# plt.show()