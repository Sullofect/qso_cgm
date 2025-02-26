import os
import aplpy
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
from muse_kin_ETP import PlaceSudoSlitOnEachGal
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

def Bin(x, y, z, bins=20):
    n, edges = np.histogram(x, bins=bins)
    y_mean, y_max, y_min = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    z_mean, z_max, z_min = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i in range(bins):
        if n[i] == 0:
            y_mean[i], y_max[i], y_min[i] = np.nan, np.nan, np.nan
            z_mean[i], z_max[i], z_min[i] = np.nan, np.nan, np.nan
        else:
            mask = (x > edges[i]) * (x <= edges[i + 1])
            y_mean[i], y_max[i], y_min[i] = np.nanmean(y[mask]), np.nanmax(y[mask]), np.nanmin(y[mask])
            z_mean[i], z_max[i], z_min[i] = np.nanmean(z[mask]), np.nanmax(z[mask]), np.nanmin(z[mask])
    return x_mean, y_mean, y_max, y_min, z_mean, z_max, z_min


# QSO information
cubename = '3C57'
str_zap = ''
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

# V50W80
path_v50 = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_plot.fits'
path_w80 = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_plot.fits'
hdul_v50 = fits.open(path_v50)
hdul_w80 = fits.open(path_w80)
v50, w80 = hdul_v50[1].data, hdul_w80[1].data

path_v50_OII = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_OII.fits'
path_w80_OII = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_OII.fits'
hdul_v50_OII = fits.open(path_v50_OII)
hdul_w80_OII = fits.open(path_w80_OII)
v50_OII, w80_OII = hdul_v50_OII[1].data, hdul_w80_OII[1].data

path_v50_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_OIII.fits'
path_w80_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_OIII.fits'
hdul_v50_OIII = fits.open(path_v50_OIII)
hdul_w80_OIII = fits.open(path_w80_OIII)
v50_OIII, w80_OIII = hdul_v50_OIII[1].data, hdul_w80_OIII[1].data

#
path_v50_NLR_OII = '../../MUSEQuBES+CUBS/fit_kin/3C57_NLR_V50_OII.fits'
path_w80_NLR_OII = '../../MUSEQuBES+CUBS/fit_kin/3C57_NLR_W80_OII.fits'
hdul_v50_NLR_OII = fits.open(path_v50_NLR_OII)
hdul_w80_NLR_OII = fits.open(path_w80_NLR_OII)
v50_NLR_OII, w80_NLR_OII = hdul_v50_NLR_OII[1].data, hdul_w80_NLR_OII[1].data

path_v50_NLR_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_NLR_V50_OIII.fits'
path_w80_NLR_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_NLR_W80_OIII.fits'
hdul_v50_NLR_OIII = fits.open(path_v50_NLR_OIII)
hdul_w80_NLR_OIII = fits.open(path_w80_NLR_OIII)
v50_NLR_OIII, w80_NLR_OIII = hdul_v50_NLR_OIII[1].data, hdul_w80_NLR_OIII[1].data

# Plot the velocity field
x, y = np.meshgrid(np.arange(v50.shape[0]), np.arange(v50.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)
pixcoord_NLR = PixCoord(x=x, y=y)
x_NLR, y_NLR = x, y

# Hedaer information
path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
w = WCS(hdr_sub_gaia, naxis=2)
center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
c2 = w.world_to_pixel(center_qso)

# Mask the center
circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
center_mask_flatten = ~circle.contains(pixcoord)
center_mask = center_mask_flatten.reshape(v50.shape)
x, y = x[center_mask_flatten], y[center_mask_flatten]
pixcoord = pixcoord[center_mask_flatten]

# Mask a slit
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=5, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0]) ** 2 + (y - c2[1]) ** 2) * 0.2 * 50 / 7
dis_mask = dis[mask]

# Mask each side
red = ((x[mask] - c2[0]) < 0) * ((y[mask] - c2[1]) > 0)
blue = ~red
dis_red = dis_mask[red] * -1
dis_blue = dis_mask[blue]

# Slit position
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
# plt.imshow(np.where(center_mask, v50, np.nan), origin='lower', cmap='coolwarm', vmin=-350, vmax=350)
# patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
# plt.plot(c2[0], c2[1], '*', markersize=15)
# fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_sudo_slit.png', bbox_inches='tight')

# Make a figure for the poster
fig, ax = plt.subplots(2, 1, figsize=(10, 7), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.0)

# ETG
gal_list = np.array(['NGC2685', 'NGC3941', 'NGC3945', 'NGC4262', 'NGC5582', 'NGC6798', 'UGC06176'])
# gal_list = np.array(['NGC2594', 'NGC2685', 'NGC2764', 'NGC3619', 'NGC3626', 'NGC3941', 'NGC3945',
#                      'NGC4203', 'NGC4262', 'NGC5582', 'NGC6798', 'UGC06176', 'UGC09519'])
# gal_list = np.array(['NGC2594', 'NGC2685', 'NGC2764', 'NGC3619', 'NGC3626', 'NGC3838', 'NGC3941',
#                      'NGC3945', 'NGC4203', 'NGC4262', 'NGC5173', 'NGC5582', 'NGC5631', 'NGC6798',
#                      'UGC06176', 'UGC09519'])

dis_red_all, v_red_all, sigma_red_all = np.array([]), np.array([]), np.array([])
dis_blue_all, v_blue_all, sigma_blue_all = np.array([]), np.array([]), np.array([])
for jj in range(len(gal_list)):
    color = 'C' + str(jj)
    dis_red_gal, dis_blue_gal, v_red_gal, v_blue_gal, \
    sigma_red_gal, sigma_blue_gal = PlaceSudoSlitOnEachGal(igal=gal_list[jj])

    dis_red_all = np.hstack((dis_red_all, dis_red_gal))
    dis_blue_all = np.hstack((dis_blue_all, dis_blue_gal))
    v_red_all = np.hstack((v_red_all, v_red_gal))
    v_blue_all = np.hstack((v_blue_all, v_blue_gal))
    sigma_red_all = np.hstack((sigma_red_all, sigma_red_gal))
    sigma_blue_all = np.hstack((sigma_blue_all, sigma_blue_gal))

    # dis_red_gal_mean, v_red_gal_mean, sigma_red_gal_mean, flux_red_gal_mean = Bin(dis_red_gal, v_red_gal, sigma_red_gal,
    #                                                                               np.zeros_like(v_red_gal), bins=20)
    # dis_blue_gal_mean, v_blue_gal_mean, sigma_blue_gal_mean, flux_blue_gal_mean = Bin(dis_blue_gal, v_blue_gal,
    #                                                                                   sigma_blue_gal,
    #                                                                                   np.zeros_like(v_blue_gal), bins=20)
    #
    # ax[0].plot(dis_red_gal_mean, v_red_gal_mean, '-', color=color, lw=2, zorder=-100, label=gal_list[jj])
    # ax[0].plot(dis_blue_gal_mean, v_blue_gal_mean, '-', color=color, lw=2, zorder=-100)
    # ax[1].plot(dis_red_gal_mean, 2.563 * sigma_red_gal_mean, '-', color=color, lw=2, zorder=-100)
    # ax[1].plot(dis_blue_gal_mean, 2.563 * sigma_blue_gal_mean, '-', color=color, lw=2, zorder=-100)


sort = np.argsort(dis_red_all)
dis_red_all, v_red_all, sigma_red_all = dis_red_all[sort], v_red_all[sort], sigma_red_all[sort]
sort = np.argsort(dis_blue_all)
dis_blue_all, v_blue_all, sigma_blue_all = dis_blue_all[sort], v_blue_all[sort], sigma_blue_all[sort]
dis_red_gal_mean, v_red_gal_mean, v_red_gal_max, v_red_gal_min, \
sigma_red_gal_mean, sigma_red_gal_max, sigma_red_gal_min = Bin(dis_red_all, v_red_all, sigma_red_all, bins=20)
dis_blue_gal_mean, v_blue_gal_mean, v_blue_gal_max, v_blue_gal_min, \
sigma_blue_gal_mean, sigma_blue_gal_max, sigma_blue_gal_min = Bin(dis_blue_all, v_blue_all, sigma_blue_all, bins=20)
#
ax[0].plot(dis_red_gal_mean, v_red_gal_mean, '-', color='C1', lw=2, zorder=-100, label='HI 21-cm from Serra et al. 2012')
ax[0].plot(dis_blue_gal_mean, v_blue_gal_mean, '-', color='C1', lw=2, zorder=-100)
ax[0].fill_between(dis_red_gal_mean, v_red_gal_min, v_red_gal_max, lw=2, color='C1', alpha=0.4, edgecolor=None)
ax[0].fill_between(dis_blue_gal_mean, v_blue_gal_min, v_blue_gal_max, lw=2, color='C1', alpha=0.4, edgecolor=None)
ax[1].plot(dis_red_gal_mean, 2.563 * sigma_red_gal_mean, '-', color='C1', lw=2, zorder=-100)
ax[1].plot(dis_blue_gal_mean, 2.563 * sigma_blue_gal_mean, '-', color='C1', lw=2, zorder=-100)
ax[1].fill_between(dis_red_gal_mean, 2.563 * sigma_red_gal_min, 2.563 * sigma_red_gal_max, lw=2, color='C1', alpha=0.4,
                   edgecolor=None)
ax[1].fill_between(dis_blue_gal_mean, 2.563 * sigma_blue_gal_min, 2.563 * sigma_blue_gal_max, lw=2, color='C1',
                   alpha=0.4, edgecolor=None)


# 3C57
# V50, W80
v50_flatten = v50.flatten()[center_mask_flatten]
w80_flatten = w80.flatten()[center_mask_flatten]
v50_blue, v50_red = v50_flatten[mask][blue], v50_flatten[mask][red]
w80_blue, w80_red = w80_flatten[mask][blue], w80_flatten[mask][red]
# dis_red, dis_blue = dis_red[~np.isnan(v50_red)], dis_blue[~np.isnan(v50_blue)]
# v50_blue, v50_red = v50_blue[~np.isnan(v50_blue)], v50_red[~np.isnan(v50_red)]
# w80_blue, w80_red = w80_blue[~np.isnan(w80_blue)], w80_red[~np.isnan(w80_red)]
d5080_blue, v50_blue_mean, _, _, w80_blue_mean, _, _ = Bin(dis_blue, v50_blue, w80_blue, bins=20)
d5080_red, v50_red_mean, _, _, w80_red_mean, _, _ = Bin(dis_red, v50_red, w80_red, bins=20)
# ax[0].scatter(d5080_red, v50_red_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(d5080_blue, v50_blue_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'$\rm 3C\,57 \, southwest$')
# ax[1].scatter(d5080_red, w80_red_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(d5080_blue, w80_blue_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')

# Plot each componenet
path_fit = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
hdul = fits.open(path_fit)
fs, hdr = hdul[1].data, hdul[2].header
v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
sigma, dsigma = hdul[5].data, hdul[6].data

# Order by velocity
sorting_indices = np.argsort(v, axis=0)
v = np.take_along_axis(v, sorting_indices, axis=0)
sigma = np.take_along_axis(sigma, sorting_indices, axis=0)

v_1, v_2, v_3 = v[0, :, :], v[1, :, :], v[2, :, :]
# plt.figure()
# plt.imshow(v_2, origin='lower', cmap='coolwarm', vmin=-350, vmax=350)
# plt.colorbar()
# plt.show()
# raise ValueError('stop')
sigma_1, sigma_2, sigma_3 = sigma[0, :, :], sigma[1, :, :], sigma[2, :, :]
v_1_flatten, v_2_flatten, v_3_flatten = v_1.flatten()[center_mask_flatten], \
                                        v_2.flatten()[center_mask_flatten], \
                                        v_3.flatten()[center_mask_flatten]
sigma_1_flatten, sigma_2_flatten, sigma_3_flatten = sigma_1.flatten()[center_mask_flatten], \
                                                    sigma_2.flatten()[center_mask_flatten], \
                                                    sigma_3.flatten()[center_mask_flatten]
v_1_blue, v_1_red = v_1_flatten[mask][blue], v_1_flatten[mask][red]
v_2_blue, v_2_red = v_2_flatten[mask][blue], v_2_flatten[mask][red]
v_3_blue, v_3_red = v_3_flatten[mask][blue], v_3_flatten[mask][red]
sigma_1_blue, sigma_1_red = sigma_1_flatten[mask][blue], sigma_1_flatten[mask][red]
sigma_2_blue, sigma_2_red = sigma_2_flatten[mask][blue], sigma_2_flatten[mask][red]
sigma_3_blue, sigma_3_red = sigma_3_flatten[mask][blue], sigma_3_flatten[mask][red]

d5080_blue, v_1_blue_mean, _, _, w_1_blue_mean, _, _ = Bin(dis_blue, v_1_blue, sigma_1_blue * 2.563, bins=20)
d5080_red, v_1_red_mean, _, _, w_1_red_mean, _, _ = Bin(dis_red, v_1_red, sigma_1_red * 2.563, bins=20)
d5080_blue, v_2_blue_mean, _, _, w_2_blue_mean, _, _ = Bin(dis_blue, v_2_blue, sigma_2_blue * 2.563, bins=20)
d5080_red, v_2_red_mean, _, _, w_2_red_mean, _, _ = Bin(dis_red, v_2_red, sigma_2_red * 2.563, bins=20)
d5080_blue, v_3_blue_mean, _, _, w_3_blue_mean, _, _ = Bin(dis_blue, v_3_blue, sigma_3_blue * 2.563, bins=20)
d5080_red, v_3_red_mean, _, _, w_3_red_mean, _, _ = Bin(dis_red, v_3_red, sigma_3_red * 2.563, bins=20)

# V50
ax[0].scatter(d5080_red, v_1_red_mean, s=50, marker="8", edgecolors='k', linewidths=0.5, color='red')
ax[0].scatter(d5080_blue, v_1_blue_mean, s=50, marker="8", edgecolors='k', linewidths=0.5, color='blue')
ax[0].scatter(d5080_red, v_2_red_mean, s=50, marker="s", edgecolors='k', linewidths=0.5, color='red')
ax[0].scatter(d5080_blue, v_2_blue_mean, s=50, marker="s", edgecolors='k', linewidths=0.5, color='blue')
ax[0].scatter(d5080_red, v_3_red_mean, s=50, marker="*", edgecolors='k', linewidths=0.5, color='red')
ax[0].scatter(d5080_blue, v_3_blue_mean, s=50, marker="*", edgecolors='k', linewidths=0.5, color='blue')

# W80
ax[1].scatter(d5080_red, w_1_red_mean, s=50, marker="8", edgecolors='k', linewidths=0.5, color='red')
ax[1].scatter(d5080_blue, w_1_blue_mean, s=50, marker="8", edgecolors='k', linewidths=0.5, color='blue')
ax[1].scatter(d5080_red, w_2_red_mean, s=50, marker="s", edgecolors='k', linewidths=0.5, color='red')
ax[1].scatter(d5080_blue, w_2_blue_mean, s=50, marker="s", edgecolors='k', linewidths=0.5, color='blue')
ax[1].scatter(d5080_red, w_3_red_mean, s=50, marker="*", edgecolors='k', linewidths=0.5, color='red')
ax[1].scatter(d5080_blue, w_3_blue_mean, s=50, marker="*", edgecolors='k', linewidths=0.5, color='blue')


# OII
v50_OII_flatten = v50_OII.flatten()[center_mask_flatten]
w80_OII_flatten = w80_OII.flatten()[center_mask_flatten]
v50_OII_blue, v50_OII_red = v50_OII_flatten[mask][blue], v50_OII_flatten[mask][red]
w80_OII_blue, w80_OII_red = w80_OII_flatten[mask][blue], w80_OII_flatten[mask][red]
d5080_blue, v50_OII_blue_mean, _, _, w80_OII_blue_mean, _, _ = Bin(dis_blue, v50_OII_blue, w80_OII_blue, bins=20)
d5080_red, v50_OII_red_mean, _, _, w80_OII_red_mean, _, _ = Bin(dis_red, v50_OII_red, w80_OII_red, bins=20)
# ax[0].scatter(d5080_red, v50_OII_red_mean, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red',
#               label=r'OII')
# ax[0].scatter(d5080_blue, v50_OII_blue_mean, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'OII')
# ax[1].scatter(d5080_red, w80_OII_red_mean, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(d5080_blue, w80_OII_blue_mean, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue')

# OIII
v50_OIII_flatten = v50_OIII.flatten()[center_mask_flatten]
w80_OIII_flatten = w80_OIII.flatten()[center_mask_flatten]
v50_OIII_blue, v50_OIII_red = v50_OIII_flatten[mask][blue], v50_OIII_flatten[mask][red]
w80_OIII_blue, w80_OIII_red = w80_OIII_flatten[mask][blue], w80_OIII_flatten[mask][red]
d5080_blue, v50_OIII_blue_mean, _, _, w80_OIII_blue_mean, _, _ = Bin(dis_blue, v50_OIII_blue, w80_OIII_blue, bins=20)
d5080_red, v50_OIII_red_mean, _, _, w80_OIII_red_mean, _, _ = Bin(dis_red, v50_OIII_red, w80_OIII_red, bins=20)
# ax[0].scatter(d5080_red, v50_OIII_red_mean, s=50, marker='o', edgecolors='k', linewidths=0.5, color='red',
#               label=r'OIII')
# ax[0].scatter(d5080_blue, v50_OIII_blue_mean, s=50, marker='o', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'OIII')
# ax[1].scatter(d5080_red, w80_OIII_red_mean, s=50, marker='o', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(d5080_blue, w80_OIII_blue_mean, s=50, marker='o', edgecolors='k', linewidths=0.5, color='blue')

# Mask each side
mask_NLR = rectangle.contains(pixcoord_NLR)
dis_NLR = np.sqrt((x_NLR - c2[0]) ** 2 + (y_NLR - c2[1]) ** 2) * 0.2 * 50 / 7
dis_mask_NLR = dis_NLR[mask_NLR]
red_NLR = ((x_NLR[mask_NLR] - c2[0]) < 0) * ((y_NLR[mask_NLR] - c2[1]) > 0)
blue_NLR = ~red_NLR
dis_red_NLR = dis_mask_NLR[red_NLR] * -1
dis_blue_NLR = dis_mask_NLR[blue_NLR]
v50_NLR_OII_flatten = v50_NLR_OII.flatten()
w80_NLR_OII_flatten = w80_NLR_OII.flatten()
v50_blue_NLR_OII, v50_red_NLR_OII = v50_NLR_OII_flatten[mask_NLR][blue_NLR], v50_NLR_OII_flatten[mask_NLR][red_NLR]
w80_blue_NLR_OII, w80_red_NLR_OII = w80_NLR_OII_flatten[mask_NLR][blue_NLR], w80_NLR_OII_flatten[mask_NLR][red_NLR]
v50_NLR_OIII_flatten = v50_NLR_OIII.flatten()
w80_NLR_OIII_flatten = w80_NLR_OIII.flatten()
v50_blue_NLR_OIII, v50_red_NLR_OIII = v50_NLR_OIII_flatten[mask_NLR][blue_NLR], v50_NLR_OIII_flatten[mask_NLR][red_NLR]
w80_blue_NLR_OIII, w80_red_NLR_OIII = w80_NLR_OIII_flatten[mask_NLR][blue_NLR], w80_NLR_OIII_flatten[mask_NLR][red_NLR]

#
d5080_blue_NLR, v50_blue_mean_NLR_OII, _, _, w80_blue_mean_NLR_OII, _, _ = Bin(dis_blue_NLR, v50_blue_NLR_OII, w80_blue_NLR_OII, bins=20)
d5080_red_NLR, v50_red_mean_NLR_OII, _, _, w80_red_mean_NLR_OII, _, _ = Bin(dis_red_NLR, v50_red_NLR_OII, w80_red_NLR_OII, bins=20)
d5080_blue_NLR, v50_blue_mean_NLR_OIII, _, _, w80_blue_mean_NLR_OIII, _, _ = Bin(dis_blue_NLR, v50_blue_NLR_OIII, w80_blue_NLR_OIII, bins=20)
d5080_red_NLR, v50_red_mean_NLR_OIII, _, _, w80_red_mean_NLR_OIII, _, _ = Bin(dis_red_NLR, v50_red_NLR_OIII, w80_red_NLR_OIII, bins=20)
# ax[0].scatter(d5080_red_NLR, v50_red_mean_NLR_OII, s=50, marker='^', edgecolors='k', linewidths=0.5, color='purple')
# ax[0].scatter(d5080_blue_NLR, v50_blue_mean_NLR_OII, s=50, marker='^', edgecolors='k', linewidths=0.5, color='C0')
# ax[1].scatter(d5080_red_NLR, w80_red_mean_NLR_OII, s=50, marker='^', edgecolors='k', linewidths=0.5, color='purple')
# ax[1].scatter(d5080_blue_NLR, w80_blue_mean_NLR_OII, s=50, marker='^', edgecolors='k', linewidths=0.5, color='C0')
# ax[0].scatter(d5080_red_NLR, v50_red_mean_NLR_OIII, s=50, marker='v', edgecolors='k', linewidths=0.5, color='violet')
# ax[0].scatter(d5080_blue_NLR, v50_blue_mean_NLR_OIII, s=50, marker='v', edgecolors='k', linewidths=0.5, color='C2')
# ax[1].scatter(d5080_red_NLR, w80_red_mean_NLR_OIII, s=50, marker='v', edgecolors='k', linewidths=0.5, color='violet')
# ax[1].scatter(d5080_blue_NLR, w80_blue_mean_NLR_OIII, s=50, marker='v', edgecolors='k', linewidths=0.5, color='C2')
ax[0].axhline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[1].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].set_xlim(-40, 40)
ax[0].set_ylim(-450, 450)
# ax[1].set_ylim(0, 200)
ax[0].set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
ax[1].set_xlabel(r'$\rm Distance \, [pkpc]$', size=25)
ax[1].set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25, labelpad=20)
ax[0].legend(loc='best', fontsize=15)
fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_velocity_profile_poster.png', bbox_inches='tight')
