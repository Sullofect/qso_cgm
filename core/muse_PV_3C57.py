import os
import aplpy
import lmfit
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

def Bin(x, y, std, flux, bins=20):
    n, edges = np.histogram(x, bins=bins)
    y_mean = np.zeros(bins)
    y_std = np.zeros(bins)
    y_flux = np.zeros(bins)
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i in range(bins):
        if n[i] == 0:
            y_mean[i] = np.nan
            y_std[i] = np.nan
            y_flux[i] = np.nan
        else:
            mask = (x > edges[i]) * (x < edges[i + 1])
            y_mean[i] = np.nanmean(y[mask])
            y_std[i] = np.nanmean(std[mask])
            y_flux[i] = np.nanmean(flux[mask])
    return x_mean, y_mean, y_std, y_flux


# QSO information
cubename = '3C57'
str_zap = ''
path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

# Measure kinematics
path_v50 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_V50.fits'
path_w80 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_W80.fits'
path_fit_N1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N1.fits'
path_fit_N2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
path_fit_N3 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N3.fits'


# v50w80
hdul_v50 = fits.open(path_v50)
hdul_w80 = fits.open(path_w80)
v50, w80 = hdul_v50[1].data, hdul_w80[1].data

# N1
hdul_N1 = fits.open(path_fit_N1)
fs_N1, hdr_N1 = hdul_N1[1].data, hdul_N1[2].header
v_N1, z_N1, dz_N1 = hdul_N1[2].data, hdul_N1[3].data, hdul_N1[4].data
sigma_N1, dsigma_N1 = hdul_N1[5].data, hdul_N1[6].data
flux_OIII_N1 = hdul_N1[9].data
chisqr_N1, redchi_N1 = hdul_N1[15].data, hdul_N1[16].data

# N2
hdul_N2 = fits.open(path_fit_N2)
fs_N2, hdr_N2 = hdul_N2[1].data, hdul_N2[2].header
v_N2, z_N2, dz_N2 = hdul_N2[2].data, hdul_N2[3].data, hdul_N2[4].data
sigma_N2, dsigma_N2 = hdul_N2[5].data, hdul_N2[6].data
flux_OIII_N2 = hdul_N2[9].data
chisqr_N2, redchi_N2 = hdul_N2[15].data, hdul_N2[16].data

# N3
hdul_N3 = fits.open(path_fit_N3)
fs_N3, hdr_N3 = hdul_N3[1].data, hdul_N3[2].header
v_N3, z_N3, dz_N3 = hdul_N3[2].data, hdul_N3[3].data, hdul_N3[4].data
sigma_N3, dsigma_N3 = hdul_N3[5].data, hdul_N3[6].data
flux_OIII_N3 = hdul_N3[9].data
chisqr_N3, redchi_N3 = hdul_N3[15].data, hdul_N3[16].data

# plot the velocity field
x, y = np.meshgrid(np.arange(v_N1.shape[1]), np.arange(v_N1.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

# Hedaer information
hdr = hdr_N1
path_sub_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
hdr['CD2_2'] = hdr_sub_gaia['CD2_2']
w = WCS(hdr, naxis=2)
center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
c2 = w.world_to_pixel(center_qso)

# mask the center
circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=3)
center_mask_flatten = ~circle.contains(pixcoord)
center_mask = center_mask_flatten.reshape(v_N1[0, :, :].shape)
x, y = x[center_mask_flatten], y[center_mask_flatten]
pixcoord = pixcoord[center_mask_flatten]

# mask a slit
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=5, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0])**2 + (y - c2[1])**2) * 0.2 * 50 / 7
dis_mask = dis[mask]
# dis_mask = dis_mask[center_mask_flatten[mask]]

# mask each side
red = ((x[mask] - c2[0]) < 0) * ((y[mask] - c2[1]) > 0)
blue = ~red
dis_red = dis_mask[red]
dis_blue = dis_mask[blue] * -1

# Slit position
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
# plt.imshow(np.where(center_mask, v_N1[0, :, :], np.nan), origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
# patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
# plt.plot(c2[0], c2[1], '*', markersize=15)
# fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_sudo_slit.png', bbox_inches='tight')


# Position-velocity diagram
i = 70
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300, sharex=True)
# fig.subplots_adjust(hspace=0, wspace=0.1)


# Components
v_N1_flatten = v_N1[0, :, :].flatten()[center_mask_flatten]
sigma_N1_flatten = sigma_N1[0, :, :].flatten()[center_mask_flatten]
flux_OIII_N1_flatten = flux_OIII_N1[0, :, :].flatten()[center_mask_flatten]
dis_blue_mean, v_N1_mean, sigma_N1_mean, flux_N1_mean = Bin(dis_blue, v_N1_flatten[mask][blue],
                                                            sigma_N1_flatten[mask][blue], flux_OIII_N1_flatten[mask][blue], bins=20)

#
v_N2_flatten_C1 = v_N2[0, :, :].flatten()[center_mask_flatten]
v_N2_flatten_C2 = v_N2[1, :, :].flatten()[center_mask_flatten]
v_N2_flatten_C1_sort = np.copy(v_N2_flatten_C1)
v_N2_flatten_C2_sort = np.copy(v_N2_flatten_C2)
sigma_N2_flatten_C1 = sigma_N2[0, :, :].flatten()[center_mask_flatten]
sigma_N2_flatten_C2 = sigma_N2[1, :, :].flatten()[center_mask_flatten]
sigma_N2_flatten_C1_sort = np.copy(sigma_N2_flatten_C1)
sigma_N2_flatten_C2_sort = np.copy(sigma_N2_flatten_C2)
flux_OIII_N2_flatten_C1 = flux_OIII_N2[0, :, :].flatten()[center_mask_flatten]
flux_OIII_N2_flatten_C2 = flux_OIII_N2[1, :, :].flatten()[center_mask_flatten]
flux_OIII_N2_flatten_C1_sort = np.copy(flux_OIII_N2_flatten_C1)
flux_OIII_N2_flatten_C2_sort = np.copy(flux_OIII_N2_flatten_C2)

#
v_sort = v_N2_flatten_C1 > v_N2_flatten_C2
v_N2_flatten_C1_sort[~v_sort] = v_N2_flatten_C2[~v_sort]
v_N2_flatten_C2_sort[~v_sort] = v_N2_flatten_C1[~v_sort]
sigma_N2_flatten_C1_sort[~v_sort] = sigma_N2_flatten_C2[~v_sort]
sigma_N2_flatten_C2_sort[~v_sort] = sigma_N2_flatten_C1[~v_sort]
flux_OIII_N2_flatten_C1_sort[~v_sort] = flux_OIII_N2_flatten_C2[~v_sort]
flux_OIII_N2_flatten_C2_sort[~v_sort] = flux_OIII_N2_flatten_C1[~v_sort]
v_N2_flatten_weight = (v_N2_flatten_C1_sort * flux_OIII_N2_flatten_C1_sort
                       + v_N2_flatten_C2_sort * flux_OIII_N2_flatten_C2_sort) / (flux_OIII_N2_flatten_C1_sort + flux_OIII_N2_flatten_C2_sort)
dis_red_mean_C1, v_N2_mean_C1, sigma_N2_mean_C1, flux_N2_mean_C1 = Bin(dis_red, v_N2_flatten_C1_sort[mask][red],
                                                                       sigma_N2_flatten_C1_sort[mask][red],
                                                                       flux_OIII_N2_flatten_C1_sort[mask][red], bins=20)
dis_red_mean_C2, v_N2_mean_C2, sigma_N2_mean_C2, flux_N2_mean_C2 = Bin(dis_red, v_N2_flatten_C2_sort[mask][red],
                                                                       sigma_N2_flatten_C2_sort[mask][red],
                                                                       flux_OIII_N2_flatten_C2_sort[mask][red], bins=20)


# Make a figure for the poster
fig, ax = plt.subplots(2, 1, figsize=(10, 7), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.05)

# ETG
# gal_list = np.array(['NGC2685', 'NGC3941', 'NGC3945', 'NGC4262', 'NGC5582', 'NGC6798', 'UGC06176'])
gal_list = np.array(['NGC2685', 'NGC4262', 'NGC5582', 'NGC6798', 'UGC06176'])
# gal_list = np.array(['NGC5582'])

# ax[0].plot(dis_red_gal_mean, v_red_gal_mean, '-k', lw=2)
# ax[0].plot(dis_blue_gal_mean, v_blue_gal_mean, '-k', label='HI 21-cm around NGC 5582 \n from Serra et al. 2012')
# ax[1].plot(dis_red_gal_mean, sigma_red_gal_mean, '-k', lw=2)
# ax[1].plot(dis_blue_gal_mean, sigma_blue_gal_mean, '-k', lw=2)

for jj in range(len(gal_list)):
    color = 'C' + str(jj)
    dis_red_gal, dis_blue_gal, v_red_gal, v_blue_gal, \
    sigma_red_gal, sigma_blue_gal = PlaceSudoSlitOnEachGal(igal=gal_list[jj])

    dis_red_gal_mean, v_red_gal_mean, sigma_red_gal_mean, flux_red_gal_mean = Bin(dis_red_gal, v_red_gal, sigma_red_gal,
                                                                                  np.zeros_like(v_red_gal), bins=20)
    dis_blue_gal_mean, v_blue_gal_mean, sigma_blue_gal_mean, flux_blue_gal_mean = Bin(dis_blue_gal, v_blue_gal,
                                                                                      sigma_blue_gal,
                                                                                      np.zeros_like(v_blue_gal), bins=20)


    ax[0].plot(dis_red_gal_mean, v_red_gal_mean, '-', color=color, lw=2, zorder=-100, label=gal_list[jj])
    ax[0].plot(dis_blue_gal_mean, v_blue_gal_mean, '-', color=color, lw=2, zorder=-100)
    ax[1].plot(dis_red_gal_mean, 2.563 * sigma_red_gal_mean, '-', color=color, lw=2, zorder=-100)
    ax[1].plot(dis_blue_gal_mean, 2.563 * sigma_blue_gal_mean, '-', color=color, lw=2, zorder=-100)


# 3C57
# v50, w80
v50_flatten = v50.flatten()[center_mask_flatten]
w80_flatten = w80.flatten()[center_mask_flatten]
v50_blue, v50_red = v50_flatten[mask][blue], v50_flatten[mask][red]
w80_blue, w80_red = w80_flatten[mask][blue], w80_flatten[mask][red]

#
d5080_blue, v50_blue_mean, w80_blue_mean, _ = Bin(dis_blue, v50_blue, w80_blue, np.zeros_like(v50_blue), bins=20)
d5080_red, v50_red_mean, w80_red_mean, _ = Bin(dis_red, v50_red, w80_red, np.zeros_like(v50_red), bins=20)
ax[0].scatter(d5080_blue, v50_blue_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[0].scatter(d5080_red, v50_red_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
ax[1].scatter(d5080_blue, w80_blue_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[1].scatter(d5080_red, w80_red_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')

# ax[0].scatter(dis_blue_mean, v_N1_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='C0', label='3C57 nebula')
# ax[0].scatter(dis_red_mean_C1, v_N2_mean_C1, s=60, marker="D", edgecolors='k', linewidths=0.5, color='red')
# ax[0].scatter(dis_red_mean_C2, v_N2_mean_C2, s=60, marker="D", edgecolors='k', linewidths=0.5, color='blue')
# ax[1].scatter(dis_blue_mean, sigma_N1_mean, s=50, marker='D', edgecolors='k', linewidths=0.5, color='C0')
# ax[1].scatter(dis_red_mean_C1, sigma_N2_mean_C1, s=60,  marker="D", edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_red_mean_C2, sigma_N2_mean_C2, s=60, marker="D", edgecolors='k', linewidths=0.5, color='blue')
ax[0].axhline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[1].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].set_xlim(-40, 40)
ax[0].set_ylim(-450, 450)
# ax[1].set_ylim(0, 200)
ax[0].set_ylabel(r'$\Delta v \rm \, [km \, s^{-1}]$', size=25)
ax[1].set_xlabel(r'$\rm Distance \, [kpc]$', size=25)
ax[1].set_ylabel(r'$W80 \rm \, [km \, s^{-1}]$', size=25)
ax[0].legend(loc='best', fontsize=19)
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_velocity_profile_poster.png', bbox_inches='tight')
