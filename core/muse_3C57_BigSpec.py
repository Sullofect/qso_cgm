import os
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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=False)
rc('ytick.minor', size=5, visible=False)
rc('xtick', direction='in', labelsize=25, top='off')
rc('ytick', direction='in', labelsize=25, right='off')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239

def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355

def expand_wave(wave, stack=True, times=3):
    if stack is True:
        wave_expand = np.array([])
    else:
        wave_expand = np.empty_like(wave)
    for i in range(len(wave)):
        wave_i = np.linspace(wave[i].min(), wave[i].max(), times * len(wave[i]))
        if stack is True:
            wave_expand = np.hstack((wave_expand, wave_i))
        else:
            wave_expand[i] = wave_i
    return wave_expand

def Gaussian(wave_vac, z, sigma_kms, flux, wave_line_vac):
    wave_obs = wave_line_vac * (1 + z)
    sigma_A = np.sqrt((sigma_kms / c_kms * wave_obs) ** 2 + (getSigma_MUSE(wave_obs)) ** 2)

    peak = flux / np.sqrt(2 * sigma_A ** 2 * np.pi)
    gaussian = peak * np.exp(-(wave_vac - wave_obs) ** 2 / 2 / sigma_A ** 2)

    return gaussian

def model_OII(wave_vac, z, sigma_kms, flux_OII, r_OII3729_3727, plot=False):
    wave_OII3727_obs = wave_OII3727_vac * (1 + z)
    wave_OII3729_obs = wave_OII3729_vac * (1 + z)

    sigma_OII3727_A = np.sqrt((sigma_kms / c_kms * wave_OII3727_obs) ** 2 + (getSigma_MUSE(wave_OII3727_obs)) ** 2)
    sigma_OII3729_A = np.sqrt((sigma_kms / c_kms * wave_OII3729_obs) ** 2 + (getSigma_MUSE(wave_OII3729_obs)) ** 2)

    flux_OII3727 = flux_OII / (1 + r_OII3729_3727)
    flux_OII3729 = flux_OII / (1 + 1.0 / r_OII3729_3727)

    peak_OII3727 = flux_OII3727 / np.sqrt(2 * sigma_OII3727_A ** 2 * np.pi)
    peak_OII3729 = flux_OII3729 / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)

    OII3727_gaussian = peak_OII3727 * np.exp(-(wave_vac - wave_OII3727_obs) ** 2 / 2 / sigma_OII3727_A ** 2)
    OII3729_gaussian = peak_OII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)

    if plot:
        return OII3727_gaussian, OII3729_gaussian
    else:
        return OII3727_gaussian + OII3729_gaussian

# QSO information
cubename = '3C57'
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

#
path_fit = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
path_v50 = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_plot.fits'
path_w80 = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_plot.fits'
hdul_v50 = fits.open(path_v50)
hdul_w80 = fits.open(path_w80)
v50, w80 = hdul_v50[1].data, hdul_w80[1].data

hdul = fits.open(path_fit)
fs, hdr = hdul[1].data, hdul[2].header
v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
sigma, dsigma = hdul[5].data, hdul[6].data
flux_OII_fit, dflux_OII_fit = hdul[7].data, hdul[8].data
flux_OIII_fit, dflux_OIII_fit = hdul[9].data, hdul[10].data
r, dr = hdul[11].data, hdul[12].data
a_OII, da_OII = hdul[13].data, hdul[14].data
a_OIII, da_OIII = hdul[17].data, hdul[18].data
b_OII, db_OII = hdul[15].data, hdul[16].data
b_OIII, db_OIII = hdul[19].data, hdul[20].data

# Load data
UseSeg = (1.5, 'gauss', 1.5, 'gauss')
UseDataSeg = (1.5, 'gauss', None, None)
line_OII, line_OIII = 'OII', 'OIII'
path_cube_OII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}.fits'.format(line_OII)
path_cube_OIII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}.fits'.format(line_OIII)
path_cube_smoothed_OII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_{}_' \
                         '{}_{}_{}.fits'.format(line_OII, *UseDataSeg)
path_cube_smoothed_OIII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_{}_' \
                          '{}_{}_{}.fits'.format(line_OIII, *UseDataSeg)
path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(line_OII, *UseSeg)
path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(line_OIII, *UseSeg)

# Load data and smoothing
cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
wave_OII_exp = expand_wave([wave_OII_vac], stack=True)
wave_OIII_exp = expand_wave([wave_OIII_vac], stack=True)
flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
# wave_OII_vac, wave_OIII_vac = wave_OII_vac[:, np.newaxis, np.newaxis], wave_OIII_vac[:, np.newaxis, np.newaxis]
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
mask_seg = mask_seg_OII + mask_seg_OIII
flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori
flux_err_seg_OII, flux_err_seg_OIII = flux_err_OII * seg_3D_OII_ori, flux_err_OIII * seg_3D_OIII_ori

# Generate model data
v_array = np.linspace(-800, 800, 500)
wave_OII3727 = wave_OII3727_vac * (1 + v_array / c_kms * (1 + z_qso) + z_qso)
wave_OII3729 = wave_OII3729_vac * (1 + v_array  / c_kms * (1 + z_qso) + z_qso)
flux_OII3727, _ = model_OII(wave_OII3727[:, np.newaxis, np.newaxis, np.newaxis], z, sigma, flux_OII_fit, r, plot=True)
_, flux_OII3729 = model_OII(wave_OII3729[:, np.newaxis, np.newaxis, np.newaxis], z, sigma, flux_OII_fit, r, plot=True)
flux_OII_C = flux_OII3727 + flux_OII3729
flux_OII_sum = np.nansum(flux_OII_C, axis=1)

# Make example plots
coord = [(65, 81), (75, 88), (80, 75)]
# fig, ax = plt.subplots(2, 1, figsize=(5, 10), dpi=300)
# fig.subplots_adjust(hspace=0.0)
#
# # OII
# flux_OII_model = model_OII(wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma, flux_OII_fit, r)[:, :, 81, 65]
# # flux_OII3727_model, flux_OII3729_model = model_OII(wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis],
# #                                                    z, sigma, flux_OII_fit, r, plot=True)
# ax[0].plot(wave_OII_vac, flux_OII[:, 81, 65] - b_OII[81, 65], '-k', drawstyle='steps-mid')
# ax[0].plot(wave_OII_vac, flux_err_OII[:, 81, 65], '-C0', drawstyle='steps-mid')
# ax[0].plot(wave_OII_exp, np.nansum(flux_OII_model, axis=1), '-r')
# ax[0].plot(wave_OII_exp, flux_OII_model[:, 0], '--r')
# ax[0].plot(wave_OII_exp, flux_OII_model[:, 1], '--b')
# ax[0].plot(wave_OII_exp, flux_OII_model[:, 2], '--C1')
#
# # OIII
# flux_OIII_model = Gaussian(wave_OIII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma,
#                             flux_OIII_fit, wave_OIII5008_vac)[:, :, 81, 65]
# ax[1].plot(wave_OIII_vac, flux_OIII[:, 81, 65] - b_OIII[81, 65], '-k', drawstyle='steps-mid')
# ax[1].plot(wave_OIII_vac, flux_err_OIII[:, 81, 65], '-C0', drawstyle='steps-mid')
# ax[1].plot(wave_OIII_exp, np.nansum(flux_OIII_model, axis=1), '-r')
# ax[1].plot(wave_OIII_exp, flux_OIII_model[:, 0], '--r')
# ax[1].plot(wave_OIII_exp, flux_OIII_model[:, 1], '--b')
# ax[1].plot(wave_OIII_exp, flux_OIII_model[:, 2], '--C1')
#
# #
# ax[0].axvline(x=(1 + z_qso) * wave_OII3727_vac, color='grey', linestyle='--', zorder=-100)
# ax[0].axvline(x=(1 + z_qso) * wave_OII3729_vac, color='grey', linestyle='--', zorder=-100)
# ax[1].axvline(x=(1 + z_qso) * wave_OIII5008_vac, color='grey', linestyle='--', zorder=-100)
# # ax[0].tick_params(axis='both', which='major', labelsize=40)
# # ax[1].tick_params(axis='both', which='major', labelsize=40)
# ax[0].set_title(r'$\mathrm{[O\,II]}$', x=0.2, y=0.85, size=40)
# ax[1].set_title(r'$\mathrm{[O\,III]}$', x=0.2, y=0.85, size=40)
# ax[1].set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=25)
# fig.supylabel(r'${f}_{\lambda} \; [10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}}]$', size=25, x=-0.12)
# figname_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_ShowFit.png'
# fig.savefig(figname_OIII, bbox_inches='tight')

# OII
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
flux_OII_model = model_OII(wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma, flux_OII_fit, r)[:, :, 81, 65]
ax.plot(wave_OII_vac, flux_OII[:, 81, 65] - b_OII[81, 65], '-k', drawstyle='steps-mid')
ax.plot(wave_OII_vac, flux_err_OII[:, 81, 65], '-C0', drawstyle='steps-mid')
ax.plot(wave_OII_exp, np.nansum(flux_OII_model, axis=1), '-r')
ax.plot(wave_OII_exp, flux_OII_model[:, 0], '--r')
ax.plot(wave_OII_exp, flux_OII_model[:, 1], '--b')
ax.plot(wave_OII_exp, flux_OII_model[:, 2], '--C1')
ax.axvline(x=(1 + z_qso) * wave_OII3727_vac, color='grey', linestyle='--', zorder=-100)
ax.axvline(x=(1 + z_qso) * wave_OII3729_vac, color='grey', linestyle='--', zorder=-100)
ax.set_xticks([6230, 6250])
ax.set_xlim(6225, 6255)
ax.set_title(r'$\mathrm{[O\,II]}$', x=0.2, y=0.85, size=40)
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=25)
# ax.set_ylabel(r'${f}_{\lambda} \; [10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}}]$', size=25, x=-0.12)
ax.set_ylabel(r'${f}_{\lambda}$', size=25, x=-0.12)
ax.tick_params(axis='y', which='major', labelleft=False, left=False)
figname_OII = '../../MUSEQuBES+CUBS/fit_kin/3C57_ShowFit_OII.png'
fig.savefig(figname_OII, bbox_inches='tight')

# OIII
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
flux_OIII_model = Gaussian(wave_OIII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma,
                            flux_OIII_fit, wave_OIII5008_vac)[:, :, 81, 65]
ax.plot(wave_OIII_vac, flux_OIII[:, 81, 65] - b_OIII[81, 65], '-k', drawstyle='steps-mid')
ax.plot(wave_OIII_vac, flux_err_OIII[:, 81, 65], '-C0', drawstyle='steps-mid')
ax.plot(wave_OIII_exp, np.nansum(flux_OIII_model, axis=1), '-r')
ax.plot(wave_OIII_exp, flux_OIII_model[:, 0], '--r')
ax.plot(wave_OIII_exp, flux_OIII_model[:, 1], '--b')
ax.plot(wave_OIII_exp, flux_OIII_model[:, 2], '--C1')
ax.axvline(x=(1 + z_qso) * wave_OIII5008_vac, color='grey', linestyle='--', zorder=-100)
ax.set_xticks([8360, 8390])
ax.set_xlim(8350, 8395)
ax.set_title(r'$\mathrm{[O\,III]}$', x=0.2, y=0.85, size=40)
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=25)
ax.set_ylabel(r'${f}_{\lambda}$', size=25, x=-0.12)
ax.tick_params(axis='y', which='major', labelleft=False, left=False)
figname_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_ShowFit_OIII.png'
fig.savefig(figname_OIII, bbox_inches='tight')

# Plot the spectra at fix place.
x, y = np.meshgrid(np.arange(v50.shape[0]), np.arange(v50.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

# Hedaer information
str_zap = ''
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

# Plot the [O II] spectra at fix place.
fig, ax = plt.subplots(1, 6, figsize=(30, 5), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0, wspace=0)

pos_array = [-30, -20, -10]
for i in range(3):
    select_red = np.argmin(np.abs(dis_red - pos_array[i]))
    x_i, y_i = x[mask][red][select_red], y[mask][red][select_red]
    flux_OII_model = model_OII(wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma, flux_OII_fit, r)[:, :, y_i,
                     x_i]
    ax[i].plot(wave_OII_vac, flux_OII[:, y_i, x_i] - b_OII[y_i, x_i], '-k', drawstyle='steps-mid')
    ax[i].plot(wave_OII_vac, flux_err_OII[:, y_i, x_i], '-C0', drawstyle='steps-mid')
    ax[i].plot(wave_OII_exp, np.nansum(flux_OII_model, axis=1), '-r')
    ax[i].plot(wave_OII_exp, flux_OII_model[:, 0], '--r')
    ax[i].plot(wave_OII_exp, flux_OII_model[:, 1], '--r')
    ax[i].plot(wave_OII_exp, flux_OII_model[:, 2], '--C1')
    ax[i].axvline(x=(1 + z_qso) * wave_OII3727_vac, color='grey', linestyle='--', zorder=-100)
    ax[i].axvline(x=(1 + z_qso) * wave_OII3729_vac, color='grey', linestyle='--', zorder=-100)
    ax[i].set_xticks([6230, 6250])
    ax[i].set_xlim(6205, 6255)
    ax[i].tick_params(axis='y', which='major', labelleft=False, left=False)
    ax[i].annotate('{} kpc'.format(pos_array[i]), (0.73, 0.9), size=25, xycoords='axes fraction')

pos_array = [10, 20, 30]
for i in range(3):
    select_blue = np.argmin(np.abs(dis_blue - pos_array[i]))
    x_i, y_i = x[mask][blue][select_blue], y[mask][blue][select_blue]
    flux_OII_model = model_OII(wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma, flux_OII_fit, r)[:, :, y_i,
                     x_i]
    ax[i+3].plot(wave_OII_vac, flux_OII[:, y_i, x_i] - b_OII[y_i, x_i], '-k', drawstyle='steps-mid')
    ax[i+3].plot(wave_OII_vac, flux_err_OII[:, y_i, x_i], '-C0', drawstyle='steps-mid')
    ax[i+3].plot(wave_OII_exp, np.nansum(flux_OII_model, axis=1), '-r')
    ax[i+3].plot(wave_OII_exp, flux_OII_model[:, 0], '--r')
    ax[i+3].plot(wave_OII_exp, flux_OII_model[:, 1], '--r')
    ax[i+3].plot(wave_OII_exp, flux_OII_model[:, 2], '--C1')
    ax[i+3].axvline(x=(1 + z_qso) * wave_OII3727_vac, color='grey', linestyle='--', zorder=-100)
    ax[i+3].axvline(x=(1 + z_qso) * wave_OII3729_vac, color='grey', linestyle='--', zorder=-100)
    ax[i+3].set_xticks([6230, 6250])
    ax[i+3].set_xlim(6205, 6255)
    ax[i+3].tick_params(axis='y', which='major', labelleft=False, left=False)
    ax[i+3].annotate('{} kpc'.format(pos_array[i]), (0.73, 0.9), size=25, xycoords='axes fraction')
ax[0].set_title(r'$\mathrm{[O\,II]}$', x=0.2, y=0.85, size=40)
fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=25, y=-0.05)
ax[0].set_ylabel(r'${f}_{\lambda}$', size=25, x=-0.12)
figname_OII = '../../MUSEQuBES+CUBS/fit_kin/3C57_OII_ref.png'
fig.savefig(figname_OII, bbox_inches='tight')

# Plot the [O III] spectra at fix place.
fig, ax = plt.subplots(1, 6, figsize=(30, 5), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0, wspace=0)

pos_array = [-30, -20, -10]
for i in range(3):
    select_red = np.argmin(np.abs(dis_red - pos_array[i]))
    x_i, y_i = x[mask][red][select_red], y[mask][red][select_red]
    flux_OIII_model = Gaussian(wave_OIII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma,
                               flux_OIII_fit, wave_OIII5008_vac)[:, :, y_i, x_i]
    ax[i].plot(wave_OIII_vac, flux_OIII[:, y_i, x_i] - b_OIII[y_i, x_i], '-k', drawstyle='steps-mid')
    ax[i].plot(wave_OIII_vac, flux_err_OIII[:, y_i, x_i], '-C0', drawstyle='steps-mid')
    ax[i].plot(wave_OIII_exp, np.nansum(flux_OIII_model, axis=1), '-r')
    ax[i].plot(wave_OIII_exp, flux_OIII_model[:, 0], '--r')
    ax[i].plot(wave_OIII_exp, flux_OIII_model[:, 1], '--r')
    ax[i].plot(wave_OIII_exp, flux_OIII_model[:, 2], '--C1')
    ax[i].axvline(x=(1 + z_qso) * wave_OIII5008_vac, color='grey', linestyle='--', zorder=-100)
    ax[i].set_xticks([8360, 8390])
    ax[i].set_xlim(8350, 8395)
    ax[i].tick_params(axis='y', which='major', labelleft=False, left=False)
    ax[i].annotate('{} kpc'.format(pos_array[i]), (0.73, 0.9), size=25, xycoords='axes fraction')

pos_array = [10, 20, 30]
for i in range(3):
    select_blue = np.argmin(np.abs(dis_blue - pos_array[i]))
    x_i, y_i = x[mask][blue][select_blue], y[mask][blue][select_blue]
    flux_OIII_model = Gaussian(wave_OIII_exp[:, np.newaxis, np.newaxis, np.newaxis], z, sigma,
                               flux_OIII_fit, wave_OIII5008_vac)[:, :, y_i, x_i]
    ax[i+3].plot(wave_OIII_vac, flux_OIII[:, y_i, x_i] - b_OIII[y_i, x_i], '-k', drawstyle='steps-mid')
    ax[i+3].plot(wave_OIII_vac, flux_err_OIII[:, y_i, x_i], '-C0', drawstyle='steps-mid')
    ax[i+3].plot(wave_OIII_exp, np.nansum(flux_OIII_model, axis=1), '-r')
    ax[i+3].plot(wave_OIII_exp, flux_OIII_model[:, 0], '--r')
    ax[i+3].plot(wave_OIII_exp, flux_OIII_model[:, 1], '--r')
    ax[i+3].plot(wave_OIII_exp, flux_OIII_model[:, 2], '--C1')
    ax[i+3].axvline(x=(1 + z_qso) * wave_OIII5008_vac, color='grey', linestyle='--', zorder=-100)
    ax[i+3].set_xticks([8360, 8390])
    ax[i+3].set_xlim(8350, 8395)
    ax[i+3].tick_params(axis='y', which='major', labelleft=False, left=False)
    ax[i+3].annotate('{} kpc'.format(pos_array[i]), (0.73, 0.9), size=25, xycoords='axes fraction')
ax[0].set_title(r'$\mathrm{[O\,III]}$', x=0.2, y=0.85, size=40)
fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=25, y=-0.05)
ax[0].set_ylabel(r'${f}_{\lambda}$', size=25, x=-0.12)
figname_OIII = '../../MUSEQuBES+CUBS/fit_kin/3C57_OIII_ref.png'
fig.savefig(figname_OIII, bbox_inches='tight')


raise ValueError('Stop here')

# Generate random 2D map data and 1D velocity data
v_rebin = v50
v_rebin = np.where(mask_seg > 0, v_rebin, np.nan)

def rebin(arr, new_shape):
    if len(np.shape(arr)) > 2:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1],
                 new_shape[2], arr.shape[2] // new_shape[2])
        return np.nanmean(np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=-2), axis=1)
    else:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=1)

# rebinning
# v_rebin = v_N1[0, 40:100, 42:102]
# flux_OIII_rebin = flux_seg_OIII[:, 40:100, 42:102]
# v_rebin = rebin(v_rebin, (20, 20))
# flux_OIII_rebin = rebin(flux_OIII_rebin, (len(flux_OIII_rebin), 20, 20))
# v_rebin = np.where(v_rebin != 0, v_N1_rebin, np.nan)
# size = np.shape(v_N1_rebin)
# x_range, y_range = np.arange(size[1]), np.arange(size[0])

# plt.figure()
# plt.imshow(v_N1_rebin, cmap='coolwarm', origin='lower')
# plt.xlim(47, 98)
# plt.ylim(47, 92)
# plt.show()
# raise ValueError('Stop here')

# Full pixels
x_range = (47, 98)
y_range = (47, 92)
size = (x_range[1] - x_range[0] + 1, y_range[1] - y_range[0] + 1)
x_range, y_range = np.arange(x_range[0], x_range[1] + 1), np.arange(y_range[0], y_range[1] + 1)
flux_OIII_rebin = flux_OIII


# Create a figure with subplots
fig, axs = plt.subplots(nrows=size[1], ncols=size[0], figsize=(40, 40))
fig.subplots_adjust(hspace=0, wspace=0)
norm = mpl.colors.Normalize(vmin=-350, vmax=350)

# Iterate over each spaxel and plot the corresponding 1D spectrum
for j in x_range:
    for i in y_range:
        ax = axs[y_range[-1] - i, j - x_range[0]]
        if ~np.isnan(v50[i, j]):
            ax.plot(v_array, flux_OII_sum[:, i, j], color='black')
            ax.axvline(x=0, color='black', alpha=0.5, linestyle='--')
            # if j == 65 and i == 81:
            #     ax.axhline(y=0, color='black', alpha=0.5, linestyle='--')
            #     ax.set_title(r"$\star$", size=40, color='green')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['right', 'top', 'left', 'bottom']].set_color('black')
            ax.spines[['right', 'top', 'left', 'bottom']].set_alpha(0.3)
            # ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

            # Color the full panel according to its velocity
            ax.set_facecolor(plt.cm.coolwarm(norm(v_rebin[i, j])))
        else:
            ax.axis('off')
plt.savefig('../../MUSEQuBES+CUBS/fit_kin/spaxel_spectra.png', bbox_inches='tight')