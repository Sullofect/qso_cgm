import os
# import aplpy
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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=False)
rc('ytick.minor', size=5, visible=False)
rc('xtick', direction='in', labelsize=25, top='off')
rc('ytick', direction='in', labelsize=25, right='off')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


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

cubename = '3C57'
# QSO information
path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

path_fit_N1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N1.fits'
path_fit_N2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N2.fits'
path_fit_N3 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N3.fits'

hdul_N1 = fits.open(path_fit_N1)
fs_N1, hdr_N1 = hdul_N1[1].data, hdul_N1[2].header
v_N1, z_N1, dz_N1 = hdul_N1[2].data, hdul_N1[3].data, hdul_N1[4].data
sigma_N1, dsigma_N1 = hdul_N1[5].data, hdul_N1[6].data
flux_OII_fit_N1, dflux_OII_fit_N1 = hdul_N1[7].data, hdul_N1[8].data
flux_OIII_fit_N1, dflux_OIII_fit_N1 = hdul_N1[9].data, hdul_N1[10].data
hdul_N2 = fits.open(path_fit_N2)
fs_N2, hdr_N2 = hdul_N2[1].data, hdul_N2[2].header
v_N2, z_N2, dz_N2 = hdul_N2[2].data, hdul_N2[3].data, hdul_N2[4].data
sigma_N2, dsigma_N2 = hdul_N2[5].data, hdul_N2[6].data
flux_OII_fit_N2, dflux_OII_fit_N2 = hdul_N2[7].data, hdul_N2[8].data
flux_OIII_fit_N2, dflux_OIII_fit_N2 = hdul_N2[9].data, hdul_N2[10].data
r_N2, dr_N2 = hdul_N2[11].data, hdul_N2[12].data
a_OII, da_OII = hdul_N2[13].data, hdul_N2[14].data
a_OIII, da_OIII = hdul_N2[17].data, hdul_N2[18].data
b_OII, db_OII = hdul_N2[15].data, hdul_N2[16].data
b_OIII, db_OIII = hdul_N2[19].data, hdul_N2[20].data

# Load data
UseSeg = (1.5, 'gauss', 1.5, 'gauss')
UseDataSeg = (1.5, 'gauss', None, None)
line_OII, line_OIII = 'OII', 'OIII'
path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}.fits'.format(line_OII)
path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}.fits'.format(line_OIII)
path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_{}_' \
                         '{}_{}_{}.fits'.format(line_OII, *UseDataSeg)
path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_{}_' \
                          '{}_{}_{}.fits'.format(line_OIII, *UseDataSeg)
path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(line_OII, *UseSeg)
path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(line_OIII, *UseSeg)

# Load data and smoothing
cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
# wave_OII_vac = expand_wave([wave_OII_vac], stack=True)
# wave_OIII_vac = expand_wave([wave_OIII_vac], stack=True)
flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
# wave_OII_vac, wave_OIII_vac = wave_OII_vac[:, np.newaxis, np.newaxis], wave_OIII_vac[:, np.newaxis, np.newaxis]
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
mask_seg = mask_seg_OII + mask_seg_OIII

# Generate random 2D map data and 1D velocity data
v_rebin = v_N1[0, :, :]
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
flux_OIII_rebin = flux_seg_OIII


# Create a figure with subplots
fig, axs = plt.subplots(nrows=size[1], ncols=size[0], figsize=(40, 40))
fig.subplots_adjust(hspace=0, wspace=0)
norm = mpl.colors.Normalize(vmin=-350, vmax=350)

# Iterate over each spaxel and plot the corresponding 1D spectrum
for j in x_range:
    for i in y_range:
        ax = axs[y_range[-1] - i, j - x_range[0]]
        if ~np.isnan(v_rebin[i, j]):
            ax.plot(wave_OIII_vac, flux_OIII_rebin[:, i, j], color='black', drawstyle='steps-mid')
            # ax.set_title(f'Spaxel ({i}, {j})')
            # ax.set_xlim(0, len(spectrum))
            # ax.set_ylim(0, 1)  # Adjust y-axis limits as needed
            ax.set_xticks([])  # Disable x-axis ticks
            ax.set_yticks([])  # Disable y-axis ticks

            # Color the full panel according to its velocity
            ax.set_facecolor(plt.cm.coolwarm(norm(v_rebin[i, j])))
        else:
            ax.axis('off')

plt.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/spaxel_spectra.png', bbox_inches='tight')