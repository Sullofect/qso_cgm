import os
import aplpy
import lmfit
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import stats
from astropy.io import ascii
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WaveCoord, Image
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
from scipy import integrate
from scipy import interpolate
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Constants
c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239


def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355


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


def model_OII_OIII(wave_vac, **params):
    if params['OIII'] == 0:
        wave_OII_vac = wave_vac
        m_OII = np.zeros_like(wave_OII_vac)
    elif params['OII'] == 0:
        wave_OIII_vac = wave_vac
        m_OIII5008 = np.zeros_like(wave_OIII_vac)
    else:
        wave_OII_vac = wave_vac[0]
        wave_OIII_vac = wave_vac[1]
        m_OII = np.zeros_like(wave_OII_vac)
        m_OIII5008 = np.zeros_like(wave_OIII_vac)

    if params['OII'] == 2:
        z_1, sigma_kms_1, flux_OII_1 = params['z_1'], params['sigma_kms_1'], params['flux_OII_1']
        z_2, sigma_kms_2, flux_OII_2 = params['z_2'], params['sigma_kms_2'], params['flux_OII_2']
        if params['ResolveOII']:
            r_OII3729_3727_1 = params['r_OII3729_3727_1']
            r_OII3729_3727_2 = params['r_OII3729_3727_2']
            m_OII_1 = model_OII(wave_OII_vac, z_1, sigma_kms_1, flux_OII_1, r_OII3729_3727_1)
            m_OII_2 = model_OII(wave_OII_vac, z_2, sigma_kms_2, flux_OII_2, r_OII3729_3727_2)
        else:
            m_OII_1 = Gaussian(wave_OII_vac, z_1, sigma_kms_1, flux_OII_1, params['OII_center'])
            m_OII_2 = Gaussian(wave_OII_vac, z_2, sigma_kms_2, flux_OII_2, params['OII_center'])
        m_OII = m_OII_1 + m_OII_2

    else:
        for i in range(params['OII']):
            z = params['z_{}'.format(i + 1)]
            sigma_kms = params['sigma_kms_{}'.format(i + 1)]
            flux_OII = params['flux_OII_{}'.format(i + 1)]
            if params['ResolveOII']:
                r_OII3729_3727 = params['r_OII3729_3727_{}'.format(i + 1)]
                m_OII_i = model_OII(wave_OII_vac, z, sigma_kms, flux_OII, r_OII3729_3727)
            else:
                m_OII_i = Gaussian(wave_OII_vac, z, sigma_kms, flux_OII, params['OII_center'])
            m_OII += m_OII_i

    #
    if params['OIII'] == 2:
        z_1, sigma_kms_1, flux_OIII5008_1 = params['z_1'], params['sigma_kms_1'], params['flux_OIII5008_1']
        z_2, sigma_kms_2, flux_OIII5008_2 = params['z_2'], params['sigma_kms_2'], params['flux_OIII5008_2']
        m_OIII5008_1 = Gaussian(wave_OIII_vac, z_1, sigma_kms_1, flux_OIII5008_1, wave_OIII5008_vac)
        m_OIII5008_2 = Gaussian(wave_OIII_vac, z_2, sigma_kms_2, flux_OIII5008_2, wave_OIII5008_vac)
        m_OIII5008 = m_OIII5008_1 + m_OIII5008_2
    else:
        for i in range(params['OIII']):
            z = params['z_{}'.format(i + 1)]
            sigma_kms = params['sigma_kms_{}'.format(i + 1)]
            flux_OIII5008 = params['flux_OIII5008_{}'.format(i + 1)]
            m_OIII5008_i = Gaussian(wave_OIII_vac, z, sigma_kms, flux_OIII5008, wave_OIII5008_vac)
            m_OIII5008 += m_OIII5008_i

    if params['OIII'] == 0:
        return m_OII + params['a'] * wave_vac + params['b']
    elif params['OII'] == 0:
        return m_OIII5008 + params['a'] * wave_vac + params['b']
    else:
        return np.hstack((m_OII + params['a_OII'] * wave_OII_vac + params['b_OII'],
                          m_OIII5008 + params['a_OIII5008'] * wave_OIII_vac + params['b_OIII5008']))


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

path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'

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
line_OII, line_OIII = 'OII', 'OIII'
path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}.fits'.format(line_OII)
path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}.fits'.format(line_OIII)
path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(line_OII, *UseSeg)
path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(line_OIII, *UseSeg)

# Load data and smoothing
cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
wave_OII_vac = expand_wave([wave_OII_vac], stack=True)
wave_OIII_vac = expand_wave([wave_OIII_vac], stack=True)
wave_OII_vac, wave_OIII_vac = wave_OII_vac[:, np.newaxis, np.newaxis], wave_OIII_vac[:, np.newaxis, np.newaxis]
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)

#
flux_OII_C2 = model_OII(wave_OII_vac, z[0, :, :], sigma[0, :, :], flux_OII_fit[0, :, :], r[0, :, :]) \
        + model_OII(wave_OII_vac, z[1, :, :], sigma[1, :, :], flux_OII_fit[1, :, :], r[1, :, :])
flux_OIII_C2 = Gaussian(wave_OIII_vac, z[0, :, :], sigma[0, :, :], flux_OIII_fit[0, :, :], wave_OIII5008_vac) \
          + Gaussian(wave_OIII_vac, z[1, :, :], sigma[1, :, :], flux_OIII_fit[1, :, :], wave_OIII5008_vac)

#
# flux_cumsum_OII = np.cumsum(flux_OII_C2, axis=0) * 1.25
flux_cumsum_OII = integrate.cumtrapz(flux_OII_C2, wave_OII_vac, initial=0, axis=0)
flux_cumsum_OII /= flux_cumsum_OII.max(axis=0)
print(flux_cumsum_OII.shape, wave_OII_vac.shape)

f = interpolate.interp1d(flux_cumsum_OII, wave_OII_vac, axis=0, fill_value='extrapolate')
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        



print('finished')
wave_10 = f(0.10)
# print(wave_10)
wave_10 = \
    np.take_along_axis(wave_OII_vac, np.argmin(np.abs(flux_cumsum_OII - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
print(wave_10)

wave_50 = \
    np.take_along_axis(wave_OII_vac, np.argmin(np.abs(flux_cumsum_OII - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
wave_90 = \
    np.take_along_axis(wave_OII_vac, np.argmin(np.abs(flux_cumsum_OII - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
z_guess_array_OII = (wave_50 - wave_OII3728_vac) / wave_OII3728_vac
v_guess_array_OII = c_kms * (z_guess_array_OII - z_qso) / (1 + z_qso)
W80_guess_array_OII = c_kms * (wave_90 - wave_10) / (wave_OII3728_vac * (1 + z_guess_array_OII))
sigma_kms_guess_array_OII = W80_guess_array_OII / 2.563  # W_80 = 2.563sigma

# Moments for OIII
# flux_cumsum_OIII = np.cumsum(flux_OIII_C2, axis=0) * 1.25
flux_cumsum_OIII = integrate.cumtrapz(flux_OIII_C2, wave_OIII_vac, initial=0, axis=0)
flux_cumsum_OIII /= flux_cumsum_OIII.max(axis=0)

wave_10 = \
    np.take_along_axis(wave_OIII_vac, np.argmin(np.abs(flux_cumsum_OIII - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
wave_50 = \
    np.take_along_axis(wave_OIII_vac, np.argmin(np.abs(flux_cumsum_OIII - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
wave_90 = \
    np.take_along_axis(wave_OIII_vac, np.argmin(np.abs(flux_cumsum_OIII - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
z_guess_array_OIII = (wave_50 - wave_OIII5008_vac) / wave_OIII5008_vac
v_guess_array_OIII = c_kms * (z_guess_array_OIII - z_qso) / (1 + z_qso)
W80_guess_array_OIII = c_kms * (wave_90 - wave_10) / (wave_OIII5008_vac * (1 + z_guess_array_OIII))
sigma_kms_guess_array_OIII = W80_guess_array_OIII / 2.563  # W_80 = 2.563sigma

z_guess_array = np.where(mask_seg_OIII != 0, z_guess_array_OIII, z_guess_array_OII)
v_guess_array = c_kms * (z_guess_array - z_qso) / (1 + z_qso)
sigma_kms_guess_array = np.where(mask_seg_OIII != 0, sigma_kms_guess_array_OIII, sigma_kms_guess_array_OII)
W80_guess_array = np.where(mask_seg_OIII != 0, W80_guess_array_OIII, W80_guess_array_OII)
v_guess_array = np.where((mask_seg_OII + mask_seg_OIII) != 0, v_guess_array, np.nan)
sigma_kms_guess_array = np.where((mask_seg_OII + mask_seg_OIII) != 0, sigma_kms_guess_array, np.nan)
W80_guess_array = np.where((mask_seg_OII + mask_seg_OIII) != 0, W80_guess_array, np.nan)


plt.figure()
plt.imshow(v_guess_array_OIII - v[0, :, :], origin='lower', cmap='coolwarm', vmin=-50, vmax=50)
plt.show()
# path_V50 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_V50.fits'
# path_W80 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_W80.fits'
#
# hdul_V50 = fits.ImageHDU(v_guess_array, header=hdr)
# hdul_V50.writeto(path_V50, overwrite=True)
# hdul_W80 = fits.ImageHDU(W80_guess_array, header=hdr)
# hdul_W80.writeto(path_W80, overwrite=True)
