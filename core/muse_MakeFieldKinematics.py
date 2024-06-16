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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

#
path_SB = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB'

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


def FitLines(cubename=None, fit_param=None, zapped=False, UseDataSeg=(1.5, 'gauss', None, None), FitType='specific',
             CheckGuess=None, width_OII=10, width_OIII=10, UseSmoothedCubes=True, UseDetectionSeg=None):
    # Define line
    if fit_param['OII'] >= 1 and fit_param['OIII'] == 0:
        line = 'OII'
    elif fit_param['OII'] == 0 and fit_param['OIII'] >= 1:
        line = 'OIII'
    else:
        line = 'OII+OIII'

    # if zapped
    if zapped:
        str_zap = '_zapped'
    else:
        str_zap = ''

    # Load qso information
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Load cubes and calculate initial conditions
    if line == 'OII+OIII':
        line_OII, line_OIII = 'OII', 'OIII'
        path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line_OII)
        path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                 '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
        path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line_OIII)
        path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                  '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDataSeg)
        if UseDetectionSeg is not None:
            path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line_OII, *UseDetectionSeg)
            path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line_OIII, *UseDetectionSeg)
        else:
            path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line_OII, *UseDataSeg)
            path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line_OIII, *UseDataSeg)
        path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
        path_cube = path_cube_OII

        # Load data and smoothing
        if UseSmoothedCubes:
            cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
        else:
            cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
        wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
        flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
        flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
        seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
        seg_3D_ori = np.vstack((seg_3D_OII_ori, seg_3D_OIII_ori))
        mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
        flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori

        # Extend over
        start_OII = (seg_3D_OII_ori != 0).argmax(axis=0)
        end_OII = start_OII + mask_seg_OII
        start_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), start_OII, start_OII - width_OII)
        end_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), end_OII, end_OII + width_OII)
        idx_OII = np.zeros_like(seg_3D_OII_ori)
        idx_OII[:] = np.arange(np.shape(seg_3D_OII_ori)[0])[:, np.newaxis, np.newaxis]
        seg_3D_OII = np.where((idx_OII >= end_OII[np.newaxis, :, :]) | (idx_OII < start_OII[np.newaxis, :, :]),
                              seg_3D_OII_ori, 1)

        # [O III]
        start_OIII = (seg_3D_OIII_ori != 0).argmax(axis=0)
        end_OIII = start_OIII + mask_seg_OIII
        start_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), start_OIII, start_OIII - width_OIII)
        end_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), end_OIII, end_OIII + width_OIII)
        idx_OIII = np.zeros_like(seg_3D_OIII_ori)
        idx_OIII[:] = np.arange(np.shape(seg_3D_OIII_ori)[0])[:, np.newaxis, np.newaxis]
        seg_3D_OIII = np.where((idx_OIII >= end_OIII[np.newaxis, :, :]) | (idx_OIII < start_OIII[np.newaxis, :, :]),
                               seg_3D_OIII_ori, 1)
        flux_OII, flux_err_OII = flux_OII * seg_3D_OII, flux_err_OII * seg_3D_OII
        flux_OIII, flux_err_OIII = flux_OIII * seg_3D_OIII, flux_err_OIII * seg_3D_OIII

        #
        mask_seg = mask_seg_OII + mask_seg_OIII
        wave_vac = np.array([wave_OII_vac, wave_OIII_vac], dtype=object)
        flux = np.vstack((flux_OII, flux_OIII))
        flux_err = np.vstack((flux_err_OII, flux_err_OIII))
        # flux_err = np.where(flux_err != 0, flux_err, np.inf)

        #
        # flux = np.where(flux_err != 0, flux, np.nan)
        # flux_err = np.where(flux_err != 0, flux_err, np.nan)

        # Moments for OII
        flux_cumsum = np.cumsum(flux_OII, axis=0) * 1.25
        flux_cumsum /= flux_cumsum.max(axis=0)
        wave_array = np.zeros_like(flux_OII)
        wave_array[:] = wave_OII_vac[:, np.newaxis, np.newaxis]

        wave_10 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array_OII = (wave_50 - wave_OII3728_vac) / wave_OII3728_vac
        sigma_kms_guess_array_OII = c_kms * (wave_90 - wave_10) / (wave_OII3728_vac * (1 + z_guess_array_OII))
        sigma_kms_guess_array_OII /= 2.563  # W_80 = 2.563sigma

        # Moments for OIII
        flux_cumsum = np.cumsum(flux_OIII, axis=0) * 1.25
        flux_cumsum /= flux_cumsum.max(axis=0)
        wave_array = np.zeros_like(flux_OIII)
        wave_array[:] = wave_OIII_vac[:, np.newaxis, np.newaxis]

        wave_10 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array_OIII = (wave_50 - wave_OIII5008_vac) / wave_OIII5008_vac
        sigma_kms_guess_array_OIII = c_kms * (wave_90 - wave_10) / (wave_OIII5008_vac * (1 + z_guess_array_OIII))
        sigma_kms_guess_array_OIII /= 2.563  # W_80 = 2.563sigma

        # Use [O III] if possible
        z_guess_array = np.where(mask_seg_OIII != 0, z_guess_array_OIII, z_guess_array_OII)
        sigma_kms_guess_array = np.where(mask_seg_OIII != 0, sigma_kms_guess_array_OIII, sigma_kms_guess_array_OII)
        z_mean, z_median, z_std = stats.sigma_clipped_stats(z_guess_array[mask_seg != 0], sigma=3, maxiters=5)
        sigma_mean, sigma_median, sigma_std = stats.sigma_clipped_stats(sigma_kms_guess_array[mask_seg != 0], sigma=3,
                                                                        maxiters=5)
        z_guess_array = np.where((z_guess_array < z_mean + 2 * z_std) * (z_guess_array > z_mean - 2 * z_std),
                                 z_guess_array, z_qso)
        sigma_kms_guess_array = np.where((sigma_kms_guess_array < sigma_mean + 1 * sigma_std) *
                                         (sigma_kms_guess_array > sigma_mean - 1 * sigma_std),
                                         sigma_kms_guess_array, sigma_mean)
        flux_guess_array_OII, flux_guess_array_OIII = np.max(flux_OII, axis=0), np.max(flux_OIII, axis=0)
        print(z_guess_array[CheckGuess[1], CheckGuess[0]])
        print(sigma_kms_guess_array[CheckGuess[1], CheckGuess[0]])
        print(flux_guess_array_OII[CheckGuess[1], CheckGuess[0]],
              flux_guess_array_OIII[CheckGuess[1], CheckGuess[0]])
    else:
        if line == 'OII':
            width = width_OII
        elif line == 'OIII':
            width = width_OIII
        path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line)
        path_cube_smoothed = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                             '{}_{}.fits'.format(cubename, str_zap, line, *UseDataSeg)
        if UseDetectionSeg is not None:
            path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'.\
                format(cubename, str_zap, line, *UseDetectionSeg)
        else:
            path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'.\
                format(cubename, str_zap, line, *UseDataSeg)
        path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)

        # Load data and smoothing
        if UseSmoothedCubes:
            cube = Cube(path_cube_smoothed)
        else:
            cube = Cube(path_cube)
        wave_vac = pyasl.airtovac2(cube.wave.coord())
        flux, flux_err = cube.data * 1e-3, np.sqrt(cube.var) * 1e-3
        seg_3D_ori = fits.open(path_3Dseg)[0].data
        mask_seg = np.sum(seg_3D_ori, axis=0)
        flux_seg = flux * seg_3D_ori
        start = (seg_3D_ori != 0).argmax(axis=0)
        end = start + mask_seg
        start = np.where((mask_seg > 20) | (mask_seg < 1), start, start - width)
        end = np.where((mask_seg > 20) | (mask_seg < 1), end, end + width)
        idx = np.zeros_like(seg_3D_ori)
        idx[:] = np.arange(np.shape(seg_3D_ori)[0])[:, np.newaxis, np.newaxis]
        seg_3D = np.where((idx >= end[np.newaxis, :, :]) | (idx < start[np.newaxis, :, :]), seg_3D_ori, 1)
        flux *= seg_3D
        flux_err *= seg_3D
        flux_err = np.where(flux_err != 0, flux_err, np.inf)

        # Moments
        flux_cumsum = np.cumsum(flux, axis=0) * 1.25
        flux_cumsum /= flux_cumsum.max(axis=0)
        wave_array = np.zeros_like(flux)
        wave_array[:] = wave_vac[:, np.newaxis, np.newaxis]

        wave_10 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50 = \
        np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90 = \
        np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array = (wave_50 - wave_OII3728_vac) / wave_OII3728_vac
        sigma_kms_guess_array = c_kms * (wave_90 - wave_10) / (wave_OII3728_vac * (1 + z_guess_array))
        sigma_kms_guess_array /= 2.563  # W_80 = 2.563sigma
        z_mean, z_median, z_std = stats.sigma_clipped_stats(z_guess_array[mask_seg != 0], sigma=3, maxiters=5)
        sigma_mean, sigma_median, sigma_std = stats.sigma_clipped_stats(sigma_kms_guess_array[mask_seg != 0],
                                                                        sigma=3, maxiters=5)
        z_guess_array = np.where((z_guess_array < z_mean + 3 * z_std) * (z_guess_array > z_mean - 3 * z_std),
                                 z_guess_array, z_qso)
        sigma_kms_guess_array = np.where((sigma_kms_guess_array < sigma_mean + 3 * sigma_std) *
                                         (sigma_kms_guess_array > sigma_mean - 3 * sigma_std),
                                         sigma_kms_guess_array, sigma_mean)
        flux_guess_array = np.max(flux, axis=0)

    # Guesses
    redshift_guess, sigma_kms_guess, flux_guess, r_OII3729_3727_guess = z_qso, 200.0, 1.0, 1.0
    parameters = lmfit.Parameters()
    model = model_OII_OIII
    parameters.add('OII', value=fit_param['OII'], vary=False, min=None, max=None, expr=None, brute_step=None)
    parameters.add('OIII', value=fit_param['OIII'], vary=False, min=None, max=None, expr=None, brute_step=None)
    parameters.add('z_qso', value=z_qso, vary=False, min=None, max=None, expr=None, brute_step=None)
    parameters.add('ResolveOII', value=fit_param['ResolveOII'], vary=False, min=None, max=None,
                   expr=None, brute_step=None)
    parameters.add('OII_center', value=fit_param['OII_center'], vary=False, min=None, max=None,
                   expr=None, brute_step=None)

    #
    size = np.shape(flux)[1:]
    max_OII, max_OIII = np.max([fit_param['OII'], fit_param['OII_2nd']]), np.max([fit_param['OIII'], fit_param['OIII_2nd']])
    max_line = np.max([max_OII, max_OIII])
    num_com = np.arange(max_line)
    size_3D = (max_line, size[0], size[1])
    fit_success, chisqr, redchi = np.zeros(size), np.zeros(size), np.zeros(size)
    chisqr_2, redchi_2 = np.zeros(size), np.zeros(size)
    fit_success_3D, chisqr_3D, redchi_3D = np.zeros(size_3D), np.zeros(size_3D), np.zeros(size_3D)
    v_fit, z_fit, dz_fit = np.zeros(size_3D), np.zeros(size_3D), np.zeros(size_3D)
    sigma_fit, dsigma_fit = np.zeros(size_3D), np.zeros(size_3D)

    # Zeros
    if line == 'OII':
        a_fit, da_fit, b_fit, db_fit = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)
        flux_fit, dflux_fit = np.zeros(size_3D), np.zeros(size_3D)
        r_fit, dr_fit = np.zeros(size_3D), np.zeros(size_3D)
        parameters.add('a', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        parameters.add('b', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)

        for i in range(max_OII):
            parameters.add_many(('z_{}'.format(i + 1), redshift_guess, True, redshift_guess - 0.05,
                                 redshift_guess + 0.05, None),
                                ('sigma_kms_{}'.format(i + 1), sigma_kms_guess, True, 50, 2000.0, None),
                                ('flux_OII_{}'.format(i + 1), flux_guess, True, 0, None, None))
            if fit_param['ResolveOII']:
                parameters.add('r_OII3729_3727_{}'.format(i + 1),
                               value=r_OII3729_3727_guess, vary=True, min=0.3, max=fit_param['r_max'])

    elif line == 'OIII':
        flux_fit, dflux_fit = np.zeros(size), np.zeros(size)
        a_fit, b_fit = np.zeros(size), np.zeros(size)
        da_fit, db_fit = np.zeros(size), np.zeros(size)

        parameters.add_many(('z', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
                            ('sigma_kms', sigma_kms_guess, True, 50, 2000.0, None),
                            ('flux_OIII5008', flux_guess, True, 0, None, None),
                            ('a', 0.0, False, None, None, None),
                            ('b', 0.0, False, None, None, None))

    else:
        a_OII_fit, b_OII_fit = np.zeros(size), np.zeros(size)
        da_OII_fit, db_OII_fit = np.zeros(size), np.zeros(size)
        a_OIII_fit, b_OIII_fit = np.zeros(size), np.zeros(size)
        da_OIII_fit, db_OIII_fit = np.zeros(size), np.zeros(size)
        flux_OII_fit, dflux_OII_fit = np.zeros(size_3D), np.zeros(size_3D)
        flux_OIII_fit, dflux_OIII_fit = np.zeros(size_3D), np.zeros(size_3D)
        r_fit, dr_fit = np.zeros(size_3D), np.zeros(size_3D)
        parameters.add('a_OII', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        parameters.add('b_OII', value=-0.0025, vary=True, min=-0.1, max=0.1, expr=None, brute_step=None)
        parameters.add('a_OIII5008', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        parameters.add('b_OIII5008', value=-0.0025, vary=True, min=-0.1, max=0.1, expr=None, brute_step=None)

        # Model
        for i in range(max_OII):
            parameters.add_many(('z_{}'.format(i + 1), redshift_guess, True, redshift_guess - 0.002,
                                 redshift_guess + 0.002, None),
                                ('sigma_kms_{}'.format(i + 1), sigma_kms_guess, True, 50, 2000.0, None),
                                ('flux_OII_{}'.format(i + 1), flux_guess, True, 0.0, None, None))
            if fit_param['ResolveOII']:
                parameters.add('r_OII3729_3727_{}'.format(i + 1), value=r_OII3729_3727_guess,
                               vary=True, min=0.3, max=fit_param['r_max'])
        for i in range(max_OIII):
            parameters.add_many(('flux_OIII5008_{}'.format(i + 1), flux_guess, True, 0.0, None, None))

    # Iterating each
    # if fit_param['OII_2nd'] != 0 or fit_param['OIII_2nd'] != 0
    if FitType == 'sequential':
        # 1st iteration
        # Fitting start
        for i in range(size[0]):  # i = p (y), j = q (x)
            for j in range(size[1]):
                if mask_seg[i, j] != 0:
                    # Give initial condition
                    flux_ij, flux_err_ij = flux[:, i, j], flux_err[:, i, j]
                    # flux_ij = np.where(flux_err_ij != 0, flux_ij, np.nan)
                    # flux_err_ij = np.where(flux_err_ij != 0, flux_err_ij, np.nan)
                    # print(np.shape(flux_ij))
                    # flux_ij = flux_ij[np.isnan(flux_ij)]
                    # flux_err_ij = flux_err_ij[np.isnan(flux_err_ij)]

                    if line == 'OII':
                        for k in range(fit_param['OII']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j]
                            parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array[i, j]
                    elif line == 'OIII':
                        for k in range(fit_param['OIII']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j]
                            parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array[i, j]
                    elif line == 'OII+OIII':
                        for k in range(fit_param['OII']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['z_{}'.format(k + 1)].max = z_guess_array[i, j] + 0.003
                            parameters['z_{}'.format(k + 1)].min = z_guess_array[i, j] - 0.003
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j]
                            parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array_OII[i, j]

                        for k in range(fit_param['OIII']):
                            # parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            # parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j]
                            parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array_OIII[i, j]

                        # flux_OII_ij, flux_err_OII_ij = flux_OII[:, i, j], flux_err_OII[:, i, j]
                        # flux_OIII_ij, flux_err_OIII_ij = flux_OIII[:, i, j], flux_err_OIII[:, i, j]
                        # mask_OII, mask_OIII = flux_OII_ij != 0, flux_OIII_ij != 0
                        # wave_vac = np.array([wave_OII_vac[mask_OII], wave_OIII_vac[mask_OIII]])
                        # flux_ij = np.hstack((flux_OII_ij[mask_OII], flux_OIII_ij[mask_OIII]))
                        # flux_err_ij = np.hstack((flux_err_OII_ij[mask_OII], flux_err_OIII_ij[mask_OIII]))


                    #
                    spec_model = lmfit.Model(model, missing='drop')
                    result = spec_model.fit(flux_ij, wave_vac=wave_vac, params=parameters, weights=1 / flux_err_ij)
                    fit_success[i, j] = result.success
                    seg_3D_i = seg_3D_ori[:, i, j]
                    residual = result.residual * seg_3D_i
                    chisqr[i, j], redchi[i, j] = result.chisqr, result.redchi
                    # chisqr[i, j] = np.sum(residual ** 2)
                    # redchi[i, j] = chisqr[i, j] / (mask_seg_OII[i, j] + mask_seg_OIII[i, j] - result.nvarys)

                    if line == 'OII':
                        a, b = result.best_values['a'], result.best_values['b']
                        da, db = result.params['a'].stderr, result.params['b'].stderr
                        a_fit[i, j], b_fit[i, j] = a, b
                        da_fit[i, j], db_fit[i, j] = da, db

                        for k in range(fit_param['OII']):
                            z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                          result.params['flux_OII_{}'.format(k + 1)].stderr
                            v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                            sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                            flux_fit[k, i, j], dflux_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                            if fit_param['ResolveOII']:
                                r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                        result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k
                    elif line == 'OIII':
                        flux_f_OIII, dflux_f_OIII = result.best_values['flux_OIII5008'], result.params['flux_OIII5008'].stderr
                        a, b = result.best_values['a'], result.best_values['b']
                        da, db = result.params['a'].stderr, result.params['b'].stderr
                        flux_fit[i, j], dflux_fit[i, j] = flux_f_OIII, dflux_f_OIII
                        a_fit[i, j], b_fit[i, j] = a, b
                        da_fit[i, j], db_fit[i, j] = da, db
                    elif line == 'OII+OIII':
                        a_OII, b_OII = result.best_values['a_OII'], result.best_values['b_OII']
                        da_OII, db_OII = result.params['a_OII'].stderr, result.params['b_OII'].stderr
                        a_OIII, b_OIII = result.best_values['a_OIII5008'], result.best_values['b_OIII5008']
                        da_OIII, db_OIII = result.params['a_OIII5008'].stderr, result.params['b_OIII5008'].stderr
                        a_OII_fit[i, j], da_OII_fit[i, j], b_OII_fit[i, j], db_OII_fit[i, j] = a_OII, da_OII, b_OII, db_OII
                        a_OIII_fit[i, j], da_OIII_fit[i, j] = a_OIII, da_OIII
                        b_OIII_fit[i, j], db_OIII_fit[i, j] = b_OIII, db_OIII

                        for k in range(fit_param['OII']):
                            z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                          result.params['flux_OII_{}'.format(k + 1)].stderr
                            v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                            sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                            flux_OII_fit[k, i, j], dflux_OII_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                            if fit_param['ResolveOII']:
                                r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                        result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k

                        for k in range(fit_param['OIII']):
                            # z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            # sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                            #                     result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OIII_k, dflux_f_OIII_k = result.best_values['flux_OIII5008_{}'.format(k + 1)], \
                                                          result.params['flux_OIII5008_{}'.format(k + 1)].stderr
                            flux_OIII_fit[k, i, j], dflux_OIII_fit[k, i, j] = flux_f_OIII_k, dflux_f_OIII_k
                else:
                    pass

        # 2nd interation  if 2nd Gaussians help to improve the fit
        # Calculate Chi^2
        if fit_param['OII_2nd'] != 0 or fit_param['OIII_2nd'] != 0:
            redchi_mean, redchi_median, redchi_std = stats.sigma_clipped_stats(redchi[redchi != 0], sigma=3, maxiters=5)
            refit_seg = np.where((redchi > 1.0), redchi, 0)
            print(len(redchi[redchi != 0]))
            print(len(refit_seg[refit_seg != 0]))
            print(redchi_mean, redchi_std)
            # plt.figure()
            # plt.hist(redchi[redchi != 0].flatten(), bins=100, range=[-10, 100])
            # plt.show()
            # raise ValueError('testing')

            # Fitting start
            # write
            path_fit_ini = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}_inigus.fits'.\
                format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
            hdul = fits.open(path_fit_ini)
            v_guess_fit, z_guess_fit, dz_guess_fit = hdul[2].data, hdul[3].data, hdul[4].data
            sigma_guess_fit, dsigma_guess_fit = hdul[5].data, hdul[6].data
            z_guess_fit = np.random.rand(*np.shape(z_guess_fit)) * 0.001 + z_guess_fit
            sigma_guess_fit = np.random.rand(*np.shape(sigma_guess_fit)) * 10 + sigma_guess_fit
            z_guess_fit = np.vstack((z_guess_fit, z_guess_fit))
            sigma_guess_fit = np.vstack((sigma_guess_fit, sigma_guess_fit))
            parameters['OII'].value = fit_param['OII_2nd']
            parameters['OIII'].value = fit_param['OIII_2nd']
            # parameters.add('bound', vary=True, min=-0.003, max=0, expr=None, brute_step=None)
            # parameters['z_1'].expr = 'z_2 + bound'
            for i in range(size[0]):
                for j in range(size[1]):
                    if refit_seg[i, j] != 0:
                        # Give initial condition
                        flux_ij, flux_err_ij = flux[:, i, j], flux_err[:, i, j]
                        if line == 'OII':
                            for k in range(fit_param['OII_2nd']):
                                parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                                parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                                parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array[i, j]
                        elif line == 'OIII':
                            for k in range(fit_param['OIII_2nd']):
                                parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                                parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                                parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array[i, j]
                        elif line == 'OII+OIII':
                            for k in range(fit_param['OII_2nd']):
                                parameters['z_{}'.format(k + 1)].value = z_guess_fit[k, i, j]
                                parameters['z_{}'.format(k + 1)].max = z_guess_fit[k, i, j] + 0.002
                                parameters['z_{}'.format(k + 1)].min = z_guess_fit[k, i, j] - 0.002
                                parameters['sigma_kms_{}'.format(k + 1)].value = sigma_guess_fit[k, i, j]
                                # parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                                # parameters['sigma_kms_{}'.format(k + 1)].max = 0.7 * sigma_fit[0, i, j]
                                # parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array_OII[i, j]
                            for k in range(fit_param['OIII_2nd']):
                                parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array_OIII[i, j]
                        #
                        spec_model = lmfit.Model(model, missing='drop')
                        result = spec_model.fit(flux_ij, wave_vac=wave_vac, params=parameters, weights=1 / flux_err_ij)
                        fit_success[i, j] = result.success
                        # residual = result.residual
                        chisqr_2[i, j], redchi_2[i, j] = result.chisqr, result.redchi

                        if line == 'OII':
                            a, b = result.best_values['a'], result.best_values['b']
                            da, db = result.params['a'].stderr, result.params['b'].stderr
                            a_fit[i, j], b_fit[i, j] = a, b
                            da_fit[i, j], db_fit[i, j] = da, db

                            for k in range(fit_param['OII_2nd']):
                                z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                                sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                    result.params['sigma_kms_{}'.format(k + 1)].stderr
                                flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                              result.params['flux_OII_{}'.format(k + 1)].stderr
                                v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                                sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                                flux_fit[k, i, j], dflux_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                                if fit_param['ResolveOII']:
                                    r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                            result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                    r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k
                        elif line == 'OIII':
                            flux_f_OIII, dflux_f_OIII = result.best_values['flux_OIII5008'], result.params['flux_OIII5008'].stderr
                            a, b = result.best_values['a'], result.best_values['b']
                            da, db = result.params['a'].stderr, result.params['b'].stderr
                            flux_fit[i, j], dflux_fit[i, j] = flux_f_OIII, dflux_f_OIII
                            a_fit[i, j], b_fit[i, j] = a, b
                            da_fit[i, j], db_fit[i, j] = da, db
                        elif line == 'OII+OIII':
                            a_OII, b_OII = result.best_values['a_OII'], result.best_values['b_OII']
                            da_OII, db_OII = result.params['a_OII'].stderr, result.params['b_OII'].stderr
                            a_OIII, b_OIII = result.best_values['a_OIII5008'], result.best_values['b_OIII5008']
                            da_OIII, db_OIII = result.params['a_OIII5008'].stderr, result.params['b_OIII5008'].stderr
                            a_OII_fit[i, j], da_OII_fit[i, j], b_OII_fit[i, j], db_OII_fit[i, j] = a_OII, da_OII, b_OII, db_OII
                            a_OIII_fit[i, j], da_OIII_fit[i, j] = a_OIII, da_OIII
                            b_OIII_fit[i, j], db_OIII_fit[i, j] = b_OIII, db_OIII

                            for k in range(fit_param['OII_2nd']):
                                z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                                sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                    result.params['sigma_kms_{}'.format(k + 1)].stderr
                                flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                              result.params['flux_OII_{}'.format(k + 1)].stderr
                                v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                                sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                                flux_OII_fit[k, i, j], dflux_OII_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                                if fit_param['ResolveOII']:
                                    r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                            result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                    r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k

                            for k in range(fit_param['OIII_2nd']):
                                # z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                                # sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                #                     result.params['sigma_kms_{}'.format(k + 1)].stderr
                                flux_f_OIII_k, dflux_f_OIII_k = result.best_values['flux_OIII5008_{}'.format(k + 1)], \
                                                              result.params['flux_OIII5008_{}'.format(k + 1)].stderr
                                flux_OIII_fit[k, i, j], dflux_OIII_fit[k, i, j] = flux_f_OIII_k, dflux_f_OIII_k
                    else:
                        pass

    elif FitType == 'find_best':
        # Fitting start
        # write
        path_fit_ini = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}_inigus.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
        hdul = fits.open(path_fit_ini)
        v_guess_fit, z_guess_fit, dz_guess_fit = hdul[2].data, hdul[3].data, hdul[4].data
        sigma_guess_fit, dsigma_guess_fit = hdul[5].data, hdul[6].data
        # z_guess_fit = np.random.rand(*np.shape(z_guess_fit)[1:]) * 0.001 + z_guess_fit
        # sigma_guess_fit = np.random.rand(*np.shape(sigma_guess_fit)[1:]) * 10 + sigma_guess_fit
        # z_guess_fit = np.vstack((z_guess_fit, z_guess_fit[1:, :, :]))
        # sigma_guess_fit = np.vstack((sigma_guess_fit, sigma_guess_fit[1:, :, :]))
        for i in range(size[0]):
            for j in range(size[1]):
                if mask_seg[i, j] != 0:
                    flux_ij, flux_err_ij = flux[:, i, j], flux_err[:, i, j]

                    # Give initial condition
                    if line == 'OII':
                        for k in range(fit_param['OII_2nd']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                            parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array[i, j]
                    elif line == 'OIII':
                        for k in range(fit_param['OIII_2nd']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                            parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array[i, j]
                    elif line == 'OII+OIII':
                        for k in range(fit_param['OII_2nd']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_fit[0, i, j]
                            parameters['z_{}'.format(k + 1)].max = z_guess_fit[0, i, j] + 0.002
                            parameters['z_{}'.format(k + 1)].min = z_guess_fit[0, i, j] - 0.002
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_guess_fit[0, i, j]
                            # parameters['z_{}'.format(k + 1)].value = z_guess_fit[k, i, j]
                            # parameters['z_{}'.format(k + 1)].max = z_guess_fit[k, i, j] + 0.002
                            # parameters['z_{}'.format(k + 1)].min = z_guess_fit[k, i, j] - 0.002
                            # parameters['sigma_kms_{}'.format(k + 1)].value = sigma_guess_fit[k, i, j]
                            # parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                            # parameters['sigma_kms_{}'.format(k + 1)].max = 0.7 * sigma_fit[0, i, j]
                            # parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array_OII[i, j]
                        for k in range(fit_param['OIII_2nd']):
                            parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array_OIII[i, j]

                    result_array = np.array([])
                    for o in range(max_line):
                        num = o + 1
                        parameters['OII'].value = num
                        parameters['OIII'].value = num

                        #
                        spec_model = lmfit.Model(model, missing='drop')
                        result = spec_model.fit(flux_ij, wave_vac=wave_vac, params=parameters, weights=1 / flux_err_ij)
                        fit_success_3D[o, i, j] = result.success
                        chisqr_3D[o, i, j], redchi_3D[o, i, j] = result.chisqr, result.redchi
                        result_array = np.hstack((result_array, result))

                    #
                    result = result_array[np.argmax(- redchi_3D[:, i, j])]
                    num_com_i = num_com[np.argmax(- redchi_3D[:, i, j])] + 1
                    fit_success = fit_success_3D[np.argmax(- redchi_3D[:, i, j])]

                    if line == 'OII':
                        a, b = result.best_values['a'], result.best_values['b']
                        da, db = result.params['a'].stderr, result.params['b'].stderr
                        a_fit[i, j], b_fit[i, j] = a, b
                        da_fit[i, j], db_fit[i, j] = da, db

                        for k in range(fit_param['OII_2nd']):
                            z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                          result.params['flux_OII_{}'.format(k + 1)].stderr
                            v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                            sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                            flux_fit[k, i, j], dflux_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                            if fit_param['ResolveOII']:
                                r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                        result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k
                    elif line == 'OIII':
                        flux_f_OIII, dflux_f_OIII = result.best_values['flux_OIII5008'], result.params['flux_OIII5008'].stderr
                        a, b = result.best_values['a'], result.best_values['b']
                        da, db = result.params['a'].stderr, result.params['b'].stderr
                        flux_fit[i, j], dflux_fit[i, j] = flux_f_OIII, dflux_f_OIII
                        a_fit[i, j], b_fit[i, j] = a, b
                        da_fit[i, j], db_fit[i, j] = da, db
                    elif line == 'OII+OIII':
                        a_OII, b_OII = result.best_values['a_OII'], result.best_values['b_OII']
                        da_OII, db_OII = result.params['a_OII'].stderr, result.params['b_OII'].stderr
                        a_OIII, b_OIII = result.best_values['a_OIII5008'], result.best_values['b_OIII5008']
                        da_OIII, db_OIII = result.params['a_OIII5008'].stderr, result.params['b_OIII5008'].stderr
                        a_OII_fit[i, j], da_OII_fit[i, j], b_OII_fit[i, j], db_OII_fit[i, j] = a_OII, da_OII, b_OII, db_OII
                        a_OIII_fit[i, j], da_OIII_fit[i, j] = a_OIII, da_OIII
                        b_OIII_fit[i, j], db_OIII_fit[i, j] = b_OIII, db_OIII

                        for k in range(num_com_i):
                            z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                          result.params['flux_OII_{}'.format(k + 1)].stderr
                            v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                            sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                            flux_OII_fit[k, i, j], dflux_OII_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                            if fit_param['ResolveOII']:
                                r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                        result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k

                            # OIII
                            flux_f_OIII_k, dflux_f_OIII_k = result.best_values['flux_OIII5008_{}'.format(k + 1)], \
                                                          result.params['flux_OIII5008_{}'.format(k + 1)].stderr
                            flux_OIII_fit[k, i, j], dflux_OIII_fit[k, i, j] = flux_f_OIII_k, dflux_f_OIII_k

                else:
                    pass

    elif FitType == 'specific':
        parameters['OII'].value = max_line
        parameters['OIII'].value = max_line
        path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}_N_com={}.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']),
                   *UseDataSeg, max_line)

        path_fit_ini = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}_inigus.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
        hdul = fits.open(path_fit_ini)
        v_guess_fit, z_guess_fit, dz_guess_fit = hdul[2].data, hdul[3].data, hdul[4].data
        sigma_guess_fit, dsigma_guess_fit = hdul[5].data, hdul[6].data
        for i in range(size[0]):
            for j in range(size[1]):
                if mask_seg[i, j] != 0:
                    flux_ij, flux_err_ij = flux[:, i, j], flux_err[:, i, j]

                    # Give initial condition
                    if line == 'OII':
                        for k in range(fit_param['OII_2nd']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                            parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array[i, j]
                    elif line == 'OIII':
                        for k in range(fit_param['OIII_2nd']):
                            parameters['z_{}'.format(k + 1)].value = z_guess_array[i, j]
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                            parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array[i, j]
                    elif line == 'OII+OIII':
                        for k in range(max_line):
                            parameters['z_{}'.format(k + 1)].value = z_guess_fit[0, i, j]
                            parameters['z_{}'.format(k + 1)].max = z_guess_fit[0, i, j] + 0.003
                            parameters['z_{}'.format(k + 1)].min = z_guess_fit[0, i, j] - 0.003
                            parameters['sigma_kms_{}'.format(k + 1)].value = sigma_guess_fit[0, i, j]
                            # parameters['z_{}'.format(k + 1)].value = z_guess_fit[k, i, j]
                            # parameters['z_{}'.format(k + 1)].max = z_guess_fit[k, i, j] + 0.002
                            # parameters['z_{}'.format(k + 1)].min = z_guess_fit[k, i, j] - 0.002
                            # parameters['sigma_kms_{}'.format(k + 1)].value = sigma_guess_fit[k, i, j]
                            # parameters['sigma_kms_{}'.format(k + 1)].value = sigma_kms_guess_array[i, j] / 2
                            # parameters['sigma_kms_{}'.format(k + 1)].max = 0.7 * sigma_fit[0, i, j]

                            # Fluxes
                            parameters['flux_OII_{}'.format(k + 1)].value = flux_guess_array_OII[i, j]
                            parameters['flux_OIII5008_{}'.format(k + 1)].value = flux_guess_array_OIII[i, j]

                    #
                    spec_model = lmfit.Model(model, missing='drop')
                    result = spec_model.fit(flux_ij, wave_vac=wave_vac, params=parameters, weights=1 / flux_err_ij)
                    fit_success[i, j] = result.success
                    chisqr[i, j], redchi[i, j] = result.chisqr, result.redchi

                    if line == 'OII':
                        a, b = result.best_values['a'], result.best_values['b']
                        da, db = result.params['a'].stderr, result.params['b'].stderr
                        a_fit[i, j], b_fit[i, j] = a, b
                        da_fit[i, j], db_fit[i, j] = da, db

                        for k in range(fit_param['OII_2nd']):
                            z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                          result.params['flux_OII_{}'.format(k + 1)].stderr
                            v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                            sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                            flux_fit[k, i, j], dflux_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                            if fit_param['ResolveOII']:
                                r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                        result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k
                    elif line == 'OIII':
                        flux_f_OIII, dflux_f_OIII = result.best_values['flux_OIII5008'], result.params['flux_OIII5008'].stderr
                        a, b = result.best_values['a'], result.best_values['b']
                        da, db = result.params['a'].stderr, result.params['b'].stderr
                        flux_fit[i, j], dflux_fit[i, j] = flux_f_OIII, dflux_f_OIII
                        a_fit[i, j], b_fit[i, j] = a, b
                        da_fit[i, j], db_fit[i, j] = da, db
                    elif line == 'OII+OIII':
                        a_OII, b_OII = result.best_values['a_OII'], result.best_values['b_OII']
                        da_OII, db_OII = result.params['a_OII'].stderr, result.params['b_OII'].stderr
                        a_OIII, b_OIII = result.best_values['a_OIII5008'], result.best_values['b_OIII5008']
                        da_OIII, db_OIII = result.params['a_OIII5008'].stderr, result.params['b_OIII5008'].stderr
                        a_OII_fit[i, j], da_OII_fit[i, j], b_OII_fit[i, j], db_OII_fit[i, j] = a_OII, da_OII, b_OII, db_OII
                        a_OIII_fit[i, j], da_OIII_fit[i, j] = a_OIII, da_OIII
                        b_OIII_fit[i, j], db_OIII_fit[i, j] = b_OIII, db_OIII

                        for k in range(max_line):
                            z_k, dz_k = result.best_values['z_{}'.format(k + 1)], result.params['z_{}'.format(k + 1)].stderr
                            sigma_k, dsigma_k = result.best_values['sigma_kms_{}'.format(k + 1)], \
                                                result.params['sigma_kms_{}'.format(k + 1)].stderr
                            flux_f_OII_k, dflux_f_OII_k = result.best_values['flux_OII_{}'.format(k + 1)], \
                                                          result.params['flux_OII_{}'.format(k + 1)].stderr
                            v_fit[k, i, j], z_fit[k, i, j], dz_fit[k, i, j] = c_kms * (z_k - z_qso) / (1 + z_qso), z_k, dz_k
                            sigma_fit[k, i, j], dsigma_fit[k, i, j] = sigma_k, dsigma_k
                            flux_OII_fit[k, i, j], dflux_OII_fit[k, i, j] = flux_f_OII_k, dflux_f_OII_k

                            if fit_param['ResolveOII']:
                                r_k, dr_k = result.best_values['r_OII3729_3727_{}'.format(k + 1)], \
                                        result.params['r_OII3729_3727_{}'.format(k + 1)].stderr
                                r_fit[k, i, j], dr_fit[k, i, j] = r_k, dr_k

                            # OIII
                            flux_f_OIII_k, dflux_f_OIII_k = result.best_values['flux_OIII5008_{}'.format(k + 1)], \
                                                          result.params['flux_OIII5008_{}'.format(k + 1)].stderr
                            flux_OIII_fit[k, i, j], dflux_OIII_fit[k, i, j] = flux_f_OIII_k, dflux_f_OIII_k

                else:
                    pass

    #
    header = fits.open(path_cube)[1].header
    header['WCSAXES'] = 2
    header.remove('CTYPE3')
    header.remove('CUNIT3')
    header.remove('CD3_3')
    header.remove('CRPIX3')
    header.remove('CRVAL3')
    header.remove('CD1_3')
    header.remove('CD2_3')
    header.remove('CD3_1')
    header.remove('CD3_2')
    try:
        header.remove('CRDER3')
    except KeyError:
        pass
    hdul_pri = fits.open(path_cube)[0]
    hdul_fs, hdul_v = fits.ImageHDU(fit_success, header=header), fits.ImageHDU(v_fit, header=header)
    hdul_z, hdul_dz = fits.ImageHDU(z_fit, header=header), fits.ImageHDU(dz_fit, header=header)
    hdul_sigma, hdul_dsigma = fits.ImageHDU(sigma_fit, header=header), fits.ImageHDU(dsigma_fit, header=header)
    hdul_chisqr, hudl_redchi = fits.ImageHDU(chisqr, header=header), fits.ImageHDU(redchi, header=header)
    hdul_chisqr_2, hudl_redchi_2 = fits.ImageHDU(chisqr_2, header=header), fits.ImageHDU(redchi_2, header=header)

    #
    if line == 'OII+OIII':
        hdul_flux_OII, hdul_dflux_OII = fits.ImageHDU(flux_OII_fit, header=header), \
                                        fits.ImageHDU(dflux_OII_fit, header=header)
        hdul_flux_OIII, hdul_dflux_OIII = fits.ImageHDU(flux_OIII_fit, header=header), \
                                          fits.ImageHDU(dflux_OIII_fit, header=header)
        hdul_r, hdul_dr = fits.ImageHDU(r_fit, header=header), fits.ImageHDU(dr_fit, header=header)
        hdul_a_OII, hdul_da_OII = fits.ImageHDU(a_OII_fit, header=header), fits.ImageHDU(da_OII_fit, header=header)
        hdul_b_OII, hdul_db_OII = fits.ImageHDU(b_OII_fit, header=header), fits.ImageHDU(db_OII_fit, header=header)
        hdul_a_OIII, hdul_da_OIII = fits.ImageHDU(a_OIII_fit, header=header), fits.ImageHDU(da_OIII_fit, header=header)
        hdul_b_OIII, hdul_db_OIII = fits.ImageHDU(b_OIII_fit, header=header), fits.ImageHDU(db_OIII_fit, header=header)
        hdul = fits.HDUList([hdul_pri, hdul_fs, hdul_v, hdul_z, hdul_dz, hdul_sigma, hdul_dsigma, hdul_flux_OII,
                             hdul_dflux_OII, hdul_flux_OIII, hdul_dflux_OIII, hdul_r, hdul_dr, hdul_a_OII,
                             hdul_da_OII, hdul_b_OII, hdul_db_OII, hdul_a_OIII, hdul_da_OIII, hdul_b_OIII,
                             hdul_db_OIII, hdul_chisqr, hudl_redchi, hdul_chisqr_2, hudl_redchi_2])

    else:
        hdul_flux, hdul_dflux = fits.ImageHDU(flux_fit, header=header), fits.ImageHDU(dflux_fit, header=header)
        hdul_r, hdul_dr = fits.ImageHDU(r_fit, header=header), fits.ImageHDU(dr_fit, header=header)
        hdul_a, hdul_da = fits.ImageHDU(a_fit, header=header), fits.ImageHDU(da_fit, header=header)
        hdul_b, hdul_db = fits.ImageHDU(b_fit, header=header), fits.ImageHDU(db_fit, header=header)
        hdul_chisqr, hudl_redchi = fits.ImageHDU(chisqr, header=header), fits.ImageHDU(redchi, header=header)
        hdul_chisqr_2, hudl_redchi_2 = fits.ImageHDU(chisqr_2, header=header), fits.ImageHDU(redchi_2, header=header)
        hdul = fits.HDUList([hdul_pri, hdul_fs, hdul_v, hdul_z, hdul_dz, hdul_sigma, hdul_dsigma, hdul_flux,
                             hdul_dflux, hdul_r, hdul_dr, hdul_a, hdul_da, hdul_b, hdul_db, hdul_chisqr,
                             hudl_redchi, hdul_chisqr_2, hudl_redchi_2])
    hdul.writeto(path_fit, overwrite=True)


def PlotKinematics(cubename=None, zapped=False, fit_param=None, UseDataSeg=(1.5, 'gauss', None, None),
                   CheckSpectra=[50, 50], S_N_thr=5, v_min=-600, v_max=600, sigma_max=400, contour_level=0.15,
                   SelectNebulae=None, width_OII=10, width_OIII=10, UseSmoothedCubes=True,
                   FixAstrometry=False, UseDetectionSeg=None, CheckSpectraSeg=True, CheckSpectra_2=False):
    # Define line
    if fit_param['OII'] >= 1 and fit_param['OIII'] == 0:
        line = 'OII'
    elif fit_param['OII'] == 0 and fit_param['OIII'] >= 1:
        line = 'OIII'
    else:
        line = 'OII+OIII'

    # QSO information
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Galaxies infomation
    path_gal = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/{}_gal_info.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        ra_gal, dec_gal, v_gal = data_gal['ra'], data_gal['dec'], data_gal['v']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal = [], [], []

    # Zapping
    if zapped:
        str_zap = '_zapped'
    else:
        str_zap = ''

    path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}.fits'.\
        format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
    path_fit_N1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}_N1.fits'.\
        format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
    path_v = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_v_{}_{}_{}_{}_{}_{}_{}.fits'.\
        format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
    figurename_v = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_v_{}_{}_{}_{}_{}_{}_{}.png'.\
        format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
    path_sigma = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_sigma_{}_{}_{}_{}_{}_{}_{}.fits'.\
        format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
    figurename_sigma = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_sigma_{}_{}_{}_{}_{}_{}_{}.png'.\
        format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
    figurename = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_{}_{}_{}_{}_{}_checkspectra.png'.\
        format(cubename, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)

    hdul = fits.open(path_fit)
    # hdul[2].header['WCSAXES'] = 2
    fs, hdr = hdul[1].data, hdul[2].header
    v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
    sigma, dsigma = hdul[5].data, hdul[6].data
    v_plot = np.copy(v)
    sigma_plot = np.copy(sigma)

    # Unpack correct one
    if fit_param['OII_2nd'] > 1 or fit_param['OIII_2nd'] > 1:
        # hdul_N1 = fits.open(path_fit_N1)
        # v_N1 = hdul_N1[2].data
        # redchi_N1 = hdul_N1[22].data
        # plt.figure()
        # plt.imshow(redchi_N1, origin='lower')
        # plt.show()
        # plt.close()
        # refit_seg = np.where((redchi_N1 > 1.0), redchi_N1, 0)

        sigma_0, sigma_1 = sigma[0], sigma[1]
        v_0, v_1 = v[0], v[1]
        z_0, z_1 = z[0], z[1]

        #
        z_0_2nd = np.where(v_1 != 0, z_0, 0)
        v_0_2nd = np.where(v_1 != 0, v_0, 0)
        sigma_0_2nd = np.where(v_1 != 0, sigma_0, 0)

        #
        v_1_2nd = np.where(z_1 > z_0_2nd, v_1, v_0_2nd)
        v_0_2nd = np.where(z_0_2nd <= z_1, v_0_2nd, v_1)
        sigma_1_2nd = np.where(z_1 > z_0_2nd, sigma_1, sigma_0_2nd)
        sigma_0_2nd = np.where(z_0_2nd <= z_1, sigma_0_2nd, sigma_1)

        #
        v_0_2nd = np.where(v_0_2nd != 0, v_0_2nd, v_0)
        v_0_2nd = np.where(v_0_2nd != 0, v_0_2nd, np.nan)
        v_1_2nd = np.where(v_1_2nd != 0, v_1_2nd, np.nan)
        sigma_0_2nd = np.where(sigma_0_2nd != 0, sigma_0_2nd, sigma_0)
        sigma_0_2nd = np.where(sigma_0_2nd != 0, sigma_0_2nd, np.nan)
        sigma_1_2nd = np.where(sigma_1_2nd != 0, sigma_1_2nd, np.nan)
        v_plot[0, :, :] = v_0_2nd
        v_plot[1, :, :] = v_1_2nd
        # v_plot[0, :, :] = np.where((redchi_N1 > 1.0), v_0_2nd, v_N1)
        # v_plot[1, :, :] = np.where((redchi_N1 > 1.0), v_1_2nd, np.nan)

        sigma_plot[0, :, :] = sigma_0_2nd
        sigma_plot[1, :, :] = sigma_1_2nd

    # Load cube and 3D seg
    if UseDetectionSeg is not None:
        UseSeg = UseDetectionSeg
    else:
        UseSeg = UseDataSeg

    if line == 'OII+OIII':
        line_OII, line_OIII = 'OII', 'OIII'
        path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)
        path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        path_SB_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)
        path_SB_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        path_SB_OII_kin = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line_OII, *UseSeg)
        figurename_SB_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'.\
            format(cubename, str_zap, line_OII, *UseSeg)
        path_SB_OIII_kin = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line_OIII, *UseSeg)
        figurename_SB_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'.\
            format(cubename, str_zap, line_OIII, *UseSeg)


        # 3D seg
        seg_3D_OII, seg_label_OII = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OII)[1].data
        seg_3D_OIII, seg_label_OIII = fits.open(path_3Dseg_OIII)[0].data, fits.open(path_3Dseg_OIII)[1].data
        seg_label = seg_label_OII + seg_label_OIII
        flux_OII_fit, dflux_OII_fit = hdul[7].data, hdul[8].data
        flux_OIII_fit, dflux_OIII_fit = hdul[9].data, hdul[10].data
        r, dr = hdul[11].data, hdul[12].data
        a_OII, da_OII = hdul[13].data, hdul[14].data
        a_OIII, da_OIII = hdul[17].data, hdul[18].data
        b_OII, db_OII = hdul[15].data, hdul[16].data
        b_OIII, db_OIII = hdul[19].data, hdul[20].data
        S_N = flux_OIII_fit / dflux_OIII_fit

    else:
        path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, *UseSeg)
        path_SB = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, *UseSeg)
        path_SB_kin = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, *UseSeg)
        figurename_SB = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'.\
            format(cubename, str_zap, line, *UseSeg)

        # 3D seg
        seg_3D, seg_label = fits.open(path_3Dseg)[0].data, fits.open(path_3Dseg)[1].data
        flux_OII_fit, dflux_OII_fit = hdul[7].data[0], hdul[8].data[0]
        r, dr = hdul[9].data[0], hdul[10].data[0]
        a, b = hdul[11].data, hdul[13].data
        S_N = flux_OII_fit / dflux_OII_fit

        #
        if SelectNebulae is not None:
            mask_select = np.zeros_like(v)
            for i in SelectNebulae:
                mask_select = np.where(seg_label != i, mask_select, 1)
            v = np.where(mask_select == 1, v, np.nan)
            sigma = np.where(mask_select == 1, sigma, np.nan)
        v_plot = np.copy(v)
        sigma_plot = np.copy(sigma)

    # print(np.std(a_OIII), np.nanmax(b_OIII))
    # Masking
    v_plot = np.where((fs * seg_label)[np.newaxis, :] != 0, v_plot, np.nan)
    # v = np.where(S_N > S_N_thr, v, np.nan)
    sigma_plot = np.where((fs * seg_label)[np.newaxis, :] != 0, sigma_plot, np.nan)
    # sigma = np.where(S_N > S_N_thr, sigma, np.nan)

    # Fix Astrometry
    if FixAstrometry:
        path_gal = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
        try:
            data_gal = fits.open(path_gal)[1].data
            ra_gal, dec_gal, v_gal = data_gal['ra'], data_gal['dec'], data_gal['v']
        except FileNotFoundError:
            print('No galaxies info')
            ra_gal, dec_gal, v_gal = [], [], []

        # Will be replaced by a table
        path_subcube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/subcubes.dat'
        data_subcube = ascii.read(path_subcube, format='fixed_width')
        data_subcube = data_subcube[data_subcube['name'] == cubename]
        ra_muse, dec_muse, radius = data_subcube['ra_center'][0], data_subcube['dec_center'][0], data_subcube['radius'][0]
        c_muse = SkyCoord(ra=ra_muse * u.degree, dec=dec_muse * u.degree, frame='icrs')
        path_muse_white = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_WHITE.fits'.\
            format(cubename)
        path_muse_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.\
            format(cubename)
        hdul_muse_white = fits.open(path_muse_white)
        hdul_muse_white_gaia = fits.open(path_muse_white_gaia)
        if cubename == 'HE0153-4520' or cubename == '3C57':
            w = WCS(hdul_muse_white[0].header, naxis=2)
        else:
            w = WCS(hdul_muse_white[1].header, naxis=2)
        w_gaia = WCS(hdul_muse_white_gaia[1].header, naxis=2)
        x, y = w.world_to_pixel(c_muse)
        c_muse_gaia = w_gaia.pixel_to_world(x, y)
        muse_white_gaia = Image(path_muse_white_gaia)
        sub_muse_white_gaia = muse_white_gaia.subimage(center=(c_muse_gaia.dec.value, c_muse_gaia.ra.value), size=30)
        path_sub_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
        sub_muse_white_gaia.write(path_sub_white_gaia)
        hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
        hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
        hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
        hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
        hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
        hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
        hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
        hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
        hdr['CD2_2'] = hdr_sub_gaia['CD2_2']
        # hdr.remove('BUNIT')

    #
    hdul_v = fits.ImageHDU(v_plot[0], header=hdr)
    hdul_v.writeto(path_v, overwrite=True)
    # hdul_v_2nd = fits.ImageHDU(v_plot[1], header=hdr)
    # hdul_v_2nd.writeto(path_v_2nd, overwrite=True)

    hdul_sigma = fits.ImageHDU(sigma_plot[0], header=hdr)
    hdul_sigma.writeto(path_sigma, overwrite=True)


    # SB map
    # Be careful with north=True sometimes it accidentally smoothes the data
    if line == 'OII+OIII':
        hdul_SB_OII_kin = fits.ImageHDU(fits.open(path_SB_OII)[1].data, header=hdr)
        hdul_SB_OII_kin.writeto(path_SB_OII_kin, overwrite=True)
        hdul_SB_OIII_kin = fits.ImageHDU(fits.open(path_SB_OIII)[1].data, header=hdr)
        hdul_SB_OIII_kin.writeto(path_SB_OIII_kin, overwrite=True)

        if cubename == 'HE0238-1904':
            ra_qso, dec_qso = 40.13564948691202, -18.864301804042814
            path_SB_OII_kin = '/Users/lzq/Dropbox/Data/CGM/cube_narrow/cube_OII_line_offset_SB_3DSeg_1.5_gauss_1.5_gauss.fits'

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB_OII_kin, figure=fig, hdu=1)
        if cubename == 'HE0238-1904':
            gc.recenter(40.1344150, -18.8656933, width=30 / 3600, height=30 / 3600)
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_SB_OII_kin, levels=[contour_level], colors='black', linewidths=2,
                        smooth=5, kernel='box', hdu=1)
        APLpyStyle(gc, type='NarrowBand', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
        gc.add_label(0.97, 0.92, r'$\rm [O\,II]$', color='black', size=30, relative=True, horizontalalignment='right')
        gc.add_label(0.08, 0.08, '(b)', color='k', size=40, relative=True)
        fig.savefig(figurename_SB_OII, bbox_inches='tight')

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB_OIII_kin, figure=fig, hdu=1)
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_SB_OIII_kin, levels=[contour_level], colors='black', linewidths=2,
                        smooth=5, kernel='box', hdu=1)
        APLpyStyle(gc, type='NarrowBand', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
        gc.add_label(0.97, 0.92, r'$\rm [O\,III]$', color='black', size=30, relative=True, horizontalalignment='right')
        gc.add_label(0.08, 0.08, '(c)', color='k', size=40, relative=True)
        fig.savefig(figurename_SB_OIII, bbox_inches='tight')
    else:
        hdul_SB_kin = fits.ImageHDU(fits.open(path_SB)[1].data, header=hdr)
        hdul_SB_kin.writeto(path_SB_kin, overwrite=True)
        #
        # plt.imshow(fits.open(path_SB_kin)[1].data, vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), origin='lower')
        # plt.show()
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB_kin, figure=fig, hdu=1)
        if cubename == 'TEX0206-048':
            ra_qso, dec_qso = 32.3784012, -4.6405697
            cubename_cor = 'TXS0206-048'
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_SB_kin, levels=[contour_level], colors='black', linewidths=2, smooth=5, kernel='box', hdu=1)
        APLpyStyle(gc, type='NarrowBand', cubename=cubename_cor, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
        fig.savefig(figurename_SB, bbox_inches='tight')

    raise ValueError('stop here')
    # LOS velocity
    path_V50 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_V50.fits'
    path_W80 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_W80.fits'
    fig = plt.figure(figsize=(8, 8), dpi=300)
    # plt.imshow(fits.open(path_v)[1].data, vmin=v_min, vmax=v_max, cmap=plt.get_cmap('coolwarm'), origin='lower')
    gc = aplpy.FITSFigure(path_v, figure=fig, hdu=1)
    if fit_param['OII_2nd'] > 1 or fit_param['OIII_2nd'] > 1:
        axins = gc.ax.inset_axes([0.0, 0.0, 0.33, 0.33], xticklabels=[], yticklabels=[], xlim=(50, 100), ylim=(50, 100),
                                 yticks=[], xticks=[])
        axins.imshow(v_plot[1], origin='lower', cmap='coolwarm', vmin=v_min, vmax=v_max)
    gc.show_colorscale(vmin=v_min, vmax=v_max, cmap='coolwarm')
    APLpyStyle(gc, type='GasMap', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    gc.show_markers(ra_gal, dec_gal, facecolor='white', marker='o', c='white', edgecolors='none', linewidths=0.8, s=100)
    gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra_gal, dec_gal, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=v_min, vmax=v_max,
                    cmap='coolwarm')
    fig.savefig(figurename_v, bbox_inches='tight')

    # Sigma map
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_sigma, figure=fig, hdu=1)
    if fit_param['OII_2nd'] > 1 or fit_param['OIII_2nd'] > 1:
        axins = gc.ax.inset_axes([0.0, 0.0, 0.33, 0.33], xticklabels=[], yticklabels=[], xlim=(50, 100), ylim=(50, 100),
                                 yticks=[], xticks=[])
        axins.imshow(sigma_plot[1], origin='lower', cmap=Dense_20_r.mpl_colormap, vmin=0, vmax=sigma_max)
    gc.show_colorscale(vmin=0, vmax=sigma_max, cmap=Dense_20_r.mpl_colormap)
    # gc.show_colorscale(vmin=0, vmax=800, cmap=Dense_20_r.mpl_colormap)
    APLpyStyle(gc, type='GasMap_sigma', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    fig.savefig(figurename_sigma, bbox_inches='tight')

    if CheckSpectra is not None:
        if line == 'OII+OIII':
            #
            line_OII, line_OIII = 'OII', 'OIII'
            path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OII)
            path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                     '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
            path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OIII)
            path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                      '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDataSeg)

            # Load data and smoothing
            if UseSmoothedCubes:
                cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
            else:
                cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
            wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(
                cube_OIII.wave.coord())
            flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
            flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
            seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
            mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
            flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori

            # Extend over
            start_OII = (seg_3D_OII_ori != 0).argmax(axis=0)
            end_OII = start_OII + mask_seg_OII
            start_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), start_OII, start_OII - width_OII)
            end_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), end_OII, end_OII + width_OII)
            idx_OII = np.zeros_like(seg_3D_OII_ori)
            idx_OII[:] = np.arange(np.shape(seg_3D_OII_ori)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OII = np.where((idx_OII >= end_OII[np.newaxis, :, :]) | (idx_OII < start_OII[np.newaxis, :, :]),
                                  seg_3D_OII_ori, 1)

            # [O III]
            start_OIII = (seg_3D_OIII_ori != 0).argmax(axis=0)
            end_OIII = start_OIII + mask_seg_OIII
            start_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), start_OIII, start_OIII - width_OIII)
            end_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), end_OIII, end_OIII + width_OIII)
            idx_OIII = np.zeros_like(seg_3D_OIII_ori)
            idx_OIII[:] = np.arange(np.shape(seg_3D_OIII_ori)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OIII = np.where((idx_OIII >= end_OIII[np.newaxis, :, :]) | (idx_OIII < start_OIII[np.newaxis, :, :]),
                                   seg_3D_OIII_ori, 1)
            flux_OII, flux_err_OII = flux_OII * seg_3D_OII, flux_err_OII * seg_3D_OII
            flux_OIII, flux_err_OIII = flux_OIII * seg_3D_OIII, flux_err_OIII * seg_3D_OIII
            flux_err_OII = np.where(flux_err_OII != 0, flux_err_OII, np.inf)
            flux_err_OIII = np.where(flux_err_OIII != 0, flux_err_OIII, np.inf)


            #
            fig_1, ax_1 = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            fig_2, ax_2 = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_OII[:, i_j, j_j] -
                                          wave_OII_vac * a_OII[i_j, j_j] - b_OII[i_j, j_j], '-k')
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_err_OII[:, i_j, j_j], '-C0')
                    if v_1[i_j, j_j] != 0:
                        if fit_param['ResolveOII']:
                            ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                  model_OII(wave_OII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                            flux_OII_fit[0, i_j, j_j], r[0, i_j, j_j]) +
                                                  model_OII(wave_OII_vac, z[1, i_j, j_j], sigma[1, i_j, j_j],
                                                            flux_OII_fit[1, i_j, j_j], r[1, i_j, j_j]), '-r')
                        else:
                            ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                  Gaussian(wave_OII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                           flux_OII_fit[0, i_j, j_j],
                                                           (wave_OII3727_vac + wave_OII3729_vac) / 2) +
                                                  Gaussian(wave_OII_vac, z[1, i_j, j_j], sigma[1, i_j, j_j],
                                                           flux_OII_fit[1, i_j, j_j],
                                                           (wave_OII3727_vac + wave_OII3729_vac) / 2), '-r')
                        for k in range(fit_param['OII_2nd']):
                            if fit_param['ResolveOII']:
                                color = ['blue', 'purple', 'violet'][k]
                                ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                      model_OII(wave_OII_vac, z[k, i_j, j_j], sigma[k, i_j, j_j],
                                                                flux_OII_fit[k, i_j, j_j], r[k, i_j, j_j]),
                                                      '-', color=color)
                                ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                      model_OII(wave_OII_vac, z[k, i_j, j_j], sigma[k, i_j, j_j],
                                                                flux_OII_fit[k, i_j, j_j], r[k, i_j, j_j],
                                                                plot=True)[0],
                                                      '--', color=color)
                                ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                      model_OII(wave_OII_vac, z[k, i_j, j_j], sigma[k, i_j, j_j],
                                                                flux_OII_fit[k, i_j, j_j], r[k, i_j, j_j],
                                                                plot=True)[1],
                                                      '--', color=color)
                            else:
                                ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                      Gaussian(wave_OII_vac, z[k, i_j, j_j], sigma[k, i_j, j_j],
                                                               flux_OII_fit[k, i_j, j_j],
                                                               fit_param['OII_center']),
                                                      '-', color=color)
                    else:
                        if fit_param['ResolveOII']:
                            ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                  model_OII(wave_OII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                            flux_OII_fit[0, i_j, j_j], r[0, i_j, j_j]),
                                                  '-r')
                            ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                  model_OII(wave_OII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                            flux_OII_fit[0, i_j, j_j], r[0, i_j, j_j],
                                                            plot=True)[0],
                                                  '--r')
                            ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                  model_OII(wave_OII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                            flux_OII_fit[0, i_j, j_j], r[0, i_j, j_j],
                                                            plot=True)[1],
                                                  '--r')
                        else:
                            ax_1[ax_i, ax_j].plot(wave_OII_vac,
                                                  Gaussian(wave_OII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                           flux_OII_fit[0, i_j, j_j],
                                                           (wave_OII3727_vac + wave_OII3729_vac) / 2), '-r')
                    ax_1[ax_i, ax_j].set_xlim(wave_OII_vac.min(), wave_OII_vac.max())
                    # ax_1[ax_i, ax_j].axvline(x=wave_OII3727_vac * (1 + z[i_j, j_j]), color='r')
                    # ax_1[ax_i, ax_j].axvline(x=wave_OII3729_vac * (1 + z[i_j, j_j]), color='r')
                    ax_1[ax_i, ax_j].axvline(x=wave_OII3727_vac * (1 + z_qso), color='C0')
                    ax_1[ax_i, ax_j].axvline(x=wave_OII3729_vac * (1 + z_qso), color='C0')
                    ax_1[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                               + '\n' + 'v=' + str(np.round(v[0, i_j, j_j], 2))
                                               + '\n' + 'sigma=' + str(np.round(sigma[0, i_j, j_j], 2))
                                               + '\n' + 'r=' + str(np.round(r[0, i_j, j_j], 2)), y=0.7, x=0.2)

                    # OIII
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_OIII[:, i_j, j_j], '-k')
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_err_OIII[:, i_j, j_j], '-C0')
                    # flux_OIII_all =  Gaussian(wave_OIII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                    #                           flux_OIII_fit[0, i_j, j_j], wave_OIII5008_vac) +
                    #                  Gaussian(wave_OIII_vac, z[1, i_j, j_j], sigma[1, i_j, j_j],
                    #                           flux_OIII_fit[1, i_j, j_j], wave_OIII5008_vac) +
                    #                  Gaussian(wave_OIII_vac, z[2, i_j, j_j], sigma[2, i_j, j_j],
                    #                           flux_OIII_fit[2, i_j, j_j], wave_OIII5008_vac)
                    flux_OIII_all = 0
                    if v_1[i_j, j_j] != 0:
                        for k in range(np.max([fit_param['OII'], fit_param['OII_2nd']])):
                            color = ['blue', 'purple', 'violet'][k]
                            ax_2[ax_i, ax_j].plot(wave_OIII_vac, Gaussian(wave_OIII_vac, z[k, i_j, j_j],
                                                                          sigma[k, i_j, j_j], flux_OIII_fit[k, i_j, j_j],
                                                                          wave_OIII5008_vac), '--', color=color)
                            flux_OIII_all += Gaussian(wave_OIII_vac, z[k, i_j, j_j], sigma[k, i_j, j_j],
                                                      flux_OIII_fit[k, i_j, j_j], wave_OIII5008_vac)
                        ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_OIII_all + wave_OIII_vac * a_OIII[i_j, j_j]
                                              + b_OIII[i_j, j_j], '-r')
                    else:
                        ax_2[ax_i, ax_j].plot(wave_OIII_vac,
                                              Gaussian(wave_OIII_vac, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                                       flux_OIII_fit[0, i_j, j_j], wave_OIII5008_vac), '-r')
                    ax_2[ax_i, ax_j].set_xlim(wave_OIII_vac.min(), wave_OIII_vac.max())
                    # ax_2[ax_i, ax_j].axvline(x=wave_OIII5008_vac * (1 + z[i_j, j_j]), color='r')
                    ax_2[ax_i, ax_j].axvline(x=wave_OIII5008_vac * (1 + z_qso), color='C0')
                    ax_2[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v[0, i_j, j_j], 2)) +
                                               '\n' + 'sigma=' + str(np.round(sigma[0, i_j, j_j], 2)), y=0.7, x=0.2)
            figname_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_checkspectra.png'.format(cubename,
                                                                                                           line, line_OII)
            figname_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_checkspectra.png'.format(cubename,
                                                                                                            line, line_OIII)
            fig_1.savefig(figname_OII, bbox_inches='tight')
            fig_2.savefig(figname_OIII, bbox_inches='tight')

        else:
            path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line)
            path_cube_smoothed = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                                 '{}_{}.fits'.format(cubename, str_zap, line, *UseDataSeg)
            path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line, *UseDataSeg)

            # Load data and smoothing
            if UseSmoothedCubes:
                cube = Cube(path_cube_smoothed)
            else:
                cube = Cube(path_cube)
            wave_vac = pyasl.airtovac2(cube.wave.coord())
            flux, flux_err = cube.data * 1e-3, np.sqrt(cube.var) * 1e-3
            seg_3D = fits.open(path_3Dseg)[0].data
            mask_seg = np.sum(seg_3D, axis=0)
            flux_seg = flux * seg_3D
            start = (seg_3D != 0).argmax(axis=0)
            end = start + mask_seg
            if line == 'OII':
                width = width_OII
            elif line == 'OIII':
                width = width_OIII
            start = np.where((mask_seg > 20) | (mask_seg < 1), start, start - width)
            end = np.where((mask_seg > 20) | (mask_seg < 1), end, end + width)
            idx = np.zeros_like(seg_3D)
            idx[:] = np.arange(np.shape(seg_3D)[0])[:, np.newaxis, np.newaxis]
            seg_3D = np.where((idx >= end[np.newaxis, :, :]) | (idx < start[np.newaxis, :, :]), seg_3D, 1)
            flux *= seg_3D
            flux_err *= seg_3D
            flux_err = np.where(flux_err != 0, flux_err, np.inf)

            fig, ax = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax[ax_i, ax_j].plot(wave_vac, flux[:, i_j, j_j], '-k')
                    # ax[ax_i, ax_j].plot(wave_vac, flux_seg[:, i_j, j_j], '-b')
                    ax[ax_i, ax_j].plot(wave_vac, flux_err[:, i_j, j_j], '-C0')
                    if fit_param['ResolveOII']:
                        ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j], r[i_j, j_j]), '-r')
                        ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j], r[i_j, j_j],
                                                                plot=True)[0], '--r')
                        ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j], r[i_j, j_j],
                                                                plot=True)[1], '--r')
                    else:
                        ax[ax_i, ax_j].plot(wave_vac, Gaussian(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j],
                                                               (wave_OII3727_vac + wave_OII3729_vac) / 2), '-r')
                    # ax[ax_i, ax_j].set_ylim(top=0.01)
                    ax[ax_i, ax_j].set_xlim(wave_vac.min(), wave_vac.max())
                    ax[ax_i, ax_j].axvline(x=3727.092 * (1 + z[i_j, j_j]), color='r')
                    ax[ax_i, ax_j].axvline(x=3729.875 * (1 + z[i_j, j_j]), color='r')
                    ax[ax_i, ax_j].axvline(x=3727.092 * (1 + z_qso), color='C0')
                    ax[ax_i, ax_j].axvline(x=3729.875 * (1 + z_qso), color='C0')
                    ax[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                    + '\n' + 'v=' + str(np.round(v[i_j, j_j], 2)), y=0.9, x=0.2)
            plt.savefig(figurename, bbox_inches='tight')
    if CheckSpectra_2 is not None:
        if line == 'OII+OIII':
            #
            line_OII, line_OIII = 'OII', 'OIII'
            path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OII)
            path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                     '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
            path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OIII)
            path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                      '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDataSeg)

            # Load data and smoothing
            if UseSmoothedCubes:
                cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
            else:
                cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
            wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(
                cube_OIII.wave.coord())
            flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
            flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
            seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
            mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
            flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori

            # Extend over
            start_OII = (seg_3D_OII_ori != 0).argmax(axis=0)
            end_OII = start_OII + mask_seg_OII
            start_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), start_OII, start_OII - width_OII)
            end_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), end_OII, end_OII + width_OII)
            idx_OII = np.zeros_like(seg_3D_OII_ori)
            idx_OII[:] = np.arange(np.shape(seg_3D_OII_ori)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OII = np.where((idx_OII >= end_OII[np.newaxis, :, :]) | (idx_OII < start_OII[np.newaxis, :, :]),
                                  seg_3D_OII_ori, 1)

            # [O III]
            start_OIII = (seg_3D_OIII_ori != 0).argmax(axis=0)
            end_OIII = start_OIII + mask_seg_OIII
            start_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), start_OIII, start_OIII - width_OIII)
            end_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), end_OIII, end_OIII + width_OIII)
            idx_OIII = np.zeros_like(seg_3D_OIII_ori)
            idx_OIII[:] = np.arange(np.shape(seg_3D_OIII_ori)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OIII = np.where((idx_OIII >= end_OIII[np.newaxis, :, :]) | (idx_OIII < start_OIII[np.newaxis, :, :]),
                                   seg_3D_OIII_ori, 1)
            flux_OII, flux_err_OII = flux_OII * seg_3D_OII, flux_err_OII * seg_3D_OII
            flux_OIII, flux_err_OIII = flux_OIII * seg_3D_OIII, flux_err_OIII * seg_3D_OIII
            flux_err_OII = np.where(flux_err_OII != 0, flux_err_OII, np.inf)
            flux_err_OIII = np.where(flux_err_OIII != 0, flux_err_OIII, np.inf)


             # OIII
            # coord = [(65, 81), (69, 81), (80, 75)]
            coord = [(65, 81), (75, 88), (80, 75)]
            v_OIII = c_kms * (wave_OIII_vac - wave_OIII5008_vac * (1 + z_qso)) / ((1 + z_qso) * wave_OIII5008_vac)
            wave_OIII_vac_exp = expand_wave([wave_OIII_vac], stack=True)
            v_OIII_exp = c_kms * (wave_OIII_vac_exp - wave_OIII5008_vac * (1 + z_qso)) / ((1 + z_qso) * wave_OIII5008_vac)
            rc('xtick.minor', size=3, visible=True)
            rc('ytick.minor', size=3, visible=True)
            rc('xtick', direction='in', labelsize=15)
            rc('ytick', direction='in', labelsize=15)
            for i_ax in range(3):
                i_j, j_j = coord[i_ax][1], coord[i_ax][0]
                fig_i, ax_i = plt.subplots(1, 1, figsize=(5, 5), sharex=True, dpi=300)
                ax_i.plot(v_OIII, flux_OIII[:, i_j, j_j] - wave_OIII_vac * a_OIII[i_j, j_j] - b_OIII[i_j, j_j], '-k')
                ax_i.plot(v_OIII, flux_err_OIII[:, i_j, j_j], '-C0')
                ax_i.plot(v_OIII_exp, Gaussian(wave_OIII_vac_exp, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                               flux_OIII_fit[0, i_j, j_j], wave_OIII5008_vac) +
                          Gaussian(wave_OIII_vac_exp, z[1, i_j, j_j], sigma[1, i_j, j_j],
                                   flux_OIII_fit[1, i_j, j_j], wave_OIII5008_vac), '-r')
                ax_i.plot(v_OIII_exp, Gaussian(wave_OIII_vac_exp, z[0, i_j, j_j], sigma[0, i_j, j_j],
                                               flux_OIII_fit[0, i_j, j_j], wave_OIII5008_vac), '--', color='blue')
                ax_i.plot(v_OIII_exp, Gaussian(wave_OIII_vac_exp, z[1, i_j, j_j], sigma[1, i_j, j_j],
                                               flux_OIII_fit[1, i_j, j_j], wave_OIII5008_vac), '--', color='purple')
                ax_i.set_xlim(v_OIII.min(), v_OIII.max())
                ax_i.axvline(x=0, color='grey', linestyle='--', zorder=-100)
                # ax_i.minorticks_on()
                ax_i.set_title(r'$\mathrm{[O\,III]}$', x=0.2, y=0.85, size=40)
                ax_i.set_xlabel(r'$\mathrm{Velocity \; [km \, s^{-1}]}$', size=40)
                ax_i.set_ylabel(r'$\mathrm{Flux}$', size=40)
                ax_i.set_xticks([-500, 0, 500])
                ax_i.tick_params(axis='both', which='major', labelsize=40)
                # ax_i.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$',
                #                 size=20, x=0.03)
                figname_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_thesis_{}.png'.\
                    format(cubename, line, line_OIII, i_ax)
                fig_i.savefig(figname_OIII, bbox_inches='tight')

        else:
            path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line)
            path_cube_smoothed = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                                 '{}_{}.fits'.format(cubename, str_zap, line, *UseDataSeg)
            path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line, *UseDataSeg)

            # Load data and smoothing
            if UseSmoothedCubes:
                cube = Cube(path_cube_smoothed)
            else:
                cube = Cube(path_cube)
            wave_vac = pyasl.airtovac2(cube.wave.coord())
            flux, flux_err = cube.data * 1e-3, np.sqrt(cube.var) * 1e-3
            seg_3D = fits.open(path_3Dseg)[0].data
            mask_seg = np.sum(seg_3D, axis=0)
            flux_seg = flux * seg_3D
            start = (seg_3D != 0).argmax(axis=0)
            end = start + mask_seg
            if line == 'OII':
                width = width_OII
            elif line == 'OIII':
                width = width_OIII
            start = np.where((mask_seg > 20) | (mask_seg < 1), start, start - width)
            end = np.where((mask_seg > 20) | (mask_seg < 1), end, end + width)
            idx = np.zeros_like(seg_3D)
            idx[:] = np.arange(np.shape(seg_3D)[0])[:, np.newaxis, np.newaxis]
            seg_3D = np.where((idx >= end[np.newaxis, :, :]) | (idx < start[np.newaxis, :, :]), seg_3D, 1)
            flux *= seg_3D
            flux_err *= seg_3D
            flux_err = np.where(flux_err != 0, flux_err, np.inf)

            fig, ax = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax[ax_i, ax_j].plot(wave_vac, flux[:, i_j, j_j], '-k')
                    # ax[ax_i, ax_j].plot(wave_vac, flux_seg[:, i_j, j_j], '-b')
                    ax[ax_i, ax_j].plot(wave_vac, flux_err[:, i_j, j_j], '-C0')
                    if fit_param['ResolveOII']:
                        ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j], r[i_j, j_j]), '-r')
                        ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j], r[i_j, j_j],
                                                                plot=True)[0], '--r')
                        ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j], r[i_j, j_j],
                                                                plot=True)[1], '--r')
                    else:
                        ax[ax_i, ax_j].plot(wave_vac, Gaussian(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                flux_OII_fit[i_j, j_j],
                                                               (wave_OII3727_vac + wave_OII3729_vac) / 2), '-r')
                    # ax[ax_i, ax_j].set_ylim(top=0.01)
                    ax[ax_i, ax_j].set_xlim(wave_vac.min(), wave_vac.max())
                    ax[ax_i, ax_j].axvline(x=3727.092 * (1 + z[i_j, j_j]), color='r')
                    ax[ax_i, ax_j].axvline(x=3729.875 * (1 + z[i_j, j_j]), color='r')
                    ax[ax_i, ax_j].axvline(x=3727.092 * (1 + z_qso), color='C0')
                    ax[ax_i, ax_j].axvline(x=3729.875 * (1 + z_qso), color='C0')
                    ax[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                    + '\n' + 'v=' + str(np.round(v[i_j, j_j], 2)), y=0.9, x=0.2)
            plt.savefig(figurename, bbox_inches='tight')

    # Regroup
    if CheckSpectraSeg:
        if line == 'OII+OIII':
            #
            line_OII, line_OIII = 'OII', 'OIII'
            path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OII)
            path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                     '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
            path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OIII)
            path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                      '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDataSeg)

            # Load data and smoothing
            if UseSmoothedCubes:
                cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
            else:
                cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
            wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(
                cube_OIII.wave.coord())
            flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
            flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
            seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
            mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
            flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori

            # Extend over
            start_OII = (seg_3D_OII_ori != 0).argmax(axis=0)
            end_OII = start_OII + mask_seg_OII
            start_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), start_OII, start_OII - width_OII)
            end_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), end_OII, end_OII + width_OII)
            idx_OII = np.zeros_like(seg_3D_OII_ori)
            idx_OII[:] = np.arange(np.shape(seg_3D_OII_ori)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OII = np.where((idx_OII >= end_OII[np.newaxis, :, :]) | (idx_OII < start_OII[np.newaxis, :, :]),
                                  seg_3D_OII_ori, 1)

            # [O III]
            start_OIII = (seg_3D_OIII_ori != 0).argmax(axis=0)
            end_OIII = start_OIII + mask_seg_OIII
            start_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), start_OIII, start_OIII - width_OIII)
            end_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), end_OIII, end_OIII + width_OIII)
            idx_OIII = np.zeros_like(seg_3D_OIII_ori)
            idx_OIII[:] = np.arange(np.shape(seg_3D_OIII_ori)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OIII = np.where((idx_OIII >= end_OIII[np.newaxis, :, :]) | (idx_OIII < start_OIII[np.newaxis, :, :]),
                                   seg_3D_OIII_ori, 1)
            flux_OII, flux_err_OII = flux_OII * seg_3D_OII, flux_err_OII * seg_3D_OII
            flux_OIII, flux_err_OIII = flux_OIII * seg_3D_OIII, flux_err_OIII * seg_3D_OIII
            # flux_err_OII = np.where(flux_err_OII != 0, flux_err_OII, np.inf)
            # flux_err_OIII = np.where(flux_err_OIII != 0, flux_err_OIII, np.inf)

            # Segmentation by redshift
            # plt.close('all')
            # plt.figure()
            # plt.hist(v_plot[1].flatten(), bins=100)
            # # plt.imshow(mask, origin='lower')
            # plt.show()

            #
            mask = np.zeros_like(z[1])
            v_thr = [200, 0, -200]
            v_mask = np.copy(v_plot[1])
            z_guess = np.array([(8382, 8375), (8377, 8366), (8373, 8367)]) / wave_OIII5008_vac - 1
            # z_guess = [(0.6725, 0.6734), (0.6722, 0.6735), (0.669, 0.671)]
            coord = [(65, 81), (75, 88), (80, 76)]
            for iax in range(3):
                # Need improvement
                # seg_z = detect_sources(v_mask, v_thr[iax], npixels=30, connectivity=4)
                # seg_z_label = seg_z.labels[0]
                # mask = np.where(seg_z.data != seg_z_label, mask, iax + 1)
                # v_mask = np.where(v_mask <= v_thr[iax], v_mask, np.nan)
                #
                # #
                # flux_OIII_iax = np.nansum(np.where(mask[np.newaxis, :, :] == iax + 1, flux_OIII, 0), axis=(1, 2))
                # flux_err_OIII_iax = np.sqrt(np.nansum(np.where(mask[np.newaxis, :, :] == iax + 1, flux_err_OIII ** 2, 0)
                #                                       , axis=(1, 2)))

                #
                i_j, j_j = coord[iax][1], coord[iax][0]
                flux_OIII_iax = flux_OIII[:, i_j, j_j]
                flux_err_OIII_iax = flux_err_OIII[:, i_j, j_j]

                fig_iax, ax_iax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, dpi=300)
                # fitting
                parameters = lmfit.Parameters()
                model = model_OII_OIII
                parameters.add('OII', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
                parameters.add('OIII', value=3, vary=False, min=None, max=None, expr=None, brute_step=None)
                parameters.add_many(('z_1', z_guess[iax][0], True, z_guess[iax][0] - 0.0003, z_guess[iax][0] + 0.0003, None),
                                    ('sigma_kms_1', 50, True, 50, 500.0, None),
                                    ('flux_OIII5008_1', 0.05, True, 0.0, None, None))
                parameters.add_many(('z_2', z_guess[iax][1], True, z_guess[iax][1] - 0.0003, z_guess[iax][1] + 0.0003, None),
                                    ('sigma_kms_2', 100, True, 50, 500.0, None),
                                    ('flux_OIII5008_2', 0.02, True, 0.0, None, None))
                parameters.add_many(('z_3', z_guess[iax][1], True, z_qso - 0.002, z_qso + 0.002, None),
                                    ('sigma_kms_3', 100, True, 50, 1000.0, None),
                                    ('flux_OIII5008_3', 0.02, True, 0.0, None, None))
                parameters.add('a', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
                parameters.add('b', value=-0.2, vary=True, min=-2, max=2, expr=None, brute_step=None)
                # parameters.add('bound', vary=True, min=-0.002, max=0, expr=None, brute_step=None)
                # parameters['z_1'].expr = 'z_2 + bound'

                spec_model = lmfit.Model(model, missing='drop')
                result = spec_model.fit(flux_OIII_iax, wave_vac=wave_OIII_vac, params=parameters,
                                        weights=1 / ( 2 * flux_err_OIII_iax))
                component_1 = Gaussian(wave_OIII_vac, result.best_values['z_1'], result.best_values['sigma_kms_1'],
                                       result.best_values['flux_OIII5008_1'], wave_OIII5008_vac)
                component_2 = Gaussian(wave_OIII_vac, result.best_values['z_2'], result.best_values['sigma_kms_2'],
                                       result.best_values['flux_OIII5008_2'], wave_OIII5008_vac)
                component_3 = Gaussian(wave_OIII_vac, result.best_values['z_3'], result.best_values['sigma_kms_3'],
                                       result.best_values['flux_OIII5008_3'], wave_OIII5008_vac)

                #
                ax_iax.plot(wave_OIII_vac, flux_OIII_iax - result.best_values['b'], '-k', drawstyle='steps-mid')
                ax_iax.plot(wave_OIII_vac, flux_err_OIII_iax, '-C0', drawstyle='steps-mid')
                ax_iax.plot(wave_OIII_vac, component_1 + component_2 + component_3, '-r')
                ax_iax.plot(wave_OIII_vac, component_1, '-', color='blue')
                ax_iax.plot(wave_OIII_vac, component_2, '-', color='purple')
                ax_iax.plot(wave_OIII_vac, component_3, '-', color='orange')
                ax_iax.set_xlim(wave_OIII_vac.min(), wave_OIII_vac.max())
                ax_iax.set_title(r'$\mathrm{[O\,III]}$'
                                 + '\n' + 'z_1={}'.format(result.best_values['z_1'])
                                 + '\n' + 'z_2={}'.format(result.best_values['z_2']), x=0.2, y=0.85, size=20)
                ax_iax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=-0.12)
                ax_iax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$',
                                  size=20, x=0.03)
                figname_OIII_iax = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_{}_{}_{}_checkspectra.png'.\
                    format(cubename, line, line_OIII, iax + 1)
                fig_iax.savefig(figname_OIII_iax, bbox_inches='tight')


def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None):
    # only for TXS0206
    gc.recenter(ra_qso, dec_qso, width=15 / 3600, height=15 / 3600)
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=3000, zorder=100)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=30)
    gc.colorbar.set_axis_label_font(size=30)
    if type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=30)
        gc.colorbar.set_axis_label_text(r'$\mathrm{SB \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        gc.colorbar.set_axis_label_font(size=30)
        gc.add_scalebar(length=7 * u.arcsecond)
        # gc.add_scalebar(length=8 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")  # 3C57
        # gc.scalebar.set_label(r"$8'' \approx 50 \mathrm{\; kpc}$")  # HE0226
        gc.scalebar.set_font_size(30)
        # gc.add_label(0.98, 0.94, cubename, size=35, relative=True, horizontalalignment='right')
        # gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        gc.add_scalebar(length=8 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$8'' \approx 50 \mathrm{\; kpc}$")  # HE0226
        gc.scalebar.set_font_size(30)

        gc.colorbar.set_ticks([-300, -150, 0, 150, 300])
        # gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
    elif type == 'GasMap_sigma':
        gc.add_scalebar(length=8 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$8'' \approx 50 \mathrm{\; kpc}$")  # HE0226
        gc.scalebar.set_font_size(30)

        # gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
        # gc.colorbar.set_axis_label_text(r'$\mathrm{W}_{80} \mathrm{\; [km \, s^{-1}]}$')
    else:
        gc.colorbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        gc.colorbar.set_axis_label_text(r'$\rm log([O \, III]/[O \, II])$')

    # Scale bar
    # gc.add_scalebar(length=3 * u.arcsecond)
    # gc.add_scalebar(length=15 * u.arcsecond)
    # gc.scalebar.set_corner('top left')
    # gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
    # gc.scalebar.set_label(r"$3'' \approx 20 \mathrm{\; pkpc}$")
    # gc.scalebar.set_font_size(20)

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

# HE0435-5304 Basically Done!
# muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 2 -l 0.2
# muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OIII -t 2.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 2 -l 0.2
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -n 2 -l 0.2 -sl 5300 5350')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -n 2 -l 0.2 -sl 5300 5350')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OIII -t 2.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -n 2 -l 0.2 -sl 7120 7180')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -n 2 -l 0.2 -sl 7120 7180')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='HE0435-5304', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[10, 10], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0435-5304', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[10, 10], v_min=-100, v_max=100, width_OII=10, S_N_thr=-np.inf,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
#                offset_gaia=True, FixAstrometry=True)
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='HE0435-5304', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[10, 10], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0435-5304', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[72, 76], v_min=-100, v_max=100, width_OII=10, S_N_thr=-np.inf,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
#                offset_gaia=True, FixAstrometry=True)


# HE0153-4520 done! no nebulae detected
# muse_MakeNBImageWith3DSeg.py -m HE0153-4520_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 1
# muse_MakeNBImageWith3DSeg.py -m HE0153-4520_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 1
# FitLines(cubename='HE0153-4520', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0153-4520', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=50,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-100, v_max=100,
#                sigma_max=200, contour_level=0.3)


# HE0226-4110
# muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 5550 5575 -l 0.1
# muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 7440 7500 -l 0.1
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 5550 5575 -l 0.1')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 5550 5575 -l 0.1')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 7440 7500 -l 0.1')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 7440 7500 -l 0.1')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='HE0226-4110', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[10, 10], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0226-4110', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[10, 10], v_min=-350, v_max=350, width_OII=10, S_N_thr=1,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
#                offset_gaia=True, FixAstrometry=True)
fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': False, 'r_max': 1.6,
             'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1, "OIII_2nd": 0}
# FitLines(cubename='HE0226-4110', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None, kernel_1D=None,
#          CheckGuess=[10, 10], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0226-4110', fit_param=fit_param, CheckSpectra=[77, 71], v_min=-350, v_max=350,
#                width_OII=10, S_N_thr=1, sigma_max=300, contour_level=0.1, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
#                FixAstrometry=True, CheckSpectraSeg=False, CheckSpectra_2=False)

# PKS0405-12
# muse_MakeNBImageWith3DSeg.py -m PKS0405-12_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False
# muse_MakeNBImageWith3DSeg.py -m PKS0405-12_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False
# FitLines(cubename='PKS0405-123', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='PKS0405-123', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=1,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-1200, v_max=1200,
#                sigma_max=300, contour_level=0.3)
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0405-123_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0405-123_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0405-123_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0405-123_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='PKS0405-123', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS0405-123', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[84, 14], v_min=-1200, v_max=1200, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='PKS0405-123', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[84, 14], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS0405-123', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[84, 14], v_min=-1200, v_max=1200, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))


# HE0238-1904
# muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False
# muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False
# FitLines(cubename='HE0238-1904', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0238-1904', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=200, contour_level=0.3, offset_gaia=True)
# FitLines(cubename='HE0238-1904', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# FitLines(cubename='HE0238-1904', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss',
#          CheckFit=True, CheckSpectra=[74, 48])
# PlotKinematics(cubename='HE0238-1904', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=-np.inf,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[111, 56], v_min=-300, v_max=300,
#                sigma_max=200, contour_level=0.3, offset_gaia=True)
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False')
# fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1, "OIII_2nd": 0}
# FitLines(cubename='HE0238-1904', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0238-1904', fit_param=fit_param, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.1, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='HE0238-1904', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0238-1904', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))


# 3C57
# muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 3
# muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 8350 8390 -n 2
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -n 3 -sl 6205 6270')
# os.system('muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -n 3 -sl 6205 6270')
# os.system('muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 8350 8390 -n 2')   # change it to 3
# os.system('muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 8350 8390 -n 2')
# FitLines(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None, kernel_1D=None)
# FitLines(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)
# FitLines(cubename='3C57', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss',
#          CheckSpectra=[72, 82], ResolveOII=True, r_max=1.6, width_OII=10, width_OIII=5)
# PlotKinematics(cubename='3C57', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckSpectra=[72, 82], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25, width_OII=10, width_OIII=5, FixAstrometry=True, offset_gaia=True)
# FitLines(cubename='3C57', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None, kernel_1D=None,
#          CheckSpectra=[71, 83], ResolveOII=True, r_max=1.6, width_OII=5, width_OIII=5)
# PlotKinematics(cubename='3C57', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None, kernel_1D=None,
#                CheckSpectra=[95, 64], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25, width_OII=10, width_OIII=2, FixAstrometry=True, offset_gaia=True)
# FitLines(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss',
#          ResolveOII=False, CheckSpectra=[70, 80])
# PlotKinematics(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)
fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': True, 'r_max': 1.6,
             'OII_center': wave_OII3728_vac, "OIII": 1, "OIII_2nd": 0}
# FitLines(cubename='3C57', fit_param=fit_param, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'), CheckGuess=[10, 10],
#          width_OII=10, width_OIII=10, FitType='sequential')
PlotKinematics(cubename='3C57', fit_param=fit_param, CheckSpectra=[64, 84], v_min=-350, v_max=350, width_OII=10,
               S_N_thr=1, sigma_max=300, contour_level=0.20, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
               FixAstrometry=True, CheckSpectraSeg=False, CheckSpectra_2=False)

# PKS0552-640
# muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OII -t 2.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6250 6290
# need sky-line subtraction
# muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 8405 8435
# FitLines(cubename='PKS0552-640', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='PKS0552-640', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)
# FitLines(cubename='PKS0552-640', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss', CheckSpectra=[70, 80])
# PlotKinematics(cubename='PKS0552-640', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)
# FitLines(cubename='PKS0552-640', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss',
#          CheckSpectra=[70, 80], ResolveOII=False)
# PlotKinematics(cubename='PKS0552-640', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckSpectra=[70, 80], v_min=-300, v_max=300, S_N_thr=1,
#                sigma_max=300, contour_level=0.25, SelectNebulae=[1,])
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OII -t 2.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 6250 6290')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 6250 6290')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 8405 8435')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-ssf False -sl 8405 8435')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='PKS0552-640', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS0552-640', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-500, v_max=500, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': wave_OII3728_vac, "OIII": 0, "OIII_2nd": 0}
# FitLines(cubename='PKS0552-640', fit_param=fit_param, CheckGuess=[58, 73], width_OII=10,
#          UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS0552-640', fit_param=fit_param, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
#                FixAstrometry=True, offset_gaia=True, S_N_thr=1)


# J0110-1648
# muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 2
# need sky-line subtraction
# muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 2
# FitLines(cubename='J0110-1648', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='J0110-1648', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=200, contour_level=0.3, offset_gaia=True)
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -n 2')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -n 2')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -n 2')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -n 2')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J0110-1648', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0110-1648', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J0110-1648', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0110-1648', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# J0454-6116
# muse_MakeNBImageWith3DSeg.py -m Q0454-6116_eso_coadd_nc_nosky_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6630 6685
# muse_MakeNBImageWith3DSeg.py -m Q0454-6116_COMBINED_CUBE_MED_FINAL_vac_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 8925 8960
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m J0454-6116_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 6630 6685')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0454-6116_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 6630 6685')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0454-6116_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 8925 8960')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0454-6116_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 8925 8960')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J0454-6116', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0454-6116', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J0454-6116', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0454-6116', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# J2135-5316
# muse_MakeNBImageWith3DSeg.py -m J2135-5316_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6730 6782
# muse_MakeNBImageWith3DSeg.py -m J2135-5316_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m J2135-5316_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 6730 6782')
# os.system('muse_MakeNBImageWith3DSeg.py -m J2135-5316_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 6730 6782')
# os.system('muse_MakeNBImageWith3DSeg.py -m J2135-5316_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3')
# os.system('muse_MakeNBImageWith3DSeg.py -m J2135-5316_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J2135-5316', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J2135-5316', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-400, v_max=400, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J2135-5316', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J2135-5316', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-400, v_max=400, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))



# J0119-2010
# muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OII -t 2.5 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6750 6810 -l 0.3
# muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.5 -sl 9060 9150
# FitLines(cubename='J0119-2010', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='J0119-2010', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-500, v_max=500,
#                sigma_max=500, contour_level=0.3, offset_gaia=True)
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OII -t 2.5 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 6750 6810 -l 0.3')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 6750 6810 -l 0.3')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OIII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.5 -sl 9060 9150')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OIII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.5 -sl 9060 9150')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J0119-2010', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0119-2010', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-500, v_max=500, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 1}
# FitLines(cubename='J0119-2010', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0119-2010', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-500, v_max=500, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE0246-4101
# muse_MakeNBImageWith3DSeg.py -m HE0246-4101_ESO-DEEP_subtracted_OII -t 2.5 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6750 6810 -l 0.2
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0246-4101_ESO-DEEP_subtracted_OII -t 2.5 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -sl 6750 6810 -l 0.2')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0246-4101_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -sl 6750 6810 -l 0.2')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0246-4101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0246-4101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0246-4101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0246-4101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# J0028-3305
# muse_MakeNBImageWith3DSeg.py -m J0028-3305_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 2 -sl 7030 7065
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m J0028-3305_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 2 -sl 7030 7065')
# os.system('muse_MakeNBImageWith3DSeg.py -m J0028-3305_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -n 2 -sl 7030 7065')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='J0028-3305', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0028-3305', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='J0028-3305', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='J0028-3305', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE0419-5657
# muse_MakeNBImageWith3DSeg.py -m HE0419-5657_ESO-DEEP_subtracted_OII -t 2.5 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 5 -sl 7250 7300
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0419-5657_ESO-DEEP_subtracted_OII -t 2.5 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 5 -sl 7250 7300')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0419-5657_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -n 5 -sl 7250 7300')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0419-5657', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0419-5657', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0419-5657', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0419-5657', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# Q0107-025
# muse_MakeNBImageWith3DSeg.py -m QPB6291_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7282 7320 -n 7
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m PB6291_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7282 7320 -n 7')
# os.system('muse_MakeNBImageWith3DSeg.py -m PB6291_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.2 -sl 7282 7320 -n 7')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PB6291', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PB6291', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PB6291', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PB6291', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# Q0107-0235
# muse_MakeNBImageWith3DSeg.py -m Q0107-0235_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7282 7320 -n 7
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m Q0107-0235_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7282 7320 -n 7')
# os.system('muse_MakeNBImageWith3DSeg.py -m Q0107-0235_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.2 -sl 7282 7320 -n 7')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='Q0107-0235', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='Q0107-0235', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='Q0107-0235', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='Q0107-0235', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# PKS2242-498
# muse_MakeNBImageWith3DSeg.py -m PKS2242-498_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -sl 7450 7490 -n 5
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS2242-498_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -sl 7450 7490 -n 5')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS2242-498_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -sl 7450 7490 -n 5')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS2242-498', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS2242-498', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS2242-498', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS2242-498', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# PKS0355-483
# muse_MakeNBImageWith3DSeg.py -m PKS0355-483_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# # -s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7485 7517
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0355-483_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7485 7517')
# os.system('muse_MakeNBImageWith3DSeg.py -m PKS0355-483_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.2 -sl 7485 7517')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS0355-483', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS0355-483', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS0355-483', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='PKS0355-483', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE0112-4145
# muse_MakeNBImageWith3DSeg.py -m HE0112-4145_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 5
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0112-4145_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 5')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0112-4145_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -n 5')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0112-4145', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0112-4145', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0112-4145', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0112-4145', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE0439-5254
# muse_MakeNBImageWith3DSeg.py -m HE0439-5254_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7630 7690 -n 5
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0439-5254_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 7630 7690 -n 5')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0439-5254_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.2 -sl 7630 7690 -n 5')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0439-5254', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0439-5254', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0439-5254', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0439-5254', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE2305-5315
# muse_MakeNBImageWith3DSeg.py -m HE2305-5315_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 10
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE2305-5315_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 10')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE2305-5315_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -n 10')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE2305-5315', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE2305-5315', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE2305-5315', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE2305-5315', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE1003+0149
# muse_MakeNBImageWith3DSeg.py -m HE1003+0149_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.8 -sl 7723 7780 -n 2
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE1003+0149_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.8 -sl 7723 7780 -n 2')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE1003+0149_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.8 -sl 7723 7780 -n 2')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE1003+0149', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE1003+0149', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE1003+0149', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE1003+0149', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE0331-4112
# muse_MakeNBImageWith3DSeg.py -m HE0331-4112_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -sl 7840 7910 -n 6
# os.chdir(path_SB)
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0331-4112_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -sl 7840 7910 -n 6')
# os.system('muse_MakeNBImageWith3DSeg.py -m HE0331-4112_ESO-DEEP_subtracted_OII -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -sl 7840 7910 -n 6')
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0331-4112', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0331-4112', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='HE0331-4112', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# PlotKinematics(cubename='HE0331-4112', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# TEX0206-048
# muse_MakeNBImageWith3DSeg.py -m TEX0206-048_ESO-DEEP_zapped_subtracted_OII -t 1.8 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 7910 7975 -n 1000 -npixels 10
# FitLines(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=2.0,
#          kernel_2D='gauss', smooth_1D=None, kernel_1D=None)
# PlotKinematics(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=2.0, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckFit=True, CheckSpectra=[89, 113])
# FitLines(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=1.5,
#          kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
fit_param = {"OII": 1, 'OII_2nd':0, 'ResolveOII': False, 'r_max': 1.6,
             'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0, 'OIII_2nd':0}
# PlotKinematics(cubename='TEX0206-048', zapped=True, fit_param=fit_param, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'),
#                S_N_thr=-np.inf, CheckSpectra=[81, 174], UseDataSeg=(1.5, 'gauss', 1.5, 'gauss'), contour_level=0.17,
#                SelectNebulae=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 22, 23,
#                               26, 27, 28, 34, 57, 60, 79, 81, 101, 107, 108, 114, 118, 317, 547, 552], FixAstrometry=False)

# Q1354+048
# muse_MakeNBImageWith3DSeg.py -m Q1354+048_ESO-DEEP_subtracted_OII -t 2.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.2 -sl 8300 8335 -n 5
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='Q1354+048', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='Q1354+048', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='Q1354+048', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='Q1354+048', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# J0154-0712
# muse_MakeNBImageWith3DSeg.py -m J0154-0712_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 5 -sl 8530 8580
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='J0154-0712', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='J0154-0712', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'), offset_gaia=True)
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='J0154-0712', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='J0154-0712', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'), offset_gaia=True)

# Q1435-0134
# muse_MakeNBImageWith3DSeg.py -m LBQS1435-0134_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 6 -sl 8590 8650
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='LBQS1435-0134', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='LBQS1435-0134', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='LBQS1435-0134', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='LBQS1435-0134', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# PG1522+101
# muse_MakeNBImageWith3DSeg.py -m PG1522+101_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 10 -sl 8663 8710
# -t 1.5 for smoothed data
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PG1522+101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='PG1522+101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PG1522+101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='PG1522+101', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))

# HE2336-5540
# muse_MakeNBImageWith3DSeg.py -m HE2336-5540_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 10
# muse_MakeNBImageWith3DSeg.py -m Q2339-5523_COMBINED_CUBE_MED_FINAL_vac_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -n 10
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS0232-04', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='PKS0232-04', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))


# PKS0232-04 Done!
# muse_MakeNBImageWith3DSeg.py -m PKS0232-04_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 7 -sl 9090 9140
# -t 1.5 for smoothed data
# fit_param = {"OII": 1, 'ResolveOII': False, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS0232-04', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73])
# PlotKinematics(cubename='PKS0232-04', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[76, 72], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
# fit_param = {"OII": 1, 'ResolveOII': True, 'r_max': 1.6,
#              'OII_center': (wave_OII3727_vac + wave_OII3729_vac) / 2, "OIII": 0}
# FitLines(cubename='PKS0232-04', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss',
#          smooth_1D=None, kernel_1D=None, CheckGuess=[58, 73], width_OII=10)
# PlotKinematics(cubename='PKS0232-04', fit_param=fit_param, smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckSpectra=[65, 52], v_min=-300, v_max=300, width_OII=10,
#                sigma_max=300, contour_level=0.25, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss'))
