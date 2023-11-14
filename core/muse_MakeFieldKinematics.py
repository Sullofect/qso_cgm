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
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=10)
rc('ytick.major', size=10)


def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355

def model_OII(wave_vac, z, sigma_kms, flux_OII, r_OII3729_3727, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OII3727_vac = 3727.092
    wave_OII3729_vac = 3729.875

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

    return OII3727_gaussian + OII3729_gaussian + a * wave_vac + b

def model_OII_single(wave_vac, z, sigma_kms, flux_OII, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OII3729_vac = 3729.875

    wave_OII3729_obs = wave_OII3729_vac * (1 + z)
    sigma_OII3729_A = np.sqrt((sigma_kms / c_kms * wave_OII3729_obs) ** 2 + (getSigma_MUSE(wave_OII3729_obs)) ** 2)

    peak_OII3729 = flux_OII / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)
    OII3729_gaussian = peak_OII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)

    return OII3729_gaussian + a * wave_vac + b

def model_Hbeta(wave_vac, z, sigma_kms, flux_Hbeta, a, b):
    # Constants
    c_kms = 2.998e5
    wave_Hbeta_vac = 4862.721

    wave_Hbeta_obs = wave_Hbeta_vac * (1 + z)
    sigma_Hbeta_A = np.sqrt((sigma_kms / c_kms * wave_Hbeta_obs) ** 2 + (getSigma_MUSE(wave_Hbeta_obs)) ** 2)

    peak_Hbeta = flux_Hbeta / np.sqrt(2 * sigma_Hbeta_A ** 2 * np.pi)
    Hbeta_gaussian = peak_Hbeta * np.exp(-(wave_vac - wave_Hbeta_obs) ** 2 / 2 / sigma_Hbeta_A ** 2)

    return Hbeta_gaussian + a * wave_vac + b


def model_OIII4960(wave_vac, z, sigma_kms, flux_OIII4960, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OIII4960_vac = 4960.295

    wave_OIII4960_obs = wave_OIII4960_vac * (1 + z)
    sigma_OIII4960_A = np.sqrt((sigma_kms / c_kms * wave_OIII4960_obs) ** 2 + (getSigma_MUSE(wave_OIII4960_obs)) ** 2)

    peak_OIII4960 = flux_OIII4960 / np.sqrt(2 * sigma_OIII4960_A ** 2 * np.pi)
    OIII4960_gaussian = peak_OIII4960 * np.exp(-(wave_vac - wave_OIII4960_obs) ** 2 / 2 / sigma_OIII4960_A ** 2)

    return OIII4960_gaussian + a * wave_vac + b


def model_OIII5008(wave_vac, z, sigma_kms, flux_OIII5008, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OIII5008_vac = 5008.239

    wave_OIII5008_obs = wave_OIII5008_vac * (1 + z)
    sigma_OIII5008_A = np.sqrt((sigma_kms / c_kms * wave_OIII5008_obs) ** 2 + (getSigma_MUSE(wave_OIII5008_obs)) ** 2)

    peak_OIII5008 = flux_OIII5008 / np.sqrt(2 * sigma_OIII5008_A ** 2 * np.pi)
    OIII5008_gaussian = peak_OIII5008 * np.exp(-(wave_vac - wave_OIII5008_obs) ** 2 / 2 / sigma_OIII5008_A ** 2)

    return OIII5008_gaussian + a * wave_vac + b

def model_OII_OIII(wave_vac, z, sigma_kms, flux_OII, flux_OIII5008, r_OII3729_3727, a_OII, b_OII,
                   a_OIII5008, b_OIII5008):
    m_OII = model_OII(wave_vac[0], z, sigma_kms, flux_OII, r_OII3729_3727, a_OII, b_OII)
    m_OIII5008 = model_OIII5008(wave_vac[1], z, sigma_kms, flux_OIII5008, a_OIII5008, b_OIII5008)
    return np.hstack((m_OII, m_OIII5008))


def FitLines(cubename=None, line='OII', ResolveOII=False, zapped=False, smooth_2D=1.5, kernel_2D='gauss',
             smooth_1D=None, kernel_1D=None, CheckFit=False, CheckSpectra=None):
    # constants
    c = 2.998e5

    # Load qso information
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    if zapped:
        str_zap = '_zapped'
    else:
        str_zap = ''

    # Load cubes
    if line == 'OII+OIII':
        line_OII, line_OIII = 'OII', 'OIII'
        path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line_OII)
        path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                 '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, smooth_2D,
                                                        kernel_2D, smooth_1D, kernel_1D)
        path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line_OIII)
        path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                  '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, smooth_2D,
                                                         kernel_2D, smooth_1D, kernel_1D)
        path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'.\
            format(cubename, str_zap, line_OII)
        path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'.\
            format(cubename, str_zap, line_OIII)
        path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
        path_cube = path_cube_OII

        # Load data and smoothing
        cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
        wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
        flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
        flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
        seg_3D_OII, seg_3D_OIII = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
        mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII, axis=0), np.sum(seg_3D_OIII, axis=0)
        flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII, flux_OIII * seg_3D_OIII

        # Extend over
        start_OII = (seg_3D_OII != 0).argmax(axis=0)
        end_OII = start_OII + mask_seg_OII
        start_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), start_OII, start_OII - 10)
        end_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), end_OII, end_OII + 10)
        idx_OII = np.zeros_like(seg_3D_OII)
        idx_OII[:] = np.arange(np.shape(seg_3D_OII)[0])[:, np.newaxis, np.newaxis]
        seg_3D_OII = np.where((idx_OII >= end_OII[np.newaxis, :, :]) | (idx_OII < start_OII[np.newaxis, :, :]),
                              seg_3D_OII, 1)

        # [O III]
        start_OIII = (seg_3D_OIII != 0).argmax(axis=0)
        end_OIII = start_OIII + mask_seg_OIII
        start_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), start_OIII, start_OIII - 10)
        end_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), end_OIII, end_OIII + 10)
        idx_OIII = np.zeros_like(seg_3D_OIII)
        idx_OIII[:] = np.arange(np.shape(seg_3D_OIII)[0])[:, np.newaxis, np.newaxis]
        seg_3D_OIII = np.where((idx_OIII >= end_OIII[np.newaxis, :, :]) | (idx_OIII < start_OIII[np.newaxis, :, :]),
                               seg_3D_OIII, 1)
        flux_OII, flux_err_OII = flux_OII * seg_3D_OII, flux_err_OII * seg_3D_OII
        flux_OIII, flux_err_OIII = flux_OIII * seg_3D_OIII, flux_err_OIII * seg_3D_OIII

        #
        mask_seg = mask_seg_OII + mask_seg_OIII
        wave_vac = np.array([wave_OII_vac, wave_OIII_vac], dtype=object)
        flux = np.vstack((flux_OII, flux_OIII))
        flux_err = np.vstack((flux_err_OII, flux_err_OIII))

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
        z_guess_array_OII = (wave_50 - 3728) / 3728
        sigma_kms_guess_array_OII = c * (wave_90 - wave_10) / (3728 * (1 + z_guess_array_OII))
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
        z_guess_array_OIII = (wave_50 - 5008.239) / 5008.239
        sigma_kms_guess_array_OIII = c * (wave_90 - wave_10) / (5008.239 * (1 + z_guess_array_OIII))
        sigma_kms_guess_array_OIII /= 2.563  # W_80 = 2.563sigma

        # Use [O III] if possible
        z_guess_array = np.where(mask_seg_OIII != 0, z_guess_array_OIII, z_guess_array_OII)
        sigma_kms_guess_array = np.where(mask_seg_OIII != 0, sigma_kms_guess_array_OIII, sigma_kms_guess_array_OII)
        z_mean, z_median, z_std = stats.sigma_clipped_stats(z_guess_array[mask_seg != 0], sigma=3, maxiters=5)
        sigma_mean, sigma_median, sigma_std = stats.sigma_clipped_stats(sigma_kms_guess_array[mask_seg != 0], sigma=3, maxiters=5)
        z_guess_array = np.where((z_guess_array < z_mean + 3 * z_std) * (z_guess_array > z_mean - 3 * z_std),
                                 z_guess_array, z_qso)
        sigma_kms_guess_array = np.where((sigma_kms_guess_array < sigma_mean + 3 * sigma_std) *
                                         (sigma_kms_guess_array > sigma_mean - 3 * sigma_std),
                                         sigma_kms_guess_array, sigma_mean)
        flux_guess_array_OII, flux_guess_array_OIII = np.max(flux_OII, axis=0), np.max(flux_OIII, axis=0)
    else:
        path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line)
        path_cube_smoothed = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                             '{}_{}.fits'.format(cubename, str_zap, line, smooth_2D,
                                                 kernel_2D, smooth_1D, kernel_1D)
        path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'.\
            format(cubename, str_zap, line)
        path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)

        # Load data and smoothing
        cube = Cube(path_cube_smoothed)
        wave_vac = pyasl.airtovac2(cube.wave.coord())
        flux, flux_err = cube.data * 1e-3, np.sqrt(cube.var) * 1e-3
        seg_3D = fits.open(path_3Dseg)[0].data
        mask_seg = np.sum(seg_3D, axis=0)
        flux_seg = flux * seg_3D
        start = (seg_3D != 0).argmax(axis=0)
        end = start + mask_seg
        start = np.where((mask_seg > 20) | (mask_seg < 1), start, start - 10)
        end = np.where((mask_seg > 20) | (mask_seg < 1), end, end + 10)
        idx = np.zeros_like(seg_3D)
        idx[:] = np.arange(np.shape(seg_3D)[0])[:, np.newaxis, np.newaxis]
        seg_3D = np.where((idx >= end[np.newaxis, :, :]) | (idx < start[np.newaxis, :, :]), seg_3D, 1)
        flux *= seg_3D
        flux_err *= seg_3D

        # Moments
        flux_cumsum = np.cumsum(flux_seg, axis=0) * 1.25
        flux_cumsum /= flux_cumsum.max(axis=0)
        wave_array = np.zeros_like(flux_seg)
        wave_array[:] = wave_vac[:, np.newaxis, np.newaxis]

        wave_10 = \
            np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50 = \
        np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90 = \
        np.take_along_axis(wave_array, np.argmin(np.abs(flux_cumsum - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array = (wave_50 - 3728) / 3728
        sigma_kms_guess_array = c * (wave_90 - wave_10) / (3728 * (1 + z_guess_array))
        sigma_kms_guess_array /= 2.563  # W_80 = 2.563sigma
        z_mean, z_median, z_std = stats.sigma_clipped_stats(z_guess_array[mask_seg != 0], sigma=3, maxiters=5)
        sigma_mean, sigma_median, sigma_std = stats.sigma_clipped_stats(sigma_kms_guess_array[mask_seg != 0], sigma=3, maxiters=5)
        z_guess_array = np.where((z_guess_array < z_mean + 3 * z_std) * (z_guess_array > z_mean - 3 * z_std),
                                 z_guess_array, z_qso)
        sigma_kms_guess_array = np.where((sigma_kms_guess_array < sigma_mean + 3 * sigma_std) *
                                         (sigma_kms_guess_array > sigma_mean - 3 * sigma_std),
                                         sigma_kms_guess_array, sigma_mean)
        flux_guess_array_OII = np.max(flux, axis=0)

    # Guesses
    redshift_guess, sigma_kms_guess, flux_guess, r_OII3729_3727_guess = z_qso, 200.0, 1.5, 2
    parameters = lmfit.Parameters()

    #
    size = np.shape(flux)[1:]
    fit_success, v_fit, z_fit, dz_fit = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)
    sigma_fit, dsigma_fit = np.zeros(size), np.zeros(size)

    if line == 'OII':
        # Zeros
        flux_fit, dflux_fit = np.zeros(size), np.zeros(size)
        a_fit, b_fit = np.zeros(size), np.zeros(size)
        da_fit, db_fit = np.zeros(size), np.zeros(size)
        r_fit, dr_fit = np.zeros(size), np.zeros(size)

        # Model
        model = model_OII
        parameters.add_many(('z', redshift_guess, True, redshift_guess - 0.05, redshift_guess + 0.05, None),
                            ('sigma_kms', sigma_kms_guess, True, 0.0, 2000.0, None),
                            ('flux_OII', flux_guess, True, 0, None, None),
                            ('r_OII3729_3727', r_OII3729_3727_guess, True, 0.3, 2, None),
                            ('a', 0.0, False, None, None, None),
                            ('b', 0.0, False, None, None, None))

        if not ResolveOII:
            parameters['r_OII3729_3727'].max = np.inf
            parameters['r_OII3729_3727'].vary = False
            parameters['r_OII3729_3727'].value = np.inf
    elif line == 'Hbeta':
        model = model_Hbeta
    elif line == 'OIII4960':
        model = model_OIII4960
    elif line == 'OIII5008':
        model = model_OIII5008
    elif line == 'OII+OIII':
        flux_OII_fit, dflux_OII_fit = np.zeros(size), np.zeros(size)
        flux_OIII_fit, dflux_OIII_fit = np.zeros(size), np.zeros(size)
        a_OII_fit, b_OII_fit = np.zeros(size), np.zeros(size)
        da_OII_fit, db_OII_fit = np.zeros(size), np.zeros(size)
        a_OIII_fit, b_OIII_fit = np.zeros(size), np.zeros(size)
        da_OIII_fit, db_OIII_fit = np.zeros(size), np.zeros(size)
        r_fit, dr_fit = np.zeros(size), np.zeros(size)

        # Model
        model = model_OII_OIII
        parameters.add_many(('z', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
                            ('sigma_kms', sigma_kms_guess, True, 0.0, 2000.0, None),
                            ('flux_OII', flux_guess, True, 0, None, None),
                            ('flux_OIII5008', flux_guess, True, 0, None, None),
                            ('r_OII3729_3727', r_OII3729_3727_guess, True, 0.3, 2, None),
                            ('a_OII', 0.0, False, None, None, None),
                            ('b_OII', 0.0, False, None, None, None),
                            ('a_OIII5008', 0.0, False, None, None, None),
                            ('b_OIII5008', 0.0, False, None, None, None))


    #
    for i in range(size[0]):  # i = p (y), j = q (x)
        for j in range(size[1]):
            if mask_seg[i, j] != 0:
                flux_ij, flux_err_ij = flux[:, i, j], flux_err[:, i, j]
                parameters['z'].value = z_guess_array[i, j]
                parameters['sigma_kms'].value = sigma_kms_guess_array[i, j]
                parameters['flux_OII'].value = flux_guess_array_OII[i, j]
                if line == 'OII+OIII':
                    parameters['flux_OIII5008'].value = flux_guess_array_OIII[i, j]

                #
                spec_model = lmfit.Model(model, missing='drop')
                result = spec_model.fit(flux_ij, wave_vac=wave_vac, params=parameters, weights=1 / flux_err_ij)

                z, dz = result.best_values['z'], result.params['z'].stderr
                sigma, dsigma = result.best_values['sigma_kms'], result.params['sigma_kms'].stderr

                fit_success[i, j] = result.success
                v_fit[i, j], z_fit[i, j], dz_fit[i, j] = c * (z - z_qso) / (1 + z_qso), z, dz
                sigma_fit[i, j], dsigma_fit[i, j] = sigma, dsigma

                if line == 'OII+OIII':
                    flux_f_OII, dflux_f_OII = result.best_values['flux_OII'], result.params['flux_OII'].stderr
                    flux_f_OIII, dflux_f_OIII = result.best_values['flux_OIII5008'], result.params['flux_OIII5008'].stderr
                    r, dr = result.best_values['r_OII3729_3727'], result.params['r_OII3729_3727'].stderr
                    a_OII, da_OII = result.best_values['a_OII'], result.params['a_OII'].stderr,
                    b_OII, db_OII = result.best_values['b_OII'], result.params['b_OII'].stderr
                    a_OIII, da_OIII = result.best_values['a_OIII5008'], result.params['a_OIII5008'].stderr
                    b_OIII, db_OIII = result.best_values['b_OIII5008'], result.params['b_OIII5008'].stderr

                    flux_OII_fit[i, j], dflux_OII_fit[i, j] = flux_f_OII, dflux_f_OII
                    flux_OIII_fit[i, j], dflux_OIII_fit[i, j] = flux_f_OIII, dflux_f_OIII
                    r_fit[i, j], dr_fit[i, j] = r, dr
                    a_OII_fit[i, j], da_OII_fit[i, j], b_OII_fit[i, j], db_OII_fit[i, j] = a_OII, da_OII, b_OII, db_OII
                    a_OIII_fit[i, j], da_OIII_fit[i, j] = a_OIII, da_OIII
                    b_OIII_fit[i, j], db_OIII_fit[i, j] =  b_OIII, db_OIII
                else:
                    flux_f_OII, dflux_f_OII = result.best_values['flux_OII'], result.params['flux_OII'].stderr
                    r, a, b = result.best_values['r_OII3729_3727'], result.best_values['a'], result.best_values['b']
                    dr, da, db = result.params['r_OII3729_3727'].stderr, result.params['a'].stderr, \
                                 result.params['b'].stderr
                    flux_fit[i, j], dflux_fit[i, j] = flux_f_OII, dflux_f_OII
                    r_fit[i, j], a_fit[i, j], b_fit[i, j] = r, a, b
                    dr_fit[i, j], da_fit[i, j], db_fit[i, j] = dr, da, db
            else:
                pass

    header = fits.open(path_cube)[1].header
    header['WCSAXES'] = 2
    header.remove('CTYPE3')
    header.remove('CUNIT3')
    header.remove('CD3_3')
    header.remove('CRPIX3')
    header.remove('CRVAL3')
    header.remove('CRDER3')
    header.remove('CD1_3')
    header.remove('CD2_3')
    header.remove('CD3_1')
    header.remove('CD3_2')
    hdul_pri = fits.open(path_cube)[0]
    hdul_fs, hdul_v = fits.ImageHDU(fit_success, header=header), fits.ImageHDU(v_fit, header=header)
    hdul_z, hdul_dz = fits.ImageHDU(z_fit, header=header), fits.ImageHDU(dz_fit, header=header)
    hdul_sigma, hdul_dsigma = fits.ImageHDU(sigma_fit, header=header), fits.ImageHDU(dsigma_fit, header=header)

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
                             hdul_dflux_OII, hdul_flux_OIII, hdul_dflux_OIII, hdul_r, hdul_dr,
                             hdul_a_OII, hdul_da_OII, hdul_b_OII, hdul_db_OII,
                             hdul_a_OIII, hdul_da_OIII, hdul_b_OIII, hdul_db_OIII])

    else:
        hdul_flux, hdul_dflux = fits.ImageHDU(flux_fit, header=header), fits.ImageHDU(dflux_fit, header=header)
        hdul_r, hdul_dr = fits.ImageHDU(r_fit, header=header), fits.ImageHDU(dr_fit, header=header)
        hdul_a, hdul_da = fits.ImageHDU(a_fit, header=header), fits.ImageHDU(da_fit, header=header)
        hdul_b, hdul_db = fits.ImageHDU(b_fit, header=header), fits.ImageHDU(db_fit, header=header)
        hdul = fits.HDUList([hdul_pri, hdul_fs, hdul_v, hdul_z, hdul_dz, hdul_sigma, hdul_dsigma, hdul_flux,
                             hdul_dflux, hdul_r, hdul_dr, hdul_a, hdul_da, hdul_b, hdul_db])
    hdul.writeto(path_fit, overwrite=True)

    if CheckFit:
        if line == 'OII+OIII':
            fig_1, ax_1 = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            fig_2, ax_2 = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_OII[:, i_j, j_j], '-k')
                    # ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_seg[:, i_j, j_j], '-b')
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_err_OII[:, i_j, j_j], '-C0')
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, model_OII(wave_OII_vac, z_fit[i_j, j_j], sigma_fit[i_j, j_j],
                                                                  flux_OII_fit[i_j, j_j], r_fit[i_j, j_j], 0, 0), '-r')
                    # ax_1[ax_i, ax_j].set_ylim(top=0.01)
                    # ax_2[ax_i, ax_j].set_ylim(top=0.01)
                    ax_1[ax_i, ax_j].set_xlim((3727.092 - 30) * (1 + z_qso), (3727.092 + 30) * (1 + z_qso))
                    ax_1[ax_i, ax_j].axvline(x=3727.092 * (1 + z_qso))
                    ax_1[ax_i, ax_j].axvline(x=3729.875 * (1 + z_qso))
                    ax_1[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v_fit[i_j, j_j], 2)), y=0.9, x=0.2)
                    #
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_OIII[:, i_j, j_j], '-k')
                    # ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_seg[:, i_j, j_j], '-b')
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_err_OIII[:, i_j, j_j], '-C0')
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, model_OIII5008(wave_OIII_vac, z_fit[i_j, j_j],
                                                                        sigma_fit[i_j, j_j], flux_OIII_fit[i_j, j_j],
                                                                        0, 0), '-r')
                    ax_2[ax_i, ax_j].set_xlim((5008 - 30) * (1 + z_qso), (5008 + 30) * (1 + z_qso))
                    ax_2[ax_i, ax_j].axvline(x=5008 * (1 + z_qso))
                    ax_2[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v_fit[i_j, j_j], 2)), y=0.9, x=0.2)
            figname_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_checkspectra.png'.format(cubename,
                                                                                                           line, line_OII)
            figname_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_checkspectra.png'.format(cubename,
                                                                                                            line, line_OIII)
            fig_1.savefig(figname_OII, bbox_inches='tight')
            fig_2.savefig(figname_OIII, bbox_inches='tight')

        else:
            fig, ax = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax[ax_i, ax_j].plot(wave_vac, flux[:, i_j, j_j], '-k')
                    ax[ax_i, ax_j].plot(wave_vac, flux_seg[:, i_j, j_j], '-b')
                    ax[ax_i, ax_j].plot(wave_vac, flux_err[:, i_j, j_j], '-C0')
                    ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                            flux_OII[i_j, j_j], r[i_j, j_j],
                                                            a[i_j, j_j], b[i_j, j_j]), '-r')
                    # ax[ax_i, ax_j].set_ylim(top=0.01)
                    ax[ax_i, ax_j].axvline(x=3727.092 * (1 + z_qso))
                    ax[ax_i, ax_j].axvline(x=3729.875 * (1 + z_qso))
                    ax[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v_fit[i_j, j_j], 2)), y=0.9, x=0.2)
            figurename = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_checkspectra.png'.format(cubename, line)
            plt.savefig(figurename, bbox_inches='tight')

def PlotKinematics(cubename='3C57', zapped=False, line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None,
                   kernel_1D=None, CheckFit=True, CheckSpectra=[50, 50], S_N_thr=5, v_min=-600, v_max=600,
                   sigma_max=400, contour_level=0.15, offset_gaia=False, SelectNebulae=None):
    if zapped:
        str_zap = '_zapped'
    else:
        str_zap = ''

    path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    path_v = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_v_{}_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    figurename_v = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_v_{}_{}_{}_{}_{}.png'. \
        format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    path_sigma = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_sigma_{}_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    figurename_sigma = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_sigma_{}_{}_{}_{}_{}.png'. \
        format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)

    hdul = fits.open(path_fit)
    # hdul[2].header['WCSAXES'] = 2
    ra_center, dec_center = hdul[2].header['CRVAL1'], hdul[2].header['CRVAL2']
    fs = hdul[1].data
    size = int(min(np.shape(fs)) * 0.2)
    v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
    sigma, dsigma = hdul[5].data, hdul[6].data

    # Load cube and 3D seg
    if line == 'OII+OIII':
        line_OII, line_OIII = 'OII', 'OIII'
        path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line_OII)
        path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                 '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, smooth_2D,
                                                        kernel_2D, smooth_1D, kernel_1D)
        path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line_OIII)
        path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                  '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, smooth_2D,
                                                         kernel_2D, smooth_1D, kernel_1D)
        path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'.\
            format(cubename, str_zap, line_OII)
        path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'.\
            format(cubename, str_zap, line_OIII)
        path_SB_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg.fits'.\
            format(cubename, str_zap, line_OII)
        figurename_SB_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'.\
            format(cubename, str_zap, line_OII, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
        path_SB_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg.fits'.\
            format(cubename, str_zap, line_OIII)
        figurename_SB_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'.\
            format(cubename, str_zap, line_OIII, smooth_2D, kernel_2D, smooth_1D, kernel_1D)

        # 3D seg
        seg_3D_OII = fits.open(path_3Dseg_OII)[0].data
        seg_label_OII = fits.open(path_3Dseg_OII)[1].data
        seg_3D_OIII = fits.open(path_3Dseg_OIII)[0].data
        seg_label_OIII = fits.open(path_3Dseg_OIII)[1].data
        flux_OII_fit, dflux_OII_fit = hdul[7].data, hdul[8].data
        flux_OIII_fit, dflux_OIII_fit = hdul[9].data, hdul[10].data
        r, dr = hdul[11].data, hdul[12].data
        a, b = hdul[13].data, hdul[14].data
        S_N = flux_OIII_fit / dflux_OIII_fit

    else:
        path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.\
            format(cubename, str_zap, line)
        path_cube_smoothed = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                             '{}_{}.fits'.format(line, cubename, str_zap, line, smooth_2D,
                                                 kernel_2D, smooth_1D, kernel_1D)
        path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'.\
            format(cubename, str_zap, line)
        path_SB = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB.fits'.\
            format(cubename, str_zap, line)
        figurename_SB = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'.\
            format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)

        # 3D seg
        seg_3D = fits.open(path_3Dseg)[0].data
        seg_label = fits.open(path_3Dseg)[1].data
        flux_OII, dflux_OII = hdul[7].data, hdul[8].data
        r, dr = hdul[9].data, hdul[10].data
        a, b = hdul[11].data, hdul[13].data
        S_N = flux_OII / dflux_OII

        #
        if SelectNebulae is not None:
            mask_select = np.zeros_like(v)
            for i in SelectNebulae:
                mask_select = np.where(seg_label != i, mask_select, 1)
            v = np.where(mask_select == 1, v, np.nan)
            sigma = np.where(mask_select == 1, sigma, np.nan)


    # v = np.where(fs == 1, v, np.nan)
    # v = np.where(S_N > S_N_thr, v, np.nan)
    # sigma = np.where(fs == 1, sigma, np.nan)
    # sigma = np.where(S_N > S_N_thr, sigma, np.nan)
    hdul_v = fits.ImageHDU(v, header=hdul[2].header)
    hdul_v.writeto(path_v, overwrite=True)
    hdul_sigma = fits.ImageHDU(sigma, header=hdul[2].header)
    hdul_sigma.writeto(path_sigma, overwrite=True)

    if CheckFit:
        if line == 'OII+OIII':
            # Load qso information
            path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
            data_qso = ascii.read(path_qso, format='fixed_width')
            data_qso = data_qso[data_qso['name'] == cubename]
            ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]


            line_OII, line_OIII = 'OII', 'OIII'
            path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OII)
            path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                     '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, smooth_2D,
                                                            kernel_2D, smooth_1D, kernel_1D)
            path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OIII)
            path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                      '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, smooth_2D,
                                                             kernel_2D, smooth_1D, kernel_1D)
            path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'. \
                format(cubename, str_zap, line_OII)
            path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg.fits'. \
                format(cubename, str_zap, line_OIII)
            path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}.fits'. \
                format(cubename, str_zap, line, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
            path_cube = path_cube_OII

            # Load data and smoothing
            cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
            wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(
                cube_OIII.wave.coord())
            flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
            flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
            seg_3D_OII, seg_3D_OIII = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
            mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII, axis=0), np.sum(seg_3D_OIII, axis=0)
            flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII, flux_OIII * seg_3D_OIII

            # Extend over
            start_OII = (seg_3D_OII != 0).argmax(axis=0)
            end_OII = start_OII + mask_seg_OII
            start_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), start_OII, start_OII - 3)
            end_OII = np.where((mask_seg_OII > 20) | (mask_seg_OII < 1), end_OII, end_OII + 3)
            idx_OII = np.zeros_like(seg_3D_OII)
            idx_OII[:] = np.arange(np.shape(seg_3D_OII)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OII = np.where((idx_OII >= end_OII[np.newaxis, :, :]) | (idx_OII < start_OII[np.newaxis, :, :]),
                                  seg_3D_OII, 1)

            # [O III]
            start_OIII = (seg_3D_OIII != 0).argmax(axis=0)
            end_OIII = start_OIII + mask_seg_OIII
            start_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), start_OIII, start_OIII - 3)
            end_OIII = np.where((mask_seg_OIII > 20) | (mask_seg_OIII < 1), end_OIII, end_OIII + 3)
            idx_OIII = np.zeros_like(seg_3D_OIII)
            idx_OIII[:] = np.arange(np.shape(seg_3D_OIII)[0])[:, np.newaxis, np.newaxis]
            seg_3D_OIII = np.where((idx_OIII >= end_OIII[np.newaxis, :, :]) | (idx_OIII < start_OIII[np.newaxis, :, :]),
                                   seg_3D_OIII, 1)
            flux_OII, flux_err_OII = flux_OII * seg_3D_OII, flux_err_OII * seg_3D_OII
            flux_OIII, flux_err_OIII = flux_OIII * seg_3D_OIII, flux_err_OIII * seg_3D_OIII

            fig_1, ax_1 = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            fig_2, ax_2 = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_OII[:, i_j, j_j], '-k')
                    # ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_seg[:, i_j, j_j], '-b')
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, flux_err_OII[:, i_j, j_j], '-C0')
                    ax_1[ax_i, ax_j].plot(wave_OII_vac, model_OII(wave_OII_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                                  flux_OII_fit[i_j, j_j], r[i_j, j_j], 0, 0), '-r')
                    # ax_1[ax_i, ax_j].set_ylim(top=0.01)
                    # ax_2[ax_i, ax_j].set_ylim(top=0.01)
                    ax_1[ax_i, ax_j].set_xlim((3727.092 - 30) * (1 + z_qso), (3727.092 + 30) * (1 + z_qso))
                    ax_1[ax_i, ax_j].axvline(x=3727.092 * (1 + z_qso))
                    ax_1[ax_i, ax_j].axvline(x=3729.875 * (1 + z_qso))
                    ax_1[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v[i_j, j_j], 2)), y=0.9, x=0.2)
                    #
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_OIII[:, i_j, j_j], '-k')
                    # ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_seg[:, i_j, j_j], '-b')
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, flux_err_OIII[:, i_j, j_j], '-C0')
                    ax_2[ax_i, ax_j].plot(wave_OIII_vac, model_OIII5008(wave_OIII_vac, z[i_j, j_j],
                                                                        sigma[i_j, j_j], flux_OIII_fit[i_j, j_j],
                                                                        0, 0), '-r')
                    ax_2[ax_i, ax_j].set_xlim((5008 - 30) * (1 + z_qso), (5008 + 30) * (1 + z_qso))
                    ax_2[ax_i, ax_j].axvline(x=5008 * (1 + z_qso))
                    ax_2[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v[i_j, j_j], 2)), y=0.9, x=0.2)
            figname_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_checkspectra.png'.format(cubename,
                                                                                                           line, line_OII)
            figname_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_{}_checkspectra.png'.format(cubename,
                                                                                                            line, line_OIII)
            fig_1.savefig(figname_OII, bbox_inches='tight')
            fig_2.savefig(figname_OIII, bbox_inches='tight')

        else:
            fig, ax = plt.subplots(5, 5, figsize=(20, 20), sharex=True)
            for ax_i in range(5):
                for ax_j in range(5):
                    i_j, j_j = ax_i + CheckSpectra[1] - 2 - 1, ax_j + CheckSpectra[0] - 2 - 1
                    i_j_idx, j_j_idx = i_j + 1, j_j + 1
                    ax[ax_i, ax_j].plot(wave_vac, flux[:, i_j, j_j], '-k')
                    ax[ax_i, ax_j].plot(wave_vac, flux_seg[:, i_j, j_j], '-b')
                    ax[ax_i, ax_j].plot(wave_vac, flux_err[:, i_j, j_j], '-C0')
                    ax[ax_i, ax_j].plot(wave_vac, model_OII(wave_vac, z[i_j, j_j], sigma[i_j, j_j],
                                                            flux_OII[i_j, j_j], r[i_j, j_j],
                                                            a[i_j, j_j], b[i_j, j_j]), '-r')
                    # ax[ax_i, ax_j].set_ylim(top=0.01)
                    ax[ax_i, ax_j].axvline(x=3727.092 * (1 + z_qso))
                    ax[ax_i, ax_j].axvline(x=3729.875 * (1 + z_qso))
                    ax[ax_i, ax_j].set_title('x={}, y={}'.format(j_j_idx, i_j_idx)
                                             + '\n' + 'v=' + str(np.round(v_fit[i_j, j_j], 2)), y=0.9, x=0.2)
            figurename = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}_fit_{}_checkspectra.png'.format(cubename, line)
            plt.savefig(figurename, bbox_inches='tight')


    from palettable.scientific.sequential import Acton_6
    from palettable.cubehelix import red_16
    from palettable.cmocean.sequential import Dense_20_r

    # SB map
    if line == 'OII+OIII':
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB_OII, figure=fig, north=True, hdu=0)
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_SB_OII, levels=[contour_level], color='k', linewidths=0.8, smooth=3, kernel='box')
        APLpyStyle(gc, type='NarrowBand', cubename=cubename, size=size,
                   offset_gaia=offset_gaia, ra_center=ra_center, dec_center=dec_center)
        fig.savefig(figurename_SB_OII, bbox_inches='tight')

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB_OIII, figure=fig, north=True, hdu=0)
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_SB_OIII, levels=[contour_level], color='k', linewidths=0.8, smooth=3, kernel='box')
        APLpyStyle(gc, type='NarrowBand', cubename=cubename, size=size,
                   offset_gaia=offset_gaia, ra_center=ra_center, dec_center=dec_center)
        fig.savefig(figurename_SB_OIII, bbox_inches='tight')
    else:
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB, figure=fig, north=True, hdu=0)
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_SB, levels=[contour_level], color='k', linewidths=0.8, smooth=3, kernel='box')
        APLpyStyle(gc, type='NarrowBand', cubename=cubename, size=size,
                   offset_gaia=offset_gaia, ra_center=ra_center, dec_center=dec_center)
        fig.savefig(figurename_SB, bbox_inches='tight')

    # LOS velocity
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_v, figure=fig, north=True, hdu=1)
    gc.show_colorscale(vmin=v_min, vmax=v_max, cmap='coolwarm')
    APLpyStyle(gc, type='GasMap', cubename=cubename, size=size,
               offset_gaia=offset_gaia, ra_center=ra_center, dec_center=dec_center)
    fig.savefig(figurename_v, bbox_inches='tight')

    # Sigma map
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_sigma, figure=fig, north=True, hdu=1)
    gc.show_colorscale(vmin=0, vmax=sigma_max, cmap=Dense_20_r.mpl_colormap)
    APLpyStyle(gc, type='GasMap_sigma', cubename=cubename, size=size,
               offset_gaia=offset_gaia, ra_center=ra_center, dec_center=dec_center)
    fig.savefig(figurename_sigma, bbox_inches='tight')


def APLpyStyle(gc, type=None, cubename=None, size=None, offset_gaia=False, ra_center=None, dec_center=None):
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    if offset_gaia:
        gc.show_markers(ra_center, dec_center, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                        linewidths=0.5, s=600, zorder=100)
        gc.recenter(ra_center, dec_center, width=size / 3600, height=size / 3600)
    else:
        path_offset = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/offsets.dat'
        data_offset = ascii.read(path_offset, format='fixed_width')
        data_offset = data_offset[data_offset['name'] == cubename]
        offset_ra_muse, offset_dec_muse = data_offset['offset_ra_muse'], data_offset['offset_dec_muse']
        gc.show_markers(ra_qso - offset_ra_muse, dec_qso - offset_dec_muse, facecolors='none', marker='*',
                        c='lightgrey', edgecolors='k', linewidths=0.5, s=600, zorder=100)
        gc.recenter(ra_qso - offset_ra_muse, dec_qso - offset_dec_muse, width=size / 3600, height=size / 3600)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)
    if type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        # gc.colorbar.set_axis_label_text('')
        # gc.colorbar.hide()
        gc.add_label(0.25, 0.90, cubename, size=30, relative=True)
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        # gc.colorbar.set_ticks([-200, -100, 0, 100, 200])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
    elif type == 'GasMap_sigma':
        # gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
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
    # xw, yw = gc.pixel2world(195, 140)  # original figure
    # xw, yw = gc.pixel2world(196, 105)
    # xw, yw = 40.1302360960119, -18.863967747328896
    # gc.show_arrows(xw, yw, -0.000035 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000035 * yw, color='k')
    # xw, yw = 40.1333130960119, -18.864847747328896
    # gc.show_arrows(xw, yw, -0.000020 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000020 * yw, color='k')
    # gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
    # gc.add_label(0.88, 0.70, r'E', size=20, relative=True)


# muse_MakeNBImageWith3DSeg.py -m HE0435-5304_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 2
# FitLines(cubename='HE0435-5304', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0435-5304', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=7,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-100, v_max=100,
#                sigma_max=200, contour_level=0.3)


# muse_MakeNBImageWith3DSeg.py -m HE0153-4520_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 1
# FitLines(cubename='HE0153-4520', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0153-4520', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=50,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-100, v_max=100,
#                sigma_max=200, contour_level=0.3)


# muse_MakeNBImageWith3DSeg.py -m HE0226-4110_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 5550 5575
# FitLines(cubename='HE0226-4110', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0226-4110', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-350, v_max=350,
#                sigma_max=300, contour_level=0.3)

# muse_MakeNBImageWith3DSeg.py -m PKS0405-12_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False
# FitLines(cubename='PKS0405-123', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='PKS0405-123', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=1,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-1200, v_max=1200,
#                sigma_max=300, contour_level=0.3)

# muse_MakeNBImageWith3DSeg.py -m HE0238-1904_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False
# FitLines(cubename='HE0238-1904', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0238-1904', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=200, contour_level=0.3, offset_gaia=True)
# FitLines(cubename='HE0238-1904', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
FitLines(cubename='HE0238-1904', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss',
         CheckFit=True, CheckSpectra=[74, 48])
PlotKinematics(cubename='HE0238-1904', line='OII+OIII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
               kernel_1D='gauss', CheckFit=True, CheckSpectra=[111, 56], v_min=-300, v_max=300,
               sigma_max=200, contour_level=0.3, offset_gaia=True)

# FitLines(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=None, kernel_1D=None)
# FitLines(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='3C57', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)

# muse_MakeNBImageWith3DSeg.py -m PKS0552-640_ESO-DEEP_subtracted_OII -t 2.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6250 6290
# FitLines(cubename='PKS0552-640', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='PKS0552-640', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)


# muse_MakeNBImageWith3DSeg.py -m J0110-1648_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 2
# FitLines(cubename='J0110-1648', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='J0110-1648', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=200, contour_level=0.3, offset_gaia=True)


# muse_MakeNBImageWith3DSeg.py -m J0119-2010_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6750 6810
# FitLines(cubename='J0119-2010', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='J0119-2010', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-500, v_max=500,
#                sigma_max=500, contour_level=0.3, offset_gaia=True)

# muse_MakeNBImageWith3DSeg.py -m HE0246-4101_ESO-DEEP_subtracted_OII -t 3.0 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 6750 6810
# FitLines(cubename='HE0246-4101', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='HE0246-4101', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, S_N_thr=5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.3, offset_gaia=True)

# muse_MakeNBImageWith3DSeg.py -m TEX0206-048_ESO-DEEP_zapped_subtracted_OII -t 1.8 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -sl 7910 7975 -n 1000 -npixels 10
# FitLines(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=2.0,
#          kernel_2D='gauss', smooth_1D=None, kernel_1D=None)
# PlotKinematics(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=2.0, kernel_2D='gauss', smooth_1D=None,
#                kernel_1D=None, CheckFit=True, CheckSpectra=[89, 113])
# FitLines(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=1.5,
#          kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='TEX0206-048', zapped=True, line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckFit=True, S_N_thr=-np.inf, CheckSpectra=[81, 174],
#                SelectNebulae=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 22, 23,
#                               26, 27, 28, 34, 57, 60, 79, 81, 101, 107, 108, 114, 118, 317, 547, 552])

# muse_MakeNBImageWith3DSeg.py -m PKS0232-04_ESO-DEEP_subtracted_OII -t 3.5 -s 1.5 -k gauss
# -s_spe 1.5 -k_spe gauss -ssf False -n 10 -sl 9090 9140
# FitLines(cubename='PKS0232-04', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5, kernel_1D='gauss')
# PlotKinematics(cubename='PKS0232-04', line='OII', smooth_2D=1.5, kernel_2D='gauss', smooth_1D=1.5,
#                kernel_1D='gauss', CheckFit=True, CheckSpectra=[70, 80], v_min=-300, v_max=300,
#                sigma_max=300, contour_level=0.25)