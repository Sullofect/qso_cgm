import os
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import rc
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


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


# def model_all(wave_vac, z, sigma_kms_OII, sigma_kms_Hbeta, sigma_kms_OIII4960, sigma_kms_OIII5008, flux_OII,
#               flux_Hbeta, flux_OIII4960, flux_OIII5008, r_OII3729_3727, a_OII, b_OII, a_Hbeta, b_Hbeta,
#               a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008):
def model_all(wave_vac, z, sigma_kms, flux_OII, flux_Hbeta, flux_OIII5008, r_OII3729_3727, a_OII, b_OII, a_Hbeta,
              b_Hbeta, a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008):

    m_OII = model_OII(wave_vac[0], z, sigma_kms, flux_OII, r_OII3729_3727, a_OII, b_OII)
    m_Hbeta = model_Hbeta(wave_vac[1], z, sigma_kms, flux_Hbeta, a_Hbeta, b_Hbeta)
    m_OIII4960 = model_OIII4960(wave_vac[2], z, sigma_kms, flux_OIII5008 / 3, a_OIII4960, b_OIII4960)
    m_OIII5008 = model_OIII5008(wave_vac[3], z, sigma_kms, flux_OIII5008, a_OIII5008, b_OIII5008)
    return np.hstack((m_OII, m_Hbeta, m_OIII4960, m_OIII5008))


def FitLines(method=None, method_spe=None, radius=80, sn_vor=30, radius_aper=1):
    # Fitting the narrow band image profile
    path_cube_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
    path_cube_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_Hbeta_line_offset.fits')
    path_cube_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_4960_line_offset.fits')
    path_cube_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_5008_line_offset.fits')
    cube_OII = Cube(path_cube_OII)
    cube_Hbeta = Cube(path_cube_Hbeta)
    cube_OIII4960 = Cube(path_cube_OIII4960)
    cube_OIII5008 = Cube(path_cube_OIII5008)
    if radius is not None:
        cube_OII = cube_OII.subcube((90, 110), radius, unit_center=None, unit_size=None)
        cube_Hbeta = cube_Hbeta.subcube((90, 110), radius, unit_center=None, unit_size=None)
        cube_OIII4960 = cube_OIII4960.subcube((90, 110), radius, unit_center=None, unit_size=None)
        cube_OIII5008 = cube_OIII5008.subcube((90, 110), radius, unit_center=None, unit_size=None)
    cube_OIII5008[0, :, :].write('/Users/lzq/Dropbox/Data/CGM/image_OOHbeta_fitline.fits')

    redshift_guess = 0.63
    sigma_kms_guess = 150.0
    # flux_OIII5008_guess = 0.01
    r_OII3729_3727_guess = 2

    parameters = lmfit.Parameters()
    parameters.add_many(('z', redshift_guess, True, 0.62, 0.64, None),
                        ('sigma_kms', sigma_kms_guess, True, 10, 500, None),
                        ('flux_OII', 0.01, True, None, None, None),
                        ('flux_Hbeta', 0.02, True, None, None, None),
                        ('flux_OIII5008', 0.1, True, None, None, None),
                        ('r_OII3729_3727', r_OII3729_3727_guess, True, 0.2, None, None),
                        ('a_OII', 0.0, False, None, None, None),
                        ('b_OII', 0.0, False, None, None, None),
                        ('a_Hbeta', 0.0, False, None, None, None),
                        ('b_Hbeta', 0.0, False, None, None, None),
                        ('a_OIII4960', 0.0, False, None, None, None),
                        ('b_OIII4960', 0.0, False, None, None, None),
                        ('a_OIII5008', 0.0, False, None, None, None),
                        ('b_OIII5008', 0.0, False, None, None, None))

    num_lines = 4
    size = np.shape(cube_OII)[1]
    fit_success = np.zeros((size, size))
    r_fit, dr_fit = np.zeros((size, size)), np.zeros((size, size))
    z_fit, dz_fit = np.zeros((size, size)), np.zeros((size, size))
    sigma_fit, dsigma_fit = np.zeros((size, size)), np.zeros((size, size))
    flux_fit, dflux_fit = np.zeros((3, size, size)), np.zeros((3, size, size))
    a_fit, b_fit = np.zeros((num_lines, size, size)), np.zeros((num_lines, size, size))
    da_fit, db_fit = np.zeros((num_lines, size, size)), np.zeros((num_lines, size, size))

    #
    wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
    wave_Hbeta_vac = pyasl.airtovac2(cube_Hbeta.wave.coord())
    wave_OIII4960_vac = pyasl.airtovac2(cube_OIII4960.wave.coord())
    wave_OIII5008_vac = pyasl.airtovac2(cube_OIII5008.wave.coord())
    wave_vac_all = np.array([wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)
    xy_array = np.array(np.meshgrid(np.arange(size), np.arange(size))).T.reshape(-1, 2)

    if method != 'aperture':
        # Data of each pixel
        flux_OII, flux_Hbeta = cube_OII.data * 1e-3, cube_Hbeta.data * 1e-3
        flux_OIII4960, flux_OIII5008 = cube_OIII4960.data * 1e-3, cube_OIII5008.data * 1e-3
        flux_OII_err, flux_Hbeta_err = np.sqrt(cube_OII.var) * 1e-3, np.sqrt(cube_Hbeta.var) * 1e-3
        flux_OIII4960_err = np.sqrt(cube_OIII4960.var) * 1e-3
        flux_OIII5008_err = np.sqrt(cube_OIII5008.var) * 1e-3
        flux_all = np.vstack((flux_OII, flux_Hbeta, flux_OIII4960, flux_OIII5008))
        flux_err_all = np.vstack((flux_OII_err, flux_Hbeta_err, flux_OIII4960_err, flux_OIII5008_err))

    if method == 'voronoi':
        # Voronoi binning
        # Signal
        OIII5008_data = cube_OIII5008.select_lambda(8140, 8180).data * 1e-3
        OIII5008_sig = np.sum(OIII5008_data, axis=0)

        # Noise #1
        OIII5008_err = np.sqrt(cube_OIII5008.select_lambda(8140, 8180).var) * 1e-3
        OIII5008_err_int = np.sqrt(np.sum(OIII5008_err ** 2, axis=0))

        # # Noise #2
        OIII5008_std = np.std(cube_OIII5008.select_lambda(8180, 8200).data * 1e-3, axis=0) \
                       * np.sqrt(np.shape(OIII5008_data)[0])
        #
        sn_array_1 = np.array([OIII5008_sig, OIII5008_err_int]).T.reshape(-1, 2)
        sn_array_2 = np.array([OIII5008_sig, OIII5008_std]).T.reshape(-1, 2)

        binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(xy_array[:, 0], xy_array[:, 1],
                                                                                    sn_array_1[:, 0],
                                                                                    sn_array_1[:, 1], sn_vor,
                                                                                    pixelsize=1,
                                                                                    quiet=1)

    for l in range(size ** 2):
        i, j = xy_array[l, 0], xy_array[l, 1]
        spec_model = lmfit.Model(model_all, missing='drop')

        if method == 'aperture':
            # For an aperture
            spe_OII = cube_OII.aperture((i, j), radius_aper, unit_center=None, is_sum=False)  # Unit in arcsec
            spe_Hbeta = cube_Hbeta.aperture((i, j), radius_aper, unit_center=None, is_sum=False)
            spe_OIII4960 = cube_OIII4960.aperture((i, j), radius_aper, unit_center=None, is_sum=False)
            spe_OIII5008 = cube_OIII5008.aperture((i, j), radius_aper, unit_center=None, is_sum=False)

            flux_OII, flux_Hbeta = spe_OII.data * 1e-3, spe_Hbeta.data * 1e-3
            flux_OIII4960, flux_OIII5008 = spe_OIII4960.data * 1e-3, spe_OIII5008.data * 1e-3
            flux_OII_err, flux_Hbeta_err = np.sqrt(spe_OII.var) * 1e-3, np.sqrt(spe_Hbeta.var) * 1e-3
            flux_OIII4960_err = np.sqrt(spe_OIII4960.var) * 1e-3
            flux_OIII5008_err = np.sqrt(spe_OIII5008.var) * 1e-3
            flux_all = np.hstack((flux_OII, flux_Hbeta, flux_OIII4960, flux_OIII5008))
            flux_err_all = np.hstack((flux_OII_err, flux_Hbeta_err, flux_OIII4960_err, flux_OIII5008_err))

            result = spec_model.fit(data=flux_all, wave_vac=wave_vac_all, params=parameters,
                                    weights=1 / flux_err_all)

        elif method == 'pixel':
            result = spec_model.fit(data=flux_all[:, i, j], wave_vac=wave_vac_all, params=parameters,
                                    weights=1 / flux_err_all[:, i, j])
        elif method == 'voronoi':
            ll = np.where(binNum == binNum[l])
            ii, jj = xy_array[ll, 0], xy_array[ll, 1]
            result = spec_model.fit(data=flux_all[:, ii, jj].sum(axis=(1, 2)), wave_vac=wave_vac_all,
                                    params=parameters,
                                    weights=1 / np.sqrt((flux_err_all[:, ii, jj] ** 2).sum(axis=(1, 2))))
        # Load parameter
        z, dz = result.best_values['z'], result.params['z'].stderr
        sigma, dsigma = result.best_values['sigma_kms'], result.params['sigma_kms'].stderr
        # sigma_Hbeta, dsigma_Hbeta = result.best_values['sigma_kms_Hbeta'], result.params['sigma_kms_Hbeta'].stderr
        # sigma_OIII4960, dsigma_OIII4960 = result.best_values['sigma_kms_OIII4960'], \
        #                                   result.params['sigma_kms_OIII4960'].stderr
        # sigma_OIII5008, dsigma_OIII5008 = result.best_values['sigma_kms_OIII5008'], \
        #                                   result.params['sigma_kms_OIII5008'].stderr
        flux_OII, dflux_OII = result.best_values['flux_OII'], result.params['flux_OII'].stderr
        flux_Hbeta, dflux_Hbeta = result.best_values['flux_Hbeta'], result.params['flux_Hbeta'].stderr
        # flux_OIII4960, dflux_OIII4960 = result.best_values['flux_OIII4960'], result.params['flux_OIII4960'].stderr
        flux_OIII5008, dflux_OIII5008 = result.best_values['flux_OIII5008'], result.params['flux_OIII5008'].stderr
        r_OII, dr_OII = result.best_values['r_OII3729_3727'], result.params['r_OII3729_3727'].stderr

        a_OII, da_OII = result.best_values['a_OII'], result.params['a_OII'].stderr
        b_OII, db_OII = result.best_values['b_OII'], result.params['b_OII'].stderr
        a_Hbeta, da_Hbeta = result.best_values['a_Hbeta'], result.params['a_Hbeta'].stderr
        b_Hbeta, db_Hbeta = result.best_values['b_Hbeta'], result.params['b_Hbeta'].stderr
        a_OIII4960, da_OIII4960 = result.best_values['a_OIII4960'], result.params['a_OIII4960'].stderr
        b_OIII4960, db_OIII4960 = result.best_values['b_OIII4960'], result.params['b_OIII4960'].stderr
        a_OIII5008, da_OIII5008 = result.best_values['a_OIII5008'], result.params['a_OIII5008'].stderr
        b_OIII5008, db_OIII5008 = result.best_values['b_OIII5008'], result.params['b_OIII5008'].stderr

        #
        z_fit[i, j], dz_fit[i, j] = z, dz
        r_fit[i, j], dr_fit[i, j] = r_OII, dr_OII
        fit_success[i, j] = result.success
        sigma_fit[i, j] = sigma
        dsigma_fit[i, j] = dsigma
        # sigma_fit[:, i, j] = [sigma_OII, sigma_Hbeta, sigma_OIII4960, sigma_OIII5008]
        # dsigma_fit[:, i, j] = [dsigma_OII, dsigma_Hbeta, dsigma_OIII4960, dsigma_OIII5008]
        flux_fit[:, i, j] = [flux_OII, flux_Hbeta, flux_OIII5008]
        dflux_fit[:, i, j] = [dflux_OII, dflux_Hbeta, dflux_OIII5008]
        a_fit[:, i, j] = [a_OII, a_Hbeta, a_OIII4960, a_OIII5008]
        da_fit[:, i, j] = [da_OII, da_Hbeta, da_OIII4960, da_OIII5008]
        b_fit[:, i, j] = [b_OII, b_Hbeta, b_OIII4960, b_OIII5008]
        db_fit[:, i, j] = [db_OII, db_Hbeta, db_OIII4960, db_OIII5008]

        # Save the fitting param
        z_qso = 0.6282144177077355
        v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)
        info = np.array([z_fit, r_fit, fit_success, sigma_fit, flux_fit[0], flux_fit[1], flux_fit[2], a_fit[0],
                         a_fit[1], a_fit[2], a_fit[3], b_fit[0], b_fit[1], b_fit[2], b_fit[3]])
        info_err = np.array([dz_fit, dr_fit, dsigma_fit, dflux_fit[0], dflux_fit[1], dflux_fit[2], da_fit[0], da_fit[1],
                             da_fit[2], da_fit[3], db_fit[0], db_fit[1], db_fit[2], db_fit[3]])
        fits.writeto('/Users/lzq/Dropbox/Data/CGM/fitOOHbeta_info_' + method  + '_' + method_spe + '.fits', info,
                     overwrite=True)
        fits.writeto('/Users/lzq/Dropbox/Data/CGM/fitOOHbeta_info_err_' + method + '_' + method_spe + '.fits', info_err,
                     overwrite=True)


# Save the fitting param
FitLines(method='voronoi', method_spe='20', radius=100, radius_aper=0.7, sn_vor=20)
