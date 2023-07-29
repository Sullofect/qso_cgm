import os
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355


def model(wave_vac, z, sigma_kms, flux_OIII5008, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OIII5008_vac = 5008.239

    wave_OIII5008_obs = wave_OIII5008_vac * (1 + z)
    sigma_OIII5008_A = np.sqrt((sigma_kms / c_kms * wave_OIII5008_obs) ** 2 + (getSigma_MUSE(wave_OIII5008_obs)) ** 2)

    peak_OIII5008 = flux_OIII5008 / np.sqrt(2 * sigma_OIII5008_A ** 2 * np.pi)
    OIII5008_gaussian = peak_OIII5008 * np.exp(-(wave_vac - wave_OIII5008_obs) ** 2 / 2 / sigma_OIII5008_A ** 2)

    return OIII5008_gaussian + a * wave_vac + b


# Fitting the narrow band image profile
path_cube_OIII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                              'CUBE_OIII_5008_line_offset_zapped.fits')
cube_OIII = Cube(path_cube_OIII)
# cube_OIII = cube_OIII.subcube((80, 100), 5, unit_center=None, unit_size=None)
cube_OIII[0, :, :].write('/Users/lzq/Dropbox/Data/CGM/image_plot/image_OIII_fitline_zapped.fits')

redshift_guess = 0.63
sigma_kms_guess = 150.0
flux_OIII5008_guess = 42

parameters = lmfit.Parameters()
parameters.add_many(('z', redshift_guess, True, 0.62, 0.64, None),
                    ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
                    ('flux_OIII5008', flux_OIII5008_guess, True, None, None, None),
                    ('a', 0.0, False, None, None, None),
                    ('b', 0.0, False, None, None, None))

size = np.shape(cube_OIII)[1]
fit_success = np.zeros((size, size))
z_fit, dz_fit = np.zeros((size, size)), np.zeros((size, size))
sigma_fit, dsigma_fit = np.zeros((size, size)), np.zeros((size, size))
flux_fit, dflux_fit = np.zeros((size, size)), np.zeros((size, size))
a_fit, b_fit = np.zeros((size, size)), np.zeros((size, size))
da_fit, db_fit = np.zeros((size, size)), np.zeros((size, size))

#
wave_OIII_vac = pyasl.airtovac2(cube_OIII.wave.coord())

#
for i in range(size):  # i = p (y), j = q (x)
    for j in range(size):
        flux_OIII = cube_OIII[:, i, j].data * 1e-3
        flux_OIII_err = np.sqrt(cube_OIII[:, i, j].var) * 1e-3
        spec_model = lmfit.Model(model, missing='drop')
        result = spec_model.fit(flux_OIII, wave_vac=wave_OIII_vac, params=parameters, weights=1 / flux_OIII_err)
        z, sigma, flux = result.best_values['z'], result.best_values['sigma_kms'], result.best_values['flux_OIII5008']
        a, b = result.best_values['a'], result.best_values['b']
        dz, dsigma, dflux = result.params['z'].stderr, result.params['sigma_kms'].stderr, \
                            result.params['flux_OIII5008'].stderr
        da, db = result.params['a'].stderr, result.params['b'].stderr

        # if i == 30:
        #     if (j > 30) and (j < 50):
        #         plt.plot(wave_OIII_vac, flux_OIII, '-')
        #         plt.plot(wave_OIII_vac, model(wave_OIII_vac, z, sigma, flux, a, b))
        #         plt.show()

        #
        fit_success[i, j] = result.success
        z_fit[i, j], dz_fit[i, j] = z, dz
        sigma_fit[i, j], dsigma_fit[i, j] = sigma, dsigma
        flux_fit[i, j], dflux_fit[i, j] = flux, dflux
        a_fit[i, j], b_fit[i, j] = a, b
        da_fit[i, j], db_fit[i, j] = da, db

z_qso = 0.6282144177077355
v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)

info = np.array([z_fit, sigma_fit, flux_fit, a_fit, b_fit])
info_err = np.array([dz_fit, dsigma_fit, dflux_fit, da_fit, db_fit])
fits.writeto('/Users/lzq/Dropbox/Data/CGM/fit_OIII/fitOIII_info_zapped.fits', info, overwrite=True)
fits.writeto('/Users/lzq/Dropbox/Data/CGM/fit_OIII/fitOIII_info_err_zapped.fits', info_err, overwrite=True)