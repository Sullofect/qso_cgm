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


# Define the model fit function
def model(wave_vac, z, sigma_kms, flux_OII, r_OII3729_3727, a, b):
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


# Fitting the narrow band image profile
path_cube_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
cube_OII = Cube(path_cube_OII)
cube_OII = cube_OII.subcube((80, 100), 50, unit_center=None, unit_size=None)
cube_OII[0, :, :].write('/Users/lzq/Dropbox/Data/CGM/image_OII_fitline.fits')

redshift_guess = 0.63
sigma_kms_guess = 150.0
flux_OII_guess = 0.01
r_OII3729_3727_guess = 2

parameters = lmfit.Parameters()
parameters.add_many(('z', redshift_guess, True, 0.62, 0.64, None),
                    ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
                    ('flux_OII', flux_OII_guess, True, None, None, None),
                    ('r_OII3729_3727', r_OII3729_3727_guess, True, 0.2, 1.6, None),
                    ('a', 0.0, False, None, None, None),
                    ('b', 0.0, False, None, None, None))

size = np.shape(cube_OII)[1]
fit_success = np.zeros((size, size))
z_fit, dz_fit = np.zeros((size, size)), np.zeros((size, size))
sigma_fit, dsigma_fit = np.zeros((size, size)), np.zeros((size, size))
flux_fit, dflux_fit = np.zeros((size, size)), np.zeros((size, size))
r_fit, dr_fit = np.zeros((size, size)), np.zeros((size, size))
a_fit, b_fit = np.zeros((size, size)), np.zeros((size, size))
da_fit, db_fit = np.zeros((size, size)), np.zeros((size, size))

#
wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())

#
for i in range(size):  # i = p (y), j = q (x)
    for j in range(size):
        flux_OII = cube_OII[:, i, j].data * 1e-3
        flux_OII_err = np.sqrt(cube_OII[:, i, j].var) * 1e-3
        spec_model = lmfit.Model(model, missing='drop')
        result = spec_model.fit(flux_OII, wave_vac=wave_OII_vac, params=parameters, weights=1 / flux_OII_err)

        #
        z, sigma, flux = result.best_values['z'], result.best_values['sigma_kms'], result.best_values['flux_OII']
        r, a, b = result.best_values['r_OII3729_3727'], result.best_values['a'], result.best_values['b']

        #
        dz, dsigma, dflux = result.params['z'].stderr, result.params['sigma_kms'].stderr, \
                            result.params['flux_OII'].stderr
        dr, da, db = result.params['r_OII3729_3727'].stderr, result.params['a'].stderr, result.params['b'].stderr
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
        r_fit[i, j], a_fit[i, j], b_fit[i, j] = r, a, b
        dr_fit[i, j], da_fit[i, j], db_fit[i, j] = dr, da, db

z_qso = 0.6282144177077355
v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)

info = np.array([z_fit, sigma_fit, flux_fit, fit_success, r_fit, a_fit, b_fit])
info_err = np.array([dz_fit, dsigma_fit, dflux_fit, dr_fit, da_fit, db_fit])
fits.writeto('/Users/lzq/Dropbox/Data/CGM/fitOII_info_test1.fits', info, overwrite=True)
fits.writeto('/Users/lzq/Dropbox/Data/CGM/fitOII_info_err_test1.fits', info_err, overwrite=True)
