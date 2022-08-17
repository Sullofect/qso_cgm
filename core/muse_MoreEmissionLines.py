import os
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
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


def model_NeV(wave_vac, z, sigma_kms, flux_NeV3346, a, b):
    # Constants
    c_kms = 2.998e5
    wave_NeV3346_vac = 3346.79

    wave_NeV3346_obs = wave_NeV3346_vac * (1 + z)
    sigma_NeV3346_A = np.sqrt((sigma_kms / c_kms * wave_NeV3346_obs) ** 2 + (getSigma_MUSE(wave_NeV3346_obs)) ** 2)

    peak_NeV3346 = flux_NeV3346 / np.sqrt(2 * sigma_NeV3346_A ** 2 * np.pi)
    NeV3346_gaussian = peak_NeV3346 * np.exp(-(wave_vac - wave_NeV3346_obs) ** 2 / 2 / sigma_NeV3346_A ** 2)

    return NeV3346_gaussian + a * wave_vac + b

def model_NeIII(wave_vac, z, sigma_kms, flux_NeIII3869, a, b):
    # Constants
    c_kms = 2.998e5
    wave_NeIII3869_vac = 3346.79

    wave_NeIII3869_obs = wave_NeIII3869_vac * (1 + z)
    sigma_NeIII3869_A = np.sqrt((sigma_kms / c_kms * wave_NeIII3869_obs) ** 2 + (getSigma_MUSE(wave_NeIII3869_obs)) ** 2)

    peak_NeIII3869 = flux_NeIII3869 / np.sqrt(2 * sigma_NeIII3869_A ** 2 * np.pi)
    NeIII3869_gaussian = peak_NeIII3869 * np.exp(-(wave_vac - wave_NeIII3869_obs) ** 2 / 2 / sigma_NeIII3869_A ** 2)

    return NeIII3869_gaussian + a * wave_vac + b

def model_HeI(wave_vac, z, sigma_kms, flux_HeI3889, a, b):
    # Constants
    c_kms = 2.998e5
    wave_HeI3889_vac = 3889.00

    wave_HeI3889_obs = wave_HeI3889_vac * (1 + z)
    sigma_HeI3889_A = np.sqrt((sigma_kms / c_kms * wave_HeI3889_obs) ** 2 + (getSigma_MUSE(wave_HeI3889_obs)) ** 2)

    peak_HeI3889 = flux_HeI3889 / np.sqrt(2 * sigma_HeI3889_A ** 2 * np.pi)
    HeI3889_gaussian = peak_HeI3889 * np.exp(-(wave_vac - wave_HeI3889_obs) ** 2 / 2 / sigma_HeI3889_A ** 2)

    return HeI3889_gaussian + a * wave_vac + b

def model_Hepi(wave_vac, z, sigma_kms, flux_Heps, a, b):
    # Constants
    c_kms = 2.998e5
    wave_Heps_vac = 3971.20

    wave_Heps_obs = wave_Heps_vac * (1 + z)
    sigma_Heps_A = np.sqrt((sigma_kms / c_kms * wave_Heps_obs) ** 2 + (getSigma_MUSE(wave_Heps_obs)) ** 2)

    peak_Heps = flux_Heps / np.sqrt(2 * sigma_Heps_A ** 2 * np.pi)
    Heps_gaussian = peak_Heps * np.exp(-(wave_vac - wave_Heps_obs) ** 2 / 2 / sigma_Heps_A ** 2)

    return Heps_gaussian + a * wave_vac + b

def model_Hdel(wave_vac, z, sigma_kms, flux_Hdel, a, b):
    # Constants
    c_kms = 2.998e5
    wave_Hdel_vac = 4102.89

    wave_Hdel_obs = wave_Hdel_vac * (1 + z)
    sigma_Hdel_A = np.sqrt((sigma_kms / c_kms * wave_Hdel_obs) ** 2 + (getSigma_MUSE(wave_Hdel_obs)) ** 2)

    peak_Hdel = flux_Hdel / np.sqrt(2 * sigma_Hdel_A ** 2 * np.pi)
    Hdel_gaussian = peak_Hdel * np.exp(-(wave_vac - wave_Hdel_obs) ** 2 / 2 / sigma_Hdel_A ** 2)

    return Hdel_gaussian + a * wave_vac + b

def model_Hgam(wave_vac, z, sigma_kms, flux_Hgam, a, b):
    # Constants
    c_kms = 2.998e5
    wave_Hgam_vac = 4341.68

    wave_Hgam_obs = wave_Hgam_vac * (1 + z)
    sigma_Hgam_A = np.sqrt((sigma_kms / c_kms * wave_Hgam_obs) ** 2 + (getSigma_MUSE(wave_Hgam_obs)) ** 2)

    peak_Hgam = flux_Hgam / np.sqrt(2 * sigma_Hgam_A ** 2 * np.pi)
    Hgam_gaussian = peak_Hgam * np.exp(-(wave_vac - wave_Hgam_obs) ** 2 / 2 / sigma_Hgam_A ** 2)

    return Hgam_gaussian + a * wave_vac + b

def model_OIII4364(wave_vac, z, sigma_kms, flux_OIII4364, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OIII4364_vac = 4364.44

    wave_OIII4364_obs = wave_OIII4364_vac * (1 + z)
    sigma_OIII4364_A = np.sqrt((sigma_kms / c_kms * wave_OIII4364_obs) ** 2 + (getSigma_MUSE(wave_OIII4364_obs)) ** 2)

    peak_OIII4364 = flux_OIII4364 / np.sqrt(2 * sigma_OIII4364_A ** 2 * np.pi)
    OIII4364_gaussian = peak_OIII4364 * np.exp(-(wave_vac - wave_OIII4364_obs) ** 2 / 2 / sigma_OIII4364_A ** 2)

    return OIII4364_gaussian + a * wave_vac + b

def model_HeII4687(wave_vac, z, sigma_kms, flux_HeII4687, a, b):
    # Constants
    c_kms = 2.998e5
    wave_HeII4687_vac = 4364.44

    wave_HeII4687_obs = wave_HeII4687_vac * (1 + z)
    sigma_HeII4687_A = np.sqrt((sigma_kms / c_kms * wave_HeII4687_obs) ** 2 + (getSigma_MUSE(wave_HeII4687_obs)) ** 2)

    peak_HeII4687 = flux_HeII4687 / np.sqrt(2 * sigma_HeII4687_A ** 2 * np.pi)
    HeII4687_gaussian = peak_HeII4687 * np.exp(-(wave_vac - wave_HeII4687_obs) ** 2 / 2 / sigma_HeII4687_A ** 2)

    return HeII4687_gaussian + a * wave_vac + b

# [Ne V] 3346.79, [Ne III] 3869, He I 3889, Hepisolon. Hdelta, Hgamma, [O III] 4364, He II 4687
# lines_more = (1 + z) * np.array([3346.79, 3869.86, 3889.00, 3971.20, 4102.89, 4341.68, 4364.44, 4687.31])

def model_all(wave_vac, z, sigma_kms, flux_OII, flux_Hbeta, flux_OIII5008, r_OII3729_3727, a_OII, b_OII, a_Hbeta,
              b_Hbeta, a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008):

    m_OII = model_OII(wave_vac[0], z, sigma_kms, flux_OII, r_OII3729_3727, a_OII, b_OII)
    m_Hbeta = model_Hbeta(wave_vac[1], z, sigma_kms, flux_Hbeta, a_Hbeta, b_Hbeta)
    m_OIII4960 = model_OIII4960(wave_vac[2], z, sigma_kms, flux_OIII5008 / 3, a_OIII4960, b_OIII4960)
    m_OIII5008 = model_OIII5008(wave_vac[3], z, sigma_kms, flux_OIII5008, a_OIII5008, b_OIII5008)
    return np.hstack((m_OII, m_Hbeta, m_OIII4960, m_OIII5008))


# Read Data
path_cube = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data',
                        'ESO_DEEP_offset_zapped.fits_SUBTRACTED.fits')
path_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                        'CUBE_OII_line_offset_zapped.fits')
path_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                          'CUBE_Hbeta_line_offset_zapped.fits')
path_bet = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                        'CUBE_bet_Hbeta_OIII_line_offset_zapped.fits')
path_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                             'CUBE_OIII_4960_line_offset_zapped.fits')
path_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                             'CUBE_OIII_5008_line_offset_zapped.fits')

cube = Cube(path_cube)
cube_OII = Cube(path_OII)
cube_Hbeta = Cube(path_Hbeta)
cube_bet = Cube(path_bet)
cube_OIII4960 = Cube(path_OIII4960)
cube_OIII5008 = Cube(path_OIII5008)
wave_cube = pyasl.airtovac2(cube.wave.coord())
wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
wave_Hbeta_vac = pyasl.airtovac2(cube_Hbeta.wave.coord())
wave_bet_vac = pyasl.airtovac2(cube_bet.wave.coord())
wave_OIII4960_vac = pyasl.airtovac2(cube_OIII4960.wave.coord())
wave_OIII5008_vac = pyasl.airtovac2(cube_OIII5008.wave.coord())
wave_vac_stack = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_stack_plot = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_bet_vac, wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_all = np.array([wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)

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




# Def Plot function
def PlotGasSpectra(ra_array, dec_array, radius_array, text_array, figname='spectra_gas_1'):
    global cube_OII, cube_Hbeta, cube_OIII4960, cube_OIII5008, wave_vac_stack, wave_vac_all
    fig, axarr = plt.subplots(len(ra_array), 2, figsize=(10, len(ra_array) * 2.5),
                              gridspec_kw={'width_ratios': [1, 3]}, dpi=300)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.1)
    # flux_info = np.zeros((len(ra_array), 6))
    # axarr = axarr.ravel()
    for i in range(len(ra_array)):
        spe_cube_i = cube.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
        spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
        spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_bet_i = cube_bet.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII4960_i = cube_OIII4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII5008_i = cube_OIII5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)

        # Continuum subtraction for faint lines
        wave_min, wave_max = wave_cube.min(), wave_cube.max()

        # For [Ne V] 3346.79
        spe_cube_i.mask_region(wave_min, 5400)
        spe_cube_i.mask_region(5440, 5460)
        spe_cube_i.mask_region(5500, wave_max)
        conti_NeV3346_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_NeV3346_i
        spe_cube_i.mask_region(wave_min, 5440)
        spe_cube_i.mask_region(5460, wave_max)
        flux_NeV3346_i, flux_NeV3346_err_i = spe_cube_i.data * 1e-3, np.sqrt(spe_cube_i.var) * 1e-3
        spe_cube_i.unmask()
        print(np.nanmedian(flux_NeV3346_i))

        # For [Ne III] 3869
        spe_cube_i.mask_region(wave_min, 6250)
        spe_cube_i.mask_region(6290, 6305)
        spe_cube_i.mask_region(6320, wave_max)
        conti_NeIII3869_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_NeIII3869_i

        # For He I 3889
        spe_cube_i.mask_region(wave_min, 6310)
        spe_cube_i.mask_region(6325, 6335)
        spe_cube_i.mask_region(6350, wave_max)
        conti_HeI3889_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_HeI3889_i

        # For Hepsilon
        spe_cube_i.mask_region(wave_min, 6430)
        spe_cube_i.mask_region(6450, 6470)
        spe_cube_i.mask_region(6490, wave_max)
        conti_Heps_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_Heps_i

        # For Hdelta
        spe_cube_i.mask_region(wave_min, 6640)
        spe_cube_i.mask_region(6670, 6685)
        spe_cube_i.mask_region(6710, wave_max)
        conti_Hdelta_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_Hdelta_i

        # For Hgamma
        spe_cube_i.mask_region(wave_min, 7020)
        spe_cube_i.mask_region(7060, 7070)
        spe_cube_i.mask_region(7090, wave_max)
        conti_Hgam_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_Hgam_i

        # For [O III] 4364
        spe_cube_i.mask_region(wave_min, 7085)
        spe_cube_i.mask_region(7098, 7108)
        spe_cube_i.mask_region(7125, wave_max)
        conti_OIII4364_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_OIII4364_i

        # For He II 4687
        spe_cube_i.mask_region(wave_min, 7595)
        spe_cube_i.mask_region(7620, 7640)
        spe_cube_i.mask_region(7660, wave_max)
        conti_HeII4687_i = spe_cube_i.poly_spec(3, weight=True)
        spe_cube_i.unmask()
        spe_cube_i -= conti_HeII4687_i

        # Second round continuum subtraction
        spe_OII_i.mask_region(6050, 6090)
        conti_OII = spe_OII_i.poly_spec(3, weight=True)
        spe_OII_i.unmask()
        spe_OII_i -= conti_OII

        spe_Hbeta_i.mask_region(7905, 7930)
        conti_Hbeta = spe_Hbeta_i.poly_spec(3, weight=True)
        spe_Hbeta_i.unmask()
        spe_Hbeta_i -= conti_Hbeta

        spe_OIII4960_i.mask_region(8060, 8100)
        conti_OIII4960 = spe_OIII4960_i.poly_spec(3, weight=True)
        spe_OIII4960_i.unmask()
        spe_OIII4960_i -= conti_OIII4960

        spe_OIII5008_i.mask_region(8140, 8180)
        conti_OIII5008 = spe_OIII5008_i.poly_spec(3, weight=True)
        spe_OIII5008_i.unmask()
        spe_OIII5008_i -= conti_OIII5008


        # Load the data
        flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
        flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
        flux_bet_i, flux_bet_err_i = spe_bet_i.data * 1e-3, np.sqrt(spe_bet_i.var) * 1e-3
        flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
        flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3
        flux_all = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
        flux_err_all = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

        spec_model = lmfit.Model(model_all, missing='drop')
        result = spec_model.fit(data=flux_all, wave_vac=wave_vac_all, params=parameters,
                                weights=1 / flux_err_all)

        # For plotting
        flux_all_plot = np.hstack((flux_OII_i, flux_Hbeta_i, flux_bet_i, flux_OIII4960_i, flux_OIII5008_i))
        flux_err_all_plot = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_bet_err_i, flux_OIII4960_err_i,
                                       flux_OIII5008_err_i))

        # Load fitted result
        z, dz = result.best_values['z'], result.params['z'].stderr
        sigma, dsigma = result.best_values['sigma_kms'], result.params['sigma_kms'].stderr
        flux_OII, dflux_OII = result.best_values['flux_OII'], result.params['flux_OII'].stderr
        flux_Hbeta, dflux_Hbeta = result.best_values['flux_Hbeta'], result.params['flux_Hbeta'].stderr
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

        # Save the fitted result
        # flux_info[i, :] = np.array([flux_OII, flux_Hbeta, flux_OIII5008, dflux_OII, dflux_Hbeta, dflux_OIII5008])

        # axarr[i, 0].plot(wave_vac_stack, flux_all, color='k', drawstyle='steps-mid', lw=1)
        # axarr[i, 0].plot(wave_vac_stack, flux_err_all, color='lightgrey', lw=1)
        # axarr[i, 0].plot(wave_vac_stack, model_all(wave_vac_all, z, sigma, flux_OII, flux_Hbeta, flux_OIII5008,
        #                                            r_OII, a_OII, b_OII, a_Hbeta, b_Hbeta, a_OIII4960,
        #                                            b_OIII4960, a_OIII5008, b_OIII5008), '-r')
        # axarr[i, 1].plot(wave_vac_stack_plot, flux_all_plot, color='k', drawstyle='steps-mid', lw=1)
        # axarr[i, 1].plot(wave_vac_stack_plot, flux_err_all_plot, color='lightgrey', lw=1)
        # axarr[i, 1].plot(wave_vac_stack, model_all(wave_vac_all, z, sigma, flux_OII, flux_Hbeta, flux_OIII5008,
        #                                            r_OII, a_OII, b_OII, a_Hbeta, b_Hbeta, a_OIII4960,
        #                                            b_OIII4960, a_OIII5008, b_OIII5008), '-r')
        axarr[i, 0].plot(wave_cube, flux_NeV3346_i, color='k', drawstyle='steps-mid', lw=1)
        axarr[i, 0].set_title(text_array[i], x=0.2, y=0.75, size=20)
        axarr[i, 0].set_xlim(5400, 5500)
        # axarr[i, 1].set_xlim(7900, 8200)
        #
        # axarr[i, 0].set_xlim(6020, 6120)
        # axarr[i, 1].set_xlim(7900, 8200)
        axarr[i, 0].spines['right'].set_visible(False)
        axarr[i, 1].spines['left'].set_visible(False)

        # Mark line info
        # [Ne V] 3346.79, [Ne III] 3869, He I 3889, Hepisolon. Hdelta, Hgamma, [O III] 4364, He II 4687
        lines_more = (1 + z) * np.array([3346.79, 3869.86, 3889.00, 3971.20, 4102.89, 4341.68, 4364.44, 4687.31])
        lines = (1 + z) * np.array([3727.092, 3729.8754960, 4862.721, 4960.295, 5008.239])
        axarr[i, 0].vlines(lines, ymin=[-5, -5, -5, -5, -5], ymax=[100, 100, 100, 100, 100], linestyles='dashed',
                           colors='grey')
        axarr[i, 1].vlines(lines, ymin=[-5, -5, -5, -5, -5], ymax=[100, 100, 100, 100, 100], linestyles='dashed',
                           colors='grey')
        # axarr[i, 0].set_ylim(flux_all.min() - 0.5, flux_all.max() + 0.5)
        # axarr[i, 1].set_ylim(flux_all.min() - 0.5, flux_all.max() + 0.5)
        axarr[i, 0].set_ylim(flux_all.min() - 0.5, 0.5)
        axarr[i, 1].set_ylim(flux_all.min() - 0.5, flux_all.max() + 0.5)
        if flux_Hbeta_i.max() > flux_OIII5008_i.max():
            axarr[i, 0].set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
            axarr[i, 1].set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
        axarr[0, 0].annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.55, 0.65), xycoords='axes fraction', size=20)
        axarr[0, 1].annotate(text=r'$\mathrm{H\beta}$', xy=(0.1, 0.65), xycoords='axes fraction', size=20)
        axarr[0, 1].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.45, 0.65), xycoords='axes fraction', size=20)
        axarr[0, 1].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.7, 0.65), xycoords='axes fraction', size=20)

        axarr[i, 0].minorticks_on()
        axarr[i, 1].minorticks_on()
        axarr[i, 0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right=False,
                                labelsize=20, size=5)
        axarr[i, 0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right=False,
                                size=3)
        axarr[i, 0].tick_params(axis='y', which='both', right=False, labelright=False)
        axarr[i, 1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False, right='on',
                                labelsize=20, size=5)
        axarr[i, 1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False, right='on',
                                size=3)
        axarr[i, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
        if i != len(ra_array) - 1:
            axarr[i, 0].tick_params(axis='x', which='both', labelbottom=False)
            axarr[i, 1].tick_params(axis='x', which='both', labelbottom=False)
    # fits.writeto('/Users/lzq/Dropbox/Data/CGM/line_profile_selected_region.fits', flux_info, overwrite=True)
    fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=0.07)
    fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20, x=0.05)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '.png', bbox_inches='tight')


# Plot the data
# Read region file
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')


PlotGasSpectra(ra_array[0:2], dec_array[0:2], radius_array[0:2], text_array[0:2], figname='spectra_gas_S2')
