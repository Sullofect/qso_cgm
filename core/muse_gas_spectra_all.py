import os
import lmfit
import extinction
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from muse_gas_spectra_S3S4 import model_all as S3S4_model_all
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
from astropy.table import Table
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


def model_NeV3346(wave_vac, z, sigma_kms, flux_NeV3346, a, b):
    # Constants
    c_kms = 2.998e5
    wave_NeV3346_vac = pyasl.airtovac2(3345.821)

    wave_NeV3346_obs = wave_NeV3346_vac * (1 + z)
    sigma_NeV3346_A = np.sqrt((sigma_kms / c_kms * wave_NeV3346_obs) ** 2 + (getSigma_MUSE(wave_NeV3346_obs)) ** 2)

    peak_NeV3346 = flux_NeV3346 / np.sqrt(2 * sigma_NeV3346_A ** 2 * np.pi)
    NeV3346_gaussian = peak_NeV3346 * np.exp(-(wave_vac - wave_NeV3346_obs) ** 2 / 2 / sigma_NeV3346_A ** 2)

    return NeV3346_gaussian + a * wave_vac + b


def model_NeIII3869(wave_vac, z, sigma_kms, flux_NeIII3869, a, b):
    # Constants
    c_kms = 2.998e5
    wave_NeIII3869_vac = pyasl.airtovac2(3868.760)

    wave_NeIII3869_obs = wave_NeIII3869_vac * (1 + z)
    sigma_NeIII3869_A = np.sqrt((sigma_kms / c_kms * wave_NeIII3869_obs) ** 2 + (getSigma_MUSE(wave_NeIII3869_obs)) ** 2)

    peak_NeIII3869 = flux_NeIII3869 / np.sqrt(2 * sigma_NeIII3869_A ** 2 * np.pi)
    NeIII3869_gaussian = peak_NeIII3869 * np.exp(-(wave_vac - wave_NeIII3869_obs) ** 2 / 2 / sigma_NeIII3869_A ** 2)

    return NeIII3869_gaussian + a * wave_vac + b


def model_HeI3889andH8(wave_vac, z, sigma_kms, flux_HeI3889, flux_H8, a, b):
    # Constants
    c_kms = 2.998e5
    wave_HeI3889_vac = pyasl.airtovac2(3888.647)
    wave_H8_vac = pyasl.airtovac2(3889.064)

    wave_HeI3889_obs = wave_HeI3889_vac * (1 + z)
    sigma_HeI3889_A = np.sqrt((sigma_kms / c_kms * wave_HeI3889_obs) ** 2 + (getSigma_MUSE(wave_HeI3889_obs)) ** 2)

    peak_HeI3889 = flux_HeI3889 / np.sqrt(2 * sigma_HeI3889_A ** 2 * np.pi)
    HeI3889_gaussian = peak_HeI3889 * np.exp(-(wave_vac - wave_HeI3889_obs) ** 2 / 2 / sigma_HeI3889_A ** 2)

    wave_H8_obs = wave_H8_vac * (1 + z)
    sigma_H8_A = np.sqrt((sigma_kms / c_kms * wave_H8_obs) ** 2 + (getSigma_MUSE(wave_H8_obs)) ** 2)

    peak_H8 = flux_H8 / np.sqrt(2 * sigma_H8_A ** 2 * np.pi)
    H8_gaussian = peak_H8 * np.exp(-(wave_vac - wave_H8_obs) ** 2 / 2 / sigma_H8_A ** 2)

    return HeI3889_gaussian + H8_gaussian + a * wave_vac + b


def model_NeIII3968andHeps(wave_vac, z, sigma_kms, flux_NeIII3968, flux_Heps, a, b):
    # Constants
    c_kms = 2.998e5
    wave_NeIII3968_vac = pyasl.airtovac2(3967.470)
    wave_Heps_vac = pyasl.airtovac2(3970.079)

    wave_NeIII3968_obs = wave_NeIII3968_vac * (1 + z)
    sigma_NeIII3968_A = np.sqrt((sigma_kms / c_kms * wave_NeIII3968_obs) ** 2 + (getSigma_MUSE(wave_NeIII3968_obs)) ** 2)

    peak_NeIII3968 = flux_NeIII3968 / np.sqrt(2 * sigma_NeIII3968_A ** 2 * np.pi)
    NeIII3968_gaussian = peak_NeIII3968 * np.exp(-(wave_vac - wave_NeIII3968_obs) ** 2 / 2 / sigma_NeIII3968_A ** 2)

    wave_Heps_obs = wave_Heps_vac * (1 + z)
    sigma_Heps_A = np.sqrt((sigma_kms / c_kms * wave_Heps_obs) ** 2 + (getSigma_MUSE(wave_Heps_obs)) ** 2)

    peak_Heps = flux_Heps / np.sqrt(2 * sigma_Heps_A ** 2 * np.pi)
    Heps_gaussian = peak_Heps * np.exp(-(wave_vac - wave_Heps_obs) ** 2 / 2 / sigma_Heps_A ** 2)

    return NeIII3968_gaussian + Heps_gaussian + a * wave_vac + b


def model_Hdel(wave_vac, z, sigma_kms, flux_Hdel, a, b):
    # Constants
    c_kms = 2.998e5
    wave_Hdel_vac = pyasl.airtovac2(4101.742)

    wave_Hdel_obs = wave_Hdel_vac * (1 + z)
    sigma_Hdel_A = np.sqrt((sigma_kms / c_kms * wave_Hdel_obs) ** 2 + (getSigma_MUSE(wave_Hdel_obs)) ** 2)

    peak_Hdel = flux_Hdel / np.sqrt(2 * sigma_Hdel_A ** 2 * np.pi)
    Hdel_gaussian = peak_Hdel * np.exp(-(wave_vac - wave_Hdel_obs) ** 2 / 2 / sigma_Hdel_A ** 2)

    return Hdel_gaussian + a * wave_vac + b


def model_Hgam(wave_vac, z, sigma_kms, flux_Hgam, a, b):
    # Constants
    c_kms = 2.998e5
    wave_Hgam_vac = pyasl.airtovac2(4340.471)

    wave_Hgam_obs = wave_Hgam_vac * (1 + z)
    sigma_Hgam_A = np.sqrt((sigma_kms / c_kms * wave_Hgam_obs) ** 2 + (getSigma_MUSE(wave_Hgam_obs)) ** 2)

    peak_Hgam = flux_Hgam / np.sqrt(2 * sigma_Hgam_A ** 2 * np.pi)
    Hgam_gaussian = peak_Hgam * np.exp(-(wave_vac - wave_Hgam_obs) ** 2 / 2 / sigma_Hgam_A ** 2)

    return Hgam_gaussian + a * wave_vac + b


def model_OIII4364(wave_vac, z, sigma_kms, flux_OIII4364, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OIII4364_vac = pyasl.airtovac2(4363.210)

    wave_OIII4364_obs = wave_OIII4364_vac * (1 + z)
    sigma_OIII4364_A = np.sqrt((sigma_kms / c_kms * wave_OIII4364_obs) ** 2 + (getSigma_MUSE(wave_OIII4364_obs)) ** 2)

    peak_OIII4364 = flux_OIII4364 / np.sqrt(2 * sigma_OIII4364_A ** 2 * np.pi)
    OIII4364_gaussian = peak_OIII4364 * np.exp(-(wave_vac - wave_OIII4364_obs) ** 2 / 2 / sigma_OIII4364_A ** 2)

    return OIII4364_gaussian + a * wave_vac + b


def model_HeII4687(wave_vac, z, sigma_kms, flux_HeII4687, a, b):
    # Constants
    c_kms = 2.998e5
    wave_HeII4687_vac = pyasl.airtovac2(4685.710)

    wave_HeII4687_obs = wave_HeII4687_vac * (1 + z)
    sigma_HeII4687_A = np.sqrt((sigma_kms / c_kms * wave_HeII4687_obs) ** 2 + (getSigma_MUSE(wave_HeII4687_obs)) ** 2)

    peak_HeII4687 = flux_HeII4687 / np.sqrt(2 * sigma_HeII4687_A ** 2 * np.pi)
    HeII4687_gaussian = peak_HeII4687 * np.exp(-(wave_vac - wave_HeII4687_obs) ** 2 / 2 / sigma_HeII4687_A ** 2)

    return HeII4687_gaussian + a * wave_vac + b


def model_all(wave_vac, z, sigma_kms, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8, flux_NeIII3968, flux_Heps,
              flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687, flux_OII, flux_Hbeta, flux_OIII5008, r_OII3729_3727,
              a_NeV3346, b_NeV3346, a_NeIII3869, b_NeIII3869, a_HeI3889, b_HeI3889, a_NeIII3968,
              b_NeIII3968, a_Hdel, b_Hdel, a_Hgam, b_Hgam, a_OIII4364, b_OIII4364, a_HeII4687,
              b_HeII4687, a_OII, b_OII, a_Hbeta, b_Hbeta, a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008):

    # Weak lines
    m_NeV3356 = model_NeV3346(wave_vac[0], z, sigma_kms, flux_NeV3346, a_NeV3346, b_NeV3346)
    m_NeIII3869 = model_NeIII3869(wave_vac[1], z, sigma_kms, flux_NeIII3869, a_NeIII3869, b_NeIII3869)
    m_HeI3889andH8 = model_HeI3889andH8(wave_vac[2], z, sigma_kms, flux_HeI3889, flux_H8, a_HeI3889, b_HeI3889)
    m_NeIII3968andHeps = model_NeIII3968andHeps(wave_vac[3], z, sigma_kms, flux_NeIII3968, flux_Heps, a_NeIII3968,
                                                b_NeIII3968)
    m_Hdel = model_Hdel(wave_vac[4], z, sigma_kms, flux_Hdel, a_Hdel, b_Hdel)
    m_Hgam = model_Hgam(wave_vac[5], z, sigma_kms, flux_Hgam, a_Hgam, b_Hgam)
    m_OIII4364 = model_OIII4364(wave_vac[6], z, sigma_kms, flux_OIII4364, a_OIII4364, b_OIII4364)
    m_HeII4687 = model_HeII4687(wave_vac[7], z, sigma_kms, flux_HeII4687, a_HeII4687, b_HeII4687)

    # Strong lines
    m_OII = model_OII(wave_vac[8], z, sigma_kms, flux_OII, r_OII3729_3727, a_OII, b_OII)
    m_Hbeta = model_Hbeta(wave_vac[9], z, sigma_kms, flux_Hbeta, a_Hbeta, b_Hbeta)
    m_OIII4960 = model_OIII4960(wave_vac[10], z, sigma_kms, flux_OIII5008 / 3, a_OIII4960, b_OIII4960)
    m_OIII5008 = model_OIII5008(wave_vac[11], z, sigma_kms, flux_OIII5008, a_OIII5008, b_OIII5008)
    return np.hstack((m_NeV3356, m_NeIII3869, m_HeI3889andH8, m_NeIII3968andHeps, m_Hdel, m_Hgam, m_OIII4364,
                      m_HeII4687, m_OII, m_Hbeta, m_OIII4960, m_OIII5008))

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

def extinction_ndarray(wave_ndarray, A_v):
    output = np.array([])
    for i in range(len(wave_ndarray)):
        ext_i = extinction.fm07(wave_ndarray[i], A_v)
        output = np.hstack((output, ext_i))
    return output


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

# Weak lines
cube = Cube(path_cube)
cube_NeV3346 = cube.select_lambda(5380, 5500)
cube_NeIII3869 = cube.select_lambda(6250, 6320)
cube_HeI3889 = cube.select_lambda(6310, 6350)
cube_Heps = cube.select_lambda(6430, 6490)
cube_Hdel = cube.select_lambda(6640, 6710)
cube_Hgam = cube.select_lambda(7020, 7090)
cube_OIII4364 = cube.select_lambda(7085, 7125)
cube_HeII4687 = cube.select_lambda(7595, 7660)

wave_NeV3346_vac = pyasl.airtovac2(cube_NeV3346.wave.coord())
wave_NeIII3869_vac = pyasl.airtovac2(cube_NeIII3869.wave.coord())
wave_HeI3889_vac = pyasl.airtovac2(cube_HeI3889.wave.coord())
wave_Heps_vac = pyasl.airtovac2(cube_Heps.wave.coord())
wave_Hdel_vac = pyasl.airtovac2(cube_Hdel.wave.coord())
wave_Hgam_vac = pyasl.airtovac2(cube_Hgam.wave.coord())
wave_OIII4364_vac = pyasl.airtovac2(cube_OIII4364.wave.coord())
wave_HeII4687_vac = pyasl.airtovac2(cube_HeII4687.wave.coord())

# Strong lines [O II], Hbeta, [O III]
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

# Strong lines
wave_vac_strong_stack = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_bet_vac, wave_OIII4960_vac, wave_OIII5008_vac))
idx_strong = (len(wave_vac_strong_stack) - len(wave_bet_vac)) * 3

# All lines
wave_vac_all = np.array([wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
                          wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac, wave_OII_vac, wave_Hbeta_vac,
                          wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)
wave_vac_all_plot = expand_wave(wave_vac_all, stack=False)
# wave_vac_all_stack = np.hstack((wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
#                                  wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac, wave_OII_vac, wave_Hbeta_vac,
#                                  wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_all_stack = expand_wave(wave_vac_all)
idx_all = len(wave_vac_all_stack)
idx_weak = idx_all - idx_strong

redshift_guess = 0.63
sigma_kms_guess = 150.0
# flux_OIII5008_guess = 0.01
r_OII3729_3727_guess = 2

parameters_all = lmfit.Parameters()
parameters_all.add_many(('z', redshift_guess, True, 0.62, 0.64, None),
                         ('sigma_kms', sigma_kms_guess, True, 10, 500, None),
                         ('flux_NeV3346', 0.01, True, None, None, None),
                         ('flux_NeIII3869', 0.05, True, None, None, None),
                         ('flux_HeI3889', 0.01, True, None, None, None),
                         ('flux_H8', 0.01, True, None, None, None),
                         ('flux_NeIII3968', 0.01, True, None, None, None),
                         ('flux_Heps', 0.03, True, None, None, None),
                         ('flux_Hdel', 0.01, True, None, None, None),
                         ('flux_Hgam', 0.01, True, None, None, None),
                         ('flux_OIII4364', 0.1, True, None, None, None),
                         ('flux_HeII4687', 0.005, True, None, None, None),
                         ('flux_OII', 0.01, True, None, None, None),
                         ('flux_Hbeta', 0.02, True, None, None, None),
                         ('flux_OIII5008', 0.1, True, None, None, None),
                         ('r_OII3729_3727', r_OII3729_3727_guess, True, 0.2, None, None),
                         ('a_NeV3346', 0.0, False, None, None, None),
                         ('b_NeV3346', 0.0, False, None, None, None),
                         ('a_NeIII3869', 0.0, False, None, None, None),
                         ('b_NeIII3869', 0.0, False, None, None, None),
                         ('a_HeI3889', 0.0, False, None, None, None),
                         ('b_HeI3889', 0.0, False, None, None, None),
                         ('a_NeIII3968', 0.0, False, None, None, None),
                         ('b_NeIII3968', 0.0, False, None, None, None),
                         ('a_Hdel', 0.0, False, None, None, None),
                         ('b_Hdel', 0.0, False, None, None, None),
                         ('a_Hgam', 0.0, False, None, None, None),
                         ('b_Hgam', 0.0, False, None, None, None),
                         ('a_OIII4364', 0.0, False, None, None, None),
                         ('b_OIII4364', 0.0, False, None, None, None),
                         ('a_HeII4687', 0.0, False, None, None, None),
                         ('b_HeII4687', 0.0, False, None, None, None),
                         ('a_OII', 0.0, False, None, None, None),
                         ('b_OII', 0.0, False, None, None, None),
                         ('a_Hbeta', 0.0, False, None, None, None),
                         ('b_Hbeta', 0.0, False, None, None, None),
                         ('a_OIII4960', 0.0, False, None, None, None),
                         ('b_OIII4960', 0.0, False, None, None, None),
                         ('a_OIII5008', 0.0, False, None, None, None),
                         ('b_OIII5008', 0.0, False, None, None, None))

# Read region file
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array_input = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array_input = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array_input = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array_input = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')


# Def Plot function
def PlotGasSpectra(region=None, figname='spectra_gas_1', deredden=True, save_table=False, save_figure=True):
    global ra_array_input, dec_array_input, radius_array_input, text_array_input
    region_mask = np.in1d(text_array_input, region)
    ra_array, dec_array, radius_array, text_array = ra_array_input[region_mask], dec_array_input[region_mask], \
                                                    radius_array_input[region_mask], text_array_input[region_mask]

    # Weak emission lines
    fig_weak, axarr_weak = plt.subplots(len(ra_array), 6, figsize=(10, len(ra_array) * 2.5),
                                        gridspec_kw={'width_ratios': [1, -0.2, 0, 1, 1, 1]}, dpi=300)
    fig_weak.subplots_adjust(hspace=0)
    fig_weak.subplots_adjust(wspace=0.2)

    # Strong emission lines
    fig_strong, axarr_strong = plt.subplots(len(ra_array), 2, figsize=(10, len(ra_array) * 2.5),
                                            gridspec_kw={'width_ratios': [1, 3]}, dpi=300)
    fig_strong.subplots_adjust(hspace=0)
    fig_strong.subplots_adjust(wspace=0.1)

    flux_info = np.zeros((len(ra_array), 32))
    for i in range(len(ra_array)):
        if len(ra_array) == 1:
            axarr_0_strong = axarr_strong[0]
            axarr_1_strong = axarr_strong[1]
            axarr_0_weak = axarr_weak
            axarr_i_weak = axarr_weak
        else:
            axarr_0_strong = axarr_strong[i, 0]
            axarr_1_strong = axarr_strong[i, 1]
            axarr_0_weak = axarr_weak[0]
            axarr_i_weak = axarr_weak[i]

        spe_NeV3346_i = cube_NeV3346.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_NeIII3869_i = cube_NeIII3869.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_HeI3889_i = cube_HeI3889.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_Heps_i = cube_Heps.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_Hdel_i = cube_Hdel.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_Hgam_i = cube_Hgam.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII4364_i = cube_OIII4364.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_HeII4687_i = cube_HeII4687.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
        spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_bet_i = cube_bet.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII4960_i = cube_OIII4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII5008_i = cube_OIII5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)

        # For [Ne V] 3346.79
        spe_NeV3346_i.mask_region(5440, 5460)
        conti_NeV3346_i = spe_NeV3346_i.poly_spec(3, weight=True)
        spe_NeV3346_i.unmask()
        spe_NeV3346_i -= conti_NeV3346_i

        # For [Ne III] 3869
        spe_NeIII3869_i.mask_region(6290, 6305)
        conti_NeIII3869_i = spe_NeIII3869_i.poly_spec(3, weight=True)
        spe_NeIII3869_i.unmask()
        spe_NeIII3869_i -= conti_NeIII3869_i

        # For He I 3889
        spe_HeI3889_i.mask_region(6325, 6335)
        conti_HeI3889_i = spe_HeI3889_i.poly_spec(3, weight=True)
        spe_HeI3889_i.unmask()
        spe_HeI3889_i -= conti_HeI3889_i

        # For Hepsilon
        spe_Heps_i.mask_region(6450, 6470)
        conti_Heps_i = spe_Heps_i.poly_spec(3, weight=True)
        spe_Heps_i.unmask()
        spe_Heps_i -= conti_Heps_i

        # For Hdelta
        spe_Hdel_i.mask_region(6670, 6685)
        conti_Hdel_i = spe_Hdel_i.poly_spec(3, weight=True)
        spe_Hdel_i.unmask()
        spe_Hdel_i -= conti_Hdel_i

        # For Hgamma
        spe_Hgam_i.mask_region(7060, 7070)
        conti_Hgam_i = spe_Hgam_i.poly_spec(3, weight=True)
        spe_Hgam_i.unmask()
        spe_Hgam_i -= conti_Hgam_i

        # For [O III] 4364
        spe_OIII4364_i.mask_region(7098, 7108)
        conti_OIII4364_i = spe_OIII4364_i.poly_spec(3, weight=True)
        spe_OIII4364_i.unmask()
        spe_OIII4364_i -= conti_OIII4364_i

        # For He II 4687
        spe_HeII4687_i.mask_region(7620, 7640)
        conti_HeII4687_i = spe_HeII4687_i.poly_spec(3, weight=True)
        spe_HeII4687_i.unmask()
        spe_HeII4687_i -= conti_HeII4687_i

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
        flux_NeV3346_i, flux_NeV3346_err_i = spe_NeV3346_i.data * 1e-3, np.sqrt(spe_NeV3346_i.var) * 1e-3
        flux_NeIII3869_i, flux_NeIII3869_err_i = spe_NeIII3869_i.data * 1e-3, np.sqrt(spe_NeIII3869_i.var) * 1e-3
        flux_HeI3889_i, flux_HeI3889_err_i = spe_HeI3889_i.data * 1e-3, np.sqrt(spe_HeI3889_i.var) * 1e-3
        flux_Heps_i, flux_Heps_err_i = spe_Heps_i.data * 1e-3, np.sqrt(spe_Heps_i.var) * 1e-3
        flux_Hdel_i, flux_Hdel_err_i = spe_Hdel_i.data * 1e-3, np.sqrt(spe_Hdel_i.var) * 1e-3
        flux_Hgam_i, flux_Hgam_err_i = spe_Hgam_i.data * 1e-3, np.sqrt(spe_Hgam_i.var) * 1e-3
        flux_OIII4364_i, flux_OIII4364_err_i = spe_OIII4364_i.data * 1e-3, np.sqrt(spe_OIII4364_i.var) * 1e-3
        flux_HeII4687_i, flux_HeII4687_err_i = spe_HeII4687_i.data * 1e-3, np.sqrt(spe_HeII4687_i.var) * 1e-3

        flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
        flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
        flux_bet_i, flux_bet_err_i = spe_bet_i.data * 1e-3, np.sqrt(spe_bet_i.var) * 1e-3
        flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
        flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3

        #
        if deredden:
            A_v = 0.087
            factor_extinction = extinction_ndarray(wave_vac_all, A_v)

            flux_NeV3346_i *= 10 ** (0.4 * factor_extinction[0])
            flux_NeV3346_err_i *= 10 ** (0.4 * factor_extinction[0])
            flux_NeIII3869_i *= 10 ** (0.4 * factor_extinction[1])
            flux_NeIII3869_err_i *= 10 ** (0.4 * factor_extinction[1])
            flux_HeI3889_i *= 10 ** (0.4 * factor_extinction[2])
            flux_HeI3889_err_i *= 10 ** (0.4 * factor_extinction[2])
            flux_Heps_i *= 10 ** (0.4 * factor_extinction[3])
            flux_Heps_err_i *= 10 ** (0.4 * factor_extinction[3])
            flux_Hdel_i *= 10 ** (0.4 * factor_extinction[4])
            flux_Hdel_err_i *= 10 ** (0.4 * factor_extinction[4])
            flux_Hgam_i *= 10 ** (0.4 * factor_extinction[5])
            flux_Hgam_err_i *= 10 ** (0.4 * factor_extinction[5])
            flux_OIII4364_i *= 10 ** (0.4 * factor_extinction[6])
            flux_OIII4364_err_i *= 10 ** (0.4 * factor_extinction[6])
            flux_HeII4687_i *= 10 ** (0.4 * factor_extinction[7])
            flux_HeII4687_err_i *= 10 ** (0.4 * factor_extinction[7])
            flux_OII_i *= 10 ** (0.4 * factor_extinction[8])
            flux_OII_err_i *= 10 ** (0.4 * factor_extinction[8])
            # no continuum
            flux_Hbeta_i *= 10 ** (0.4 * factor_extinction[9])
            flux_Hbeta_err_i *= 10 ** (0.4 * factor_extinction[9])
            flux_OIII4960_i *= 10 ** (0.4 * factor_extinction[10])
            flux_OIII4960_err_i *= 10 ** (0.4 * factor_extinction[10])
            flux_OIII5008_i *= 10 ** (0.4 * factor_extinction[11])
            flux_OIII5008_err_i *= 10 ** (0.4 * factor_extinction[11])


            #
        flux_strong = np.hstack((flux_OII_i, flux_Hbeta_i, flux_bet_i, flux_OIII4960_i, flux_OIII5008_i))
        flux_err_strong = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_bet_err_i, flux_OIII4960_err_i,
                                     flux_OIII5008_err_i))
        
        flux_all = np.hstack((flux_NeV3346_i, flux_NeIII3869_i, flux_HeI3889_i, flux_Heps_i, flux_Hdel_i, flux_Hgam_i,
                              flux_OIII4364_i, flux_HeII4687_i, flux_OII_i, flux_Hbeta_i, flux_OIII4960_i,
                              flux_OIII5008_i))
        flux_err_all = np.hstack((flux_NeV3346_err_i, flux_NeIII3869_err_i, flux_HeI3889_err_i, flux_Heps_err_i,
                                   flux_Hdel_err_i, flux_Hgam_err_i, flux_OIII4364_err_i, flux_HeII4687_err_i,
                                   flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

        # Save spec data for proposal
        # Table_spec_data = Table()
        # Table_spec_data['wave'] = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac))
        # Table_spec_data['flux'] = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
        # Table_spec_data['flux_err'] = np.hstack((flux_OII_err_i, flux_Hbeta_err_i,
        #                                          flux_OIII4960_err_i, flux_OIII5008_err_i))
        # Table_spec_data.write('/Users/lzq/Dropbox/Data/CGM/RegionsSpecData/StrongSpecData_' + region + '.fits',
        #                       format='fits', overwrite=True)
        # wave_vac_all = np.array([wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
        #                          wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac, wave_OII_vac, wave_Hbeta_vac,
        #                          wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)


        # Fit
        spec_model_all = lmfit.Model(model_all, missing='drop')
        result_all = spec_model_all.fit(data=flux_all, wave_vac=wave_vac_all, params=parameters_all,
                                        weights=1 / flux_err_all)

        # Load fitted result
        z, dz = result_all.best_values['z'], result_all.params['z'].stderr
        sigma, dsigma = result_all.best_values['sigma_kms'], result_all.params['sigma_kms'].stderr

        # Strong lines
        flux_OII, dflux_OII = result_all.best_values['flux_OII'], result_all.params['flux_OII'].stderr
        flux_Hbeta, dflux_Hbeta = result_all.best_values['flux_Hbeta'], result_all.params['flux_Hbeta'].stderr
        flux_OIII5008, dflux_OIII5008 = result_all.best_values['flux_OIII5008'], result_all.params['flux_OIII5008'].stderr
        r_OII, dr_OII = result_all.best_values['r_OII3729_3727'], result_all.params['r_OII3729_3727'].stderr

        # Strong lines conti
        a_OII, da_OII = result_all.best_values['a_OII'], result_all.params['a_OII'].stderr
        b_OII, db_OII = result_all.best_values['b_OII'], result_all.params['b_OII'].stderr
        a_Hbeta, da_Hbeta = result_all.best_values['a_Hbeta'], result_all.params['a_Hbeta'].stderr
        b_Hbeta, db_Hbeta = result_all.best_values['b_Hbeta'], result_all.params['b_Hbeta'].stderr
        a_OIII4960, da_OIII4960 = result_all.best_values['a_OIII4960'], result_all.params['a_OIII4960'].stderr
        b_OIII4960, db_OIII4960 = result_all.best_values['b_OIII4960'], result_all.params['b_OIII4960'].stderr
        a_OIII5008, da_OIII5008 = result_all.best_values['a_OIII5008'], result_all.params['a_OIII5008'].stderr
        b_OIII5008, db_OIII5008 = result_all.best_values['b_OIII5008'], result_all.params['b_OIII5008'].stderr

        # Weak lines
        flux_NeV3346, dflux_NeV3346 = result_all.best_values['flux_NeV3346'], result_all.params['flux_NeV3346'].stderr
        flux_NeIII3869, dflux_NeIII3869 = result_all.best_values['flux_NeIII3869'], \
                                          result_all.params['flux_NeIII3869'].stderr
        flux_HeI3889, dflux_HeI3889 = result_all.best_values['flux_HeI3889'], result_all.params['flux_HeI3889'].stderr
        flux_H8, dflux_H8 = result_all.best_values['flux_H8'], result_all.params['flux_H8'].stderr
        flux_NeIII3968, dflux_NeIII3968 = result_all.best_values['flux_NeIII3968'], \
                                          result_all.params['flux_NeIII3968'].stderr
        flux_Heps, dflux_Heps = result_all.best_values['flux_Heps'], result_all.params['flux_Heps'].stderr
        flux_Hdel, dflux_Hdel = result_all.best_values['flux_Hdel'], result_all.params['flux_Hdel'].stderr
        flux_Hgam, dflux_Hgam = result_all.best_values['flux_Hgam'], result_all.params['flux_Hgam'].stderr
        flux_OIII4364, dflux_OIII4364 = result_all.best_values['flux_OIII4364'], \
                                        result_all.params['flux_OIII4364'].stderr
        flux_HeII4687, dflux_HeII4687 = result_all.best_values['flux_HeII4687'], \
                                        result_all.params['flux_HeII4687'].stderr

        # Weak lines conti
        a_NeV3346, da_NeV3346 = result_all.best_values['a_NeV3346'], result_all.params['a_NeV3346'].stderr
        b_NeV3346, db_NeV3346 = result_all.best_values['b_NeV3346'], result_all.params['b_NeV3346'].stderr
        a_NeIII3869, da_NeIII3869 = result_all.best_values['a_NeIII3869'], result_all.params['a_NeIII3869'].stderr
        b_NeIII3869, db_NeIII3869 = result_all.best_values['b_NeIII3869'], result_all.params['b_NeIII3869'].stderr
        a_HeI3889, da_HeI3889 = result_all.best_values['a_HeI3889'], result_all.params['a_HeI3889'].stderr
        b_HeI3889, db_HeI3889 = result_all.best_values['b_HeI3889'], result_all.params['b_HeI3889'].stderr
        a_NeIII3968, da_NeIII3968 = result_all.best_values['a_NeIII3968'], result_all.params['a_NeIII3968'].stderr
        b_NeIII3968, db_NeIII3968 = result_all.best_values['b_NeIII3968'], result_all.params['b_NeIII3968'].stderr
        a_Hdel, da_Hdel = result_all.best_values['a_Hdel'], result_all.params['a_Hdel'].stderr
        b_Hdel, db_Hdel = result_all.best_values['b_Hdel'], result_all.params['b_Hdel'].stderr
        a_Hgam, da_Hgam = result_all.best_values['a_Hgam'], result_all.params['a_Hgam'].stderr
        b_Hgam, db_Hgam = result_all.best_values['b_Hgam'], result_all.params['b_Hgam'].stderr
        a_OIII4364, da_OIII4364 = result_all.best_values['a_OIII4364'], result_all.params['a_OIII4364'].stderr
        b_OIII4364, db_OIII4364 = result_all.best_values['b_OIII4364'], result_all.params['b_OIII4364'].stderr
        a_HeII4687, da_HeII4687 = result_all.best_values['a_HeII4687'], result_all.params['a_HeII4687'].stderr
        b_HeII4687, db_HeII4687 = result_all.best_values['b_HeII4687'], result_all.params['b_HeII4687'].stderr

        # Save the fitted result
        flux_info[i, :] = np.array([z, dz, sigma, dsigma, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8,
                                    flux_NeIII3968, flux_Heps, flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687,
                                    flux_OII, r_OII, flux_Hbeta,
                                    flux_OIII5008, dflux_NeV3346, dflux_NeIII3869, dflux_HeI3889, dflux_H8,
                                    dflux_NeIII3968, dflux_Heps, dflux_Hdel, dflux_Hgam, dflux_OIII4364,
                                    dflux_HeII4687, dflux_OII, dr_OII, dflux_Hbeta, dflux_OIII5008])

        # Use different line profile if specified
        if text_array[i] == 'S3' or text_array[i] == 'S4':
            if deredden:
                lineRatioS3S4 = Table.read('/Users/lzq/Dropbox/Data/CGM/RegionLinesRatio/'
                                           'RegionLinesRatio_S3S4_dered.fits')
                lineRatioS3S4 = lineRatioS3S4[lineRatioS3S4['region'] == text_array[i]]
            else:
                lineRatioS3S4 = Table.read('/Users/lzq/Dropbox/Data/CGM/RegionLinesRatio/'
                                           'RegionLinesRatio_S3S4.fits')
                lineRatioS3S4 = lineRatioS3S4[lineRatioS3S4['region'] == text_array[i]]

            z, dz_wing = lineRatioS3S4['z'], lineRatioS3S4['dz_wing']
            sigma, sigma_wing = lineRatioS3S4['sigma'], lineRatioS3S4['sigma_wing']
            flux_NeV3346, flux_NeIII3869 = lineRatioS3S4['flux_NeV3346'], lineRatioS3S4['flux_NeIII3869']
            flux_NeV3346_wing, flux_NeIII3869_wing = lineRatioS3S4['flux_NeV3346_wing'], \
                                                     lineRatioS3S4['flux_NeIII3869_wing']
            flux_HeI3889, flux_H8 = lineRatioS3S4['flux_HeI3889'], lineRatioS3S4['flux_H8']
            flux_HeI3889_wing, flux_H8_wing = lineRatioS3S4['flux_HeI3889_wing'], lineRatioS3S4['flux_H8_wing']
            flux_NeIII3968, flux_Heps = lineRatioS3S4['flux_NeIII3968'], lineRatioS3S4['flux_Heps']
            flux_NeIII3968_wing, flux_Heps_wing = lineRatioS3S4['flux_NeIII3968_wing'], lineRatioS3S4['flux_Heps_wing']
            flux_Hdel, flux_Hgam = lineRatioS3S4['flux_Hdel'], lineRatioS3S4['flux_Hgam']
            flux_Hdel_wing, flux_Hgam_wing = lineRatioS3S4['flux_Hdel_wing'], lineRatioS3S4['flux_Hgam_wing']
            flux_OIII4364, flux_HeII4687 = lineRatioS3S4['flux_OIII4364'], lineRatioS3S4['flux_HeII4687']
            flux_OIII4364_wing, flux_HeII4687_wing = lineRatioS3S4['flux_OIII4364_wing'], \
                                                     lineRatioS3S4['flux_HeII4687_wing']
            flux_OII, flux_OII_wing = lineRatioS3S4['flux_OII'], lineRatioS3S4['flux_OII_wing']
            flux_Hbeta, flux_OIII5008 = lineRatioS3S4['flux_Hbeta'], lineRatioS3S4['flux_OIII5008']
            flux_Hbeta_wing, flux_OIII5008_wing = lineRatioS3S4['flux_Hbeta_wing'], lineRatioS3S4['flux_OIII5008_wing']
            r_OII, r_OII_wing = lineRatioS3S4['r_OII'], lineRatioS3S4['r_OII_wing']

            line_model_all = S3S4_model_all(wave_vac_all_plot, z, dz_wing, sigma, sigma_wing, flux_NeV3346,
                                            flux_NeV3346_wing, flux_NeIII3869, flux_NeIII3869_wing, flux_HeI3889,
                                            flux_HeI3889_wing, flux_H8, flux_H8_wing, flux_NeIII3968,
                                            flux_NeIII3968_wing, flux_Heps, flux_Heps_wing, flux_Hdel, flux_Hdel_wing,
                                            flux_Hgam, flux_Hgam_wing, flux_OIII4364, flux_OIII4364_wing, flux_HeII4687,
                                            flux_HeII4687_wing, flux_OII, flux_OII_wing, flux_Hbeta, flux_Hbeta_wing,
                                            flux_OIII5008, flux_OIII5008_wing, r_OII, r_OII_wing, a_NeV3346, b_NeV3346,
                                            a_NeIII3869, b_NeIII3869, a_HeI3889, b_HeI3889, a_NeIII3968, b_NeIII3968,
                                            a_Hdel, b_Hdel, a_Hgam, b_Hgam, a_OIII4364, b_OIII4364, a_HeII4687,
                                            b_HeII4687, a_OII, b_OII, a_Hbeta, b_Hbeta, a_OIII4960, b_OIII4960,
                                            a_OIII5008, b_OIII5008)

        else:
            line_model_all = model_all(wave_vac_all_plot, z, sigma, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8,
                                        flux_NeIII3968, flux_Heps, flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687,
                                        flux_OII, flux_Hbeta, flux_OIII5008, r_OII, a_NeV3346, b_NeV3346,
                                        a_NeIII3869, b_NeIII3869, a_HeI3889, b_HeI3889, a_NeIII3968, b_NeIII3968,
                                        a_Hdel, b_Hdel, a_Hgam, b_Hgam, a_OIII4364, b_OIII4364, a_HeII4687, b_HeII4687,
                                        a_OII, b_OII, a_Hbeta, b_Hbeta, a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008)


        # Set title
        text_i = text_array[i]
        if len(text_i) > 5:
            text_i = text_i[:-4]

        # Weak lines
        axarr_i_weak[0].plot(wave_NeV3346_vac, flux_NeV3346_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i_weak[0].plot(wave_NeV3346_vac, flux_NeV3346_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i_weak[0].plot(wave_vac_all_stack[:idx_weak], line_model_all[:idx_weak], '-r', lw=1)
        axarr_i_weak[0].set_xlim(5380, 5530)
        axarr_i_weak[0].set_title(text_i, x=0.3, y=0.75, size=20)

        # axarr_i_weak[1].plot(wave_NeIII3869_vac, flux_NeIII3869_i, color='k', drawstyle='steps-mid', lw=1)
        # axarr_i_weak[1].plot(wave_NeIII3869_vac, flux_NeIII3869_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        # axarr_i_weak[1].plot(wave_HeI3889_vac, flux_HeI3889_i, color='k', drawstyle='steps-mid', lw=1)
        # axarr_i_weak[1].plot(wave_HeI3889_vac, flux_HeI3889_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        # axarr_i_weak[1].plot(wave_vac_all_stack[:idx_weak], line_model_all[:idx_weak], '-r', lw=1)
        # axarr_i_weak[1].set_xlim(6250, 6400)
        # axarr_i_weak[1].set_xticks([6300, 6400], ['6300', ''])
        #
        # axarr_i_weak[2].plot(wave_Heps_vac, flux_Heps_i, color='k', drawstyle='steps-mid', lw=1)
        # axarr_i_weak[2].plot(wave_Heps_vac, flux_Heps_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        # axarr_i_weak[2].plot(wave_vac_all_stack[:idx_weak], line_model_all[:idx_weak], '-r', lw=1)
        # axarr_i_weak[2].set_xlim(6400, 6550)

        axarr_i_weak[3].plot(wave_Hdel_vac, flux_Hdel_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i_weak[3].plot(wave_Hdel_vac, flux_Hdel_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i_weak[3].plot(wave_vac_all_stack[:idx_weak], line_model_all[:idx_weak], '-r', lw=1)
        axarr_i_weak[3].set_xlim(6600, 6750)

        axarr_i_weak[4].plot(wave_Hgam_vac, flux_Hgam_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i_weak[4].plot(wave_Hgam_vac, flux_Hgam_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i_weak[4].plot(wave_OIII4364_vac, flux_OIII4364_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i_weak[4].plot(wave_OIII4364_vac, flux_OIII4364_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i_weak[4].plot(wave_vac_all_stack[:idx_weak], line_model_all[:idx_weak], '-r', lw=1)
        axarr_i_weak[4].set_xlim(7000, 7150)

        axarr_i_weak[5].plot(wave_HeII4687_vac, flux_HeII4687_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i_weak[5].plot(wave_HeII4687_vac, flux_HeII4687_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i_weak[5].plot(wave_vac_all_stack[:idx_weak], line_model_all[:idx_weak], '-r', lw=1)
        axarr_i_weak[5].set_xlim(7550, 7700)

        #
        axarr_i_weak[0].spines['right'].set_visible(False)
        axarr_i_weak[1].spines['right'].set_visible(False)
        axarr_i_weak[1].spines['left'].set_visible(False)
        axarr_i_weak[2].spines['right'].set_visible(False)
        axarr_i_weak[2].spines['left'].set_visible(False)
        axarr_i_weak[3].spines['right'].set_visible(False)
        axarr_i_weak[3].spines['left'].set_visible(False)
        axarr_i_weak[4].spines['right'].set_visible(False)
        axarr_i_weak[4].spines['left'].set_visible(False)
        axarr_i_weak[5].spines['left'].set_visible(False)
        axarr_i_weak[1].set_visible(False)
        axarr_i_weak[2].set_visible(False)

        # Mark line info
        # [Ne V] 3346.79, [O II] 3727, 3729, [Ne III] 3869, He I 3889 and H8, NeIII3968 and Hepsilon. Hdelta, Hgamma,
        # [O III] 4364, He II 4687 Hbeta, [O III] 4960 5008
        lines = (1 + z) * np.array([pyasl.airtovac2(3345.821), pyasl.airtovac2(3726.032),
                                    pyasl.airtovac2(3728.815), pyasl.airtovac2(3868.760),
                                    pyasl.airtovac2(3888.647), pyasl.airtovac2(3889.064),
                                    pyasl.airtovac2(3967.470), pyasl.airtovac2(3970.079),
                                    pyasl.airtovac2(4101.742), pyasl.airtovac2(4340.471),
                                    pyasl.airtovac2(4363.210), pyasl.airtovac2(4685.710),
                                    pyasl.airtovac2(4861.333), pyasl.airtovac2(4958.911),
                                    pyasl.airtovac2(5006.843)])
        ymin, ymax = -5 * np.ones_like(lines), 100 * np.ones_like(lines)
        axarr_i_weak[0].vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        # axarr_i_weak[1].vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        # axarr_i_weak[2].vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        axarr_i_weak[3].vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        axarr_i_weak[4].vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        axarr_i_weak[5].vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)


        axarr_i_weak[0].set_ylim(-0.05, flux_Hgam_i.max() + 0.10)
        axarr_i_weak[1].set_ylim(-0.05, flux_Hgam_i.max() + 0.10)
        axarr_i_weak[2].set_ylim(-0.05, flux_Hgam_i.max() + 0.10)
        axarr_i_weak[3].set_ylim(-0.05, flux_Hgam_i.max() + 0.10)
        axarr_i_weak[4].set_ylim(-0.05, flux_Hgam_i.max() + 0.10)
        axarr_i_weak[5].set_ylim(-0.05, flux_Hgam_i.max() + 0.10)

        axarr_0_weak[0].annotate(text=r'$\mathrm{[Ne \, V]}$', xy=(0.1, 0.65), xycoords='axes fraction', size=15)
        # axarr_0_weak[1].annotate(text=r'$\mathrm{[Ne \, III]}$', xy=(-0.25, 0.65), xycoords='axes fraction', size=15)
        # axarr_0_weak[1].annotate(text=r'$\mathrm{He \, I}$' + '\n' + r'$\mathrm{H8}$', xy=(0.6, 0.53),
        #                          xycoords='axes fraction', size=15)
        # axarr_0_weak[2].annotate(text=r'$\mathrm{[Ne \, III]}$', xy=(-0.20, 0.65), xycoords='axes fraction', size=15)
        # axarr_0_weak[2].annotate(text=r'$\mathrm{H \epsilon}$', xy=(0.60, 0.65), xycoords='axes fraction', size=15)
        axarr_0_weak[3].annotate(text=r'$\mathrm{H \delta}$', xy=(0.2, 0.65), xycoords='axes fraction', size=15)
        axarr_0_weak[4].annotate(text=r'$\mathrm{H \gamma}$', xy=(0.2, 0.65), xycoords='axes fraction', size=15)
        axarr_0_weak[4].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.74, 0.65), xycoords='axes fraction', size=15)
        axarr_0_weak[5].annotate(text=r'$\mathrm{He \, II}$', xy=(0.6, 0.65), xycoords='axes fraction', size=15)

        axarr_i_weak[0].minorticks_on()
        axarr_i_weak[1].minorticks_on()
        axarr_i_weak[2].minorticks_on()
        axarr_i_weak[3].minorticks_on()
        axarr_i_weak[4].minorticks_on()
        axarr_i_weak[5].minorticks_on()
        axarr_i_weak[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on',
                                    right=False, labelsize=20, size=5)
        axarr_i_weak[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on',
                                    right=False, size=3)
        axarr_i_weak[0].tick_params(axis='y', which='both', right=False, labelright=False)
        axarr_i_weak[5].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False,
                                    right='on', labelsize=20, size=5)
        axarr_i_weak[5].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False,
                                    right='on', size=3)
        axarr_i_weak[5].tick_params(axis='y', which='both', left=False, labelleft=False)
        for j in [1, 2, 3, 4]:
            axarr_i_weak[j].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False,
                                        right=False, labelsize=20, size=5)
            axarr_i_weak[j].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False,
                                        right=False, size=3)
            axarr_i_weak[j].tick_params(axis='y', which='both', right=False, labelright=False, left=False, labelleft=False)

        if i != len(ra_array) - 1:
            axarr_i_weak[0].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i_weak[1].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i_weak[2].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i_weak[3].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i_weak[4].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i_weak[5].tick_params(axis='x', which='both', labelbottom=False)
        
        # Strong lines
        axarr_0_strong.plot(wave_vac_strong_stack, flux_strong, color='k', drawstyle='steps-mid', lw=1)
        axarr_0_strong.plot(wave_vac_strong_stack, flux_err_strong, color='lightgrey', lw=1)
        axarr_0_strong.plot(wave_vac_all_stack[idx_weak:idx_all], line_model_all[idx_weak:idx_all], '-r', lw=1)

        axarr_1_strong.plot(wave_vac_strong_stack, flux_strong, color='k', drawstyle='steps-mid', lw=1)
        axarr_1_strong.plot(wave_vac_strong_stack, flux_err_strong, color='lightgrey', lw=1)
        axarr_1_strong.plot(wave_vac_all_stack[idx_weak:idx_all], line_model_all[idx_weak:idx_all], '-r', lw=1)

        axarr_0_strong.set_title(text_i, x=0.2, y=0.75, size=20)
        axarr_0_strong.set_xlim(6020, 6120)
        axarr_1_strong.set_xlim(7900, 8200)
        axarr_0_strong.spines['right'].set_visible(False)
        axarr_1_strong.spines['left'].set_visible(False)

        # Mark line info
        axarr_0_strong.vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        axarr_1_strong.vlines(lines, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1, zorder=-10)
        axarr_0_strong.set_ylim(flux_strong.min() - 0.5, flux_strong.max() + 0.5)
        axarr_1_strong.set_ylim(flux_strong.min() - 0.5, flux_strong.max() + 0.5)

        if flux_Hbeta_i.max() > flux_OIII5008_i.max():
            axarr_0_strong.set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
            axarr_1_strong.set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)

        if i == 0:
            axarr_0_strong.annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.6, 0.65), xycoords='axes fraction', size=20)
            axarr_1_strong.annotate(text=r'$\mathrm{H\beta}$', xy=(0.1, 0.65), xycoords='axes fraction', size=20)
            axarr_1_strong.annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.45, 0.65), xycoords='axes fraction', size=20)
            axarr_1_strong.annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.7, 0.65), xycoords='axes fraction', size=20)

        axarr_0_strong.minorticks_on()
        axarr_1_strong.minorticks_on()
        axarr_0_strong.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on',
                                   right=False, labelsize=20, size=5)
        axarr_0_strong.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on',
                                   right=False, size=3)
        axarr_0_strong.tick_params(axis='y', which='both', right=False, labelright=False)
        axarr_1_strong.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False,
                                   right='on', labelsize=20, size=5)
        axarr_1_strong.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False,
                                   right='on', size=3)
        axarr_1_strong.tick_params(axis='y', which='both', left=False, labelleft=False)
        if i != len(ra_array) - 1:
            axarr_0_strong.tick_params(axis='x', which='both', labelbottom=False)
            axarr_1_strong.tick_params(axis='x', which='both', labelbottom=False)

    # Finish the table
    t = Table(flux_info, names=('z', 'dz', 'sigma', 'dsigma', 'flux_NeV3346', 'flux_NeIII3869', 'flux_HeI3889',
                                'flux_H8', 'flux_NeIII3968', 'flux_Heps', 'flux_Hdel', 'flux_Hgam', 'flux_OIII4364',
                                'flux_HeII4687', 'flux_OII', 'r_OII', 'flux_Hbeta', 'flux_OIII5008', 'dflux_NeV3346',
                                'dflux_NeIII3869', 'dflux_HeI3889', 'dflux_H8', 'dflux_NeIII3968',
                                'dflux_Heps', 'dflux_Hdel', 'dflux_Hgam', 'dflux_OIII4364',
                                'dflux_HeII4687', 'dflux_OII', 'dr_OII', 'dflux_Hbeta', 'dflux_OIII5008'))
    t['region'] = text_array
    if save_table is True:
        if deredden:
            t.write('/Users/lzq/Dropbox/Data/CGM/RegionLinesRatio/RegionLinesRatio_dered.fits', format='fits',
                    overwrite=True)
        else:
            t.write('/Users/lzq/Dropbox/Data/CGM/RegionLinesRatio/RegionLinesRatio.fits', format='fits', overwrite=True)

    if len(ra_array) == 1:
        fig_weak.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=-0.12)
        fig_weak.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$',
                           size=20, x=0.03)
        fig_strong.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=-0.12)
        fig_strong.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$',
                             size=20, x=0.03)
    else:
        fig_weak.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=0.05)
        fig_weak.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$',
                           size=20,)
        fig_strong.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=0.05)
        fig_strong.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$',
                             size=20)
    if save_figure:
        if deredden:
            fig_weak.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '_weak_dered.png', bbox_inches='tight')
            fig_strong.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '_strong_dered.png', bbox_inches='tight')
        else:
            fig_weak.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '_weak.png', bbox_inches='tight')
            fig_strong.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '_strong.png', bbox_inches='tight')


# Plot the data

# PlotGasSpectra(region=text_array, figname='spectra_gas/all',
#                save_table=True, save_figure=True, deredden=True)

# region_paper = ['S2', 'S4', 'S6', 'S8', 'S9', 'B2']
# PlotGasSpectra(region=region_paper, figname='spectra_gas/spectra_gas_paper',
#                save_table=False, save_figure=True, deredden=True)

print(text_array_input)
for i, text_i in enumerate(text_array_input[-2:]):
    PlotGasSpectra(region=text_i, figname='spectra_gas/spectra_gas_' + str(text_i))
