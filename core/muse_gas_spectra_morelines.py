import os
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
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

# def model_H8(wave_vac, z, sigma_kms, flux_H8, a, b):
#     # Constants
#     c_kms = 2.998e5
#     wave_H8_vac = 3890.16
#
#     wave_H8_obs = wave_H8_vac * (1 + z)
#     sigma_H8_A = np.sqrt((sigma_kms / c_kms * wave_H8_obs) ** 2 + (getSigma_MUSE(wave_H8_obs)) ** 2)
#
#     peak_H8 = flux_H8 / np.sqrt(2 * sigma_H8_A ** 2 * np.pi)
#     H8_gaussian = peak_H8 * np.exp(-(wave_vac - wave_H8_obs) ** 2 / 2 / sigma_H8_A ** 2)
#
#     return H8_gaussian + a * wave_vac + b

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

# def model_Heps(wave_vac, z, sigma_kms, flux_Heps, a, b):
#     # Constants
#     c_kms = 2.998e5
#     wave_Heps_vac = 3971.20
#
#     wave_Heps_obs = wave_Heps_vac * (1 + z)
#     sigma_Heps_A = np.sqrt((sigma_kms / c_kms * wave_Heps_obs) ** 2 + (getSigma_MUSE(wave_Heps_obs)) ** 2)
#
#     peak_Heps = flux_Heps / np.sqrt(2 * sigma_Heps_A ** 2 * np.pi)
#     Heps_gaussian = peak_Heps * np.exp(-(wave_vac - wave_Heps_obs) ** 2 / 2 / sigma_Heps_A ** 2)
#
#     return Heps_gaussian + a * wave_vac + b

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

# [Ne V] 3346.79, [Ne III] 3869, He I 3889, Hepsilon. Hdelta, Hgamma, [O III] 4364, He II 4687
# lines_more = (1 + z) * np.array([3346.79, 3869.86, 3889.00, 3971.20, 4102.89, 4341.68, 4364.44, 4687.31])

def model_more(wave_vac, z, sigma_kms, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8, flux_NeIII3968, flux_Heps,
              flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687, flux_OII, flux_Hbeta, flux_OIII5008, r_OII3729_3727,
              a_NeV3346, b_NeV3346, a_NeIII3869, b_NeIII3869, a_HeI3889, b_HeI3889, a_NeIII3968,
              b_NeIII3968, a_Hdel, b_Hdel, a_Hgam, b_Hgam, a_OIII4364, b_OIII4364, a_HeII4687,
              b_HeII4687, a_OII, b_OII, a_Hbeta, b_Hbeta, a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008):

    # Weak lines
    m_NeV3356 = model_NeV3346(wave_vac[0], z, sigma_kms, flux_NeV3346, a_NeV3346, b_NeV3346)
    m_NeIII3869 = model_NeIII3869(wave_vac[1], z, sigma_kms, flux_NeIII3869, a_NeIII3869, b_NeIII3869)
    m_HeI3889andH8 = model_HeI3889andH8(wave_vac[2], z, sigma_kms, flux_HeI3889, flux_H8, a_HeI3889, b_HeI3889)
    # m_H8 = model_H8(wave_vac[2], z, sigma_kms, flux_H8, a_H8, b_H8) # wavelength same as HeI
    m_NeIII3968andHeps = model_NeIII3968andHeps(wave_vac[3], z, sigma_kms, flux_NeIII3968, flux_Heps, a_NeIII3968,
                                                b_NeIII3968)
    # wavelength same as Heps

    # m_Heps = model_Heps(wave_vac[3], z, sigma_kms, flux_Heps, a_Heps, b_Heps)
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

def model_all(wave_vac, z, sigma_kms, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8, flux_NeIII3968,
              flux_Heps, flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687, a_NeV3346, b_NeV3346, a_NeIII3869,
              b_NeIII3869, a_HeI3889, b_HeI3889, a_NeIII3968, b_NeIII3968, a_Hdel, b_Hdel, a_Hgam, b_Hgam, a_OIII4364,
              b_OIII4364, a_HeII4687, b_HeII4687):

    m_NeV3356 = model_NeV3346(wave_vac[0], z, sigma_kms, flux_NeV3346, a_NeV3346, b_NeV3346)
    m_NeIII3869 = model_NeIII3869(wave_vac[1], z, sigma_kms, flux_NeIII3869, a_NeIII3869, b_NeIII3869)
    m_HeI3889andH8 = model_HeI3889andH8(wave_vac[2], z, sigma_kms, flux_HeI3889, flux_H8, a_HeI3889, b_HeI3889)
    m_NeIII3968andHeps = model_NeIII3968andHeps(wave_vac[3], z, sigma_kms, flux_NeIII3968, flux_Heps, a_NeIII3968,
                                                b_NeIII3968)
    m_Hdel = model_Hdel(wave_vac[4], z, sigma_kms, flux_Hdel, a_Hdel, b_Hdel)
    m_Hgam = model_Hgam(wave_vac[5], z, sigma_kms, flux_Hgam, a_Hgam, b_Hgam)
    m_OIII4364 = model_OIII4364(wave_vac[6], z, sigma_kms, flux_OIII4364, a_OIII4364, b_OIII4364)
    m_HeII4687 = model_HeII4687(wave_vac[7], z, sigma_kms, flux_HeII4687, a_HeII4687, b_HeII4687)
    return np.hstack((m_NeV3356, m_NeIII3869, m_HeI3889andH8, m_NeIII3968andHeps, m_Hdel, m_Hgam, m_OIII4364, m_HeII4687))


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

# More lines
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


# Classical [O II], Hbeta, [O III]
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

wave_vac_more = np.array([wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
                          wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac, wave_OII_vac, wave_Hbeta_vac,
                          wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)
wave_vac_more_stack = np.hstack((wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
                                 wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac, wave_OII_vac, wave_Hbeta_vac,
                                 wave_OIII4960_vac, wave_OIII5008_vac))

wave_vac_more_plot = np.array([wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
                          wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac], dtype=object)
wave_vac_more_stack_plot = np.hstack((wave_NeV3346_vac, wave_NeIII3869_vac, wave_HeI3889_vac, wave_Heps_vac, wave_Hdel_vac,
                                 wave_Hgam_vac, wave_OIII4364_vac, wave_HeII4687_vac))

redshift_guess = 0.63
sigma_kms_guess = 150.0
# flux_OIII5008_guess = 0.01
r_OII3729_3727_guess = 2

# parameters = lmfit.Parameters()
# parameters.add_many(('z', redshift_guess, True, 0.62, 0.64, None),
#                     ('sigma_kms', sigma_kms_guess, True, 10, 500, None),
#                     ('flux_OII', 0.01, True, None, None, None),
#                     ('flux_Hbeta', 0.02, True, None, None, None),
#                     ('flux_OIII5008', 0.1, True, None, None, None),
#                     ('r_OII3729_3727', r_OII3729_3727_guess, True, 0.2, None, None),
#                     ('a_OII', 0.0, False, None, None, None),
#                     ('b_OII', 0.0, False, None, None, None),
#                     ('a_Hbeta', 0.0, False, None, None, None),
#                     ('b_Hbeta', 0.0, False, None, None, None),
#                     ('a_OIII4960', 0.0, False, None, None, None),
#                     ('b_OIII4960', 0.0, False, None, None, None),
#                     ('a_OIII5008', 0.0, False, None, None, None),
#                     ('b_OIII5008', 0.0, False, None, None, None))

parameters_more = lmfit.Parameters()
parameters_more.add_many(('z', redshift_guess, True, 0.62, 0.64, None),
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


# Def Plot function
def PlotGasSpectra(ra_array, dec_array, radius_array, text_array, figname='spectra_gas_1'):
    global cube_OII, cube_Hbeta, cube_OIII4960, cube_OIII5008, wave_vac_stack, wave_vac_all
    fig, axarr = plt.subplots(len(ra_array), 6, figsize=(10, len(ra_array) * 2.5),
                              gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1]}, dpi=300)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.2)
    flux_info = np.zeros((len(ra_array), 28))
    # axarr = axarr.ravel()
    for i in range(len(ra_array)):
        if len(ra_array) == 1:
            axarr_0 = axarr
            axarr_i = axarr
        else:
            axarr_0 = axarr[0]
            axarr_i = axarr[i]

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

        # Continuum subtraction for faint lines
        wave_min, wave_max = wave_cube.min(), wave_cube.max()

        # For [Ne V] 3346.79
        spe_NeV3346_i.mask_region(5440, 5460)
        # print(len(spe_cube_i.data.data[~spe_cube_i.data.mask]))
        conti_NeV3346_i = spe_NeV3346_i.poly_spec(3, weight=True)
        spe_NeV3346_i.unmask()
        # flux_i_ori, flux_err_i_ori = spe_cube_i.data * 1e-3, np.sqrt(spe_cube_i.var) * 1e-3
        # flux_i_conti = conti_NeV3346_i.data * 1e-3
        spe_NeV3346_i -= conti_NeV3346_i
        # spe_cube_i.mask_region(wave_min, 5440)
        # spe_cube_i.mask_region(5460, wave_max)
        # spe_cube_i.unmask()
        # print(np.median(flux_NeV3346_i.data))

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

        # [Ne V] 3346.79, [Ne III] 3869, He I 3889, Hepisolon. Hdelta, Hgamma, [O III] 4364, He II 4687
        # Load the data
        flux_NeV3346_i, flux_NeV3346_err_i = spe_NeV3346_i.data * 1e-3, np.sqrt(spe_NeV3346_i.var) * 1e-3
        flux_NeIII3869_i, flux_NeIII3869_err_i = spe_NeIII3869_i.data * 1e-3, np.sqrt(spe_NeIII3869_i.var) * 1e-3
        flux_HeI3889_i, flux_HeI3889_err_i = spe_HeI3889_i.data * 1e-3, np.sqrt(spe_HeI3889_i.var) * 1e-3
        flux_Heps_i, flux_Heps_err_i = spe_Heps_i.data * 1e-3, np.sqrt(spe_Heps_i.var) * 1e-3
        flux_Hdel_i, flux_Hdel_err_i = spe_Hdel_i.data * 1e-3, np.sqrt(spe_Hdel_i.var) * 1e-3
        flux_Hgam_i, flux_Hgam_err_i = spe_Hgam_i.data * 1e-3, np.sqrt(spe_Hgam_i.var) * 1e-3
        flux_OIII4364_i, flux_OIII4364_err_i = spe_OIII4364_i.data * 1e-3, np.sqrt(spe_OIII4364_i.var) * 1e-3
        flux_HeII4687_i, flux_HeII4687_err_i = spe_HeII4687_i.data * 1e-3, np.sqrt(spe_HeII4687_i.var) * 1e-3

        # flux_i, flux_err_i = spe_cube_i.data * 1e-3, np.sqrt(spe_cube_i.var) * 1e-3
        flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
        flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
        flux_bet_i, flux_bet_err_i = spe_bet_i.data * 1e-3, np.sqrt(spe_bet_i.var) * 1e-3
        flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
        flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3
        flux_all = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
        flux_err_all = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

        flux_more = np.hstack((flux_NeV3346_i, flux_NeIII3869_i, flux_HeI3889_i, flux_Heps_i, flux_Hdel_i, flux_Hgam_i,
                               flux_OIII4364_i, flux_HeII4687_i, flux_OII_i, flux_Hbeta_i, flux_OIII4960_i,
                               flux_OIII5008_i))
        flux_err_more =  np.hstack((flux_NeV3346_err_i, flux_NeIII3869_err_i, flux_HeI3889_err_i, flux_Heps_err_i,
                                    flux_Hdel_err_i, flux_Hgam_err_i, flux_OIII4364_err_i, flux_HeII4687_err_i,
                                    flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

        # spec_model = lmfit.Model(model_all, missing='drop')
        # result = spec_model.fit(data=flux_all, wave_vac=wave_vac_all, params=parameters,
        #                         weights=1 / flux_err_all)

        spec_model_more = lmfit.Model(model_more, missing='drop')
        result_more = spec_model_more.fit(data=flux_more, wave_vac=wave_vac_more, params=parameters_more,
                                weights=1 / flux_err_more)

        # For plotting
        # flux_all_plot = np.hstack((flux_OII_i, flux_Hbeta_i, flux_bet_i, flux_OIII4960_i, flux_OIII5008_i))
        # flux_err_all_plot = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_bet_err_i, flux_OIII4960_err_i,
        #                                flux_OIII5008_err_i))

        # Load fitted result
        z, dz = result_more.best_values['z'], result_more.params['z'].stderr
        sigma, dsigma = result_more.best_values['sigma_kms'], result_more.params['sigma_kms'].stderr
        flux_OII, dflux_OII = result_more.best_values['flux_OII'], result_more.params['flux_OII'].stderr
        flux_Hbeta, dflux_Hbeta = result_more.best_values['flux_Hbeta'], result_more.params['flux_Hbeta'].stderr
        flux_OIII5008, dflux_OIII5008 = result_more.best_values['flux_OIII5008'], result_more.params['flux_OIII5008'].stderr
        r_OII, dr_OII = result_more.best_values['r_OII3729_3727'], result_more.params['r_OII3729_3727'].stderr

        a_OII, da_OII = result_more.best_values['a_OII'], result_more.params['a_OII'].stderr
        b_OII, db_OII = result_more.best_values['b_OII'], result_more.params['b_OII'].stderr
        a_Hbeta, da_Hbeta = result_more.best_values['a_Hbeta'], result_more.params['a_Hbeta'].stderr
        b_Hbeta, db_Hbeta = result_more.best_values['b_Hbeta'], result_more.params['b_Hbeta'].stderr
        a_OIII4960, da_OIII4960 = result_more.best_values['a_OIII4960'], result_more.params['a_OIII4960'].stderr
        b_OIII4960, db_OIII4960 = result_more.best_values['b_OIII4960'], result_more.params['b_OIII4960'].stderr
        a_OIII5008, da_OIII5008 = result_more.best_values['a_OIII5008'], result_more.params['a_OIII5008'].stderr
        b_OIII5008, db_OIII5008 = result_more.best_values['b_OIII5008'], result_more.params['b_OIII5008'].stderr

        # [Ne V] 3346.79, [Ne III] 3869, He I 3889, Hepsilon. Hdelta, Hgamma, [O III] 4364, He II 4687

        flux_NeV3346, dflux_NeV3346 = result_more.best_values['flux_NeV3346'], result_more.params['flux_NeV3346'].stderr
        flux_NeIII3869, dflux_NeIII3869 = result_more.best_values['flux_NeIII3869'], \
                                          result_more.params['flux_NeIII3869'].stderr
        flux_HeI3889, dflux_HeI3889 = result_more.best_values['flux_HeI3889'], result_more.params['flux_HeI3889'].stderr
        flux_H8, dflux_H8 = result_more.best_values['flux_H8'], result_more.params['flux_H8'].stderr
        flux_NeIII3968, dflux_NeIII3968 = result_more.best_values['flux_NeIII3968'], \
                                          result_more.params['flux_NeIII3968'].stderr
        flux_Heps, dflux_Heps = result_more.best_values['flux_Heps'], result_more.params['flux_Heps'].stderr
        flux_Hdel, dflux_Hdel = result_more.best_values['flux_Hdel'], result_more.params['flux_Hdel'].stderr
        flux_Hgam, dflux_Hgam = result_more.best_values['flux_Hgam'], result_more.params['flux_Hgam'].stderr
        flux_OIII4364, dflux_OIII4364 = result_more.best_values['flux_OIII4364'], \
                                        result_more.params['flux_OIII4364'].stderr
        flux_HeII4687, dflux_HeII4687 = result_more.best_values['flux_HeII4687'], \
                                        result_more.params['flux_HeII4687'].stderr
        print(flux_NeV3346)
        print(flux_OIII4364)

        a_NeV3346, da_NeV3346 = result_more.best_values['a_NeV3346'], result_more.params['a_NeV3346'].stderr
        b_NeV3346, db_NeV3346 = result_more.best_values['b_NeV3346'], result_more.params['b_NeV3346'].stderr
        a_NeIII3869, da_NeIII3869 = result_more.best_values['a_NeIII3869'], result_more.params['a_NeIII3869'].stderr
        b_NeIII3869, db_NeIII3869 = result_more.best_values['b_NeIII3869'], result_more.params['b_NeIII3869'].stderr
        a_HeI3889, da_HeI3889 = result_more.best_values['a_HeI3889'], result_more.params['a_HeI3889'].stderr
        b_HeI3889, db_HeI3889 = result_more.best_values['b_HeI3889'], result_more.params['b_HeI3889'].stderr
        a_NeIII3968, da_NeIII3968 = result_more.best_values['a_NeIII3968'], result_more.params['a_NeIII3968'].stderr
        b_NeIII3968, db_NeIII3968 = result_more.best_values['b_NeIII3968'], result_more.params['b_NeIII3968'].stderr
        a_Hdel, da_Hdel = result_more.best_values['a_Hdel'], result_more.params['a_Hdel'].stderr
        b_Hdel, db_Hdel = result_more.best_values['b_Hdel'], result_more.params['b_Hdel'].stderr
        a_Hgam, da_Hgam = result_more.best_values['a_Hgam'], result_more.params['a_Hgam'].stderr
        b_Hgam, db_Hgam = result_more.best_values['b_Hgam'], result_more.params['b_Hgam'].stderr
        a_OIII4364, da_OIII4364 = result_more.best_values['a_OIII4364'], result_more.params['a_OIII4364'].stderr
        b_OIII4364, db_OIII4364 = result_more.best_values['b_OIII4364'], result_more.params['b_OIII4364'].stderr
        a_HeII4687, da_HeII4687 = result_more.best_values['a_HeII4687'], result_more.params['a_HeII4687'].stderr
        b_HeII4687, db_HeII4687 = result_more.best_values['b_HeII4687'], result_more.params['b_HeII4687'].stderr

        # Save the fitted result
        flux_info[i, :] = np.array([flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8, flux_NeIII3968, flux_Heps,
                                    flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687, flux_OII, r_OII, flux_Hbeta,
                                    flux_OIII5008, dflux_NeV3346, dflux_NeIII3869, dflux_HeI3889, dflux_H8,
                                    dflux_NeIII3968, dflux_Heps, dflux_Hdel, dflux_Hgam, dflux_OIII4364,
                                    dflux_HeII4687, dflux_OII, dr_OII, dflux_Hbeta, dflux_OIII5008])

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

        # More lines
        # axarr[i, 0].plot(wave_cube, flux_NeV3346_i, color='k', drawstyle='steps-mid', lw=1)
        # axarr[i, 0].plot(wave_cube, flux_NeV3346_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        # axarr[i, 0].plot(wave_cube, flux_i_ori, color='r', drawstyle='steps-mid', lw=1)
        # axarr[i, 0].plot(wave_cube, flux_i_conti, color='b', drawstyle='steps-mid', lw=1)

        # line_model = model_more(wave_vac_more, z, sigma, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8,
        #                         flux_NeIII3968, flux_Heps, flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687,
        #                         flux_OII, flux_Hbeta, flux_OIII5008, r_OII,
        #                         a_NeV3346, b_NeV3346, a_NeIII3869, b_NeIII3869, a_HeI3889, b_HeI3889,
        #                         a_NeIII3968, b_NeIII3968, a_Hdel, b_Hdel, a_Hgam, b_Hgam,
        #                         a_OIII4364, b_OIII4364, a_HeII4687, b_HeII4687, a_OII, b_OII, a_Hbeta, b_Hbeta,
        #                         a_OIII4960, b_OIII4960, a_OIII5008, b_OIII5008)


        line_model = model_all(wave_vac_more_plot, z, sigma, flux_NeV3346, flux_NeIII3869, flux_HeI3889, flux_H8,
                                flux_NeIII3968, flux_Heps, flux_Hdel, flux_Hgam, flux_OIII4364, flux_HeII4687,
                                a_NeV3346, b_NeV3346, a_NeIII3869, b_NeIII3869, a_HeI3889, b_HeI3889,
                                a_NeIII3968, b_NeIII3968, a_Hdel, b_Hdel, a_Hgam, b_Hgam,
                                a_OIII4364, b_OIII4364, a_HeII4687, b_HeII4687)


        axarr_i[0].plot(wave_NeV3346_vac, flux_NeV3346_i, color='k', drawstyle='steps-mid', lw=1)
        # axarr_i[0].plot(wave_NeV3346_vac, flux_i_conti, color='b', drawstyle='steps-mid', lw=1)
        axarr_i[0].plot(wave_NeV3346_vac, flux_NeV3346_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[0].plot(wave_vac_more_stack_plot, line_model, '-r', lw=1)
        axarr_i[0].set_xlim(5350, 5500)
        axarr_i[0].set_title(text_array[i], x=0.2, y=0.75, size=20)

        axarr_i[1].plot(wave_NeIII3869_vac, flux_NeIII3869_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[1].plot(wave_NeIII3869_vac, flux_NeIII3869_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[1].plot(wave_HeI3889_vac, flux_HeI3889_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[1].plot(wave_HeI3889_vac, flux_HeI3889_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[1].plot(wave_vac_more_stack_plot, line_model, '-r', lw=1)
        axarr_i[1].set_xlim(6250, 6400)
        axarr_i[1].set_xticks([6300, 6400], ['6300', ''])


        axarr_i[2].plot(wave_Heps_vac, flux_Heps_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[2].plot(wave_Heps_vac, flux_Heps_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[2].plot(wave_vac_more_stack_plot, line_model, '-r', lw=1)
        axarr_i[2].set_xlim(6400, 6550)

        axarr_i[3].plot(wave_Hdel_vac, flux_Hdel_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[3].plot(wave_Hdel_vac, flux_Hdel_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[3].plot(wave_vac_more_stack_plot, line_model, '-r', lw=1)
        axarr_i[3].set_xlim(6600, 6750)

        axarr_i[4].plot(wave_Hgam_vac, flux_Hgam_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[4].plot(wave_Hgam_vac, flux_Hgam_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[4].plot(wave_OIII4364_vac, flux_OIII4364_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[4].plot(wave_OIII4364_vac, flux_OIII4364_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[4].plot(wave_vac_more_stack_plot, line_model, '-r', lw=1)
        axarr_i[4].set_xlim(7000, 7150)

        axarr_i[5].plot(wave_HeII4687_vac, flux_HeII4687_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i[5].plot(wave_HeII4687_vac, flux_HeII4687_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i[5].plot(wave_vac_more_stack_plot, line_model, '-r', lw=1)
        axarr_i[5].set_xlim(7550, 7700)
        # axarr_i[0].set_xlim(6300, 6350)
        # axarr_i[1].set_xlim(7900, 8200)
        #
        # axarr_i[0].set_xlim(6020, 6120)
        # axarr_i[1].set_xlim(7900, 8200)
        axarr_i[0].spines['right'].set_visible(False)
        axarr_i[1].spines['right'].set_visible(False)
        axarr_i[1].spines['left'].set_visible(False)
        axarr_i[2].spines['right'].set_visible(False)
        axarr_i[2].spines['left'].set_visible(False)
        axarr_i[3].spines['right'].set_visible(False)
        axarr_i[3].spines['left'].set_visible(False)
        axarr_i[4].spines['right'].set_visible(False)
        axarr_i[4].spines['left'].set_visible(False)
        axarr_i[5].spines['left'].set_visible(False)

        # Mark line info
        # [Ne V] 3346.79, [Ne III] 3869, He I 3889 and H8, NeIII3968 and Hepsilon. Hdelta, Hgamma, [O III] 4364, He II 4687
        lines_more = (1 + z) * np.array([3346.79, 3869.86, 3889.00, 3890.16, 3968.59, 3971.20, 4102.89, 4341.68,
                                         4364.44, 4687.31])
        # [O II] 3727, 3729, Hbeta, [O III] 4960 5008
        lines = (1 + z) * np.array([3727.092, 3729.8754960, 4862.721, 4960.295, 5008.239])
        ymin, ymax = [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        # axarr_i[0].vlines(lines, ymin=[-5, -5, -5, -5, -5,], ymax=[100, 100, 100, 100, 100], linestyles='dashed',
        #                    colors='grey')
        # axarr_i[1].vlines(lines, ymin=[-5, -5, -5, -5, -5], ymax=[100, 100, 100, 100, 100], linestyles='dashed',
        #                    colors='grey')
        axarr_i[0].vlines(lines_more, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1)
        axarr_i[1].vlines(lines_more, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1)
        axarr_i[2].vlines(lines_more, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1)
        axarr_i[3].vlines(lines_more, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1)
        axarr_i[4].vlines(lines_more, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1)
        axarr_i[5].vlines(lines_more, ymin=ymin, ymax=ymax, linestyles='dashed', colors='grey', lw=1)
        # axarr_i[0].set_ylim(flux_all.min() - 0.5, flux_all.max() + 0.5)
        # axarr_i[1].set_ylim(flux_all.min() - 0.5, flux_all.max() + 0.5)
        axarr_i[0].set_ylim(-0.15, flux_NeIII3968.max() + 0.05)
        axarr_i[1].set_ylim(-0.15, flux_NeIII3968.max() + 0.05)
        axarr_i[2].set_ylim(-0.15, flux_NeIII3968.max() + 0.05)
        axarr_i[3].set_ylim(-0.15, flux_NeIII3968.max() + 0.05)
        axarr_i[4].set_ylim(-0.15, flux_NeIII3968.max() + 0.05)
        axarr_i[5].set_ylim(-0.15, flux_NeIII3968.max() + 0.05)
        # if flux_Hbeta_i.max() > flux_OIII5008_i.max():
        #     axarr_i[0].set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
        #     axarr_i[1].set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
        axarr_0[0].annotate(text=r'$\mathrm{[Ne \, V]}$', xy=(0.1, 0.65), xycoords='axes fraction', size=15)
        axarr_0[1].annotate(text=r'$\mathrm{[Ne \, III]}$', xy=(-0.25, 0.65), xycoords='axes fraction', size=15)
        axarr_0[1].annotate(text=r'$\mathrm{He \, I}$' + '\n' + r'$\mathrm{H8}$', xy=(0.6, 0.53),
                             xycoords='axes fraction', size=15)
        axarr_0[2].annotate(text=r'$\mathrm{[Ne \, III]}$', xy=(-0.20, 0.65), xycoords='axes fraction', size=15)
        axarr_0[2].annotate(text=r'$\mathrm{H \epsilon}$', xy=(0.74, 0.65), xycoords='axes fraction', size=15)
        axarr_0[3].annotate(text=r'$\mathrm{H \delta}$', xy=(0.1, 0.65), xycoords='axes fraction', size=15)
        axarr_0[4].annotate(text=r'$\mathrm{H \gamma}$', xy=(0.1, 0.65), xycoords='axes fraction', size=15)
        axarr_0[4].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.74, 0.65), xycoords='axes fraction', size=15)
        axarr_0[5].annotate(text=r'$\mathrm{He \, II}$', xy=(0.6, 0.65), xycoords='axes fraction', size=15)

        axarr_i[0].minorticks_on()
        axarr_i[1].minorticks_on()
        axarr_i[2].minorticks_on()
        axarr_i[3].minorticks_on()
        axarr_i[4].minorticks_on()
        axarr_i[5].minorticks_on()
        axarr_i[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on',
                                right=False,
                                labelsize=20, size=5)
        axarr_i[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on',
                                right=False,
                                size=3)
        axarr_i[0].tick_params(axis='y', which='both', right=False, labelright=False)
        axarr_i[5].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False,
                                right='on',
                                labelsize=20, size=5)
        axarr_i[5].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False,
                                right='on',
                                size=3)
        axarr_i[5].tick_params(axis='y', which='both', left=False, labelleft=False)
        for j in [1, 2, 3, 4]:
            axarr_i[j].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False,
                                    right=False,
                                    labelsize=20, size=5)
            axarr_i[j].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False,
                                    right=False,
                                    size=3)
            axarr_i[j].tick_params(axis='y', which='both', right=False, labelright=False, left=False, labelleft=False)

        if i != len(ra_array) - 1:
            axarr_i[0].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i[1].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i[2].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i[3].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i[4].tick_params(axis='x', which='both', labelbottom=False)
            axarr_i[5].tick_params(axis='x', which='both', labelbottom=False)
    t = Table(flux_info, names=('flux_NeV3346', 'flux_NeIII3869', 'flux_HeI3889', 'flux_H8', f'lux_NeIII3968',
                                'flux_Heps', 'flux_Hdel', 'flux_Hgam', 'flux_OIII4364', 'flux_HeII4687',
                                'flux_OII', 'r_OII', 'flux_Hbeta', 'flux_OIII5008', 'dflux_NeV3346',
                                'dflux_NeIII3869', 'dflux_HeI3889', 'dflux_H8', 'dflux_NeIII3968',
                                'dflux_Heps', 'dflux_Hdel', 'dflux_Hgam', 'dflux_OIII4364',
                                'dflux_HeII4687', 'dflux_OII', 'dr_OII', 'dflux_Hbeta', 'dflux_OIII5008'))
    t.write('/Users/lzq/Dropbox/Data/CGM/moreline_profile_selected_region.fits', format='fits', overwrite=True)
    if len(ra_array) == 1:
        fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=-0.12)
        fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20, x=0.03)
    else:
        fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=0.0)
        fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20, x=0.02)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '.png', bbox_inches='tight')


# Plot the data
# Read region file
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

PlotGasSpectra(ra_array, dec_array, radius_array, text_array, figname='spectra_gas_all')

#
# PlotGasSpectra([ra_array[1]], [dec_array[1]], [radius_array[1]], [text_array[1]], figname='spectra_gas_S2')
