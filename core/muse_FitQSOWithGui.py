import os
import sys
import aplpy
import lmfit
import numpy as np
import pyqtgraph as pg
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pyqtgraph.parametertree as pt
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
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
from scipy.ndimage import rotate
from astropy.table import Table
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout
from matplotlib import cm
from scipy import integrate, interpolate
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

# QSO table
path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'

def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355

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

    if params['OII'] == 1:
        z_1, sigma_kms_1, flux_OII_1 = params['z_1'], params['sigma_kms_1'], params['flux_OII_1']
        if params['ResolveOII']:
            r_OII3729_3727_1 = params['r_OII3729_3727_1']
            m_OII = model_OII(wave_OII_vac, z_1, sigma_kms_1, flux_OII_1, r_OII3729_3727_1)
        else:
            m_OII = Gaussian(wave_OII_vac, z_1, sigma_kms_1, flux_OII_1, wave_OII3728_vac)

    elif params['OII'] == 2:
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

    elif params['OII'] == 3:
        z_1, sigma_kms_1, flux_OII_1 = params['z_1'], params['sigma_kms_1'], params['flux_OII_1']
        z_2, sigma_kms_2, flux_OII_2 = params['z_2'], params['sigma_kms_2'], params['flux_OII_2']
        z_3, sigma_kms_3, flux_OII_3 = params['z_3'], params['sigma_kms_3'], params['flux_OII_3']
        if params['ResolveOII']:
            r_OII3729_3727_1 = params['r_OII3729_3727_1']
            r_OII3729_3727_2 = params['r_OII3729_3727_2']
            r_OII3729_3727_3 = params['r_OII3729_3727_3']
            m_OII_1 = model_OII(wave_OII_vac, z_1, sigma_kms_1, flux_OII_1, r_OII3729_3727_1)
            m_OII_2 = model_OII(wave_OII_vac, z_2, sigma_kms_2, flux_OII_2, r_OII3729_3727_2)
            m_OII_3 = model_OII(wave_OII_vac, z_3, sigma_kms_3, flux_OII_3, r_OII3729_3727_3)
        else:
            m_OII_1 = Gaussian(wave_OII_vac, z_1, sigma_kms_1, flux_OII_1, params['OII_center'])
            m_OII_2 = Gaussian(wave_OII_vac, z_2, sigma_kms_2, flux_OII_2, params['OII_center'])
            m_OII_3 = Gaussian(wave_OII_vac, z_3, sigma_kms_3, flux_OII_3, params['OII_center'])
        m_OII = m_OII_1 + m_OII_2 + m_OII_3

    # [O III]
    if params['OIII'] == 1:
        z_1, sigma_kms_1, flux_OIII5008_1 = params['z_1'], params['sigma_kms_1'], params['flux_OIII5008_1']
        m_OIII5008 = Gaussian(wave_OIII_vac, z_1, sigma_kms_1, flux_OIII5008_1, wave_OIII5008_vac)

    elif params['OIII'] == 2:
        z_1, sigma_kms_1, flux_OIII5008_1 = params['z_1'], params['sigma_kms_1'], params['flux_OIII5008_1']
        z_2, sigma_kms_2, flux_OIII5008_2 = params['z_2'], params['sigma_kms_2'], params['flux_OIII5008_2']
        m_OIII5008_1 = Gaussian(wave_OIII_vac, z_1, sigma_kms_1, flux_OIII5008_1, wave_OIII5008_vac)
        m_OIII5008_2 = Gaussian(wave_OIII_vac, z_2, sigma_kms_2, flux_OIII5008_2, wave_OIII5008_vac)
        m_OIII5008 = m_OIII5008_1 + m_OIII5008_2
    elif params['OIII'] == 3:
        z_1, sigma_kms_1, flux_OIII5008_1 = params['z_1'], params['sigma_kms_1'], params['flux_OIII5008_1']
        z_2, sigma_kms_2, flux_OIII5008_2 = params['z_2'], params['sigma_kms_2'], params['flux_OIII5008_2']
        z_3, sigma_kms_3, flux_OIII5008_3 = params['z_3'], params['sigma_kms_3'], params['flux_OIII5008_3']
        m_OIII5008_1 = Gaussian(wave_OIII_vac, z_1, sigma_kms_1, flux_OIII5008_1, wave_OIII5008_vac)
        m_OIII5008_2 = Gaussian(wave_OIII_vac, z_2, sigma_kms_2, flux_OIII5008_2, wave_OIII5008_vac)
        m_OIII5008_3 = Gaussian(wave_OIII_vac, z_3, sigma_kms_3, flux_OIII5008_3, wave_OIII5008_vac)
        m_OIII5008 = m_OIII5008_1 + m_OIII5008_2 + m_OIII5008_3

    if params['OIII'] == 0:
        return m_OII + params['a'] * wave_vac + params['b']
    elif params['OII'] == 0:
        return m_OIII5008 + params['a'] * wave_vac + params['b']
    else:
        return np.hstack((m_OII + params['a_OII'] * wave_OII_vac + params['b_OII'],
                          m_OIII5008 + params['a_OIII5008'] * wave_OIII_vac + params['b_OIII5008']))

class PlotWindow(QMainWindow):
    def __init__(self, cubename='3C57', zapped=False, UseDataSeg=(1.5, 'gauss', None, None),
                 width_OII=50, width_OIII=50, UseSmoothedCubes=True, UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss')):
        super().__init__()

        fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': True, 'r_max': 1.6,
                     'OII_center': wave_OII3728_vac, "OIII": 1, "OIII_2nd": 0}

        # Define lines
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
        data_qso = ascii.read(path_qso, format='fixed_width')
        data_qso = data_qso[data_qso['name'] == cubename]
        self.ra_qso, self.dec_qso, self.z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

        # Load cubes
        if line == 'OII+OIII':
            line_OII, line_OIII = 'OII', 'OIII'
            path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line_OII)
            path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                                     '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
            path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
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
            path_cube = path_cube_OII

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
            self.wave_OII_vac, self.wave_OIII_vac = wave_OII_vac, wave_OIII_vac
            self.flux_OII, self.flux_OIII = flux_OII, flux_OIII
            self.flux_err_OII, self.flux_err_OIII = flux_err_OII, flux_err_OIII


            #
            mask_seg = mask_seg_OII + mask_seg_OIII
            self.wave_vac = np.array([wave_OII_vac, wave_OIII_vac], dtype=object)
            self.flux = np.vstack((flux_OII, flux_OIII))
            self.flux_err = np.vstack((flux_err_OII, flux_err_OIII))
            # self.flux_err = np.where(self.flux_err != 0, self.flux_err, np.inf)

            #
            # flux = np.where(flux_err != 0, flux, np.nan)
            # flux_err = np.where(flux_err != 0, flux_err, np.nan)
        else:
            if line == 'OII':
                width = width_OII
            elif line == 'OIII':
                width = width_OIII
            path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line)
            path_cube_smoothed = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                                 '{}_{}.fits'.format(cubename, str_zap, line, *UseDataSeg)
            if UseDetectionSeg is not None:
                path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line, *UseDetectionSeg)
            else:
                path_3Dseg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line, *UseDataSeg)

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


        # Mask
        self.mask_OII = mask_seg_OII
        self.mask_OIII = mask_seg_OIII
        self.mask = mask_seg


        # Load the QSO field fit
        self.path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
        redshift_guess, sigma_kms_guess, flux_guess, r_OII3729_3727_guess = self.z_qso, 200.0, 1.0, 1.0
        self.model = model_OII_OIII
        self.parameters = lmfit.Parameters()
        self.parameters.add_many(('z_1', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
                                 ('z_2', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
                                 ('z_3', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
                                 ('sigma_kms_1', sigma_kms_guess, True, 50, 2000.0, None),
                                 ('sigma_kms_2', sigma_kms_guess, True, 50, 2000.0, None),
                                 ('sigma_kms_3', sigma_kms_guess, True, 50, 2000.0, None),
                                 ('flux_OII_1', flux_guess, True, 0.0, None, None),
                                 ('flux_OII_2', flux_guess, True, 0.0, None, None),
                                 ('flux_OII_3', flux_guess, True, 0.0, None, None),
                                 ('r_OII3729_3727_1', r_OII3729_3727_guess, True, 0.3, fit_param['r_max'], None),
                                 ('r_OII3729_3727_2', r_OII3729_3727_guess, True, 0.3, fit_param['r_max'], None),
                                 ('r_OII3729_3727_3', r_OII3729_3727_guess, True, 0.3, fit_param['r_max'], None),
                                 ('flux_OIII5008_1', flux_guess, True, 0, None, None),
                                 ('flux_OIII5008_2', flux_guess, True, 0, None, None),
                                 ('flux_OIII5008_3', flux_guess, True, 0, None, None),
                                 ('a_OII', 0.0, False, None, None, None),
                                 ('b_OII', 0.0, True, -0.1, 0.1, None),
                                 ('a_OIII5008', 0.0, False, None, None, None),
                                 ('b_OIII5008', 0.0, True, -0.1, 0.1, None),
                                 ('OII', 1, False, None, None, None),
                                 ('OIII', 1, False, None, None, None),
                                 ('z_qso', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
                                 ('ResolveOII', fit_param['ResolveOII'], False, None, None, None),
                                 ('OII_center', fit_param['OII_center'], False, None, None, None))
        if os.path.exists(self.path_fit) is False:
            print('Fitting result file does not exist, start fitting from scratch.')

            # if num_Gaussian == 1:
            #
            #     self.fit()
            # elif num_Gaussian == 2:
            #
            # elif num_Gaussian == 3:
            #
            # else:
            #
            # self.fit()
        self.hdul_fit = fits.open(self.path_fit)
        pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, \
        da_OII, b_OII, db_OII, a_OIII, da_OIII, b_OIII, db_OIII, chisqr, redchi = self.hdul_fit[:23]
        pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, \
        da_OII, b_OII, db_OII, a_OIII, da_OIII, b_OIII, db_OIII, \
        chisqr, redchi = pri.data, fs.data, v.data, z.data, dz.data, sigma.data, dsigma.data, \
                         flux_OII_fit.data, dflux_OII_fit.data, flux_OIII_fit.data, dflux_OIII_fit.data, r.data, \
                         dr.data, a_OII.data, da_OII.data, b_OII.data, db_OII.data, a_OIII.data, da_OIII.data, \
                         b_OIII.data, db_OIII.data, chisqr.data, redchi.data
        self.size = fs.shape

        # Convert zeros to nan
        # v, z, dz = np.where(v != 0, v, np.nan), np.where(z != 0, z, np.nan), np.where(dz != 0, dz, np.nan)
        # sigma, dsigma = np.where(sigma != 0, sigma, np.nan), np.where(dsigma != 0, dsigma, np.nan)
        # flux_OII_fit, dflux_OII_fit = np.where(flux_OII_fit != 0, flux_OII_fit, np.nan), np.where(dflux_OII_fit != 0, dflux_OII_fit, np.nan)
        # flux_OIII_fit, dflux_OIII_fit = np.where(flux_OIII_fit != 0, flux_OIII_fit, np.nan), np.where(dflux_OIII_fit != 0, dflux_OIII_fit, np.nan)
        # r, dr = np.where(r != 0, r, np.nan), np.where(dr != 0, dr, np.nan)
        # self.hdul_fit[2].data, self.hdul_fit[3].data, self.hdul_fit[4].data, self.hdul_fit[5].data, \
        # self.hdul_fit[6].data, self.hdul_fit[7].data, self.hdul_fit[8].data, self.hdul_fit[9].data, \
        # self.hdul_fit[10].data, self.hdul_fit[11].data, self.hdul_fit[12].data = v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, \
        #                                                                          flux_OIII_fit, dflux_OIII_fit, r, dr
        # self.hdul_fit.writeto(self.path_fit, overwrite=True)
        #
        # raise ValueError('Convert zeros')
        # if v.shape[0] != 3:
        #     print('Number of Gaussian components is not 3, please check the fitting results.')
        #     row = np.zeros((1, *self.size))
        #     v, z, dz = np.concatenate((v, row), axis=0), np.concatenate((z, row), axis=0), np.concatenate((dz, row), axis=0)
        #     sigma, dsigma = np.concatenate((sigma, row), axis=0), np.concatenate((dsigma, row), axis=0)
        #     flux_OII_fit, dflux_OII_fit = np.concatenate((flux_OII_fit, row), axis=0), np.concatenate((dflux_OII_fit, row), axis=0)
        #     flux_OIII_fit, dflux_OIII_fit = np.concatenate((flux_OIII_fit, row), axis=0), np.concatenate((dflux_OIII_fit, row), axis=0)
        #     r, dr = np.concatenate((r, row), axis=0), np.concatenate((dr, row), axis=0)
        #     self.hdul_fit[2].data, self.hdul_fit[3].data, self.hdul_fit[4].data, self.hdul_fit[5].data, \
        #     self.hdul_fit[6].data, self.hdul_fit[7].data, self.hdul_fit[8].data, self.hdul_fit[9].data, \
        #     self.hdul_fit[10].data, self.hdul_fit[11].data, self.hdul_fit[12].data = v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, \
        #                                                                              flux_OIII_fit, dflux_OIII_fit, r, dr
        #     self.hdul_fit.writeto(self.path_fit, overwrite=True)


        self.pri, self.fs, self.v, self.z, self.dz, self.sigma, self.dsigma, self.flux_OII_fit, \
        self.dflux_OII_fit, self.flux_OIII_fit, self.dflux_OIII_fit, self.r, self.dr, self.a_OII, \
        self.da_OII, self.b_OII, self.db_OII, self.a_OIII, self.da_OIII, self.b_OIII, \
        self.db_OIII, self.chisqr, self.redchi = pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, \
                                                 flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, da_OII, b_OII, db_OII, \
                                                 a_OIII, da_OIII, b_OIII, db_OIII, chisqr, redchi

        # Calculae flux of each component
        self.wave_OII_exp = expand_wave([self.wave_OII_vac], stack=True)
        self.wave_OIII_exp = expand_wave([self.wave_OIII_vac], stack=True)
        self.flux_OII_array = model_OII(self.wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], self.z, self.sigma,
                                        self.flux_OII_fit, self.r, plot=False)
        self.flux_OIII_array = Gaussian(self.wave_OIII_exp[:, np.newaxis, np.newaxis, np.newaxis], self.z, self.sigma,
                                        self.flux_OIII_fit, wave_OIII5008_vac)

        # Calculate V_50 and W_80
        flux_OII_v50 = model_OII(self.wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], self.z, self.sigma,
                                 self.flux_OII_fit, self.r, plot=True)[0]
        wave_OII_exp_ = self.wave_OII_exp[:, np.newaxis, np.newaxis]
        wave_OIII_exp_ = self.wave_OIII_exp[:, np.newaxis, np.newaxis]
        flux_OII_sum = np.nansum(flux_OII_v50, axis=1)
        flux_OIII_sum = np.nansum(self.flux_OIII_array, axis=1)

        # Moments
        flux_cumsum_OII = integrate.cumtrapz(flux_OII_sum, wave_OII_exp_, initial=0, axis=0)
        flux_cumsum_OIII = integrate.cumtrapz(flux_OIII_sum, wave_OIII_exp_, initial=0, axis=0)
        flux_cumsum_OII /= flux_cumsum_OII.max(axis=0)
        flux_cumsum_OIII /= flux_cumsum_OIII.max(axis=0)

        wave_10_OII, wave_10_OIII = np.nan * np.zeros(self.size), np.nan * np.zeros(self.size)
        wave_50_OII, wave_50_OIII = np.nan * np.zeros(self.size), np.nan * np.zeros(self.size)
        wave_90_OII, wave_90_OIII = np.nan * np.zeros(self.size), np.nan * np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.mask[i, j] != 0:
                    f_OII = interpolate.interp1d(flux_cumsum_OII[:, i, j], self.wave_OII_exp, fill_value='extrapolate')
                    f_OIII = interpolate.interp1d(flux_cumsum_OIII[:, i, j], self.wave_OIII_exp, fill_value='extrapolate')
                    wave_10_OII[i, j], wave_10_OIII[i, j] = f_OII(0.1), f_OIII(0.1)
                    wave_50_OII[i, j], wave_50_OIII[i, j] = f_OII(0.5), f_OIII(0.5)
                    wave_90_OII[i, j], wave_90_OIII[i, j] = f_OII(0.9), f_OIII(0.9)
                else:
                    pass

        z50_OII = (wave_50_OII - wave_OII3727_vac) / wave_OII3727_vac
        w80_OII = c_kms * (wave_90_OII - wave_10_OII) / (wave_OII3727_vac * (1 + z50_OII))
        z50_OIII = (wave_50_OIII - wave_OIII5008_vac) / wave_OIII5008_vac
        w80_OIII = c_kms * (wave_90_OIII - wave_10_OIII) / (wave_OIII5008_vac * (1 + z50_OIII))

        z50 = np.where(self.mask_OIII != 0, z50_OIII, z50_OII)
        v50 = c_kms * (z50 - self.z_qso) / (1 + self.z_qso)
        w80 = np.where(self.mask_OIII > 0, w80_OIII, w80_OII)
        v50 = np.where(self.mask != 0, v50, np.nan)
        w80 = np.where(self.mask != 0, w80, np.nan)
        self.v50 = v50
        self.w80 = w80

        # Define a top-level widget
        self.widget = QWidget()
        self.widget.resize(2000, 2000)
        self.setCentralWidget(self.widget)
        self.layout = QtGui.QGridLayout()
        self.widget.setLayout(self.layout)

        # Set title
        self.setWindowTitle("Check fitting")

        # Create plot widgets
        self.widget1 = pg.GraphicsLayoutWidget()
        self.widget2 = pg.GraphicsLayoutWidget()
        self.widget3 = pg.GraphicsLayoutWidget()
        self.widget4 = pg.GraphicsLayoutWidget()
        self.widget5 = pg.GraphicsLayoutWidget()
        self.widget1_plot = self.widget1.addPlot()
        self.widget2_plot = self.widget2.addPlot()
        self.widget3_plot = self.widget3.addPlot()
        self.widget4_plot = self.widget4.addPlot()
        self.widget5_plot = self.widget5.addPlot()
        self.widget1.setFixedSize(450, 450)
        self.widget2.setFixedSize(450, 450)
        self.widget3.setFixedSize(450, 450)
        self.widget4.setFixedSize(450, 450)
        self.widget5.setFixedSize(450, 450)

        # Set background color
        self.widget1.setBackground((235, 233, 221, 100))
        self.widget2.setBackground((235, 233, 221, 100))
        self.widget3.setBackground((235, 233, 221, 100))
        self.widget4.setBackground((235, 233, 221, 100))
        self.widget5.setBackground((235, 233, 221, 100))
        self.widget1_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget2_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget3_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])

        # Set param
        self.paramSpec = [dict(name='chi=', type='float', value=0, dec=False, readonly=False),
                          dict(name='v_1=', type='float', value=0, dec=False, readonly=False),
                          dict(name='sigma_1=', type='float', value=0, readonly=False),
                          dict(name='OII_1=', type='float', value=0, readonly=False),
                          dict(name='OIII_1=', type='float', value=0, readonly=False),
                          dict(name='v_2=', type='float', value=0, dec=False, readonly=False),
                          dict(name='sigma_2=', type='float', value=0, readonly=False),
                          dict(name='OII_2=', type='float', value=0, readonly=False),
                          dict(name='OIII_2=', type='float', value=0, readonly=False),
                          dict(name='v_3=', type='float', value=0, dec=False, readonly=False),
                          dict(name='sigma_3=', type='float', value=0, readonly=False),
                          dict(name='OII_3=', type='float', value=0, readonly=False),
                          dict(name='OIII_3=', type='float', value=0, readonly=False)]
        self.param = pt.Parameter.create(name='Options', type='group', children=self.paramSpec)
        self.tree = pt.ParameterTree()
        self.tree.setParameters(self.param)

        # Buttons
        btn_1 = QPushButton("Refit")
        btn_2 = QPushButton("1 Gauss")
        btn_3 = QPushButton("2 Gauss")
        btn_4 = QPushButton("3 Gauss")
        btn_5 = QPushButton("Save v50w80")
        btn_6 = QPushButton("Clear")
        btn_1.clicked.connect(self.update_fit)
        btn_2.clicked.connect(self.N_1)
        btn_3.clicked.connect(self.N_2)
        btn_4.clicked.connect(self.N_3)
        btn_5.clicked.connect(self.save_v50w80)
        btn_6.clicked.connect(self.clear_scatter)

        layout_RHS = QVBoxLayout()
        layout_btn = QHBoxLayout()
        layout_btn.addWidget(btn_1)
        layout_btn.addWidget(btn_2)
        layout_btn.addWidget(btn_3)
        layout_btn.addWidget(btn_4)
        layout_btn.addWidget(btn_5)
        layout_btn.addWidget(btn_6)
        layout_RHS.addLayout(layout_btn)
        layout_RHS.addWidget(self.tree)

        #
        self.layout.addWidget(self.widget1, 0, 0, 1, 1)
        self.layout.addWidget(self.widget2, 0, 1, 1, 1)
        self.layout.addWidget(self.widget3, 0, 2, 1, 1)
        self.layout.addWidget(self.widget4, 1, 0, 1, 1)
        self.layout.addWidget(self.widget5, 1, 1, 1, 1)
        self.layout.addLayout(layout_RHS, 1, 2, 1, 1)
        # self.layout.addWidget(self.tree, 1, 2, 1, 1)


        # Plot the 2D map in the first plot
        self.v_map = pg.ImageItem()
        self.widget1_plot.addItem(self.v_map)
        colormap = cm.get_cmap("coolwarm")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.v_map.setLookupTable(lut)
        self.v_map.updateImage(image=self.v50.T, levels=(-350, 350))

        # Plot the 2D map in the second plot
        self.sigma_map = pg.ImageItem()
        self.widget2_plot.addItem(self.sigma_map)
        colormap = Dense_20_r.mpl_colormap
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.sigma_map.setLookupTable(lut)
        self.sigma_map.updateImage(image=self.w80.T, levels=(0, 800))

        # Plot the chi 2D map in the third plot
        self.chi_map = pg.ImageItem()
        self.widget3_plot.addItem(self.chi_map)
        colormap = cm.get_cmap("viridis")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.chi_map.setLookupTable(lut)
        self.chi_map.updateImage(image=self.redchi.T, levels=(0, 2))

        # Mark the locations
        self.scatter_1 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
        self.scatter_2 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
        self.scatter_3 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))

        # Link the axis of the plots
        self.widget1_plot.setXLink(self.widget2_plot)
        self.widget1_plot.setYLink(self.widget2_plot)
        self.widget2_plot.setXLink(self.widget1_plot)
        self.widget2_plot.setYLink(self.widget1_plot)
        self.widget3_plot.setXLink(self.widget2_plot)
        self.widget3_plot.setYLink(self.widget2_plot)

        # Plot initial data in the second plot
        self.widget4_plot.setLabel('bottom', 'Wavelength ($\AA$)')
        self.widget4_plot.setLabel('left', 'Flux')
        self.widget5_plot.setLabel('bottom', r'Wavelength ($\AA$)')
        self.widget5_plot.setLabel('left', 'Flux')

        # Connect mouse click event to update plot
        self.widget1_plot.scene().sigMouseClicked.connect(self.update_plot)
        self.widget2_plot.scene().sigMouseClicked.connect(self.update_plot)
        self.widget3_plot.scene().sigMouseClicked.connect(self.update_plot)

    def fit(self):
        # fitting starts
        fit_success = np.zeros(self.size)
        v_fit, dv_fit = np.zeros(self.size), np.zeros(self.size)
        sigma_fit, dsigma_fit = np.zeros(self.size), np.zeros(self.size)
        flux_fit, dflux_fit = np.zeros(self.size), np.zeros(self.size)

        for i in range(self.size[0]):  # i = p (y), j = q (x)
            for j in range(self.size[1]):
                if self.mask[i, j]:
                    self.parameters['v'].value = self.v_Serra[i, j]
                    flux_ij = self.flux[:, i, j]
                    flux_err_ij = self.flux_err[:, 0, 0]
                    spec_model = lmfit.Model(self.model, missing='drop')
                    result = spec_model.fit(flux_ij, velocity=self.v_array, params=self.parameters,
                                            weights=1 / flux_err_ij)

                    # Access the fitting results
                    fit_success[i, j] = result.success
                    v, dv = result.best_values['v'], result.params['v'].stderr
                    sigma, dsigma = result.best_values['sigma'], result.params['sigma'].stderr
                    flux, dflux = result.best_values['flux'], \
                                      result.params['flux'].stderr

                    # fill the value
                    v_fit[i, j], dv_fit[i, j] = v, dv
                    sigma_fit[i, j], dsigma_fit[i, j] = sigma, dsigma
                    flux_fit[i, j], dflux_fit[i, j] = flux, dflux
                else:
                    pass

        # Save fitting results
        hdul_fs = fits.PrimaryHDU(fit_success, header=self.hdr)
        hdul_v, hdul_dv = fits.ImageHDU(v_fit, header=self.hdr), fits.ImageHDU(dv_fit, header=self.hdr)
        hdul_sigma, hdul_dsigma = fits.ImageHDU(sigma_fit, header=self.hdr), fits.ImageHDU(dsigma_fit, header=self.hdr)
        hdul_flux, hdul_dflux = fits.ImageHDU(flux_fit, header=self.hdr), fits.ImageHDU(dflux_fit, header=self.hdr)
        hdul = fits.HDUList([hdul_fs, hdul_v, hdul_dv, hdul_sigma, hdul_dsigma, hdul_flux, hdul_dflux])
        hdul.writeto(self.path_fit, overwrite=True)

    def update_plot(self, event):
        if event.double():
            # Get pixel coordinates
            pos = event.pos()
            # pos = self.widget1_plot.vb.mapSceneToView(pos)
            pos = self.v_map.mapFromScene(pos)
            # print(pos.x(), pos.y())
            self.xpixel, self.ypixel = int(np.floor(pos.x() + 1)), int(np.floor(pos.y()))

            self.show_fit()
            self.plot_OII()
            self.plot_OIII()

    def show_fit(self):
        i, j = self.ypixel, self.xpixel
        # Get the fitting results
        if self.mask[i, j]:
            # Show the fitting results
            self.param['chi='] = '{:.2f}'.format(self.redchi[i, j])
            self.param['v_1='] = '{:.0f}'.format(self.v[0, i, j])
            self.param['sigma_1='] = '{:.0f}'.format(self.sigma[0, i, j])
            self.param['OII_1='] = '{:.4f}'.format(self.flux_OII_fit[0, i, j])
            self.param['OIII_1='] = '{:.4f}'.format(self.flux_OIII_fit[0, i, j])
            self.param['v_2='] = '{:.0f}'.format(self.v[1, i, j])
            self.param['sigma_2='] = '{:.0f}'.format(self.sigma[1, i, j])
            self.param['OII_2='] = '{:.4f}'.format(self.flux_OII_fit[1, i, j])
            self.param['OIII_2='] = '{:.4f}'.format(self.flux_OIII_fit[1, i, j])
            self.param['v_3='] = '{:.0f}'.format(self.v[2, i, j])
            self.param['sigma_3='] = '{:.0f}'.format(self.sigma[2, i, j])
            self.param['OII_3='] = '{:.4f}'.format(self.flux_OII_fit[2, i, j])
            self.param['OIII_3='] = '{:.4f}'.format(self.flux_OIII_fit[2, i, j])

            # Draw points
            self.scatter_1.addPoints([self.xpixel + 0.5], [self.ypixel + 0.5])
            self.scatter_2.addPoints([self.xpixel + 0.5], [self.ypixel + 0.5])
            self.scatter_3.addPoints([self.xpixel + 0.5], [self.ypixel + 0.5])
            self.widget1_plot.addItem(self.scatter_1)
            self.widget2_plot.addItem(self.scatter_2)
            self.widget3_plot.addItem(self.scatter_3)

    def plot_OII(self):
        i, j = self.ypixel, self.xpixel
        self.widget4_plot.clear()
        if self.mask_OII[i, j]:
            # Plot the data
            self.widget4_plot.plot(self.wave_OII_vac, self.flux_OII[:, i, j], pen='k')
            self.widget4_plot.plot(self.wave_OII_vac, self.flux_err_OII[:, i, j], pen='g')

            # Plot each individual component
            self.widget4_plot.plot(self.wave_OII_exp, self.flux_OII_array[:, 0, i, j],
                                   pen=pg.mkPen(color=(91, 28, 237), width=5, style=QtCore.Qt.DashLine))
            self.widget4_plot.plot(self.wave_OII_exp, self.flux_OII_array[:, 1, i, j],
                                   pen=pg.mkPen(color=(119, 73, 227), width=3, style=QtCore.Qt.DashLine))
            self.widget4_plot.plot(self.wave_OII_exp, self.flux_OII_array[:, 2, i, j],
                                   pen=pg.mkPen(color=(151, 116, 232), width=2, style=QtCore.Qt.DashLine))
            self.widget4_plot.plot(self.wave_OII_exp, np.nansum(self.flux_OII_array[:, :, i, j], axis=1)
                                   + self.b_OII[i, j] + self.a_OII[i, j] * self.wave_OII_exp, pen='r')

    def plot_OIII(self):
        i, j = self.ypixel, self.xpixel
        self.widget5_plot.clear()
        if self.mask_OIII[i, j]:
            # Plot the data and the fit
            self.widget5_plot.plot(self.wave_OIII_vac, self.flux_OIII[:, i, j], pen='k')
            self.widget5_plot.plot(self.wave_OIII_vac, self.flux_err_OIII[:, i, j], pen='g')

            # Plot each individual component
            self.widget5_plot.plot(self.wave_OIII_exp, self.flux_OIII_array[:, 0, i, j],
                                   pen=pg.mkPen(color=(91, 28, 237), width=5, style=QtCore.Qt.DashLine))
            self.widget5_plot.plot(self.wave_OIII_exp, self.flux_OIII_array[:, 1, i, j],
                                   pen=pg.mkPen(color=(119, 73, 227), width=3, style=QtCore.Qt.DashLine))
            self.widget5_plot.plot(self.wave_OIII_exp, self.flux_OIII_array[:, 2, i, j],
                                   pen=pg.mkPen(color=(151, 116, 232), width=2, style=QtCore.Qt.DashLine))
            self.widget5_plot.plot(self.wave_OIII_exp, np.nansum(self.flux_OIII_array[:, :, i, j], axis=1)
                                   + self.b_OIII[i, j] + self.a_OIII[i, j] * self.wave_OIII_exp, pen='r')

    def N_1(self):
        self.parameters['OII'].value = 1
        self.parameters['OIII'].value = 1
        self.parameters['z_2'].value = np.nan
        self.parameters['z_2'].vary = False
        self.parameters['z_3'].value = np.nan
        self.parameters['z_3'].vary = False
        self.parameters['sigma_kms_2'].value = np.nan
        self.parameters['sigma_kms_2'].vary = False
        self.parameters['sigma_kms_3'].value = np.nan
        self.parameters['sigma_kms_3'].vary = False
        self.parameters['flux_OII_2'].value = np.nan
        self.parameters['flux_OII_2'].vary = False
        self.parameters['flux_OII_3'].value = np.nan
        self.parameters['flux_OII_3'].vary = False
        self.parameters['r_OII3729_3727_2'].value = np.nan
        self.parameters['r_OII3729_3727_2'].vary = False
        self.parameters['r_OII3729_3727_3'].value = np.nan
        self.parameters['r_OII3729_3727_3'].vary = False
        self.parameters['flux_OIII5008_2'].value = np.nan
        self.parameters['flux_OIII5008_2'].vary = False
        self.parameters['flux_OIII5008_3'].value = np.nan
        self.parameters['flux_OIII5008_3'].vary = False

    def N_2(self):
        if self.parameters['OII'].value == 1:
            self.param['v_2='] = 0
            self.param['sigma_2='] = 0
        self.parameters['OII'].value = 2
        self.parameters['OIII'].value = 2
        self.parameters['z_2'].value = self.z_qso
        self.parameters['z_2'].vary = True
        self.parameters['z_3'].value = np.nan
        self.parameters['z_3'].vary = False
        self.parameters['sigma_kms_2'].value = 200
        self.parameters['sigma_kms_2'].vary = True
        self.parameters['sigma_kms_3'].value = np.nan
        self.parameters['sigma_kms_3'].vary = False
        self.parameters['flux_OII_2'].value = 1
        self.parameters['flux_OII_2'].vary = True
        self.parameters['flux_OII_3'].value = np.nan
        self.parameters['flux_OII_3'].vary = False
        self.parameters['r_OII3729_3727_2'].value = 1
        self.parameters['r_OII3729_3727_2'].vary = True
        self.parameters['r_OII3729_3727_3'].value = np.nan
        self.parameters['r_OII3729_3727_3'].vary = False
        self.parameters['flux_OIII5008_2'].value = 1
        self.parameters['flux_OIII5008_2'].vary = True
        self.parameters['flux_OIII5008_3'].value = np.nan
        self.parameters['flux_OIII5008_3'].vary = False

    def N_3(self):
        self.parameters['OII'].value = 3
        self.parameters['OIII'].value = 3
        self.parameters['z_2'].value = self.z_qso
        self.parameters['z_2'].vary = True
        self.parameters['z_3'].value = self.z_qso
        self.parameters['z_3'].vary = True
        self.parameters['sigma_kms_2'].value = 200
        self.parameters['sigma_kms_2'].vary = True
        self.parameters['sigma_kms_3'].value = 200
        self.parameters['sigma_kms_3'].vary = True
        self.parameters['flux_OII_2'].value = 1
        self.parameters['flux_OII_2'].vary = True
        self.parameters['flux_OII_3'].value = 1
        self.parameters['flux_OII_3'].vary = True
        self.parameters['r_OII3729_3727_2'].value = 1
        self.parameters['r_OII3729_3727_2'].vary = True
        self.parameters['r_OII3729_3727_3'].value = 1
        self.parameters['r_OII3729_3727_3'].vary = True
        self.parameters['flux_OIII5008_2'].value = 1
        self.parameters['flux_OIII5008_2'].vary = True
        self.parameters['flux_OIII5008_3'].value = 1
        self.parameters['flux_OIII5008_3'].vary = True

    def compute_v50w80(self, i, j):
        flux_OII_ij = model_OII(self.wave_OII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                self.flux_OII_fit[:, i, j], self.r[:, i, j], plot=True)[0]
        flux_OIII_ij = Gaussian(self.wave_OIII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                self.flux_OIII_fit[:, i, j], wave_OIII5008_vac)

        # Calculate V_50 and W_80 for a single pixel
        flux_OII_sum = np.nansum(flux_OII_ij, axis=1)
        flux_OIII_sum = np.nansum(flux_OIII_ij, axis=1)

        # Moments
        flux_cumsum_OII = integrate.cumtrapz(flux_OII_sum, self.wave_OII_exp, initial=0, axis=0)
        flux_cumsum_OII /= flux_cumsum_OII.max(axis=0)
        flux_cumsum_OIII = integrate.cumtrapz(flux_OIII_sum, self.wave_OIII_exp, initial=0, axis=0)
        flux_cumsum_OIII /= flux_cumsum_OIII.max(axis=0)

        f_OII = interpolate.interp1d(flux_cumsum_OII, self.wave_OII_exp, fill_value='extrapolate')
        f_OIII = interpolate.interp1d(flux_cumsum_OIII, self.wave_OIII_exp, fill_value='extrapolate')

        wave_10_OII, wave_10_OIII = f_OII(0.1), f_OIII(0.1)
        wave_50_OII, wave_50_OIII = f_OII(0.5), f_OIII(0.5)
        wave_90_OII, wave_90_OIII = f_OII(0.9), f_OIII(0.9)

        z50_OII = (wave_50_OII - wave_OII3727_vac) / wave_OII3727_vac
        w80_OII = c_kms * (wave_90_OII - wave_10_OII) / (wave_OII3727_vac * (1 + z50_OII))
        z50_OIII = (wave_50_OIII - wave_OIII5008_vac) / wave_OIII5008_vac
        w80_OIII = c_kms * (wave_90_OIII - wave_10_OIII) / (wave_OIII5008_vac * (1 + z50_OIII))

        if self.mask_OIII[i, j] != 0:
            z50 = z50_OIII
            w80 = w80_OIII
        else:
            z50 = z50_OII
            w80 = w80_OII
        v50 = c_kms * (z50 - self.z_qso) / (1 + self.z_qso)
        return v50, w80

    def update_fit(self):
        i, j = self.ypixel, self.xpixel

        # Refit that specific pixel
        z_1 = (self.param['v_1='] / c_kms * (1 + self.z_qso)) + self.z_qso
        self.parameters['z_1'].value = z_1
        self.parameters['z_1'].max = z_1 + (100 / c_kms * (1 + self.z_qso))
        self.parameters['z_1'].min = z_1 - (100 / c_kms * (1 + self.z_qso))
        self.parameters['sigma_kms_1'].value = self.param['sigma_1=']
        self.parameters['sigma_kms_1'].max = self.param['sigma_1='] + 100
        self.parameters['sigma_kms_1'].min = 30

        if self.parameters['OII'].value == 2:
            z_2 = (self.param['v_2='] / c_kms * (1 + self.z_qso)) + self.z_qso
            self.parameters['z_2'].value = z_2
            self.parameters['z_2'].max = z_2 + (100 / c_kms * (1 + self.z_qso))
            self.parameters['z_2'].min = z_2 - (100 / c_kms * (1 + self.z_qso))
            self.parameters['sigma_kms_2'].value = self.param['sigma_2=']
            self.parameters['sigma_kms_2'].max = self.param['sigma_2='] + 100
            self.parameters['sigma_kms_2'].min = 30

        elif self.parameters['OII'].value == 3:
            z_3 = (self.param['v_3='] / c_kms * (1 + self.z_qso)) + self.z_qso
            self.parameters['z_3'].value = z_3
            self.parameters['z_3'].max = z_3 + (100 / c_kms * (1 + self.z_qso))
            self.parameters['z_3'].min = z_3 - (100 / c_kms * (1 + self.z_qso))
            self.parameters['sigma_kms_3'].value = self.param['sigma_3=']
            self.parameters['sigma_kms_3'].max = self.param['sigma_3='] + 100
            self.parameters['sigma_kms_3'].min = 30

        #
        flux_ij = self.flux[:, i, j]
        flux_err_ij = self.flux_err[:, i, j]
        spec_model = lmfit.Model(self.model, missing='drop')
        result = spec_model.fit(flux_ij, wave_vac=self.wave_vac, params=self.parameters, weights=1 / flux_err_ij)
        self.fs[i, j] = result.success
        self.chisqr[i, j], self.redchi[i, j] = result.chisqr, result.redchi

        # fill the value
        a_OII, b_OII = result.best_values['a_OII'], result.best_values['b_OII']
        da_OII, db_OII = result.params['a_OII'].stderr, result.params['b_OII'].stderr
        a_OIII, b_OIII = result.best_values['a_OIII5008'], result.best_values['b_OIII5008']
        da_OIII, db_OIII = result.params['a_OIII5008'].stderr, result.params['b_OIII5008'].stderr
        self.a_OII[i, j], self.da_OII[i, j], self.b_OII[i, j], self.db_OII[i, j] = a_OII, da_OII, b_OII, db_OII
        self.a_OIII[i, j], self.da_OIII[i, j] = a_OIII, da_OIII
        self.b_OIII[i, j], self.db_OIII[i, j] = b_OIII, db_OIII

        z_1, dz_1 = result.best_values['z_1'], result.params['z_1'].stderr
        z_2, dz_2 = result.best_values['z_2'], result.params['z_2'].stderr
        z_3, dz_3 = result.best_values['z_3'], result.params['z_3'].stderr
        sigma_1, dsigma_1 = result.best_values['sigma_kms_1'], result.params['sigma_kms_1'].stderr
        sigma_2, dsigma_2 = result.best_values['sigma_kms_2'], result.params['sigma_kms_2'].stderr
        sigma_3, dsigma_3 = result.best_values['sigma_kms_3'], result.params['sigma_kms_3'].stderr
        self.z[:, i, j], self.sigma[:, i, j] = [z_1, z_2, z_3], [sigma_1, sigma_2, sigma_3]
        self.dz[:, i, j], self.dsigma[:, i, j] = [dz_1, dz_2, dz_3], [dsigma_1, dsigma_2, dsigma_3]
        self.v[:, i, j] = [c_kms * (z_1 - self.z_qso) / (1 + self.z_qso),
                           c_kms * (z_2 - self.z_qso) / (1 + self.z_qso),
                           c_kms * (z_3 - self.z_qso) / (1 + self.z_qso)]


        flux_OII_1, dflux_OII_1 = result.best_values['flux_OII_1'], result.params['flux_OII_1'].stderr
        flux_OII_2, dflux_OII_2 = result.best_values['flux_OII_2'], result.params['flux_OII_2'].stderr
        flux_OII_3, dflux_OII_3 = result.best_values['flux_OII_3'], result.params['flux_OII_3'].stderr
        r_1, dr_1 = result.best_values['r_OII3729_3727_1'], result.params['r_OII3729_3727_1'].stderr
        r_2, dr_2 = result.best_values['r_OII3729_3727_2'], result.params['r_OII3729_3727_2'].stderr
        r_3, dr_3 = result.best_values['r_OII3729_3727_3'], result.params['r_OII3729_3727_3'].stderr
        self.flux_OII_fit[:, i, j] = [flux_OII_1, flux_OII_2, flux_OII_3]
        self.dflux_OII_fit[:, i, j] = [dflux_OII_1, dflux_OII_2, dflux_OII_3]
        self.r[:, i, j] = [r_1, r_2, r_3]
        self.dr[:, i, j] = [dr_1, dr_2, dr_3]

        flux_OIII_1, dflux_OIII_1 = result.best_values['flux_OIII5008_1'], result.params['flux_OIII5008_1'].stderr
        flux_OIII_2, dflux_OIII_2 = result.best_values['flux_OIII5008_2'], result.params['flux_OIII5008_2'].stderr
        flux_OIII_3, dflux_OIII_3 = result.best_values['flux_OIII5008_3'], result.params['flux_OIII5008_3'].stderr
        self.flux_OIII_fit[:, i, j] = [flux_OIII_1, flux_OIII_2, flux_OIII_3]
        self.dflux_OIII_fit[:, i, j] = [dflux_OIII_1, dflux_OIII_2, dflux_OIII_3]

        self.hdul_fit[1].data = self.fs
        self.hdul_fit[2].data, self.hdul_fit[3].data, self.hdul_fit[4].data, self.hdul_fit[5].data, \
        self.hdul_fit[6].data, self.hdul_fit[7].data, self.hdul_fit[8].data, self.hdul_fit[9].data, \
        self.hdul_fit[10].data, self.hdul_fit[11].data, \
        self.hdul_fit[12].data = self.v, self.z, self.dz, self.sigma, self.dsigma, self.flux_OII_fit, \
                                 self.dflux_OII_fit, self.flux_OIII_fit, self.dflux_OIII_fit, self.r, self.dr
        self.hdul_fit[21].data, self.hdul_fit[22].data = self.chisqr, self.redchi

        # Save fitting results
        self.hdul_fit.writeto(self.path_fit, overwrite=True)

        # Re initiolize
        self.v50[i, j], self.w80[i, j] = self.compute_v50w80(i, j)

        # Update fit
        self.flux_OII_array[:, :, i, j] = model_OII(self.wave_OII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                        self.flux_OII_fit[:, i, j], self.r[:, i, j], plot=False)
        self.flux_OIII_array[:, :, i, j] = Gaussian(self.wave_OIII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                        self.flux_OIII_fit[:, i, j], wave_OIII5008_vac)

        # Replot
        self.show_fit()
        self.plot_OII()
        self.plot_OIII()
        self.v_map.updateImage(image=self.v50.T)
        self.sigma_map.updateImage(image=self.w80.T)
        self.chi_map.updateImage(image=self.redchi.T)

    def save_v50w80(self):
        path_v50 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_V50.fits'
        path_w80 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_W80.fits'

        hdul_v50 = fits.ImageHDU(self.v50, header=self.hdul_fit[2].header)
        hdul_v50.writeto(path_v50, overwrite=True)
        hdul_w80 = fits.ImageHDU(self.w80, header=self.hdul_fit[2].header)
        hdul_w80.writeto(path_w80, overwrite=True)

    def clear_scatter(self):
        self.scatter_1.clear()
        self.scatter_2.clear()
        self.scatter_3.clear()






if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotWindow(cubename='3C57')
    window.show()
    sys.exit(app.exec_())
