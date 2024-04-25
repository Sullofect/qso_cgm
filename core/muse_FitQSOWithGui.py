import os
import sys
import aplpy
import lmfit
import numpy as np
import pyqtgraph as pg
import matplotlib as mpl
import gala.potential as gp
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib import cm
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

class PlotWindow(QMainWindow):
    def __init__(self, cubename='3C57', zapped=False, UseDataSeg=(1.5, 'gauss', None, None), FitType='specific',
             CheckGuess=None, width_OII=10, width_OIII=10, UseSmoothedCubes=True, UseDetectionSeg=None):
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
        ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

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
            wave_vac = np.array([wave_OII_vac, wave_OIII_vac], dtype=object)
            flux = np.vstack((flux_OII, flux_OIII))
            flux_err = np.vstack((flux_err_OII, flux_err_OIII))
            # flux_err = np.where(flux_err != 0, flux_err, np.inf)

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
        self.mask = mask_seg
        # Zeros
        # if line == 'OII':
        #     a_fit, da_fit, b_fit, db_fit = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)
        #     flux_fit, dflux_fit = np.zeros(size_3D), np.zeros(size_3D)
        #     r_fit, dr_fit = np.zeros(size_3D), np.zeros(size_3D)
        #     parameters.add('a', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        #     parameters.add('b', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        #
        #     for i in range(max_OII):
        #         parameters.add_many(('z_{}'.format(i + 1), redshift_guess, True, redshift_guess - 0.05,
        #                              redshift_guess + 0.05, None),
        #                             ('sigma_kms_{}'.format(i + 1), sigma_kms_guess, True, 50, 2000.0, None),
        #                             ('flux_OII_{}'.format(i + 1), flux_guess, True, 0, None, None))
        #         if fit_param['ResolveOII']:
        #             parameters.add('r_OII3729_3727_{}'.format(i + 1),
        #                            value=r_OII3729_3727_guess, vary=True, min=0.3, max=fit_param['r_max'])
        #
        # elif line == 'OIII':
        #     flux_fit, dflux_fit = np.zeros(size), np.zeros(size)
        #     a_fit, b_fit = np.zeros(size), np.zeros(size)
        #     da_fit, db_fit = np.zeros(size), np.zeros(size)
        #
        #     parameters.add_many(('z', redshift_guess, True, redshift_guess - 0.02, redshift_guess + 0.02, None),
        #                         ('sigma_kms', sigma_kms_guess, True, 50, 2000.0, None),
        #                         ('flux_OIII5008', flux_guess, True, 0, None, None),
        #                         ('a', 0.0, False, None, None, None),
        #                         ('b', 0.0, False, None, None, None))
        #
        # else:
        #     a_OII_fit, b_OII_fit = np.zeros(size), np.zeros(size)
        #     da_OII_fit, db_OII_fit = np.zeros(size), np.zeros(size)
        #     a_OIII_fit, b_OIII_fit = np.zeros(size), np.zeros(size)
        #     da_OIII_fit, db_OIII_fit = np.zeros(size), np.zeros(size)
        #     flux_OII_fit, dflux_OII_fit = np.zeros(size_3D), np.zeros(size_3D)
        #     flux_OIII_fit, dflux_OIII_fit = np.zeros(size_3D), np.zeros(size_3D)
        #     r_fit, dr_fit = np.zeros(size_3D), np.zeros(size_3D)
        #     parameters.add('a_OII', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        #     parameters.add('b_OII', value=-0.0025, vary=True, min=-0.1, max=0.1, expr=None, brute_step=None)
        #     parameters.add('a_OIII5008', value=0, vary=False, min=None, max=None, expr=None, brute_step=None)
        #     parameters.add('b_OIII5008', value=-0.0025, vary=True, min=-0.1, max=0.1, expr=None, brute_step=None)
        #
        #     # Model
        #     for i in range(max_OII):
        #         parameters.add_many(('z_{}'.format(i + 1), redshift_guess, True, redshift_guess - 0.002,
        #                              redshift_guess + 0.002, None),
        #                             ('sigma_kms_{}'.format(i + 1), sigma_kms_guess, True, 50, 2000.0, None),
        #                             ('flux_OII_{}'.format(i + 1), flux_guess, True, 0.0, None, None))
        #         if fit_param['ResolveOII']:
        #             parameters.add('r_OII3729_3727_{}'.format(i + 1), value=r_OII3729_3727_guess,
        #                            vary=True, min=0.3, max=fit_param['r_max'])
        #     for i in range(max_OIII):
        #         parameters.add_many(('flux_OIII5008_{}'.format(i + 1), flux_guess, True, 0.0, None, None))


        # Load the cube


        # Load the QSO field fit
        self.path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}_{}_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
        v_guess, sigma_guess, flux_guess = 0, 50, 0.5
        self.model = Gaussian
        self.parameters = lmfit.Parameters()
        self.parameters.add_many(('v', v_guess, True, -300, 300, None),
                                 ('sigma', sigma_guess, True, 5, 70, None),
                                 ('flux', flux_guess, True, 0, None, None))
        if os.path.exists(self.path_fit) is False:
            # if num_Gaussian == 1:
            #     12
            # elif num_Gaussian == 2:
            #
            # elif num_Gaussian == 3:
            #
            # else:

            print('Fitting result file does not exist, start fitting from scratch.')
            hdr_Serra = hdul_Serra[0].header
            hdr_Serra['NAXIS'] = 2
            hdr_Serra.remove('NAXIS3')
            hdr_Serra.remove('CTYPE3')
            hdr_Serra.remove('CDELT3')
            hdr_Serra.remove('CRPIX3')
            hdr_Serra.remove('CRVAL3')
            self.hdr = hdr_Serra
            self.fit()
        hdul_fit = fits.open(self.path_fit)
        self.pri, self.fs, self.v, self.z, self.dz, self.sigma, self.dsigma, self.flux_OII_fit, \
        self.dflux_OII_fit, self.flux_OIII_fit, self.dflux_OIII_fit, self.r, self.dr, self.a_OII, \
        self.da_OII, self.b_OII, self.db_OII, self.a_OIII, self.da_OIII, self.b_OIII, \
        self.db_OIII, self.chisqr, self.redchi = hdul_fit[:23]
        self.size = self.fs.data.shape
        self.v.data = self.v.data[0, :, :]
        self.sigma.data = self.sigma.data[0, :, :]


        # Calculae flux of each component
        self.flux_OIII_fit_array = Gaussian(self.wave_OIII_vac[:, np.newaxis, np.newaxis], self.z, self.sigma, flux, wave_line_vac
                                      self.size, self.model, self.v, self.sigma, self.flux_OIII_fit)



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
        # self.widget4_plot.setLimits(xMin=-1000, xMax=1000)
        # self.widget5_plot.setLimits(xMin=-1000, xMax=1000)

        # Set param
        self.paramSpec = [dict(name='Re-fit', type='action'),
                          dict(name='chi=', type='float', value=None, dec=False, readonly=False),
                          dict(name='v1=', type='float', value=None, dec=False, readonly=False),
                          dict(name='sigma1=', type='float', value=None, readonly=False),
                          dict(name='flux1=', type='float', value=None, readonly=False),
                          dict(name='v2=', type='float', value=None, dec=False, readonly=False),
                          dict(name='sigma2=', type='float', value=None, readonly=False),
                          dict(name='flux2=', type='float', value=None, readonly=False),
                          dict(name='v3=', type='float', value=None, dec=False, readonly=False),
                          dict(name='sigma3=', type='float', value=None, readonly=False),
                          dict(name='flux3=', type='float', value=None, readonly=False)]
        self.param = pt.Parameter.create(name='Options', type='group', children=self.paramSpec)
        self.tree = pt.ParameterTree()
        self.tree.setParameters(self.param)
        self.param.children()[0].sigStateChanged.connect(self.update_fit)

        #
        self.layout.addWidget(self.widget1, 0, 0, 1, 1)
        self.layout.addWidget(self.widget2, 0, 1, 1, 1)
        self.layout.addWidget(self.widget3, 0, 2, 1, 1)
        self.layout.addWidget(self.widget4, 1, 0, 1, 2)
        self.layout.addWidget(self.widget5, 1, 1, 1, 1)
        self.layout.addWidget(self.tree, 1, 2, 1, 1)


        # Plot the 2D map in the first plot
        self.v_map = pg.ImageItem()
        self.widget1_plot.addItem(self.v_map)
        colormap = cm.get_cmap("coolwarm")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.v_map.setLookupTable(lut)
        self.v_map.updateImage(image=self.v.data.T, levels=(-350, 350))

        # Plot the 2D map in the second plot
        self.sigma_map = pg.ImageItem()
        self.widget2_plot.addItem(self.sigma_map)
        colormap = Dense_20_r.mpl_colormap
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.sigma_map.setLookupTable(lut)
        self.sigma_map.updateImage(image=self.sigma.data.T, levels=(0, 300))

        # Plot the chi 2D map in the third plot
        self.chi_map = pg.ImageItem()
        self.widget3_plot.addItem(self.chi_map)
        colormap = cm.get_cmap("viridis")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.chi_map.setLookupTable(lut)
        self.chi_map.updateImage(image=self.chisqr.data.T, levels=(0, 50))

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
            # Clear plot
            self.widget4_plot.clear()

            # Get pixel coordinates
            pos = event.pos()
            # pos = self.widget1_plot.vb.mapSceneToView(pos)
            pos = self.v_map.mapFromScene(pos)
            # print(pos.x(), pos.y())
            self.xpixel, self.ypixel = int(np.floor(pos.x() + 1)), int(np.floor(pos.y()))
            self.plot_OII()
            self.plot_OIII()

    def plot_OII(self):
        i, j = self.ypixel, self.xpixel
        if self.mask[i, j]:
            # Plot new data
            self.widget4_plot.clear()

            # Plot the data
            self.widget4_plot.plot(self.wave_OII_vac, self.flux_OII[:, i, j], pen='k')
            self.widget4_plot.plot(self.wave_OII_vac, self.flux_err_OII[:, i, j], pen='g')
            # self.widget4_plot.plot(self.wave_OII_vac, self.flux_OII_fit, pen='r')

            # Plot each individual component

    def plot_OIII(self):
        i, j = self.ypixel, self.xpixel
        if self.mask[i, j]:
            # Plot new data
            self.widget5_plot.clear()

            # Plot the data and the fit
            self.widget5_plot.plot(self.wave_OIII_vac, self.flux_OIII[:, i, j], pen='k')
            self.widget5_plot.plot(self.wave_OIII_vac, self.flux_err_OIII[:, i, j], pen='g')
            # self.widget5_plot.plot(self.wave_OIII_vac, self.flux_OIII_fit, pen='r')

            # Plot each individual component

    def plot(self):
        i, j = self.ypixel, self.xpixel
        if self.mask[i, j]:
            # Plot new data
            self.widget4_plot.clear()
            self.param['v='] = '{:.0f}'.format(self.v_fit[i, j])
            self.param['sigma='] = '{:.0f}'.format(self.sigma_fit[i, j])
            self.param['flux='] = '{:.2f}'.format(self.flux_fit[i, j])
            self.param['chi='] = '{:.2f}'.format(self.chi_fit[i, j])

            # Plot spectrum
            scatter_1 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
            scatter_1.addPoints([j + 0.5], [i + 0.5])
            scatter_2 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
            scatter_2.addPoints([j + 0.5], [i + 0.5])
            scatter_3 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
            scatter_3.addPoints([j + 0.5], [i + 0.5])
            self.widget1_plot.addItem(scatter_1)
            self.widget2_plot.addItem(scatter_2)
            self.widget3_plot.addItem(scatter_3)

            # Plot spectrum
            self.widget4_plot.plot(self.v_array, self.flux[:, i, j], pen='k')
            self.widget4_plot.plot(self.v_array, self.flux_fit_array[:, i, j], pen='r')
            self.widget4_plot.plot(self.v_array, self.flux_err[:, 0, 0], pen='g')
            self.widget4_plot.setLabel('top', 'x={}, y={}'.format(i, j))
            self.widget4_plot.addItem(pg.InfiniteLine(self.v_fit[i, j],
                                                      pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine),
                                                      labelOpts={'position': 0.8, 'rotateAxis': [1, 0]}))
            self.widget4_plot.addItem(pg.InfiniteLine(self.v_fit[i, j] + self.sigma_fit[i, j],
                                                      pen=pg.mkPen('b', width=2, style=QtCore.Qt.DashLine),
                                                      labelOpts={'position': 0.8, 'rotateAxis': [1, 0]}))
            self.widget4_plot.addItem(pg.InfiniteLine(self.v_fit[i, j] - self.sigma_fit[i, j],
                                                      pen=pg.mkPen('b', width=2, style=QtCore.Qt.DashLine),
                                                      labelOpts={'position': 0.8, 'rotateAxis': [1, 0]}))

    def update_fit(self):
        i, j = self.ypixel, self.xpixel

        # Refite that specific pixel
        self.parameters['v'].value = self.param['v=']
        self.parameters['sigma'].value = self.param['sigma=']
        self.parameters['v'].max = self.param['v='] + 10
        self.parameters['v'].min = self.param['v='] - 10
        self.parameters['sigma'].max = self.param['sigma='] + 10
        self.parameters['sigma'].min = self.param['sigma='] - 10

        #
        flux_ij = self.flux[:, i, j]
        flux_err_ij = self.flux_err[:, 0, 0]
        spec_model = lmfit.Model(self.model, missing='drop')
        result = spec_model.fit(flux_ij, velocity=self.v_array, params=self.parameters, weights=1 / flux_err_ij)

        # fill the value
        hdul_fit = fits.open(self.path_fit)
        hdul_fit[0].data[i, j] = result.success
        hdul_fit[1].data[i, j], hdul_fit[2].data[i, j] = result.best_values['v'], result.params['v'].stderr
        hdul_fit[3].data[i, j], hdul_fit[4].data[i, j] = result.best_values['sigma'], result.params['sigma'].stderr
        hdul_fit[5].data[i, j], hdul_fit[6].data[i, j] = result.best_values['flux'], result.params['flux'].stderr

        # Re initiolize
        self.v_fit[i, j], self.sigma_fit[i, j], self.flux_fit[i, j] = result.best_values['v'], \
                                                                      result.best_values['sigma'], \
                                                                      result.best_values['flux']
        self.flux_fit_array[:, i, j] = Gaussian(self.v_array, self.v_fit[i, j],
                                                self.sigma_fit[i, j], self.flux_fit[i, j])

        chi2_ij = ((flux_ij - self.flux_fit_array[:, i, j]) / flux_err_ij) ** 2
        chi2_ij = np.where((self.v_array > self.v_fit[i, j] - 4 * self.sigma_fit[i, j])
                           * (self.v_array < self.v_fit[i, j] + 4 * self.sigma_fit[i, j]),
                           chi2_ij, np.nan)
        chi_fit_ij = np.nansum(chi2_ij, axis=0)
        self.chi_fit[i, j] = chi_fit_ij

        # Save fitting results
        hdul_fit.writeto(self.path_fit, overwrite=True)

        # Replot
        self.plot()
        self.v_map.updateImage(image=self.v_fit.T)
        self.sigma_map.updateImage(image=self.sigma_fit.T)
        self.chi_map.updateImage(image=self.chi_fit.T)

    def update_fit_G2(self):
        print('in progress')





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotWindow(cubename='3C57')
    window.show()
    sys.exit(app.exec_())
