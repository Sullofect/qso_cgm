#!/usr/bin/env python
import os
import sys
import lmfit
import warnings
import argparse
import numpy as np
import pyqtgraph as pg
import astropy.io.fits as fits
import pyqtgraph.parametertree as pt
from astropy.io import ascii
from matplotlib import rc, cm
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from palettable.cmocean.sequential import Dense_20_r
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QHBoxLayout, QWidget, QPushButton, QVBoxLayout
from scipy import integrate, interpolate
warnings.filterwarnings("ignore")
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

# Set up the parser
parser = argparse.ArgumentParser(description='2D kinematics fitting Gui')
parser.add_argument('-m', metavar='cubename', help='MUSE cube name (without .fits), required', required=True, type=str)
parser.add_argument('-seg', metavar='segmentation file', help='which set of segmentation map', required=False,
                    type=str, default='(1.5, "gauss", 1.5, "gauss")')
args = parser.parse_args()  # parse the arguments

# QSO table
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'

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
        wave_OII_vac = np.array(wave_vac[0], dtype=float)
        wave_OIII_vac = np.array(wave_vac[1], dtype=float)
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
        return m_OII + params['a_OII'] * wave_vac + params['b_OII']
    elif params['OII'] == 0:
        return m_OIII5008 + params['a_OIII5008'] * wave_vac + params['b_OIII5008']
    else:
        return np.hstack((m_OII + params['a_OII'] * wave_OII_vac + params['b_OII'],
                          m_OIII5008 + params['a_OIII5008'] * wave_OIII_vac + params['b_OIII5008']))

class PlotWindow(QMainWindow):
    def __init__(self, cubename='3C57', zapped=False, NLR='', UseDataSeg=(1.5, 'gauss', None, None),
                 width_OII=50, width_OIII=50, UseSmoothedCubes=True, extend_over=False,
                 UseDetectionSeg=(1.5, 'gauss', 1.5, 'gauss')):
        super().__init__()

        # Define parameters
        fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': True, 'r_max': 1.5,
                     'OII_center': wave_OII3728_vac, "OIII": 1, "OIII_2nd": 0}

        # if zapped
        if cubename == 'TEX0206-048':
            zapped = True

        if zapped:
            str_zap = '_zapped'
        else:
            str_zap = ''

        # Load qso information
        data_qso = ascii.read(path_qso, format='fixed_width')
        data_qso = data_qso[data_qso['name'] == cubename]
        self.ra_qso, self.dec_qso, self.z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

        # Check OIII coverage
        if (1 + self.z_qso) * wave_OIII5008_vac >= 9350:
            print('OIII is not covered')
            fit_param['OIII'] = 0
        else:
            print('OIII coverage is covered')

        # Define lines
        if fit_param['OII'] >= 1 and fit_param['OIII'] == 0:
            line = 'OII'
        elif fit_param['OII'] == 0 and fit_param['OIII'] >= 1:
            line = 'OIII'
        else:
            line = 'OII+OIII'
        self.line = line

        # Save V50 and W80
        self.path_v50_OII = '../../MUSEQuBES+CUBS/fit_kin/{}{}_V50_OII.fits'.format(cubename, NLR)
        self.path_w80_OII = '../../MUSEQuBES+CUBS/fit_kin/{}{}_W80_OII.fits'.format(cubename, NLR)
        self.path_v50_OIII = '../../MUSEQuBES+CUBS/fit_kin/{}{}_V50_OIII.fits'.format(cubename, NLR)
        self.path_w80_OIII = '../../MUSEQuBES+CUBS/fit_kin/{}{}_W80_OIII.fits'.format(cubename, NLR)
        self.path_v50 = '../../MUSEQuBES+CUBS/fit_kin/{}{}_V50.fits'.format(cubename, NLR)
        self.path_w80 = '../../MUSEQuBES+CUBS/fit_kin/{}{}_W80.fits'.format(cubename, NLR)

        # Load cubes
        if line == 'OII+OIII':
            line_OII, line_OIII = 'OII', 'OIII'
            path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}.fits'. \
                format(cubename, str_zap, line_OII, NLR)
            path_cube_smoothed_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}_{}_' \
                                     '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, NLR, *UseDataSeg)
            path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}.fits'. \
                format(cubename, str_zap, line_OIII, NLR)
            path_cube_smoothed_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}_{}_' \
                                      '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, NLR, *UseDataSeg)
            if UseDetectionSeg is not None:
                path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line_OII, NLR, *UseDetectionSeg)
                path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line_OIII, NLR, *UseDetectionSeg)
            else:
                path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line_OII, NLR, *UseDataSeg)
                path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line_OIII, NLR, *UseDataSeg)
            self.path_cube_hdr = path_cube_OII

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
            # seg_3D_ori = np.vstack((seg_3D_OII_ori, seg_3D_OIII_ori))
            mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
            flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori
            flux_err_seg_OII, flux_err_seg_OIII = flux_err_OII * seg_3D_OII_ori, flux_err_OIII * seg_3D_OIII_ori
            S_N_OII = np.sum(flux_seg_OII / flux_err_seg_OII, axis=0).filled(np.nan)
            S_N_OIII = np.sum(flux_seg_OIII / flux_err_seg_OIII, axis=0).filled(np.nan)
            self.S_N = np.sqrt(np.nan_to_num(S_N_OII, nan=0.0)**2 + np.nan_to_num(S_N_OIII, nan=0.0)**2)

            # Extend over
            if extend_over:
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


            # Choose a S/N cut
            mask_seg = mask_seg_OII + mask_seg_OIII
            self.wave_vac = np.array([wave_OII_vac, wave_OIII_vac], dtype=object)
            self.flux = np.vstack((flux_OII, flux_OIII))
            self.flux_err = np.vstack((flux_err_OII, flux_err_OIII))
            # self.flux_err = np.where(self.flux_err != 0, self.flux_err, np.inf)

            #
            # flux = np.where(flux_err != 0, flux, np.nan)
            # flux_err = np.where(flux_err != 0, flux_err, np.nan)

            wave_OII_min, wave_OII_max = np.min(wave_OII_vac), np.max(wave_OII_vac)
            wave_OIII_min, wave_OIII_max = np.min(wave_OIII_vac), np.max(wave_OIII_vac)
            z_lb, z_ub = np.max([wave_OII_min / wave_OII3728_vac - 1, wave_OIII_min / wave_OIII5008_vac - 1]), \
                         np.min([wave_OII_max / wave_OII3728_vac - 1, wave_OIII_max / wave_OIII5008_vac - 1])
        else:
            path_cube = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
                format(cubename, str_zap, line)
            path_cube_smoothed = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_' \
                                 '{}_{}.fits'.format(cubename, str_zap, line, *UseDataSeg)
            if UseDetectionSeg is not None:
                path_3Dseg = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line, *UseDetectionSeg)
            else:
                path_3Dseg = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
                    format(cubename, str_zap, line, *UseDataSeg)
            self.path_cube_hdr = path_cube

            # Load data and smoothing
            if UseSmoothedCubes:
                cube = Cube(path_cube_smoothed)
            else:
                cube = Cube(path_cube)
            self.wave_vac = pyasl.airtovac2(cube.wave.coord())
            flux, flux_err = cube.data * 1e-3, np.sqrt(cube.var) * 1e-3
            seg_3D_ori = fits.open(path_3Dseg)[0].data
            mask_seg = np.sum(seg_3D_ori, axis=0)
            flux_seg, flux_err_seg = flux * seg_3D_ori, flux_err * seg_3D_ori
            self.S_N = np.sum(flux_seg / flux_err_seg, axis=0)

            if extend_over:
                width = width_OII if line == 'OII' else width_OIII
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

            self.flux, self.flux_err = flux, flux_err

            if line == 'OII':
                mask_seg_OII = mask_seg
                mask_seg_OIII = np.zeros_like(mask_seg)
                self.wave_OII_vac, self.wave_OIII_vac = self.wave_vac, np.zeros_like(self.wave_vac)
                self.flux_OII, self.flux_OIII = flux, np.zeros_like(flux)
                self.flux_err_OII, self.flux_err_OIII = flux_err, np.zeros_like(flux_err)

                wave_OII_min, wave_OII_max = np.min(self.wave_OII_vac), np.max(self.wave_OII_vac)
                z_lb, z_ub = wave_OII_min / wave_OII3728_vac - 1, wave_OII_max / wave_OII3728_vac - 1

            elif line == 'OIII':
                mask_seg_OII = np.zeros_like(mask_seg)
                mask_seg_OIII = mask_seg
                self.wave_OII_vac, self.wave_OIII_vac = np.zeros_like(self.wave_vac), self.wave_vac
                self.flux_OII, self.flux_OIII = np.zeros_like(flux), flux
                self.flux_err_OII, self.flux_err_OIII = np.zeros_like(flux_err), flux_err,

                wave_OIII_min, wave_OIII_max = np.min(self.wave_OIII_vac), np.max(self.wave_OIII_vac)
                z_lb, z_ub = wave_OIII_min / wave_OIII5008_vac - 1, wave_OIII_max / wave_OIII5008_vac - 1



        # Mask
        self.mask_OII, self.mask_OII_ori = mask_seg_OII, mask_seg_OII
        self.mask_OIII, self.mask_OIII_ori = mask_seg_OIII, mask_seg_OIII
        self.mask, self.mask_ori = mask_seg, mask_seg
        self.size = self.mask.shape


        # Load the QSO field fit
        self.path_fit = '../../MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}{}_{}_{}_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line, NLR, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)
        redshift_guess, sigma_kms_guess, flux_guess, r_OII3729_3727_guess = self.z_qso, 200.0, 1.0, 1.0
        self.spec_model = lmfit.Model(model_OII_OIII, missing='drop')
        self.parameters = lmfit.Parameters()
        self.parameters.add_many(('z_1', redshift_guess, True, z_lb, z_ub, None),
                                 ('z_2', redshift_guess, True, z_lb, z_ub, None),
                                 ('z_3', redshift_guess, True, z_lb, z_ub, None),
                                 ('sigma_kms_1', sigma_kms_guess, True, 50, 2000.0, None),
                                 ('sigma_kms_2', sigma_kms_guess, True, 50, 2000.0, None),
                                 ('sigma_kms_3', sigma_kms_guess, True, 50, 2000.0, None),
                                 ('flux_OII_1', flux_guess, True, 0.0, None, None),
                                 ('flux_OII_2', flux_guess, True, 0.0, None, None),
                                 ('flux_OII_3', flux_guess, True, 0.0, None, None),
                                 ('r_OII3729_3727_1', r_OII3729_3727_guess, True, 0.35, fit_param['r_max'], None),
                                 ('r_OII3729_3727_2', r_OII3729_3727_guess, True, 0.35, fit_param['r_max'], None),
                                 ('r_OII3729_3727_3', r_OII3729_3727_guess, True, 0.35, fit_param['r_max'], None),
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
            self.fit()
        self.hdul_fit = fits.open(self.path_fit)
        pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, \
        da_OII, b_OII, db_OII, a_OIII, da_OIII, b_OIII, db_OIII, chisqr, redchi = self.hdul_fit[:23]
        pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, \
        da_OII, b_OII, db_OII, a_OIII, da_OIII, b_OIII, db_OIII, \
        chisqr, redchi = pri.data, fs.data, v.data, z.data, dz.data, sigma.data, dsigma.data, \
                         flux_OII_fit.data, dflux_OII_fit.data, flux_OIII_fit.data, dflux_OIII_fit.data, r.data, \
                         dr.data, a_OII.data, da_OII.data, b_OII.data, db_OII.data, a_OIII.data, da_OIII.data, \
                         b_OIII.data, db_OIII.data, chisqr.data, redchi.data

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
        self.redchi_show = redchi
        self.num_comp, self.num_comp_show = np.sum(~np.isnan(self.v), axis=0), np.sum(~np.isnan(self.v), axis=0)

        # Calculae flux of each component
        self.wave_OII_exp = expand_wave([self.wave_OII_vac], stack=True)
        self.wave_OIII_exp = expand_wave([self.wave_OIII_vac], stack=True)
        self.flux_OII_array = model_OII(self.wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], self.z, self.sigma,
                                        self.flux_OII_fit, self.r, plot=False)
        self.flux_OIII_array = Gaussian(self.wave_OIII_exp[:, np.newaxis, np.newaxis, np.newaxis], self.z, self.sigma,
                                        self.flux_OIII_fit, wave_OIII5008_vac)

        # Calculate V_50 and W_80
        flux_OII_v50 = model_OII(self.wave_OII_exp[:, np.newaxis, np.newaxis, np.newaxis], self.z, self.sigma,
                                 self.flux_OII_fit, self.r, plot=True)[0] * (1 + self.r)
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
        v50_OII = c_kms * (z50_OII - self.z_qso) / (1 + self.z_qso)
        w80_OII = c_kms * (wave_90_OII - wave_10_OII) / (wave_OII3727_vac * (1 + z50_OII))
        z50_OIII = (wave_50_OIII - wave_OIII5008_vac) / wave_OIII5008_vac
        v50_OIII = c_kms * (z50_OIII - self.z_qso) / (1 + self.z_qso)
        w80_OIII = c_kms * (wave_90_OIII - wave_10_OIII) / (wave_OIII5008_vac * (1 + z50_OIII))

        z50 = np.where(self.mask_OIII != 0, z50_OIII, z50_OII)
        v50 = c_kms * (z50 - self.z_qso) / (1 + self.z_qso)
        w80 = np.where(self.mask_OIII > 0, w80_OIII, w80_OII)
        v50 = np.where(self.mask != 0, v50, np.nan)
        w80 = np.where(self.mask != 0, w80, np.nan)
        self.redchi_show = np.where(self.mask != 0, self.redchi_show, np.nan)
        self.num_comp_show = np.where(self.mask != 0, self.num_comp_show, np.nan)
        self.v50_OII = np.where(self.mask_OII, v50_OII, np.nan)
        self.w80_OII = np.where(self.mask_OII, w80_OII, np.nan)
        self.v50_OIII = np.where(self.mask_OIII, v50_OIII, np.nan)
        self.w80_OIII = np.where(self.mask_OIII, w80_OIII, np.nan)
        self.v50_ori = v50
        self.w80_ori = w80
        self.v50 = v50
        self.w80 = w80

        # Define a top-level widget
        self.widget = QWidget()
        self.widget.resize(2000, 2000)
        self.setCentralWidget(self.widget)
        self.layout = QGridLayout()
        self.widget.setLayout(self.layout)

        # Set title
        self.setWindowTitle('{}'.format(cubename))

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
        self.paramSpec = [dict(name='Select 3rd panel', type='list', values=['chi^2', 'N_comp'], value='chi^2'),
                          dict(name='S_N=', type='float', value=15, dec=False, readonly=False),
                          dict(name='Fitting ranges', type='group', children=[
                              dict(name='OII', type='group', children=[
                                  dict(name='enabled', type='bool', value=False),
                                  dict(name='min', type='float', value=self.wave_OII_vac.min(), step=1.0),
                                  dict(name='max', type='float', value=self.wave_OII_vac.max(), step=1.0)]),
                              dict(name='OIII', type='group', children=[
                                  dict(name='enabled', type='bool', value=False),
                                  dict(name='min', type='float', value=self.wave_OIII_vac.min(), step=1.0),
                                  dict(name='max', type='float', value=self.wave_OIII_vac.max(), step=1.0)]),
                          ]),
                          dict(name='AIC, BIC=', type='str', value='', readonly=False),
                          dict(name='chi=', type='float', value=0, dec=False, readonly=False),
                          dict(name='v_1=', type='float', value=0, dec=False, readonly=False),
                          dict(name='sigma_1=', type='float', value=0, readonly=False),
                          dict(name='OII_1=', type='float', value=0, readonly=False),
                          dict(name='OIII_1=', type='float', value=0, readonly=False),
                          dict(name='r_1=', type='float', value=0, readonly=False),
                          dict(name='v_2=', type='float', value=0, dec=False, readonly=False),
                          dict(name='sigma_2=', type='float', value=0, readonly=False),
                          dict(name='OII_2=', type='float', value=0, readonly=False),
                          dict(name='OIII_2=', type='float', value=0, readonly=False),
                          dict(name='r_2=', type='float', value=0, readonly=False),
                          dict(name='v_3=', type='float', value=0, dec=False, readonly=False),
                          dict(name='sigma_3=', type='float', value=0, readonly=False),
                          dict(name='OII_3=', type='float', value=0, readonly=False),
                          dict(name='OIII_3=', type='float', value=0, readonly=False),
                          dict(name='r_3=', type='float', value=0, readonly=False)]
        self.param = pt.Parameter.create(name='Options', type='group', children=self.paramSpec)
        self.tree = pt.ParameterTree()
        self.tree.setParameters(self.param)
        self.param.child('Select 3rd panel').sigValueChanged.connect(self.plot_3rd_panel)

        # Buttons
        self.state = 0
        btn_11 = QPushButton("Refit")
        btn_12 = QPushButton("Fit region")
        btn_13 = QPushButton("only OII")
        btn_14 = QPushButton("only OIII")
        btn_21 = QPushButton("1 Gauss")
        btn_22 = QPushButton("2 Gauss")
        btn_23 = QPushButton("3 Gauss")
        btn_24 = QPushButton("Fix ratio")
        btn_31 = QPushButton("Save v50w80")
        btn_32 = QPushButton("Clear")
        btn_33 = QPushButton("Remask")
        btn_11.clicked.connect(self.update_fit)
        btn_12.clicked.connect(self.draw_region)
        btn_13.clicked.connect(self.only_OII)
        btn_14.clicked.connect(self.only_OIII)
        btn_21.clicked.connect(self.N_1)
        btn_22.clicked.connect(self.N_2)
        btn_23.clicked.connect(self.N_3)
        btn_24.clicked.connect(self.toggle_OII_ratio)
        btn_31.clicked.connect(self.save_v50w80)
        btn_32.clicked.connect(self.clear_scatter)
        btn_33.clicked.connect(self.re_mask)

        layout_RHS = QVBoxLayout()
        layout_btn = QVBoxLayout()
        layout_btn_r1 = QHBoxLayout()
        layout_btn_r2 = QHBoxLayout()
        layout_btn_r3 = QHBoxLayout()
        layout_btn_r1.addWidget(btn_11)
        layout_btn_r1.addWidget(btn_12)
        layout_btn_r1.addWidget(btn_13)
        layout_btn_r1.addWidget(btn_14)
        layout_btn_r2.addWidget(btn_21)
        layout_btn_r2.addWidget(btn_22)
        layout_btn_r2.addWidget(btn_23)
        layout_btn_r2.addWidget(btn_24)
        layout_btn_r3.addWidget(btn_31)
        layout_btn_r3.addWidget(btn_32)
        layout_btn_r3.addWidget(btn_33)
        layout_btn.addLayout(layout_btn_r1)
        layout_btn.addLayout(layout_btn_r2)
        layout_btn.addLayout(layout_btn_r3)
        layout_RHS.addLayout(layout_btn)
        layout_RHS.addWidget(self.tree)

        #
        self.layout.addWidget(self.widget1, 0, 0, 1, 1)
        self.layout.addWidget(self.widget2, 0, 1, 1, 1)
        self.layout.addWidget(self.widget3, 0, 2, 1, 1)
        self.layout.addWidget(self.widget4, 1, 0, 1, 1)
        self.layout.addWidget(self.widget5, 1, 1, 1, 1)
        self.layout.addLayout(layout_RHS, 1, 2, 1, 1)

        # Plot the 2D map in the first plot
        self.v_map = pg.ImageItem()
        self.widget1_plot.addItem(self.v_map)
        colormap = cm.get_cmap("coolwarm")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.v_map.setLookupTable(lut)
        bar = pg.ColorBarItem(values=(-350, 350), colorMap=pg.ColorMap(np.linspace(0, 1, len(lut)), lut),
                              orientation='horizontal', limits=(-1500, 1500))
        self.v_map.updateImage(image=self.v50.T)
        bar.setImageItem(self.v_map, insert_in=self.widget1_plot)

        # Plot the 2D map in the second plot
        self.sigma_map = pg.ImageItem()
        self.widget2_plot.addItem(self.sigma_map)
        colormap = Dense_20_r.mpl_colormap
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.sigma_map.setLookupTable(lut)
        bar = pg.ColorBarItem(values=(0, 1500), colorMap=pg.ColorMap(np.linspace(0, 1, len(lut)), lut),
                              orientation='horizontal', limits=(0, 3000))
        self.sigma_map.updateImage(image=self.w80.T)
        bar.setImageItem(self.sigma_map, insert_in=self.widget2_plot)

        # Plot the chi 2D map in the third plot
        self.chi_map = pg.ImageItem()
        self.widget3_plot.addItem(self.chi_map)
        colormap = cm.get_cmap("viridis")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.chi_map.setLookupTable(lut)
        bar = pg.ColorBarItem(values=(0, 2), colorMap=pg.ColorMap(np.linspace(0, 1, len(lut)), lut),
                              orientation='horizontal', limits=(0, 5))
        self.chi_map.updateImage(image=self.redchi_show.T)
        bar.setImageItem(self.chi_map, insert_in=self.widget3_plot)

        # Mark the locations
        self.scatter_1 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
        self.scatter_2 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
        self.scatter_3 = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))
        self.scatter_1_mask = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 35, 255))
        self.scatter_2_mask = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 35, 255))
        self.scatter_3_mask = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 35, 255))

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

        # Initiate region fit
        self.current_roi = None
        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None

        # Default to S_N = 15 and gaussian componenet = 1
        self.re_mask()
        self.N_1()

    def calculate_iniguess(self):
        # Moments
        wave_OII_vac_ = self.wave_OII_vac[:, np.newaxis, np.newaxis]
        wave_OIII_vac_ = self.wave_OIII_vac[:, np.newaxis, np.newaxis]
        flux_cumsum_OII = integrate.cumtrapz(self.flux_OII, wave_OII_vac_, initial=0, axis=0)
        flux_cumsum_OIII = integrate.cumtrapz(self.flux_OIII, wave_OIII_vac_, initial=0, axis=0)
        flux_cumsum_OII /= flux_cumsum_OII.max(axis=0)
        flux_cumsum_OIII /= flux_cumsum_OIII.max(axis=0)
        wave_array_OII = np.zeros_like(self.flux_OII)
        wave_array_OII[:] = self.wave_OII_vac[:, np.newaxis, np.newaxis]
        wave_array_OIII = np.zeros_like(self.flux_OIII)
        wave_array_OIII[:] = self.wave_OIII_vac[:, np.newaxis, np.newaxis]

        wave_10_OII = \
            np.take_along_axis(wave_array_OII, np.argmin(np.abs(flux_cumsum_OII - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50_OII = \
            np.take_along_axis(wave_array_OII, np.argmin(np.abs(flux_cumsum_OII - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90_OII = \
            np.take_along_axis(wave_array_OII, np.argmin(np.abs(flux_cumsum_OII - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array_OII = (wave_50_OII - wave_OII3728_vac) / wave_OII3728_vac
        sigma_kms_guess_array_OII = c_kms * (wave_90_OII - wave_10_OII) / (wave_OII3728_vac * (1 + z_guess_array_OII))
        sigma_kms_guess_array_OII /= 2.563  # W_80 = 2.563sigma

        # Moments for OIII
        wave_10_OIII = \
            np.take_along_axis(wave_array_OIII, np.argmin(np.abs(flux_cumsum_OIII - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50_OIII = \
            np.take_along_axis(wave_array_OIII, np.argmin(np.abs(flux_cumsum_OIII - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90_OIII = \
            np.take_along_axis(wave_array_OIII, np.argmin(np.abs(flux_cumsum_OIII - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array_OIII = (wave_50_OIII - wave_OIII5008_vac) / wave_OIII5008_vac
        sigma_kms_guess_array_OIII = c_kms * (wave_90_OIII - wave_10_OIII) / (wave_OIII5008_vac * (1 + z_guess_array_OIII))
        sigma_kms_guess_array_OIII /= 2.563  # W_80 = 2.563sigma

        # Use [O III] if possible
        z_guess_array = np.where(self.mask_OIII != 0, z_guess_array_OIII, z_guess_array_OII)
        sigma_kms_guess_array = np.where(self.mask_OIII != 0, sigma_kms_guess_array_OIII, sigma_kms_guess_array_OII)

        return z_guess_array, sigma_kms_guess_array

    def calculate_iniguess_OII(self):
        # Moments for only OII or OIII
        wave_OII_vac_ = self.wave_vac[:, np.newaxis, np.newaxis]
        flux_cumsum_OII = integrate.cumtrapz(self.flux, wave_OII_vac_, initial=0, axis=0)
        flux_cumsum_OII /= flux_cumsum_OII.max(axis=0)
        wave_array_OII = np.zeros_like(self.flux)
        wave_array_OII[:] = self.wave_vac[:, np.newaxis, np.newaxis]

        wave_10_OII = \
            np.take_along_axis(wave_array_OII, np.argmin(np.abs(flux_cumsum_OII - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_50_OII = \
            np.take_along_axis(wave_array_OII, np.argmin(np.abs(flux_cumsum_OII - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
        wave_90_OII = \
            np.take_along_axis(wave_array_OII, np.argmin(np.abs(flux_cumsum_OII - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
        z_guess_array_OII = (wave_50_OII - wave_OII3728_vac) / wave_OII3728_vac
        sigma_kms_guess_array_OII = c_kms * (wave_90_OII - wave_10_OII) / (wave_OII3728_vac * (1 + z_guess_array_OII))
        sigma_kms_guess_array_OII /= 2.563  # W_80 = 2.563sigma

        return z_guess_array_OII, sigma_kms_guess_array_OII

    def fit(self):
        # Make inital condition
        self.N_1()

        header = fits.open(self.path_cube_hdr)[1].header
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

        # fitting starts
        fs, v = np.full(self.size, np.nan), np.full((3, *self.size), np.nan)
        z, dz, \
        sigma, dsigma = np.full((3, *self.size), np.nan), np.full((3, *self.size), np.nan), \
                        np.full((3, *self.size), np.nan), np.full((3, *self.size), np.nan)
        flux_OII_fit, dflux_OII_fit, \
        flux_OIII_fit, dflux_OIII_fit, \
        r, dr = np.full((3, *self.size), np.nan), np.full((3, *self.size), np.nan), \
                np.full((3, *self.size), np.nan), np.full((3, *self.size), np.nan), \
                np.full((3, *self.size), np.nan), np.full((3, *self.size), np.nan)
        a_OII_fit, da_OII_fit, b_OII_fit, \
        db_OII_fit, a_OIII_fit, da_OIII_fit, \
        b_OIII_fit, db_OIII_fit, \
        chisqr, redchi = np.full(self.size, np.nan), np.full(self.size, np.nan), np.full(self.size, np.nan), \
                         np.full(self.size, np.nan), np.full(self.size, np.nan), np.full(self.size, np.nan), \
                         np.full(self.size, np.nan), np.full(self.size, np.nan), \
                         np.full(self.size, np.nan), np.full(self.size, np.nan)

        z_guess_array, sigma_kms_guess_array = self.calculate_iniguess()
        print('Fitting starts')
        for i in range(self.size[0]):  # i = p (y), j = q (x)
            for j in range(self.size[1]):
                if self.mask[i, j] != 0:
                    self.parameters['z_1'].value = z_guess_array[i, j]

                    flux_ij = self.flux[:, i, j]
                    flux_err_ij = self.flux_err[:, i, j]
                    # spec_model = lmfit.Model(self.model, missing='drop')
                    result = self.spec_model.fit(flux_ij, wave_vac=self.wave_vac, params=self.parameters,
                                            weights=1 / flux_err_ij)
                    fs[i, j] = result.success
                    chisqr[i, j], redchi[i, j] = result.chisqr, result.redchi

                    # fill the value
                    a_OII, b_OII = result.best_values['a_OII'], result.best_values['b_OII']
                    da_OII, db_OII = result.params['a_OII'].stderr, result.params['b_OII'].stderr
                    a_OIII, b_OIII = result.best_values['a_OIII5008'], result.best_values['b_OIII5008']
                    da_OIII, db_OIII = result.params['a_OIII5008'].stderr, result.params['b_OIII5008'].stderr
                    a_OII_fit[i, j], da_OII_fit[i, j], b_OII_fit[i, j], db_OII_fit[i, j] = a_OII, da_OII, b_OII, db_OII
                    a_OIII_fit[i, j], da_OIII_fit[i, j] = a_OIII, da_OIII
                    b_OIII_fit[i, j], db_OIII_fit[i, j] = b_OIII, db_OIII

                    z_1, dz_1 = result.best_values['z_1'], result.params['z_1'].stderr
                    z_2, dz_2 = result.best_values['z_2'], result.params['z_2'].stderr
                    z_3, dz_3 = result.best_values['z_3'], result.params['z_3'].stderr
                    sigma_1, dsigma_1 = result.best_values['sigma_kms_1'], result.params['sigma_kms_1'].stderr
                    sigma_2, dsigma_2 = result.best_values['sigma_kms_2'], result.params['sigma_kms_2'].stderr
                    sigma_3, dsigma_3 = result.best_values['sigma_kms_3'], result.params['sigma_kms_3'].stderr
                    z[:, i, j], sigma[:, i, j] = [z_1, z_2, z_3], [sigma_1, sigma_2, sigma_3]
                    dz[:, i, j], dsigma[:, i, j] = [dz_1, dz_2, dz_3], [dsigma_1, dsigma_2, dsigma_3]
                    v[:, i, j] = [c_kms * (z_1 - self.z_qso) / (1 + self.z_qso),
                                  c_kms * (z_2 - self.z_qso) / (1 + self.z_qso),
                                  c_kms * (z_3 - self.z_qso) / (1 + self.z_qso)]

                    flux_OII_1, dflux_OII_1 = result.best_values['flux_OII_1'], result.params['flux_OII_1'].stderr
                    flux_OII_2, dflux_OII_2 = result.best_values['flux_OII_2'], result.params['flux_OII_2'].stderr
                    flux_OII_3, dflux_OII_3 = result.best_values['flux_OII_3'], result.params['flux_OII_3'].stderr
                    r_1, dr_1 = result.best_values['r_OII3729_3727_1'], result.params['r_OII3729_3727_1'].stderr
                    r_2, dr_2 = result.best_values['r_OII3729_3727_2'], result.params['r_OII3729_3727_2'].stderr
                    r_3, dr_3 = result.best_values['r_OII3729_3727_3'], result.params['r_OII3729_3727_3'].stderr
                    flux_OII_fit[:, i, j] = [flux_OII_1, flux_OII_2, flux_OII_3]
                    dflux_OII_fit[:, i, j] = [dflux_OII_1, dflux_OII_2, dflux_OII_3]
                    r[:, i, j] = [r_1, r_2, r_3]
                    dr[:, i, j] = [dr_1, dr_2, dr_3]

                    flux_OIII_1, dflux_OIII_1 = result.best_values['flux_OIII5008_1'], result.params[
                        'flux_OIII5008_1'].stderr
                    flux_OIII_2, dflux_OIII_2 = result.best_values['flux_OIII5008_2'], result.params[
                        'flux_OIII5008_2'].stderr
                    flux_OIII_3, dflux_OIII_3 = result.best_values['flux_OIII5008_3'], result.params[
                        'flux_OIII5008_3'].stderr
                    flux_OIII_fit[:, i, j] = [flux_OIII_1, flux_OIII_2, flux_OIII_3]
                    dflux_OIII_fit[:, i, j] = [dflux_OIII_1, dflux_OIII_2, dflux_OIII_3]
                else:
                    pass

        # Save results
        hdul_pri = fits.PrimaryHDU()
        hdul_fs, hdul_v = fits.ImageHDU(fs, header=header), fits.ImageHDU(v, header=header)
        hdul_z, hdul_dz = fits.ImageHDU(z, header=header), fits.ImageHDU(dz, header=header)
        hdul_sigma, hdul_dsigma = fits.ImageHDU(sigma, header=header), fits.ImageHDU(dsigma, header=header)
        hdul_chisqr, hudl_redchi = fits.ImageHDU(chisqr, header=header), fits.ImageHDU(redchi, header=header)
        hdul_flux_OII, hdul_dflux_OII = fits.ImageHDU(flux_OII_fit, header=header), \
                                        fits.ImageHDU(dflux_OII_fit, header=header)
        hdul_flux_OIII, hdul_dflux_OIII = fits.ImageHDU(flux_OIII_fit, header=header), \
                                          fits.ImageHDU(dflux_OIII_fit, header=header)
        hdul_r, hdul_dr = fits.ImageHDU(r, header=header), fits.ImageHDU(dr, header=header)
        hdul_a_OII, hdul_da_OII = fits.ImageHDU(a_OII_fit, header=header), fits.ImageHDU(da_OII_fit, header=header)
        hdul_b_OII, hdul_db_OII = fits.ImageHDU(b_OII_fit, header=header), fits.ImageHDU(db_OII_fit, header=header)
        hdul_a_OIII, hdul_da_OIII = fits.ImageHDU(a_OIII_fit, header=header), fits.ImageHDU(da_OIII_fit, header=header)
        hdul_b_OIII, hdul_db_OIII = fits.ImageHDU(b_OIII_fit, header=header), fits.ImageHDU(db_OIII_fit, header=header)
        hdul = fits.HDUList([hdul_pri, hdul_fs, hdul_v, hdul_z, hdul_dz, hdul_sigma, hdul_dsigma, hdul_flux_OII,
                             hdul_dflux_OII, hdul_flux_OIII, hdul_dflux_OIII, hdul_r, hdul_dr, hdul_a_OII,
                             hdul_da_OII, hdul_b_OII, hdul_db_OII, hdul_a_OIII, hdul_da_OIII, hdul_b_OIII,
                             hdul_db_OIII, hdul_chisqr, hudl_redchi])
        hdul.writeto(self.path_fit, overwrite=True)

    def draw_region(self):
        # Initially, ROI is inactive
        if self.current_roi is not None:
            self.widget1_plot.removeItem(self.current_roi)
        self.current_roi = pg.RectROI([75, 75], [10, 10], pen='r')
        self.widget1_plot.addItem(self.current_roi)

        # Connect ROI updated signal to the function
        self.current_roi.sigRegionChanged.connect(self.get_region)
        self.widget1_plot.keyPressEvent = self.fit_region

    def get_region(self):
        # Get the bounding box of the ROI
        pos = self.current_roi.pos()  # (x, y) of the top-left corner
        size = self.current_roi.size()  # (width, height)

        # Calculate the coordinates of the selected region
        self.x1, self.y1 = int(pos[0]) + 1, int(pos[1]) + 1
        self.x2, self.y2 = int(pos[0] + size[0]), int(pos[1] + size[1])

    def fit_region(self, event):
        print(f"Selected Region: ({self.x1}, {self.y1}) to ({self.x2}, {self.y2})")
        if event.text() == 'f':
            for i in range(self.y1, self.y2):
                for j in range(self.x1, self.x2):
                    if self.mask[i, j] != 0:
                        self.ypixel, self.xpixel = i, j
                        self.update_fit()

    def update_plot(self, event):
        if event.double():
            # Get pixel coordinates
            pos = event.pos()
            pos = self.v_map.mapFromScene(pos)
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
            self.param['r_1='] = '{:.4f}'.format(self.r[0, i, j])
            self.param['v_2='] = '{:.0f}'.format(self.v[1, i, j])
            self.param['sigma_2='] = '{:.0f}'.format(self.sigma[1, i, j])
            self.param['OII_2='] = '{:.4f}'.format(self.flux_OII_fit[1, i, j])
            self.param['OIII_2='] = '{:.4f}'.format(self.flux_OIII_fit[1, i, j])
            self.param['r_2='] = '{:.4f}'.format(self.r[1, i, j])
            self.param['v_3='] = '{:.0f}'.format(self.v[2, i, j])
            self.param['sigma_3='] = '{:.0f}'.format(self.sigma[2, i, j])
            self.param['OII_3='] = '{:.4f}'.format(self.flux_OII_fit[2, i, j])
            self.param['OIII_3='] = '{:.4f}'.format(self.flux_OIII_fit[2, i, j])
            self.param['r_3='] = '{:.4f}'.format(self.r[2, i, j])

            # Update the parameter display if N=2 or 3 but N from previous fit=1
            if self.parameters['OII'].value == 2 and np.isnan(self.param['v_2=']):
                self.N_2()
            elif self.parameters['OII'].value == 3 and np.isnan(self.param['v_3=']):
                self.N_3()

            # Draw points
            self.scatter_1.addPoints([j + 0.5], [i + 0.5])
            self.scatter_2.addPoints([j + 0.5], [i + 0.5])
            self.scatter_3.addPoints([j + 0.5], [i + 0.5])
            self.widget1_plot.addItem(self.scatter_1)
            self.widget2_plot.addItem(self.scatter_2)
            self.widget3_plot.addItem(self.scatter_3)
        else:
            self.scatter_1_mask.addPoints([j + 0.5], [i + 0.5])
            self.scatter_2_mask.addPoints([j + 0.5], [i + 0.5])
            self.scatter_3_mask.addPoints([j + 0.5], [i + 0.5])
            self.widget1_plot.addItem(self.scatter_1_mask)
            self.widget2_plot.addItem(self.scatter_2_mask)
            self.widget3_plot.addItem(self.scatter_3_mask)

    def plot_OII(self):
        i, j = self.ypixel, self.xpixel
        self.widget4_plot.clear()

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
        self.set_num_components(1)

    def N_2(self):
        self.set_num_components(2)

    def N_3(self):
        self.set_num_components(3)

    def set_num_components(self, N):
        self.parameters['OII'].value = N
        self.parameters['OIII'].value = N

        for k in [1, 2, 3]:
            active = k <= N
            self._reset_ties_if_inactive(k, np.isnan(self.param[f'v_{k}=']))
            self._set_component_active(k, active)

        if self.line == 'OII':
            self.only_OII()

    def _set_component_active(self, k, active):
        suffix = f"_{k}"

        for p in [
            f'z{suffix}',
            f'sigma_kms{suffix}',
            f'flux_OII{suffix}',
            f'r_OII3729_3727{suffix}',
            f'flux_OIII5008{suffix}'
        ]:
            self.parameters[p].vary = active
            self.parameters[p].value = (
                self.z_qso if 'z' in p and active else
                200 if 'sigma' in p and active else
                1 if active else np.nan
            )

    def _reset_ties_if_inactive(self, k, is_nan):
        if not is_nan:
            return

        # GUI tie flags (if they exist)
        for tie in ['v', 'sigma', 'r']:
            self.param[f"{tie}_{k}="] = 0

    def only_OII(self):
        self.parameters['OIII'].value = 0
        self.parameters['flux_OIII5008_1'].value = np.nan
        self.parameters['flux_OIII5008_1'].vary = False
        self.parameters['flux_OIII5008_2'].value = np.nan
        self.parameters['flux_OIII5008_2'].vary = False
        self.parameters['flux_OIII5008_3'].value = np.nan
        self.parameters['flux_OIII5008_3'].vary = False
        self.parameters['a_OIII5008'].value = 0
        self.parameters['a_OIII5008'].vary = False
        self.parameters['b_OIII5008'].value = 0
        self.parameters['b_OIII5008'].vary = False

    def only_OIII(self):
        ###
        print('to be written')
        self.parameters['OII'].value = 0
        self.parameters['flux_OII_1'].value = np.nan
        self.parameters['flux_OII_1'].vary = False
        self.parameters['flux_OII_2'].value = np.nan
        self.parameters['flux_OII_2'].vary = False
        self.parameters['flux_OII_3'].value = np.nan
        self.parameters['flux_OII_3'].vary = False
        self.parameters['a_OII'].value = 0
        self.parameters['a_OII'].vary = False
        self.parameters['b_OII'].value = 0
        self.parameters['b_OII'].vary = False

    def toggle_OII_ratio(self):
        if self.state == 0:
            print('OII ratio is fixed')
            self.parameters['r_OII3729_3727_1'].vary = False
            self.parameters['r_OII3729_3727_2'].vary = False
            self.parameters['r_OII3729_3727_3'].vary = False
            self.state = not self.state
        elif self.state == 1:
            print('OII ratio is unfixed')
            self.parameters['r_OII3729_3727_1'].vary = True
            if self.parameters['OII'].value == 2:
                self.parameters['r_OII3729_3727_2'].vary = True
            elif self.parameters['OII'].value == 3:
                self.parameters['r_OII3729_3727_2'].vary = True
                self.parameters['r_OII3729_3727_3'].vary = True
            self.state = not self.state

    def compute_v50w80(self, i, j):
        flux_OII_ij = model_OII(self.wave_OII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                self.flux_OII_fit[:, i, j], self.r[:, i, j], plot=True)[0] * (1 + self.r[:, i, j])
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
        self.parameters['z_1'].max = z_1 + (100 / c_kms * (1 + z_1))
        self.parameters['z_1'].min = z_1 - (100 / c_kms * (1 + z_1))
        self.parameters['sigma_kms_1'].value = self.param['sigma_1=']
        self.parameters['sigma_kms_1'].max = self.param['sigma_1='] + 100
        self.parameters['sigma_kms_1'].min = 30
        self.parameters['r_OII3729_3727_1'].value = self.param['r_1=']

        if self.parameters['OII'].value == 2:
            z_2 = (self.param['v_2='] / c_kms * (1 + self.z_qso)) + self.z_qso
            self.parameters['z_2'].value = z_2
            self.parameters['z_2'].max = z_2 + (100 / c_kms * (1 + z_2))
            self.parameters['z_2'].min = z_2 - (100 / c_kms * (1 + z_2))
            self.parameters['sigma_kms_2'].value = self.param['sigma_2=']
            self.parameters['sigma_kms_2'].max = self.param['sigma_2='] + 100
            self.parameters['sigma_kms_2'].min = 30
            self.parameters['r_OII3729_3727_2'].value = self.param['r_2=']

        elif self.parameters['OII'].value == 3:
            z_3 = (self.param['v_3='] / c_kms * (1 + self.z_qso)) + self.z_qso
            self.parameters['z_3'].value = z_3
            self.parameters['z_3'].max = z_3 + (100 / c_kms * (1 + z_3))
            self.parameters['z_3'].min = z_3 - (100 / c_kms * (1 + z_3))
            self.parameters['sigma_kms_3'].value = self.param['sigma_3=']
            self.parameters['sigma_kms_3'].max = self.param['sigma_3='] + 100
            self.parameters['sigma_kms_3'].min = 30
            self.parameters['r_OII3729_3727_3'].value = self.param['r_3=']


        # Fit
        mask_spec_OII = np.ones(len(self.wave_vac[0]), dtype=bool)
        mask_spec_OIII = np.ones(len(self.wave_vac[1]), dtype=bool)

        # If select wavelength is enabled
        if self.param['Fitting ranges', 'OII', 'enabled']:
            mask_spec_OII = (self.wave_vac[0] >= self.param['Fitting ranges', 'OII', 'min']) \
                            & (self.wave_vac[0] <= self.param['Fitting ranges', 'OII', 'max'])
        if self.param['Fitting ranges', 'OIII', 'enabled']:
            mask_spec_OIII = (self.wave_vac[1] >= self.param['Fitting ranges', 'OIII', 'min']) \
                            & (self.wave_vac[1] <= self.param['Fitting ranges', 'OIII', 'max'])
        mask_spec = np.hstack((mask_spec_OII, mask_spec_OIII))
        wave_vac_mask_spec = np.array([self.wave_vac[0][mask_spec_OII], self.wave_vac[1][mask_spec_OIII]], dtype='object')

        result = self.spec_model.fit(self.flux[:, i, j][mask_spec], wave_vac=wave_vac_mask_spec, params=self.parameters,
                                     weights=1 / self.flux_err[:, i, j][mask_spec])
        self.fs[i, j] = result.success
        self.chisqr[i, j], self.redchi[i, j] = result.chisqr, result.redchi
        self.redchi_show[i, j] = result.redchi

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
        self.num_comp[i, j], self.num_comp_show[i, j] = np.sum(~np.isnan(self.v[:, i, j])), \
                                                        np.sum(~np.isnan(self.v[:, i, j]))

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
        self.v50_ori[i, j], self.w80_ori[i, j] = self.v50[i, j], self.w80[i, j]

        # Update fit
        self.flux_OII_array[:, :, i, j] = model_OII(self.wave_OII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                        self.flux_OII_fit[:, i, j], self.r[:, i, j], plot=False)
        self.flux_OIII_array[:, :, i, j] = Gaussian(self.wave_OIII_exp[:, np.newaxis], self.z[:, i, j], self.sigma[:, i, j],
                                        self.flux_OIII_fit[:, i, j], wave_OIII5008_vac)

        # Replot
        self.param['AIC, BIC='] = '{:.2f}, {:.2f}'.format(result.aic, result.bic)
        self.show_fit()
        self.plot_OII()
        self.plot_OIII()
        self.v_map.updateImage(image=self.v50.T)
        self.sigma_map.updateImage(image=self.w80.T)
        self.plot_3rd_panel()

    def save_v50w80(self):
        hdul_v50 = fits.ImageHDU(self.v50, header=self.hdul_fit[2].header)
        hdul_v50.writeto(self.path_v50, overwrite=True)
        hdul_w80 = fits.ImageHDU(self.w80, header=self.hdul_fit[2].header)
        hdul_w80.writeto(self.path_w80, overwrite=True)

        hdul_v50 = fits.ImageHDU(self.v50_OII, header=self.hdul_fit[2].header)
        hdul_v50.writeto(self.path_v50_OII, overwrite=True)
        hdul_w80 = fits.ImageHDU(self.w80_OII, header=self.hdul_fit[2].header)
        hdul_w80.writeto(self.path_w80_OII, overwrite=True)

        hdul_v50 = fits.ImageHDU(self.v50_OIII, header=self.hdul_fit[2].header)
        hdul_v50.writeto(self.path_v50_OIII, overwrite=True)
        hdul_w80 = fits.ImageHDU(self.w80_OIII, header=self.hdul_fit[2].header)
        hdul_w80.writeto(self.path_w80_OIII, overwrite=True)

    def re_mask(self):
        S_N_thr = self.param['S_N=']
        self.mask_OII = np.where(self.S_N > S_N_thr, self.mask_OII_ori, 0)
        self.mask_OIII = np.where(self.S_N > S_N_thr, self.mask_OIII_ori, 0)
        self.mask = np.where(self.S_N > S_N_thr, self.mask_ori, 0)
        self.v50 = np.where(self.mask, self.v50_ori, np.nan)
        self.w80 = np.where(self.mask, self.w80_ori, np.nan)
        self.redchi_show = np.where(self.mask, self.redchi, np.nan)
        self.num_comp_show = np.where(self.mask, self.num_comp, np.nan)

        self.v_map.updateImage(image=self.v50.T)
        self.sigma_map.updateImage(image=self.w80.T)
        self.plot_3rd_panel()

    def clear_scatter(self):
        self.scatter_1.clear()
        self.scatter_2.clear()
        self.scatter_3.clear()
        self.scatter_1_mask.clear()
        self.scatter_2_mask.clear()
        self.scatter_3_mask.clear()
        self.widget1_plot.removeItem(self.current_roi)

    def plot_3rd_panel(self):
        if self.param['Select 3rd panel'] == 'chi^2':
            self.chi_map.updateImage(image=self.redchi_show.T)
        elif self.param['Select 3rd panel'] == 'N_comp':
            self.chi_map.updateImage(image=self.num_comp_show.T)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotWindow(cubename=args.m, NLR='', UseDetectionSeg=eval(args.seg))
    window.show()
    sys.exit(app.exec_())
