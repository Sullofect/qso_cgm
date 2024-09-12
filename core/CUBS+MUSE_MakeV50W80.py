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
from mpdaf.obj import Cube, WaveCoord, Image
from astropy.wcs import WCS
from regions import PixCoord
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
import palettable.scientific.sequential as sequential_s
import palettable
from scipy import integrate
from scipy import interpolate
import time as tm
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

def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None):
    gc.recenter(ra_qso, dec_qso, width=30 / 3600, height=30 / 3600)
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=1200, zorder=100)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=30)
    gc.colorbar.set_axis_label_font(size=30)
    if type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=30)
        gc.colorbar.set_axis_label_text(r'$\mathrm{SB \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        gc.colorbar.set_axis_label_font(size=30)
        # gc.add_scalebar(length=7 * u.arcsecond)
        gc.add_scalebar(length=8 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        # gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")  # 3C57
        gc.scalebar.set_label(r"$8'' \approx 50 \mathrm{\; kpc}$")  # HE0226
        gc.scalebar.set_font_size(30)
        # gc.add_label(0.98, 0.94, cubename, size=35, relative=True, horizontalalignment='right')
        # gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)

        gc.colorbar.set_ticks([-300, -150, 0, 150, 300])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        gc.add_label(0.98, 0.90, '{}'.format(cubename), size=60, relative=True, horizontalalignment='right')
        # gc.colorbar.set_axis_label_text(r'$\rm V_{50} \mathrm{\; [km \, s^{-1}]}$')
    elif type == 'GasMap_slit':
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('bottom left')
        gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)
        gc.colorbar.hide()
        # gc.add_label(0.98, 0.94, r'$\rm 3C\,57$', size=35, relative=True, horizontalalignment='right')
        gc.add_label(0.98, 0.90, r'$\rm 3C\,57$', size=60, relative=True, horizontalalignment='right')
        # gc.colorbar.set_ticks([-300, -150, 0, 150, 300])
        # gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        # gc.colorbar.set_axis_label_text(r'$\rm V_{50} \mathrm{\; [km \, s^{-1}]}$')
    elif type == 'GasMap_sigma':
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)

        gc.colorbar.set_ticks([0, 150, 300, 450, 600, 750])
        gc.colorbar.set_axis_label_text(r'$\rm W_{80} \mathrm{\; [km \, s^{-1}]}$')
        # gc.colorbar.set_axis_label_text(r'$\mathrm{W}_{80} \mathrm{\; [km \, s^{-1}]}$')
    elif type == 'N':
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)
        gc.colorbar.set_ticks([0, 1, 2, 3])
        gc.colorbar.set_axis_label_text(r'$\rm Number \, of \, Gaussians$')
    else:
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)

        gc.colorbar.set_ticks([-1, -0.5, 0.0, 0.5, 1.0])
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
    # xw, yw = gc.pixel2world(146, 140)  # original figure
    xw, yw = gc.pixel2world(140, 140)
    # gc.show_arrows(xw, yw, -0.000035 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000035 * yw, color='k')
    # xw, yw = 40.1333130960119, -18.864847747328896
    # gc.show_arrows(xw, yw, -0.000020 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000020 * yw, color='k')
    # gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
    # gc.add_label(0.88, 0.70, r'E', size=20, relative=True)


def MakeV50W80(cubename=None):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Load fitting
    path_fit = '../../MUSEQuBES+CUBS/fit_kin/{}_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'.format(cubename)
    hdul = fits.open(path_fit)
    fs, hdr = hdul[1].data, hdul[2].header
    v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
    sigma, dsigma = hdul[5].data, hdul[6].data
    flux_OII_fit, dflux_OII_fit = hdul[7].data, hdul[8].data
    flux_OIII_fit, dflux_OIII_fit = hdul[9].data, hdul[10].data
    r, dr = hdul[11].data, hdul[12].data
    a_OII, da_OII = hdul[13].data, hdul[14].data
    a_OIII, da_OIII = hdul[17].data, hdul[18].data
    b_OII, db_OII = hdul[15].data, hdul[16].data
    b_OIII, db_OIII = hdul[19].data, hdul[20].data

    # Load data
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    UseDataSeg=(1.5, 'gauss', None, None)
    line = 'OII+OIII'
    line_OII, line_OIII = 'OII', 'OIII'
    path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}.fits'.format(cubename, line_OII)
    path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}.fits'.format(cubename, line_OIII)
    path_cube_smoothed_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_{}_' \
                             '{}_{}_{}.fits'.format(cubename, line_OII, *UseDataSeg)
    path_cube_smoothed_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_{}_' \
                              '{}_{}_{}.fits'.format(cubename, line_OIII, *UseDataSeg)
    path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, line_OII, *UseSeg)
    path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, line_OIII, *UseSeg)
    figurename_V50 = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_{}_{}_{}_{}_{}_{}_{}.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    figurename_V50_slit = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_{}_{}_{}_{}_{}_{}_{}_slit.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    figurename_W80 = '../../MUSEQuBES+CUBS/fit_kin/{}_W80_{}_{}_{}_{}_{}_{}_{}.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    figurename_OIII_OII = '../../MUSEQuBES+CUBS/fit_kin/{}_OIII_OII_{}_{}_{}_{}_{}_{}_{}.png'.\
        format(cubename, line, True, 3728, *UseDataSeg)
    figurename_N = '../../MUSEQuBES+CUBS/fit_kin/{}_N_Com_{}_{}_{}_{}_{}_{}_{}.png'.\
        format(cubename, line, True, 3728, *UseDataSeg)

    # Load data and smoothing
    cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
    wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
    flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
    flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
    # wave_OII_vac = expand_wave([wave_OII_vac], stack=True)
    # wave_OIII_vac = expand_wave([wave_OIII_vac], stack=True)
    # wave_OII_exp, wave_OIII_exp = wave_OII_vac[:, np.newaxis, np.newaxis], wave_OIII_vac[:, np.newaxis, np.newaxis]
    seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
    mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
    mask_seg = mask_seg_OII + mask_seg_OIII
    flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori
    flux_err_seg_OII, flux_err_seg_OIII = flux_err_OII * seg_3D_OII_ori, flux_err_OIII * seg_3D_OIII_ori
    S_N_OII = np.sum(flux_seg_OII / flux_err_seg_OII, axis=0)
    S_N_OIII = np.sum(flux_seg_OIII / flux_err_seg_OIII, axis=0)
    S_N = np.nansum(np.dstack((S_N_OII, S_N_OIII)), axis=2) / 2
    OIII_OII = np.log10(np.nansum(flux_OIII_fit, axis=0) / np.nansum(flux_OII_fit, axis=0))

    # Compute 3sigma limit
    wave_array_OII, wave_array_OIII = np.zeros_like(flux_OII), np.zeros_like(flux_OIII)
    wave_array_OII[:], wave_array_OIII[:] = wave_OII_vac[:, np.newaxis, np.newaxis], wave_OIII_vac[:, np.newaxis, np.newaxis]
    win_OII_vel = (mask_seg_OII * 1.25) / (wave_OII3728_vac * (1 + z_qso)) * c_kms  # They have the same velocity window
    win_OIII = win_OII_vel / c_kms * (wave_OIII5008_vac * (1 + z_qso))
    OII_start = np.take_along_axis(wave_array_OII,
                                   np.argmin(np.abs(np.cumsum(seg_3D_OII_ori, axis=0) - 1), axis=0)[np.newaxis, :, :], axis=0)[0]
    OIII_start = wave_OIII5008_vac * OII_start / wave_OII3727_vac
    OIII_end = OIII_start + win_OIII
    flux_OIII_3sig = np.where((wave_array_OIII >= OIII_start) * (wave_array_OIII <= OIII_end), 3 * flux_err_OIII, np.nan)
    OIII_OII_3sig = np.log10(np.nansum(flux_OIII_3sig, axis=0) * 1.25 / np.nansum(flux_OII_fit, axis=0))
    OIII_OII = np.where(mask_seg_OIII != 0, OIII_OII, OIII_OII_3sig)

    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        ra_gal, dec_gal, v_gal = data_gal['ra'], data_gal['dec'], data_gal['v']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal = [], [], []

    # Replace coordinate
    path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}_WCS_subcube.fits'.format(cubename)
    hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
    w = WCS(hdr_sub_gaia, naxis=2)
    center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
    c2 = w.world_to_pixel(center_qso)

    path_v50 = '../../MUSEQuBES+CUBS/fit_kin/{}_V50.fits'.format(cubename)
    path_w80 = '../../MUSEQuBES+CUBS/fit_kin/{}_W80.fits'.format(cubename)
    hdul_v50 = fits.open(path_v50)
    hdul_w80 = fits.open(path_w80)
    hdr = hdul_v50[1].header
    hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
    hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
    hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
    hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
    hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
    hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
    hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
    hdr['CD2_2'] = hdr_sub_gaia['CD2_2']

    #
    path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
    path_w80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_W80_plot.fits'.format(cubename)
    path_OIII_OII_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_OIII_OII_plot.fits'.format(cubename)
    path_N_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_N_plot.fits'.format(cubename)
    v50, w80 = hdul_v50[1].data, hdul_w80[1].data

    # Plot the velocity field
    x, y = np.meshgrid(np.arange(v50.shape[0]), np.arange(v50.shape[1]))
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)

    # Mask the center
    circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
    center_mask_flatten = ~circle.contains(pixcoord)
    center_mask = center_mask_flatten.reshape(v50.shape)
    x, y = x[center_mask_flatten], y[center_mask_flatten]
    pixcoord = pixcoord[center_mask_flatten]
    #
    # # Mask a slit
    # rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=5, angle=Angle(-30, 'deg'))
    # mask = rectangle.contains(pixcoord)
    # dis = np.sqrt((x - c2[0])**2 + (y - c2[1])**2) * 0.2 * 50 / 7
    # dis_mask = dis[mask]
    #
    # # Mask each side
    # red = ((x[mask] - c2[0]) < 0) * ((y[mask] - c2[1]) > 0)
    # blue = ~red
    # dis_red = dis_mask[red] * -1
    # dis_blue = dis_mask[blue]

    hdul_v50[1].data = np.where(center_mask, hdul_v50[1].data, np.nan)
    hdul_w80[1].data = np.where(center_mask, hdul_w80[1].data, np.nan)
    OIII_OII = np.where(center_mask, OIII_OII, np.nan)
    hdul_v50[1].data = np.where(S_N > 10, hdul_v50[1].data, np.nan)
    hdul_w80[1].data = np.where(S_N > 10, hdul_w80[1].data, np.nan)
    OIII_OII = np.where(S_N > 10, OIII_OII, np.nan)
    print('Median O32', np.nanmean(np.where(mask_seg_OIII != 0, OIII_OII, np.nan)))

    hdul_v50[1].header = hdr
    hdul_w80[1].header = hdr
    hdul_v50.writeto(path_v50_plot, overwrite=True)
    hdul_w80.writeto(path_w80_plot, overwrite=True)
    hdul_OIII_OII = fits.ImageHDU(OIII_OII, header=hdul_v50[1].header)
    hdul_OIII_OII.writeto(path_OIII_OII_plot, overwrite=True)

    N = np.where(np.isnan(v), v, 1)
    N = np.nansum(N, axis=0)
    N = np.where(N !=0, N, np.nan)
    hdul_N = fits.ImageHDU(N, header=hdul_v50[1].header)
    hdul_N.writeto(path_N_plot, overwrite=True)

    # V50
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_v50_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=-350, vmax=350, cmap='coolwarm')
    APLpyStyle(gc, type='GasMap', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    gc.show_markers(ra_gal, dec_gal, facecolor='white', marker='o', c='white', edgecolors='none', linewidths=0.8, s=100)
    gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra_gal, dec_gal, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=-350, vmax=350, cmap='coolwarm')
    # gc.add_label(0.08, 0.08, '(e)', color='k', size=40, relative=True)
    fig.savefig(figurename_V50, bbox_inches='tight')

    # V50_slit
    # fig = plt.figure(figsize=(8, 8), dpi=300)
    # gc = aplpy.FITSFigure(path_v50_plot, figure=fig, hdu=1)
    # gc.show_colorscale(vmin=-350, vmax=350, cmap='coolwarm')
    # patch = rectangle.plot(ax=gc.ax, facecolor='none', edgecolor='k', lw=1.0, linestyle='--', label='Rectangle')
    # APLpyStyle(gc, type='GasMap_slit', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    # gc.scalebar.set_font_size(50)
    # fig.savefig(figurename_V50_slit, bbox_inches='tight')

    # W80 map
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_w80_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=0, vmax=800, cmap=Dense_20_r.mpl_colormap)
    APLpyStyle(gc, type='GasMap_sigma', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    # gc.add_label(0.08, 0.08, '(f)', color='k', size=40, relative=True)
    # gc.colorbar.hide()
    # gc.scalebar.set_font_size(50)
    # gc.scalebar.hide()
    fig.savefig(figurename_W80, bbox_inches='tight')

    # OIII/OII map
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_OIII_OII_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=-1, vmax=1, cmap=sequential_s.Buda_20.mpl_colormap)
    APLpyStyle(gc, type='else', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    # gc.add_label(0.08, 0.08, '(d)', color='k', size=40, relative=True)
    fig.savefig(figurename_OIII_OII, bbox_inches='tight')

    # Number of components
    import cmasher as cmr
    # import palettable.scientific.sequential as Safe_3
    # cmap = cmr.get_sub_cmap('Pastel1', 0, 0.4)
    cmap = mpl.colors.ListedColormap(palettable.cmocean.sequential.Matter_3.mpl_colors)
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_N_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=0.5, vmax=3.5, cmap=cmap)
    APLpyStyle(gc, type='N', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso)
    # gc.add_label(0.08, 0.08, '(f)', color='k', size=40, relative=True)
    fig.savefig(figurename_N, bbox_inches='tight')



MakeV50W80(cubename='HE0435-5304')