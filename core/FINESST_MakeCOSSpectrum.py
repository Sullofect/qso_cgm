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
from astropy.coordinates import SkyCoord
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
from astropy.convolution import convolve
from astropy.convolution import Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.minor', size=3, visible=True)
rc('ytick.minor', size=3, visible=True)
rc('xtick', direction='in', labelsize=15)
rc('ytick', direction='in', labelsize=15)
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# line information
wave_NeVIII = 770.40
wave_OVI = 1031.92
wave_OV = 629.73
wave_NIV = 765.147
for i in ['3C57']:
    cubename = i

    # Load qso information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Spectrum
    path_cos = '../../MUSEQuBES+CUBS/COS/{}_FUV.fits'.format(cubename)
    data_cos = fits.getdata(path_cos, 1, ignore_missing_end=True)
    wave = data_cos['wave'] / (1 + z_qso)
    flux = data_cos['flux'] / data_cos['continuum']

    kernel = Box1DKernel(5)
    flux = convolve(flux, kernel)

    window_OVI = 3
    mask_OVI = np.where((wave > wave_OVI - window_OVI) * (wave < wave_OVI + window_OVI))
    wave_OVI_array = wave[mask_OVI]
    v_OVI_array = (wave_OVI_array - wave_OVI) / wave_OVI * 3e5
    flux_OVI_array = flux[mask_OVI]

    window_NeVIII = 3
    mask_NeVIII = np.where((wave > wave_NeVIII - window_NeVIII) * (wave < wave_NeVIII + window_NeVIII))
    wave_NeVIII_array = wave[mask_NeVIII]
    v_NeVIII_array = (wave_NeVIII_array - wave_NeVIII) / wave_NeVIII * 3e5
    flux_NeVIII_array = flux[mask_NeVIII]

    plt.figure(figsize=(5, 2.5), dpi=300)
    plt.plot(v_OVI_array, flux_OVI_array, 'k-', lw=0.5, drawstyle='steps-mid')
    plt.axvline(0, ls='--', color='grey', lw=1)
    plt.axhline(1, ls='--', color='grey', lw=1)
    plt.xlim(-800, 0)
    plt.xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
    plt.ylabel(r'$\mathrm{Normalized \; Flux}$', size=20)
    plt.title(r'$\rm O \, VI$', size=20, x=0.8, y=0.1)
    plt.savefig('../../Proposal/FINESST_25/{}_{}_FUV.pdf'.format(cubename, 'OVI'), bbox_inches='tight')

    # Spectrum
    plt.figure(figsize=(5, 2.5), dpi=300)
    plt.plot(v_NeVIII_array, flux_NeVIII_array, 'k-', lw=0.5, drawstyle='steps-mid')
    plt.axvline(0, ls='--', color='grey', lw=1)
    plt.axhline(1, ls='--', color='grey', lw=1)
    plt.xlim(-800, 0)
    plt.xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
    plt.ylabel(r'$\mathrm{Normalized \; Flux}$', size=20)
    plt.title(r'$\rm Ne \, VIII$', size=20, x=0.8, y=0.1)
    plt.savefig('../../Proposal/FINESST_25/{}_{}_FUV.pdf'.format(cubename, 'NeVIII'), bbox_inches='tight')


for i in ['HE0238-1904']:
    cubename = i

    # Load qso information
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Spectrum
    path_cos = '../../MUSEQuBES+CUBS/COS/{}_FUV.fits'.format(cubename)
    data_cos = fits.getdata(path_cos, 1, ignore_missing_end=True)
    wave = data_cos['wave'] / (1 + z_qso)
    flux = data_cos['flux'] / data_cos['continuum']

    kernel = Box1DKernel(5)
    flux = convolve(flux, kernel)

    window_OVI = 40
    mask_OVI = np.where((wave > wave_OVI - window_OVI) * (wave < wave_OVI + window_OVI))
    wave_OVI_array = wave[mask_OVI]
    v_OVI_array = (wave_OVI_array - wave_OVI) / wave_OVI * 3e5
    flux_OVI_array = flux[mask_OVI]

    window_NeVIII = 40
    mask_NeVIII = np.where((wave > wave_NeVIII - window_NeVIII) * (wave < wave_NeVIII + window_NeVIII))
    wave_NeVIII_array = wave[mask_NeVIII]
    v_NeVIII_array = (wave_NeVIII_array - wave_NeVIII) / wave_NeVIII * 3e5
    flux_NeVIII_array = flux[mask_NeVIII]

    plt.figure(figsize=(5, 2.5), dpi=300)
    plt.plot(v_OVI_array, flux_OVI_array, 'k-', lw=0.5, drawstyle='steps-mid')
    plt.axvline(0, ls='--', color='grey', lw=1)
    plt.axhline(1, ls='--', color='grey', lw=1)
    plt.xlim(-5200, -3000)
    plt.ylim(0, 1.2)
    plt.xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
    plt.ylabel(r'$\mathrm{Normalized \; Flux}$', size=20)
    plt.title(r'$\rm O \, VI$', size=20, x=0.8, y=0.1)
    plt.savefig('../../Proposal/FINESST_25/{}_{}_FUV.pdf'.format(cubename, 'OVI'), bbox_inches='tight')

    plt.figure(figsize=(5, 2.5), dpi=300)
    plt.plot(v_NeVIII_array, flux_NeVIII_array, 'k-', lw=0.5, drawstyle='steps-mid')
    plt.axvline(0, ls='--', color='grey', lw=1)
    plt.axhline(1, ls='--', color='grey', lw=1)
    plt.xlim(-5200, -3000)
    plt.ylim(0, 1.2)
    plt.xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
    plt.ylabel(r'$\mathrm{Normalized \; Flux}$', size=20)
    plt.title(r'$\rm Ne \, VIII$', size=20, x=0.8, y=0.1)
    plt.savefig('../../Proposal/FINESST_25/{}_{}_FUV.pdf'.format(cubename, 'NeVIII'), bbox_inches='tight')

for i in ['TEX0206-048']:
    cubename = i

    # Load qso information
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    print(z_qso)

    # Spectrum
    path_cos = '../../MUSEQuBES+CUBS/COS/{}_FUV.fits'.format(cubename)
    data_cos = fits.getdata(path_cos, 1, ignore_missing_end=True)
    wave = data_cos['wave'] / (1 + z_qso)
    flux = data_cos['flux'] / data_cos['continuum']

    kernel = Box1DKernel(5)
    flux = convolve(flux, kernel)

    window_NeVIII = 5
    mask_NeVIII = np.where((wave > wave_NeVIII - window_NeVIII) * (wave < wave_NeVIII + window_NeVIII))
    wave_NeVIII_array = wave[mask_NeVIII]
    v_NeVIII_array = (wave_NeVIII_array - wave_NeVIII) / wave_NeVIII * 3e5
    flux_NeVIII_array = flux[mask_NeVIII]

    window_OV = 5
    mask_OV = np.where((wave > wave_OV - window_OV) * (wave < wave_OV + window_OV))
    wave_OV_array = wave[mask_OV]
    v_OV_array = (wave_OV_array - wave_OV) / wave_OV * 3e5
    flux_OV_array = flux[mask_OV]

    window_NIV = 5
    mask_NIV = np.where((wave > wave_NIV - window_NIV) * (wave < wave_NIV + window_NIV))
    wave_NIV_array = wave[mask_NIV]
    v_NIV_array = (wave_NIV_array - wave_NIV) / wave_NIV * 3e5
    flux_NIV_array = flux[mask_NIV]

    # plt.figure(figsize=(5, 2.5), dpi=300)
    # plt.plot(v_NeVIII_array, flux_NeVIII_array, 'k-', lw=1, drawstyle='steps-mid')
    # plt.fill_between([63, 175], [0, 0], [1.5, 1.5], color='red', alpha=0.5)
    # plt.axvline(0, ls='--', color='grey', lw=1)
    # plt.axhline(1, ls='--', color='grey', lw=1)
    # plt.xlim(-850, 250)
    # plt.ylim(0, 1.5)
    # plt.xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
    # plt.ylabel(r'$\mathrm{Normalized \; Flux}$', size=20)
    # plt.title(r'$\rm Ne \, VIII$', size=20, x=0.15, y=0.1)
    # plt.savefig('../../Proposal/FINESST_25/{}_FUV.pdf'.format(cubename), bbox_inches='tight')

    fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.0)
    ax[0].plot(v_OV_array, flux_OV_array, 'k-', lw=1, drawstyle='steps-mid')
    ax[1].plot(v_NIV_array, flux_NIV_array, 'k-', lw=1, drawstyle='steps-mid')
    ax[0].fill_between([150, 250], [0, 0], [1.5, 1.5], color='red', alpha=0.5)
    ax[1].fill_between([150, 250], [0, 0], [1.5, 1.5], color='red', alpha=0.5)
    ax[0].axvline(0, ls='--', color='grey', lw=1)
    ax[0].axhline(1, ls='--', color='grey', lw=1)
    ax[1].axvline(0, ls='--', color='grey', lw=1)
    ax[1].axhline(1, ls='--', color='grey', lw=1)
    ax[0].set_xlim(-850, 300)
    ax[0].set_ylim(0, 1.4)
    ax[1].set_xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
    fig.supylabel(r'$\mathrm{Normalized \; Flux}$', size=20)
    # ax[1].set_title(r'$\rm Ne \, VIII$', size=20, x=0.15, y=0.80)
    ax[0].set_title(r'$\rm O \, V$', size=20, x=0.15, y=0.80)
    ax[1].set_title(r'$\rm N \, IV$', size=20, x=0.15, y=0.80)
    plt.savefig('../../Proposal/FINESST_25/{}_FUV.pdf'.format(cubename), bbox_inches='tight')





# Spectrum
# path_cos = '../../MUSEQuBES+CUBS/COS/HE0226-4110_COS_FUV_wavecal.fits'
# data_cos = fits.getdata(path_cos, 1, ignore_missing_end=True)
# wave = data_cos['wave']
# flux = data_cos['flux']
#
#
# plt.figure(figsize=(16, 4), dpi=300)
# plt.plot(wave[::5], flux[::5] / 1e-14, 'k-', lw=0.5)
# plt.xlabel(r'Observed wavelength [$\rm \AA$]', size=20)
# plt.ylabel(r'${f}_{\lambda} \; [10^{-14} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}}]$', size=20, x=0.03)
# plt.title('HE0226-4110 COS', size=20, x=0.2, y=0.8)
# plt.xlim(1150, 1800)
# plt.savefig('../../MUSEQuBES+CUBS/COS/HE0226-4110_COS_FUV_wavecal.pdf', bbox_inches='tight')
