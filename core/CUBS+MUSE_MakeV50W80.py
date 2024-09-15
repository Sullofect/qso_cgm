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
from astropy.cosmology import FlatLambdaCDM
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


def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None):
    gc.recenter(ra_qso, dec_qso, width=30 / 3600, height=30 / 3600)
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=1000, zorder=100)
    gc.set_system_latex(True)

    # calculate angular diameter distance
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
    arcsec_str = '{:.0f}'.format(((50 / d_A_kpc) * 206265))

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
        gc.add_scalebar(length=float(arcsec_str) * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(arcsec_str + r"$'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)
        gc.colorbar.set_ticks([-300, -150, 0, 150, 300])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        gc.colorbar.hide()
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
        gc.add_scalebar(length=float(arcsec_str) * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(arcsec_str + r"${}'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(30)

        gc.colorbar.set_ticks([0, 150, 300, 450, 600, 750])
        gc.colorbar.set_axis_label_text(r'$\rm W_{80} \mathrm{\; [km \, s^{-1}]}$')
        gc.colorbar.hide()
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

    # Cubename and redshift
    if cubename == 'LBQS1435-0134':
        cubename = 'Q1435-0134'
    elif cubename == 'TEX0206-048':
        cubename = 'TXS0206-048'

    try:
        name_1, name_2 = cubename.split('-')
    except ValueError:
        name_1, name_2 = cubename.split('+')

    for i in range(len(cubename)):
        if not cubename[i].isalpha():
            break

    if i == 1:
        cubename_label = name_1 + r'$-$' + name_2
    else:
        cubename_label = cubename[:i] + r'$\,$' + name_1[i:] + r'$-$' + name_2
    gc.add_label(0.98, 0.95, cubename_label, size=35, relative=True, horizontalalignment='right')
    gc.add_label(0.98, 0.87, r'$z=$' + ' {:.4f}'.format(z_qso), size=35, relative=True, horizontalalignment='right')

    # Hide
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.ticks.set_length(30)


def MakeV50W80(cubename=None, v_max=300, sigma_max=300):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Load data
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    UseDataSeg=(1.5, 'gauss', None, None)
    line = 'OII+OIII'
    line_OII, line_OIII = 'OII', 'OIII'
    figurename_V50 = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_{}_{}_{}_{}_{}_{}_{}.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    figurename_S80 = '../../MUSEQuBES+CUBS/fit_kin/{}_W80_{}_{}_{}_{}_{}_{}_{}.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        ra_gal, dec_gal, v_gal = data_gal['ra'], data_gal['dec'], data_gal['v']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal = [], [], []

    # V50, S80
    path_v50 = '../../MUSEQuBES+CUBS/fit_kin/{}_V50.fits'.format(cubename)
    path_w80 = '../../MUSEQuBES+CUBS/fit_kin/{}_W80.fits'.format(cubename)
    hdul_v50 = fits.open(path_v50)
    hdul_s80 = fits.open(path_w80)
    hdul_s80[1].data = hdul_s80[1].data / 2.563
    hdr = hdul_v50[1].header

    # Replace coordinate to Gaia
    path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}_WCS_subcube.fits'.format(cubename)
    if os.path.exists(path_sub_white_gaia):
        print('correcting to gaia')
        hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
        w = WCS(hdr_sub_gaia, naxis=2)

        hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
        hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
        hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
        hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
        hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
        hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
        hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
        hdr['CD2_2'] = hdr_sub_gaia['CD2_2']
    else:
        w = WCS(hdr, naxis=2)
        print('No gaia correction info')

    center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
    c2 = w.world_to_pixel(center_qso)

    #
    path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
    path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)

    # Plot the velocity field
    x, y = np.meshgrid(np.arange(hdul_v50[1].data.shape[0]), np.arange(hdul_v50[1].data.shape[1]))
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)

    # Mask the center
    circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
    center_mask_flatten = ~circle.contains(pixcoord)
    center_mask = center_mask_flatten.reshape(hdul_v50[1].data.shape)

    hdul_v50[1].data = np.where(center_mask, hdul_v50[1].data, np.nan)
    hdul_s80[1].data = np.where(center_mask, hdul_s80[1].data, np.nan)

    hdul_v50[1].header = hdr
    hdul_s80[1].header = hdr
    hdul_v50.writeto(path_v50_plot, overwrite=True)
    hdul_s80.writeto(path_s80_plot, overwrite=True)

    # V50
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_v50_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=-v_max, vmax=v_max, cmap='coolwarm')
    APLpyStyle(gc, type='GasMap', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
    gc.add_label(0.05, 0.08, '[{}, {}]'.format(-v_max, v_max), size=30, relative=True, horizontalalignment='left')
    gc.show_markers(ra_gal, dec_gal, facecolor='white', marker='o', c='white', edgecolors='none', linewidths=0.8, s=100)
    gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra_gal, dec_gal, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=-v_max, vmax=v_max, cmap='coolwarm')
    fig.savefig(figurename_V50, bbox_inches='tight')

    # S80 map converted from W80 to sigma
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_s80_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=0, vmax=sigma_max, cmap=Dense_20_r.mpl_colormap)
    gc.add_label(0.05, 0.08, '[{}, {}]'.format(0, sigma_max), size=30, relative=True, horizontalalignment='left')
    APLpyStyle(gc, type='GasMap_sigma', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
    fig.savefig(figurename_S80, bbox_inches='tight')




# MakeV50W80(cubename='HE0435-5304', v_max=100, sigma_max=300)

# MakeV50W80(cubename='TEX0206-048', v_max=600, sigma_max=400)
# MakeV50W80(cubename='Q1354+048', v_max=400, sigma_max=300)
# MakeV50W80(cubename='J0154-0712', v_max=300, sigma_max=300)
MakeV50W80(cubename='LBQS1435-0134', v_max=400, sigma_max=400)
# MakeV50W80(cubename='PG1522+101', v_max=300, sigma_max=300)

# MakeV50W80(cubename='PKS0232-04', v_max=300, sigma_max=300)