import os
import aplpy
# import lmfit
import numpy as np
import matplotlib as mpl
import gala.potential as gp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
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
import biconical_outflow_model_3d as bicone
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from gala.units import galactic, solarsystem, dimensionless
from photutils.isophote import EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.isophote import Ellipse
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
from scipy.ndimage import rotate
from astropy.table import Table
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
Halpha = 6564.614
c_kms = 2.998e5

def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None, name_gal='NGC 3945', dis_gal=None):
    scale_phy_3C57 = 30 * 50 / 7
    scale = np.pi * dis_gal * 3600 / 180 * 1e3
    width_gal = np.round(scale_phy_3C57 / scale, 0)
    if np.isnan(dis_gal):
        # gc.recenter(ra_qso, dec_qso, width=1500 / 3600, height=1500 / 3600)
        gc.recenter(ra_qso, dec_qso, width=50 / 3600, height=50 / 3600)
    else:
        gc.recenter(ra_qso, dec_qso, width=width_gal / 3600, height=width_gal / 3600)
        # gc.recenter(ra_qso, dec_qso, width=100 / 3600, height=100 / 3600)
    # gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
    #                 linewidths=0.5, s=600, zorder=100)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)
    if type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$6'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(35)
        gc.add_label(0.98, 0.94, cubename, size=35, relative=True, horizontalalignment='right')
        gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        gc.colorbar.set_ticks([-300, -200, -100, 0, 100, 200, 300])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        gc.colorbar.hide()
        gc.add_label(0.98, 0.94, name_gal, size=35, relative=True, horizontalalignment='right')
    elif type == 'GasMap_sigma':
        # gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')

    # Scale bar
    gc.add_scalebar(length=50 / scale * u.arcsecond)
    gc.scalebar.set_corner('top left')
    gc.scalebar.set_label('{:.0f}'.format(50 / scale) +  r"$'' \approx 50 \mathrm{\; pkpc}$")
    gc.scalebar.set_font_size(30)

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

path_gals_3D = '../../MUSEQuBES+CUBS/CALIFA/eCALIFA.pyPipe3D.fits'
data_gals_3D = fits.open(path_gals_3D)[1].data
ID_3D = data_gals_3D['ID']
FoV = data_gals_3D['FoV']
M_ste = data_gals_3D['log_Mass']
M_ste = M_ste[np.argsort(ID_3D)]
FoV = FoV[np.argsort(ID_3D)]

# Path
path_gals = '../../MUSEQuBES+CUBS/CALIFA/galaxies_properties.fits'
data_gals = fits.open(path_gals)[1].data
ID = data_gals['ID']
name = data_gals['cubename']
col = np.arange(len(name))
type = data_gals['type']
Mgas = data_gals['Mgas']
ra, dec = data_gals['RA'], data_gals['DEC']
# mask = ((type == 'S0') | (type == 'S0a') | (type == 'E0') | (type == 'E1') | (type == 'E2')
#         | (type == 'E3') | (type == 'E4') | (type == 'E5') | (type == 'E6') | (type == 'E7')) * (Mgas > 8)
# mask = (type == 'S0a') * ((M_ste > 11.5))
# mask = ((type == 'S0')) * (M_ste > 11.5)
mask = ((type == 'E0') | (type == 'E1') | (type == 'E2') | (type == 'E3') | (type == 'E4')
        | (type == 'E5') | (type == 'E6') | (type == 'E7')) * (M_ste > 11.5) * (FoV > 1)
name_ETP = name[mask]
col_ETP = col[mask]
print(col_ETP // 3, name_ETP)

# raise ValueError('testing')

#
# gal_list = ['CGCG251-041', 'NGC3182', 'NGC0693', 'NGC5784', 'SN2001bf', 'UGC10205', 'CGCG229-009', 'UGC05771']
# v_sys_list =  [5275, 2099, 1565, 5370, np.nan, 6556, 8488, 7398]
# dis_list = [65, 33, 19, 82, 59.4, 122.550, 122.990, 93.6]
gal_list = ['UGC10205', 'CGCG229-009', 'UGC05771']
v_sys_list =  [6556, 8488, 7398]
dis_list = [122.550, 122.990, 93.6]
for i, ival in enumerate(gal_list):
    path_i = '../../MUSEQuBES+CUBS/CALIFA/{}.Pipe3D.cube.fits'.format(ival)
    hdr = fits.open(path_i)[0].header
    v_i = fits.open(path_i)[4].data[0, :, :] - v_sys_list[i]
    FWHM_i = fits.open(path_i)[4].data[1, :, :]
    flux_Halpha = fits.open(path_i)[4].data[6, :, :]
    sigma_i = c_kms / Halpha * np.sqrt((FWHM_i / 2.354) ** 2 - 2.6 ** 2)  # Instrumental resolution 2.6A
    # sigma_i = np.where(flux_Halpha > 0.1 * np.std(flux_Halpha), sigma_i, np.nan)
    # v_i = np.where(flux_Halpha > 0.1 * np.std(flux_Halpha), v_i, np.nan)

    #
    hdr['NAXIS'] = 2
    hdr.remove('CTYPE3')
    hdr.remove('CDELT3')
    hdr.remove('CRPIX3')
    hdr.remove('CRVAL3')
    hdr.remove('CUNIT3')
    hdr.remove('CD1_3')
    hdr.remove('CD2_3')
    hdr.remove('CD3_1')
    hdr.remove('CD3_2')
    hdr.remove('CD3_3')

    #
    path_v_i = '../../MUSEQuBES+CUBS/CALIFA/{}_v.fits'.format(ival)
    path_figure_v_i = '../../MUSEQuBES+CUBS/CALIFA/{}_v.png'.format(ival)
    hdul_v_i = fits.ImageHDU(v_i, header=hdr)
    hdul_v_i.writeto(path_v_i, overwrite=True)

    #
    # fig = plt.figure(figsize=(8, 8), dpi=300)
    # gc = aplpy.FITSFigure(path_v_i, figure=fig, hdu=1)
    # gc.show_colorscale(vmin=-350, vmax=350, cmap='coolwarm')
    # # patch = rectangle_gal.plot(ax=gc.ax, facecolor='none', edgecolor='k', lw=1, linestyle='--', label='Rectangle')
    # APLpyStyle(gc, type='GasMap', ra_qso=ra[name == ival], dec_qso=dec[name == ival], name_gal=gal_list[i], dis_gal=dis_list[i])
    # fig.savefig(path_figure_v_i, bbox_inches='tight')


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(v_i, origin='lower', vmin=-300, vmax=300, cmap='coolwarm')
    ax[1].imshow(sigma_i, origin='lower', vmin=0, vmax=350, cmap=Dense_20_r.mpl_colormap)
    plt.show()