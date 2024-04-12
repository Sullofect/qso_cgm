import os
import aplpy
import lmfit
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
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from photutils.isophote import EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.isophote import Ellipse
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
from scipy.ndimage import rotate
from astropy.table import Table
import mpl_interactions.ipyplot as iplt
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None, name_gal='NGC 3945', dis_gal=None):
    scale_phy_3C57 = 30 * 50 / 7
    scale = np.pi * dis_gal * 1 / 3600 / 180 * 1e3
    width_gal = np.round(scale_phy_3C57 / scale, 0)
    if np.isnan(dis_gal):
        gc.recenter(ra_qso, dec_qso, width=1500 / 3600, height=1500 / 3600)
    else:
        gc.recenter(ra_qso, dec_qso, width=width_gal / 3600, height=width_gal / 3600)
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=600, zorder=100)
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
        # gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'GasMap_sigma':
        # gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
    else:
        gc.colorbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        gc.colorbar.set_axis_label_text(r'$\rm log([O \, III]/[O \, II])$')

    # Scale bar
    # gc.add_scalebar(length=3 * u.arcsecond)
    # gc.add_scalebar(length=50 / scale * u.arcsecond)
    # gc.scalebar.set_corner('top left')
    # gc.scalebar.set_label(r"$450'' \approx \,$" + '{:.0f}'.format(450 * scale) + r"$\mathrm{\; pkpc}$")
    # gc.scalebar.set_label(r"$3'' \approx 20 \mathrm{\; pkpc}$")
    # gc.scalebar.set_label('{:.0f}'.format(50 / scale) +  r"$'' \approx 50 \mathrm{\; pkpc}$")
    # gc.scalebar.set_font_size(30)

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

def Gaussian(v, v_c, sigma, flux):
    peak = flux / np.sqrt(2 * sigma ** 2 * np.pi)
    gaussian = peak * np.exp(-(v - v_c) ** 2 / 2 / sigma ** 2)

    return gaussian

# load
gal_name = 'NGC5582'
path_table_gals = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/table_gals.fits'
table_gals = fits.open(path_table_gals)[1].data
gal_name_ = gal_name.replace('C', 'C ')
name_sort = table_gals['Object Name'] == gal_name_
ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
v_sys_gal = table_gals[name_sort]['cz (Velocity)']

# NGC5582
path_ETG = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(gal_name)
path_ETG_new = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1_new.fits'.format(gal_name)
path_ETG_mom2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom2.fits'.format(gal_name)
path_ETG_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/allcubes/{}_cube.fits'.format(gal_name)
path_figure_mom1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_mom1.png'.format(gal_name)
path_figure_mom2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_mom2.png'.format(gal_name)
path_figure_spec = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_spec.png'.format(gal_name)

# Load the kinematic map
hdul_ETG = fits.open(path_ETG)
hdr_ETG = hdul_ETG[0].header
hdr_ETG['NAXIS'] = 2
hdr_ETG.remove('NAXIS3')
hdr_ETG.remove('CTYPE3')
hdr_ETG.remove('CDELT3')
hdr_ETG.remove('CRPIX3')
hdr_ETG.remove('CRVAL3')
v_ETG = hdul_ETG[0].data[0, :, :] - v_sys_gal
hdul_ETG_new = fits.ImageHDU(v_ETG, header=hdr_ETG)
hdul_ETG_new.writeto(path_ETG_new, overwrite=True)

# Load the cube
hdul_ETG_cube = fits.open(path_ETG_cube)
hdr_ETG_cube = hdul_ETG_cube[0].header
flux = hdul_ETG_cube[0].data
flux = np.where(~np.isnan(v_ETG)[np.newaxis, :, :], flux, np.nan)
v_array = np.arange(hdr_ETG_cube['CRVAL3'], hdr_ETG_cube['CRVAL3'] + flux.shape[0] * hdr_ETG_cube['CDELT3'],
                    hdr_ETG_cube['CDELT3']) / 1e3 - v_sys_gal # Convert from m/s to km/s,
mask = ~np.isnan(v_ETG)
size = np.shape(flux)[1:]


# Load ETG fit
path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_fit.fits'.format(gal_name)
hdul_fit = fits.open(path_fit)
v_fit, sigma_fit, flux_fit = hdul_fit[1].data, hdul_fit[3].data, hdul_fit[5].data
flux_fit_array = Gaussian(v_array[:, np.newaxis, np.newaxis], v_fit, sigma_fit, flux_fit)

def flux_cube(v, i, j):
    return flux[:, j, i]
def flux_fit(v, j, i):
    return flux_fit_array[:, j, i]

# print(flux_cube(v_array, 174, 197))

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# controls = iplt.plot(v_array, flux_cube, i=np.arange(360), j=np.arange(360),
#                      label="data", drawstyle='steps-mid', color='k')
# iplt.plot(v_array, flux_fit, controls=controls, label="fit", color='r')
# _ = plt.legend()
# plt.ylim(flux_cube.min(), flux_cube.max())
# plt.show()


from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)

        # Set the central widget of the Window.
        self.setCentralWidget(button)

    def the_button_was_clicked(self):
        print("Clicked!")

app = QApplication(sys.argv)
widget = MainWindow()
widget.show()

app.exec()