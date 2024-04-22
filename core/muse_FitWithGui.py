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

def Gaussian(velocity, v, sigma, flux):
    peak = flux / np.sqrt(2 * sigma ** 2 * np.pi)
    gaussian = peak * np.exp(-(velocity - v) ** 2 / 2 / sigma ** 2)

    return gaussian

#
path_table_gals = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/table_gals.fits'



class PlotWindow(QMainWindow):
    def __init__(self, gal_name='NGC5582'):
        super().__init__()

        # Load the data
        self.gal_name = gal_name
        table_gals = fits.open(path_table_gals)[1].data
        gal_name_ = gal_name.replace('C', 'C ')
        name_sort = table_gals['Object Name'] == gal_name_
        ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
        v_sys_gal = table_gals[name_sort]['cz (Velocity)']

        # Load data
        path_Serra = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(gal_name)
        path_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/allcubes/{}_cube.fits'.format(gal_name)
        hdul_Serra = fits.open(path_Serra)
        self.v_Serra = hdul_Serra[0].data[0, :, :] - v_sys_gal

        # Load the cube
        hdul_cube = fits.open(path_cube)
        hdr_cube = hdul_cube[0].header
        flux = hdul_cube[0].data
        self.flux = np.where(~np.isnan(self.v_Serra)[np.newaxis, :, :], flux, np.nan)
        self.v_array = np.arange(hdr_cube['CRVAL3'], hdr_cube['CRVAL3'] + flux.shape[0] * hdr_cube['CDELT3'],
                                 hdr_cube['CDELT3']) / 1e3 - v_sys_gal  # Convert from m/s to km/s,
        self.mask = ~np.isnan(self.v_Serra)
        self.size = np.shape(flux)[1:]


        # Load ETG fit
        self.path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_fit.fits'.\
            format(self.gal_name)
        if os.path.exists(self.path_fit) is False:
            print('Fitting result file does not exist, start fitting from scratch.')
            hdr_Serra = hdul_Serra[0].header
            hdr_Serra['NAXIS'] = 2
            hdr_Serra.remove('NAXIS3')
            hdr_Serra.remove('CTYPE3')
            hdr_Serra.remove('CDELT3')
            hdr_Serra.remove('CRPIX3')
            hdr_Serra.remove('CRVAL3')
            self.hdr = hdr_Serra
            v_guess, sigma_guess, flux_guess = 0, 50, 0.5
            self.model = Gaussian
            self.parameters = lmfit.Parameters()
            self.parameters.add_many(('v', v_guess, True, -300, 300, None),
                                     ('sigma', sigma_guess, True, 0, 150, None),
                                     ('flux', flux_guess, True, 0, None, None))
            self.fit()
        hdul_fit = fits.open(self.path_fit)
        self.v_fit, self.sigma_fit, self.flux_fit = hdul_fit[1].data, hdul_fit[3].data, hdul_fit[5].data
        self.flux_fit_array = Gaussian(self.v_array[:, np.newaxis, np.newaxis], self.v_fit, self.sigma_fit, self.flux_fit)
        self.v_fit = np.where(self.mask, self.v_fit, np.nan)
        self.sigma_fit = np.where(self.mask, self.sigma_fit, np.nan)


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
        self.widget1_plot = self.widget1.addPlot()
        self.widget2_plot = self.widget2.addPlot()
        self.widget3_plot = self.widget3.addPlot()
        self.widget1.setFixedSize(450, 450)
        self.widget2.setFixedSize(450, 450)
        self.widget3.setFixedSize(900, 450)

        # Set background color
        self.widget1.setBackground((112, 181, 116))
        self.widget2.setBackground((230, 161, 92))
        # self.widget3.setBackground("w")
        self.widget1_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget2_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget3_plot.setLimits(xMin=-1000, xMax=1000)

        # Set param
        self.paramSpec = [dict(name='v=', type='float', value=None, dec=False, readonly=False),
                          dict(name='sigma=', type='float', value=None, readonly=False),
                          dict(name='flux=', type='float', value=None, readonly=False)]
        self.param = pt.Parameter.create(name='Options', type='group', children=self.paramSpec)
        self.tree = pt.ParameterTree()
        self.tree.setParameters(self.param)

        #
        self.layout.addWidget(self.widget1, 0, 0, 1, 1)
        self.layout.addWidget(self.widget2, 0, 1, 1, 1)
        self.layout.addWidget(self.widget3, 1, 0, 1, 2)
        self.layout.addWidget(self.tree, 0, 2, 1, 1)


        # Plot the 2D map in the first plot
        self.v_map = pg.ImageItem()
        self.widget1_plot.addItem(self.v_map)
        colormap = cm.get_cmap("coolwarm")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.v_map.setLookupTable(lut)
        self.v_map.updateImage(image=self.v_fit.T, levels=(-350, 350))

        # Plot the 2D map in the second plot
        self.sigma_map = pg.ImageItem()
        self.widget2_plot.addItem(self.sigma_map)
        colormap = Dense_20_r.mpl_colormap
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        self.sigma_map.setLookupTable(lut)
        self.sigma_map.updateImage(image=self.sigma_fit.T, levels=(0, 300))
        self.widget2_plot.setXLink(self.widget1_plot)
        self.widget2_plot.setYLink(self.widget1_plot)
        self.widget1_plot.setXLink(self.widget2_plot)
        self.widget1_plot.setYLink(self.widget2_plot)

        # Plot initial data in the second plot
        self.widget3_plot.setLabel('bottom', 'Velocity (km/s)')
        self.widget3_plot.setLabel('left', 'Flux')

        # Connect mouse click event to update plot
        self.widget1_plot.scene().sigMouseClicked.connect(self.update_plot)
        self.widget2_plot.scene().sigMouseClicked.connect(self.update_plot)

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
                    spec_model = lmfit.Model(self.model, missing='drop')
                    result = spec_model.fit(flux_ij, velocity=self.v_array, params=self.parameters)

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
            self.widget3_plot.clear()

            # Get pixel coordinates
            pos = event.pos()
            # pos = self.widget1_plot.vb.mapSceneToView(pos)
            pos = self.v_map.mapFromScene(pos)
            # print(pos.x(), pos.y())
            x_pixel, y_pixel = int(np.floor(pos.x() + 1)), int(np.floor(pos.y()))

            if self.mask[y_pixel, x_pixel]:
                # Plot new data
                # self.widget1_plot.setLabel('top', 'v={:.0f}'.format(self.v_fit[y_pixel, x_pixel]))
                # self.widget2_plot.setLabel('top', 'sigma={:.0f}'.format(self.sigma_fit[y_pixel, x_pixel]))
                self.param['v='] = '{:.0f}'.format(self.v_fit[y_pixel, x_pixel])
                self.param['sigma='] = '{:.0f}'.format(self.sigma_fit[y_pixel, x_pixel])
                self.param['flux='] = '{:.2f}'.format(self.flux_fit[y_pixel, x_pixel])

                # Plot spectrum
                self.widget3_plot.plot(self.v_array, self.flux[:, y_pixel, x_pixel], pen='w')
                self.widget3_plot.plot(self.v_array, self.flux_fit_array[:, y_pixel, x_pixel], pen='r')
                self.widget3_plot.setLabel('top', 'x={}, y={}'.format(x_pixel, y_pixel))
                self.widget3_plot.addItem(pg.InfiniteLine(self.v_fit[y_pixel, x_pixel],
                                                          pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine),
                                                          label=None,
                                                         labelOpts={'position': 0.8, 'rotateAxis': [1, 0]}))
                self.widget3_plot.addItem(pg.InfiniteLine(self.v_fit[y_pixel, x_pixel] + self.sigma_fit[y_pixel, x_pixel],
                                                          pen=pg.mkPen('b', width=2, style=QtCore.Qt.DashLine),
                                                          label=None,
                                                          labelOpts={'position': 0.8, 'rotateAxis': [1, 0]}))
                self.widget3_plot.addItem(pg.InfiniteLine(self.v_fit[y_pixel, x_pixel] - self.sigma_fit[y_pixel, x_pixel],
                                                          pen=pg.mkPen('b', width=2, style=QtCore.Qt.DashLine),
                                                          label=None,
                                                          labelOpts={'position': 0.8, 'rotateAxis': [1, 0]}))

    def update_fit(self):
        print('U')
        hdul_fit = fits.open(self.path_fit)
        self.v_fit, self.sigma_fit, flux_fit = hdul_fit[1].data, hdul_fit[3].data, hdul_fit[5].data
        self.flux_fit_array = Gaussian(self.v_array[:, np.newaxis, np.newaxis], self.v_fit, self.sigma_fit, flux_fit)

        # Refite that specific pixel
        self.parameters['v'].value = self.v_Serra[i, j]
        self.parameters['sigma'].value = self.sigma_Serra[i, j]

        flux_ij = self.flux[:, i, j]
        spec_model = lmfit.Model(self.model, missing='drop')
        result = spec_model.fit(flux_ij, velocity=self.v_array, params=self.parameters)

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







if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotWindow(gal_name='NGC3945')
    window.show()
    sys.exit(app.exec_())
