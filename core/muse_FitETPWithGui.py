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

def Gaussian(velocity, v, sigma, flux):
    peak = flux / np.sqrt(2 * sigma ** 2 * np.pi)
    gaussian = peak * np.exp(-(velocity - v) ** 2 / 2 / sigma ** 2)

    return gaussian

#
path_table_gals = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/table_gals.fits'


class PlotWindow(QMainWindow):
    def __init__(self, gal_name='NGC5582'):
        super().__init__()

        gal_name = gal_name
        if gal_name == 'NGC2594':
            gal_cube = 'NGC2592'
        elif gal_name == 'NGC3619':
            gal_cube = 'NGC3613'
        else:
            gal_cube = gal_name

        # Load the data
        table_gals = fits.open(path_table_gals)[1].data
        gal_name_ = gal_name.replace('C', 'C ')
        name_sort = table_gals['Object Name'] == gal_name_
        ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
        v_sys_gal = table_gals[name_sort]['cz (Velocity)']

        # Load data
        path_Serra = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(gal_cube)
        path_cube = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/allcubes/{}_cube.fits'.format(gal_cube)
        hdul_Serra = fits.open(path_Serra)
        self.v_Serra = hdul_Serra[0].data[0, :, :] - v_sys_gal

        # Load the cube
        hdul_cube = fits.open(path_cube)
        hdr_cube = hdul_cube[0].header
        flux = hdul_cube[0].data
        self.mask = ~np.isnan(self.v_Serra)
        flux_err = np.where(~self.mask[np.newaxis, :, :], flux, np.nan)
        self.flux_err = np.nanstd(flux_err, axis=(1, 2))[:, np.newaxis, np.newaxis] + 0.0004
        self.flux = np.where(self.mask[np.newaxis, :, :], flux, np.nan)
        self.v_array = np.arange(hdr_cube['CRVAL3'], hdr_cube['CRVAL3'] + flux.shape[0] * hdr_cube['CDELT3'],
                                 hdr_cube['CDELT3']) / 1e3 - v_sys_gal  # Convert from m/s to km/s, and shift to v_sys
        self.size = np.shape(flux)[1:]


        # Load ETG fit
        self.path_fit = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_fit/{}_fit.fits'.\
            format(gal_name)
        v_guess, sigma_guess, flux_guess = 0, 50, 0.5
        self.model = Gaussian
        self.parameters = lmfit.Parameters()
        self.parameters.add_many(('v', v_guess, True, -300, 300, None),
                                 ('sigma', sigma_guess, True, 5, 70, None),
                                 ('flux', flux_guess, True, 0, None, None))
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
            self.fit()
        hdul_fit = fits.open(self.path_fit)
        self.v_fit, self.sigma_fit, self.flux_fit = hdul_fit[1].data, hdul_fit[3].data, hdul_fit[5].data
        self.flux_fit_array = Gaussian(self.v_array[:, np.newaxis, np.newaxis], self.v_fit, self.sigma_fit, self.flux_fit)
        self.v_fit = np.where(self.mask, self.v_fit, np.nan)
        self.sigma_fit = np.where(self.mask, self.sigma_fit, np.nan)

        # Calculate chi-square
        chi2 = ((self.flux - self.flux_fit_array) / self.flux_err) ** 2
        chi2 = np.where((self.v_array[:, np.newaxis, np.newaxis] > self.v_fit - 4 * self.sigma_fit)
                        * (self.v_array[:, np.newaxis, np.newaxis] < self.v_fit + 4 * self.sigma_fit), chi2, np.nan)
        self.chi_fit = np.nansum(chi2, axis=0)
        self.chi_fit = np.where(self.mask, self.chi_fit, np.nan)


        # Define a top-level widget
        self.widget = QWidget()
        self.widget.resize(2000, 2000)
        self.setCentralWidget(self.widget)
        self.layout = QtGui.QGridLayout()
        self.widget.setLayout(self.layout)
        # self.setStyleSheet("background-color: rgb(235, 233, 221);")
        # self.setStyleSheet("background-color: white;")

        # Set title
        self.setWindowTitle("Check fitting")

        # Create plot widgets
        self.widget1 = pg.GraphicsLayoutWidget()
        self.widget2 = pg.GraphicsLayoutWidget()
        self.widget3 = pg.GraphicsLayoutWidget()
        self.widget4 = pg.GraphicsLayoutWidget()
        self.widget1_plot = self.widget1.addPlot()
        self.widget2_plot = self.widget2.addPlot()
        self.widget3_plot = self.widget3.addPlot()
        self.widget4_plot = self.widget4.addPlot()
        self.widget1.setFixedSize(450, 450)
        self.widget2.setFixedSize(450, 450)
        self.widget3.setFixedSize(450, 450)
        self.widget4.setFixedSize(900, 450)

        # Set background color
        self.widget1.setBackground((235, 233, 221, 100))
        self.widget2.setBackground((235, 233, 221, 100))
        self.widget3.setBackground((235, 233, 221, 100))
        self.widget4.setBackground((235, 233, 221, 100))
        self.widget1_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget2_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget3_plot.setLimits(xMin=0, xMax=self.size[0], yMin=0, yMax=self.size[1])
        self.widget4_plot.setLimits(xMin=-1000, xMax=1000)

        # Set param
        self.paramSpec = [dict(name='v=', type='float', value=None, dec=False, readonly=False),
                          dict(name='sigma=', type='float', value=None, readonly=False),
                          dict(name='flux=', type='float', value=None, readonly=False),
                          dict(name='Re-fit', type='action'),
                          dict(name='chi=', type='float', value=None, dec=False, readonly=False)]
        self.param = pt.Parameter.create(name='Options', type='group', children=self.paramSpec)
        self.tree = pt.ParameterTree()
        self.tree.setParameters(self.param)
        self.param.children()[3].sigStateChanged.connect(self.update_fit)

        #
        self.layout.addWidget(self.widget1, 0, 0, 1, 1)
        self.layout.addWidget(self.widget2, 0, 1, 1, 1)
        self.layout.addWidget(self.widget3, 0, 2, 1, 1)
        self.layout.addWidget(self.widget4, 1, 0, 1, 2)
        self.layout.addWidget(self.tree, 1, 2, 1, 1)


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
        lut = (colormap._lut * 255).view(np.ndarray)
        self.sigma_map.setLookupTable(lut)
        self.sigma_map.updateImage(image=self.sigma_fit.T, levels=(0, 300))

        # Plot the chi 2D map in the third plot
        self.chi_map = pg.ImageItem()
        self.widget3_plot.addItem(self.chi_map)
        colormap = cm.get_cmap("viridis")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.chi_map.setLookupTable(lut)
        self.chi_map.updateImage(image=self.chi_fit.T, levels=(0, 50))


        self.widget1_plot.setXLink(self.widget2_plot)
        self.widget1_plot.setYLink(self.widget2_plot)
        self.widget2_plot.setXLink(self.widget1_plot)
        self.widget2_plot.setYLink(self.widget1_plot)
        self.widget3_plot.setXLink(self.widget2_plot)
        self.widget3_plot.setYLink(self.widget2_plot)

        # Plot initial data in the second plot
        self.widget4_plot.setLabel('bottom', 'Velocity (km/s)')
        self.widget4_plot.setLabel('left', 'Flux')

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
            self.plot()

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




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotWindow(gal_name='NGC2685')
    window.show()
    sys.exit(app.exec_())
