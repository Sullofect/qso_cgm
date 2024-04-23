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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

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

# fitting starts
v_c_guess, sigma_guess, flux_guess = 0, 50, 0.5
parameters = lmfit.Parameters()
model = Gaussian
parameters.add_many(('v_c', v_c_guess, True, -300, 300, None),
                    ('sigma', sigma_guess, True, 0, 150, None),
                    ('flux', flux_guess, True, 0, None, None))
fit_success = np.zeros(size)
v_c_fit, dv_c_fit = np.zeros(size), np.zeros(size)
sigma_fit, dsigma_fit = np.zeros(size), np.zeros(size)
flux_fit, dflux_fit = np.zeros(size), np.zeros(size)

for i in range(size[0]):  # i = p (y), j = q (x)
    for j in range(size[1]):
        if mask[i, j]:
            parameters['v_c'].value = v_ETG[i, j]
            flux_ij = flux[:, i, j]
            spec_model = lmfit.Model(model, missing='drop')
            result = spec_model.fit(flux_ij, v=v_array, params=parameters)
            fit_success[i, j] = result.success

            v_c, dv_c = result.best_values['v_c'], result.params['v_c'].stderr
            sigma, dsigma = result.best_values['sigma'], result.params['sigma'].stderr
            flux_b, dflux_b = result.best_values['flux'], \
                          result.params['flux'].stderr

            # fill the value
            v_c_fit[i, j], dv_c_fit[i, j] = v_c, dv_c
            sigma_fit[i, j], dsigma_fit[i, j] = sigma, dsigma
            flux_fit[i, j], dflux_fit[i, j] = flux_b, dflux_b
        else:
            pass

# Save fitting results
path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_fit.fits'.format(gal_name)
hdul_fs = fits.PrimaryHDU(fit_success, header=hdr_ETG)
hdul_v_c, hdul_dv_c = fits.ImageHDU(v_c_fit, header=hdr_ETG), fits.ImageHDU(dv_c_fit, header=hdr_ETG)
hdul_sigma, hdul_dsigma = fits.ImageHDU(sigma_fit, header=hdr_ETG), fits.ImageHDU(dsigma_fit, header=hdr_ETG)
hdul_flux, hdul_dflux = fits.ImageHDU(flux_fit, header=hdr_ETG), fits.ImageHDU(dflux_fit, header=hdr_ETG)
hdul = fits.HDUList([hdul_fs, hdul_v_c, hdul_dv_c, hdul_sigma, hdul_dsigma, hdul_flux, hdul_dflux])
hdul.writeto(path_fit, overwrite=True)



