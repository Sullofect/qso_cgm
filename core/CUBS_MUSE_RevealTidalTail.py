import os
import glob
import aplpy
import coord
import shutil
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import rc
from astropy import stats
from scipy import interpolate
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.wcs import WCS
from mpdaf.obj import Cube, WaveCoord, Image
from astropy.coordinates import SkyCoord
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12

def RevealTidalTail(cubename):
    path_muse_zap = '../../MUSEQuBES+CUBS/CUBS/DATACUBE-RXSJ02282-4057-v01-PROPVAR-ZAP.fits'
    path_muse = '../../MUSEQuBES+CUBS/CUBS/DATACUBE-RXSJ02282-4057-v01-PROPVAR.fits'
    hdr = fits.getheader(path_muse, ext=1)

    # MUSE cube
    cube = Cube(path_muse)
    wave_vac = cube.wave.coord()  # Already in vacuum wavelength
    flux = cube.data  # Keep the initial unit 1e-20 erg/s/cm2/Ang
    median = np.nanmedian(ma.filled(flux, np.nan), axis=0)
    # kernel = Gaussian2DKernel(x_stddev=3, y_stddev=3)
    # kernel_1 = Kernel(kernel.array[np.newaxis, :, :])
    # median = convolve(median, kernel, boundary='extend')

    # Filter
    F814W = '../../pyobs/data/kcorrect/filters/ACS_F814W.fits'
    F814W = fits.open(F814W)[1].data
    wave, trans = F814W['wave'], F814W['transmission']
    wave_mask = (wave > wave_vac[0]) & (wave < wave_vac[-1])
    wave = wave[wave_mask]
    trans = trans[wave_mask]
    f = interpolate.interp1d(wave, trans, kind='linear', bounds_error=False)

    #     # Save result with gaia header
    path_muse_tidaltail = '../../MUSEQuBES+CUBS/CUBS/{}_TidalTail.fits'.format(cubename)
    # i_band = np.nansum(flux * f(wave_vac)[:, np.newaxis, np.newaxis], axis=0) / np.nansum(f(wave_vac))
    # median = np.nanmedian(flux, axis=0)
    # median = np.nanmedian(ma.filled(flux, np.nan), axis=0)
    hdr['NAXIS'] = 2
    del hdr['NAXIS3']
    del hdr['CUNIT3']
    del hdr['CTYPE3']
    del hdr['CD3_3']
    del hdr['CRPIX3']
    del hdr['CRVAL3']
    del hdr['CD1_3']
    del hdr['CD2_3']
    del hdr['CD3_1']
    del hdr['CD3_2']
    hdul = fits.ImageHDU(median, header=hdr)
    hdul.writeto(path_muse_tidaltail, overwrite=True)


def Plot(cubename):
    path_muse_tidaltail = '../../MUSEQuBES+CUBS/CUBS/{}_TidalTail.fits'.format(cubename)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_muse_tidaltail, figure=fig, hdu=1)
    gc.show_colorscale(cmap=plt.get_cmap('Greys'), vmin=-0.1, vmax=0.7, vmid=0.15, stretch='arcsinh')
    gc.recenter(37.063497, -40.954101, width=45 / 3600, height=23 / 3600)
    gc.set_system_latex(True)

    # Colorbar
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.ticks.set_length(30)
    gc.savefig('../../MUSEQuBES+CUBS/CUBS/HE0226-4110_TidalTail.png')
# Reveal tidal tail for HE0226-4110
# RevealTidalTail(cubename='HE0226-4110')
Plot(cubename='HE0226-4110')
