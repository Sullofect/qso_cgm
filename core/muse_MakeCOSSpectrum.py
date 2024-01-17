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


# Spectrum
path_cos = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/COS/HE0226-4110_COS_FUV_wavecal.fits'
data_cos = fits.getdata(path_cos, 1, ignore_missing_end=True)
wave = data_cos['wave']
flux = data_cos['flux']


plt.figure(figsize=(16, 4), dpi=300)
plt.plot(wave[::5], flux[::5] / 1e-14, 'k-', lw=0.5)
plt.xlabel(r'Observed wavelength [$\rm \AA$]', size=20)
plt.ylabel(r'${f}_{\lambda} \; [10^{-14} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}}]$', size=20, x=0.03)
plt.title('HE0226-4110 COS', size=20, x=0.2, y=0.8)
plt.xlim(1150, 1800)
plt.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/COS/HE0226-4110_COS_FUV_wavecal.pdf', bbox_inches='tight')
