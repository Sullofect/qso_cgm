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
from regions import PixCoord
from regions import RectangleSkyRegion, RectanglePixelRegion
from astropy.coordinates import Angle
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


cubename = '3C57'

# QSO information
path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

#
path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
hdul = fits.open(path_fit)
fs, hdr = hdul[1].data, hdul[2].header
v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
sigma, dsigma = hdul[5].data, hdul[6].data
x, y = np.meshgrid(np.arange(v.shape[1]), np.arange(v.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

#
str_zap = ''
path_sub_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
hdr['CD2_2'] = hdr_sub_gaia['CD2_2']

w = WCS(hdr, naxis=2)
center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
c2 = w.world_to_pixel(center_qso)
# region_sky = RectangleSkyRegion(center=center_qso, width=3 * u.deg, height=4 * u.deg, angle=5 * u.deg)
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=30, height=1, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0])**2 + (y - c2[1])**2) * 0.2

fig, ax = plt.subplots(1, 1)
plt.imshow(v[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
plt.plot(c2[0], c2[1], '*', markersize=15)
# plt.show()


fig, ax = plt.subplots(2, 1, figsize=(5, 10))
fig.tight_layout()
ax[0].plot(v[0, :, :].flatten()[mask], dis[mask], 'k.')
ax[1].plot(v[1, :, :].flatten()[mask], dis[mask], 'k.')
# ax[2].plot(v[0, :, :].flatten(), dis, 'k.')
ax[0].set_ylim(3, 0)
ax[1].set_ylim(3, 0)
# ax[2].set_ylim(5, 0)
ax[0].set_xlabel('Velocity (km/s)')
ax[0].set_ylabel('Distance (pixel)')
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_velocity_profile.png', bbox_inches='tight')
# plt.show()
