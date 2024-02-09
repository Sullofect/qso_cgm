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
import biconical_outflow_model_3d as bicone
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# QSO information
cubename = '3C57'
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

# plot the velocity field
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
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=35, height=1, angle=Angle(-33, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0])**2 + (y - c2[1])**2) * 0.2 * 50 / 7

fig, ax = plt.subplots(1, 1)
plt.imshow(v[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
plt.plot(c2[0], c2[1], '*', markersize=15)
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_sudo_slit.png', bbox_inches='tight')


# NFW profile
# constants
h = 0.7
G = 6.674e-8
M_sun = 1.989e33
M_halo = 10 ** 13 * M_sun
rho_c = 1.8788e-29 * h ** 2
kpc = 3.086e21
c = 20

#
x = np.linspace(0, 0.2, 1000)
R_vir = (3 * M_halo / 4 / np.pi / 200 / rho_c) ** (1 / 3)
f_cx = np.log(1 + c * x) - c * x / (1 + c * x)
f_c = np.log(1 + c) - c / (1 + c)
v_vir = np.sqrt(G * M_halo / R_vir) / 1e5
v_c = v_vir * np.sqrt(f_cx / x / f_c)

fig, ax = plt.subplots(1, 1)
ax.plot(dis[mask], np.abs(v[0, :, :].flatten()[mask]), 'k.')
ax.plot(dis[mask], np.abs(v[1, :, :].flatten()[mask]), 'b.')
ax.plot(x * R_vir / kpc, v_c, '-r')
ax.set_xlabel('R (kpc)')
ax.set_ylabel('V (km/s)')
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_NFW_profile.png', bbox_inches='tight')

# VVD
v1, v2 = v[0, :, :], v[1, :, :]
sigma1, sigma2 = sigma[0, :, :], sigma[1, :, :]
v1 = np.where(np.abs(v1) > 500, np.nan, v1)

fig, ax = plt.subplots(1, 1)
# ax.plot(v[0, :, :].flatten(), sigma[0, :, :].flatten(), 'k.')
ax.plot(v[0, :, :].flatten(), sigma[0, :, :].flatten(), 'k.')
ax.plot(v[1, :, :].flatten(), sigma[1, :, :].flatten(), 'r.')
ax.set_xlim(-500, 500)
ax.set_ylim(0, 500)
ax.set_xlabel('V (km/s)')
ax.set_ylabel(r'$\sigma$ (km/s)')
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_VVD_profile.png', bbox_inches='tight')

# Biconical outflow model
# Model Parameters
A = 0.90 # dust extinction level (0.0 - 1.0)
tau = 5.00  # shape of flux profile
D = 1.0  # length of bicone (arbitrary units)
fn = 1.0e3  # initial flux value at center

theta_in_deg = 20.0     # inner opening angle (degrees)
theta_out_deg = 40.0    # outer opening angle (degrees)

# Bicone inclination and PA
theta_B1_deg = 60.0  # rotation along x
theta_B2_deg = 45.0     # rotation along y
theta_B3_deg = 45.0     # rotation along z
# Dust plane inclination and PA
theta_D1_deg = 135.0    # rotation along x
theta_D2_deg = 0.0     # rotation along y
theta_D3_deg = 0.0     # rotation along z

# Velocity profile parameters
vmax = 1000.0  # km/s
vtype = 'decreasing'  # 'increasing','decreasing', or 'constant'
# vtype = 'constant'
# vtype = 'increasing'

# Sampling paramters
sampling = 100  # point sampling
# 3d Plot orientation
azim = 45
elev = 15

# 2d Map options
map_interpolation = 'none'
# emission model options
obs_res = 68.9  # resolution of SDSS for emission line model
nbins = 60  # number of bins for emission line histogram

# Bicone coordinate, flux, and velocity grids
# xbgrid,ybgrid,zbgrid,fgrid,vgrid = bicone.generate_bicone(theta_in_deg, theta_out_deg, theta_B1_deg, theta_B2_deg,
#                                                           theta_B3_deg, theta_D1_deg, theta_D2_deg, theta_D3_deg,
#                                                           D=D, tau=tau, fn=fn, A=A, vmax=vmax, vtype=vtype,
#                                                           sampling=sampling, plot=False, orientation=(azim, elev),
#                                                           save_fig=False)
#
# fmap, vmap, dmap,v_int,d_int = bicone.map_2d(xbgrid,ybgrid,zbgrid,fgrid,vgrid,
#                                              D=D,sampling=sampling,interpolation=map_interpolation,
#                                              plot=True,save_fig=True)





# Velocity profile
fig, ax = plt.subplots(2, 1, figsize=(5, 10))
fig.tight_layout()
ax[0].plot(v[0, :, :].flatten()[mask], dis[mask], 'k.')
ax[1].plot(v[1, :, :].flatten()[mask], dis[mask], 'k.')
# ax[2].plot(v[0, :, :].flatten(), dis, 'k.')
ax[0].set_ylim(5, 0)
ax[1].set_ylim(5, 0)
# ax[2].set_ylim(5, 0)
ax[0].set_xlabel('Velocity (km/s)')
ax[0].set_ylabel('Distance (pixel)')
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_velocity_profile.png', bbox_inches='tight')
# plt.show()
