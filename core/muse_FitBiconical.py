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
rc('xtick.minor', size=3, visible=True)
rc('ytick.minor', size=3, visible=True)
rc('xtick', direction='in', labelsize=15)
rc('ytick', direction='in', labelsize=15)
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# QSO information
cubename = '3C57'
path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

#
path_fit_N1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N1.fits'
path_fit_N2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N2.fits'
path_fit_N3 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N3.fits'

# N1
hdul_N1 = fits.open(path_fit_N1)
fs_N1, hdr_N1 = hdul_N1[1].data, hdul_N1[2].header
v_N1, z_N1, dz_N1 = hdul_N1[2].data, hdul_N1[3].data, hdul_N1[4].data
sigma_N1, dsigma_N1 = hdul_N1[5].data, hdul_N1[6].data
flux_OIII_N1 = hdul_N1[9].data

# N2
hdul_N2 = fits.open(path_fit_N2)
fs_N2, hdr_N2 = hdul_N2[1].data, hdul_N2[2].header
v_N2, z_N2, dz_N2 = hdul_N2[2].data, hdul_N2[3].data, hdul_N2[4].data
sigma_N2, dsigma_N2 = hdul_N2[5].data, hdul_N2[6].data
flux_OIII_N2 = hdul_N2[9].data

# N3
hdul_N3 = fits.open(path_fit_N3)
fs_N3, hdr_N3 = hdul_N3[1].data, hdul_N3[2].header
v_N3, z_N3, dz_N3 = hdul_N3[2].data, hdul_N3[3].data, hdul_N3[4].data
sigma_N3, dsigma_N3 = hdul_N3[5].data, hdul_N3[6].data
flux_OIII_N3 = hdul_N3[9].data


# plot the velocity field
x, y = np.meshgrid(np.arange(v_N1.shape[1]), np.arange(v_N1.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

#
hdr = hdr_N1
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
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=2, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0])**2 + (y - c2[1])**2) * 0.2 * 50 / 7
dis_mask = dis[mask]

#
red = ((x[mask] - c2[0]) < 0) * ((y[mask] - c2[1]) > 0)
blue = ~red
dis_red = dis_mask[red]
dis_blue = dis_mask[blue] * -1

# Slit position
fig, ax = plt.subplots(1, 1)
plt.imshow(v_N1[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
plt.plot(c2[0], c2[1], '*', markersize=15)
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_sudo_slit.png', bbox_inches='tight')


# Position-velocity diagram
i = 70
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)

# NFW profile
h = 0.7
G = 6.674e-8
M_sun = 1.989e33
M_halo = 10 ** 13 * M_sun
rho_c = 1.8788e-29 * h ** 2
kpc = 3.086e21
c = 15
x = np.linspace(0, 0.2, 1000)
R_vir = (3 * M_halo / 4 / np.pi / 200 / rho_c) ** (1 / 3)
f_cx = np.log(1 + c * x) - c * x / (1 + c * x)
f_c = np.log(1 + c) - c / (1 + c)
v_vir = np.sqrt(G * M_halo / R_vir) / 1e5
v_c = v_vir * np.sqrt(f_cx / x / f_c) * np.sin(i * np.pi / 180)
ax.plot(x * R_vir / kpc, v_c, '-r', label='NFW profile with i={} deg'.format(i))
ax.plot(- x * R_vir / kpc, -v_c, '-r')


# Velocity profile
def v(index=2, v_max=800, i_deg=5, R_max=30):
    #
    # theta = np.arcsin(d / R)
    # P1 = - 2 * d * (np.tan(i_rad) * np.cos(theta_rad) - np.tan(i_rad) ** 2 * np.sin(theta_rad))
    # P2 = np.sqrt(4 * d ** 2 * (np.tan(i_rad) ** 2 * np.cos(theta_rad) ** 2 + np.tan(i_rad) * np.sin(2 * theta_rad) + np.sin(theta_rad) ** 2))
    # P3 = 2 * (np.sin(theta_rad) ** 2 * (np.tan(i_rad) ** 2 - 1) - np.tan(i_rad)* np.sin(2 * theta_rad))
    # R = - 2 * d * (np.tan(i_deg * np.pi / 180) + np.tan(theta))
    # r = d / np.sin(i_deg * np.pi / 180)

    #
    R = np.arange(0, R_max + 0.1, 0.1)
    i_rad = i_deg * np.pi / 180
    theta_rad = np.arccos((R / 2) / (R_max / 2))
    v_3D = v_max * (R / R_max) ** index

    #
    d_minus = R * np.sin(np.pi / 2 - theta_rad - i_rad)
    d_plus = R * np.sin(np.pi / 2 + theta_rad - i_rad)

    #
    v_minus = -1 * v_3D * np.cos(np.pi / 2 - theta_rad - i_rad)
    v_plus = -1 * v_3D * np.cos(np.pi / 2 + theta_rad - i_rad)
    return d_minus, d_plus, v_minus, v_plus

#
ax.plot(v()[0], v()[2], '-', linewidth=1, color='k')
ax.plot(v()[1], v()[3], '-', linewidth=1, color='k')


# Components
v_N1_flatten = v_N1[0, :, :].flatten()
ax.errorbar(dis_blue, v_N1_flatten[mask][blue], np.zeros_like(dis[mask][blue]),
            fmt='.k', capsize=0, elinewidth=0.7, mfc='C0', ms=8, markeredgewidth=0.7)
# ax.plot(dis[mask][red], v_N1_flatten[mask][red], '.k')

#
v_N2_flatten_C1 = v_N2[0, :, :].flatten()
v_N2_flatten_C2 = v_N2[1, :, :].flatten()
flux_OIII_N2_flatten_C1 = flux_OIII_N2[0, :, :].flatten()
flux_OIII_N2_flatten_C2 = flux_OIII_N2[1, :, :].flatten()
v_N2_flatten_weight = (v_N2_flatten_C1 * flux_OIII_N2_flatten_C1
                       + v_N2_flatten_C2 * flux_OIII_N2_flatten_C2) / (flux_OIII_N2_flatten_C1 + flux_OIII_N2_flatten_C2)
ax.plot(dis_red, v_N2_flatten_C1[mask][red], '.r')
ax.plot(dis_red, v_N2_flatten_C2[mask][red], '.r')
# ax.plot(dis[mask][red], v_N2_flatten_weight[mask][red], '.b', ms=10)
ax.errorbar(dis[mask][red], v_N2_flatten_weight[mask][red], np.zeros_like(dis[mask][red]),
            fmt='.k', capsize=0, elinewidth=0.7, mfc='C1', ms=8, markeredgewidth=0.7)

#
v_N3_flatten_C1 = v_N3[0, :, :].flatten()
v_N3_flatten_C2 = v_N3[1, :, :].flatten()
v_N3_flatten_C3 = v_N3[2, :, :].flatten()
flux_OIII_N3_flatten_C1 = flux_OIII_N3[0, :, :].flatten()
flux_OIII_N3_flatten_C2 = flux_OIII_N3[1, :, :].flatten()
flux_OIII_N3_flatten_C3 = flux_OIII_N3[2, :, :].flatten()
v_N3_flatten_weight = (v_N3_flatten_C1 * flux_OIII_N3_flatten_C1 + v_N3_flatten_C2 * flux_OIII_N3_flatten_C2
                       + v_N3_flatten_C3 * flux_OIII_N3_flatten_C3) \
                      / (flux_OIII_N3_flatten_C1 + flux_OIII_N3_flatten_C2 + flux_OIII_N3_flatten_C3)
# ax.plot(v_N3[0, :, :].flatten()[mask], dis[mask], '.C2')
# ax.plot(v_N3[1, :, :].flatten()[mask], dis[mask], '.C2')
# ax.plot(v_N3[2, :, :].flatten()[mask], dis[mask], '.C2')
# ax.plot(v_N3_flatten_weight[mask][red], dis[mask][red], '.', color='C6', ms=5)

ax.axhline(0, linestyle='--', color='k', linewidth=1)
ax.axvline(0, linestyle='--', color='k', linewidth=1)
ax.set_xlim(-40, 40)
ax.set_ylim(-450, 450)
ax.set_xlabel(r'$\rm Distance \, [kpc]$', size=20)
ax.set_ylabel(r'$\Delta v \rm \, [km \, s^{-1}]$', size=20)
ax.legend(loc=4, fontsize=15)
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_velocity_profile.png', bbox_inches='tight')
# plt.show()


# Biconical outflow model
# Model Parameters
A = 0.00 # dust extinction level (0.0 - 1.0)
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
vtype = 'increasing'  # 'increasing','decreasing', or 'constant'
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
xbgrid, ybgrid, zbgrid, fgrid, vgrid = bicone.generate_bicone(theta_in_deg, theta_out_deg, theta_B1_deg, theta_B2_deg,
                                                              theta_B3_deg, theta_D1_deg, theta_D2_deg, theta_D3_deg,
                                                              D=D, tau=tau, fn=fn, A=A, vmax=vmax, vtype=vtype,
                                                              sampling=sampling, plot=False, orientation=(azim, elev),
                                                              save_fig=False)

fmap, vmap, dmap, v_int, d_int = bicone.map_2d(xbgrid, ybgrid, zbgrid, fgrid, vgrid,
                                               D=D, sampling=sampling, interpolation=map_interpolation,
                                               plot=True, save_fig=True)





# # VVD
# v1, v2 = v[0, :, :], v[1, :, :]
# sigma1, sigma2 = sigma[0, :, :], sigma[1, :, :]
# v1 = np.where(v2 != 0, v1, 0)
# sigma1 = np.where(sigma2 != 0, sigma1, 0)
#
# #
# fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
# # ax.plot(v[0, :, :].flatten(), sigma[0, :, :].flatten(), 'k.')
# ax.plot(v1.flatten(), sigma1.flatten(), 'k.', ms=3)
# ax.plot(v2.flatten(), sigma2.flatten(), 'r.', ms=3)
# ax.set_xlim(-500, 500)
# ax.set_ylim(0, 500)
# ax.set_xlabel('V (km/s)')
# ax.set_ylabel(r'$\sigma$ (km/s)')
# fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_VVD_profile.png', bbox_inches='tight')