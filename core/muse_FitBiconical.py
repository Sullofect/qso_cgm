import os
# import aplpy
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
from regions import RectangleSkyRegion, RectanglePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
import biconical_outflow_model_3d as bicone
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from gala.units import galactic, solarsystem, dimensionless
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
str_zap = ''
path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

#
UseDetectionSeg = (1.5, 'gauss', 1.5, 'gauss')
UseSmoothedCubes = True
line_OII, line_OIII = 'OII', 'OIII'
path_3Dseg_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_3Dseg_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OIII, *UseDetectionSeg)
path_SB_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_SB_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OIII, *UseDetectionSeg)
path_cube_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
    format(cubename, str_zap, line_OII)
path_cube_smoothed_OII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                         '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_cube_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
    format(cubename, str_zap, line_OIII)
path_cube_smoothed_OIII = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                          '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDetectionSeg)

# Measure isophotal SB
hdul_SB_OII = fits.open(path_SB_OII)
hdul_SB_OIII = fits.open(path_SB_OIII)
data_SB_OII = hdul_SB_OII[1].data
data_SB_OIII = hdul_SB_OIII[1].data
kernel = Gaussian2DKernel(x_stddev=1.0, y_stddev=1.0)
data_SB_OII_smoothed = convolve(data_SB_OII, kernel)
data_SB_OIII_smoothed = convolve(data_SB_OIII, kernel)

from photutils.isophote import EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.isophote import Ellipse
geometry = EllipseGeometry(x0=72.5681, y0=74.3655, sma=20, eps=0.5,
                           pa=-25.0 * np.pi / 180.0, fix_pa=True)
ellipse_OII = Ellipse(data_SB_OII, geometry)
isolist_OII = ellipse_OII.fit_image()
geometry = EllipseGeometry(x0=72.5681, y0=74.3655, sma=10, eps=0.5,
                           pa=-25.0 * np.pi / 180.0, fix_pa=False, fix_center=False)
ellipse_OIII = Ellipse(data_SB_OIII_smoothed, geometry)
isolist_OIII = ellipse_OIII.fit_image()

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
ax[0].imshow(data_SB_OII, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=5)
ax[1].imshow(data_SB_OIII, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=5)
smas = np.linspace(10, 50, 10)[1:2]
for sma in smas:
    iso_OII = isolist_OII.get_closest(sma)
    x_OII, y_OII = iso_OII.sampled_coordinates()
    ax[0].plot(x_OII, y_OII, color='white')
    print(iso_OII.x0, iso_OII.y0)
    f, sma = iso_OII.eps, iso_OII.sma
    smia = sma * (1 - f)
    print(sma, smia)
    i = (np.pi / 2 - np.arcsin(smia / sma)) * 180 / np.pi
    print('inclination angle is', i)

smas = np.linspace(5, 10, 3)[2:]
for sma in smas:
    iso_OIII = isolist_OIII.get_closest(sma)
    x_OIII, y_OIII, = iso_OIII.sampled_coordinates()
    ax[1].plot(x_OIII, y_OIII, color='white')
    print(iso_OII.x0, iso_OII.y0)
    f, sma = iso_OIII.eps, iso_OIII.sma
    smia = sma * (1 - f)
    print(sma, smia)
    i = (np.pi / 2 - np.arcsin(smia / sma)) * 180 / np.pi
    print('inclination angle is', i)
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_SB_OII_isophote.png', bbox_inches='tight')
# raise ValueError('stop')

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
chisqr_N1, redchi_N1 = hdul_N1[15].data, hdul_N1[16].data

# N2
hdul_N2 = fits.open(path_fit_N2)
fs_N2, hdr_N2 = hdul_N2[1].data, hdul_N2[2].header
v_N2, z_N2, dz_N2 = hdul_N2[2].data, hdul_N2[3].data, hdul_N2[4].data
sigma_N2, dsigma_N2 = hdul_N2[5].data, hdul_N2[6].data
flux_OIII_N2 = hdul_N2[9].data
chisqr_N2, redchi_N2 = hdul_N2[15].data, hdul_N2[16].data

# N3
hdul_N3 = fits.open(path_fit_N3)
fs_N3, hdr_N3 = hdul_N3[1].data, hdul_N3[2].header
v_N3, z_N3, dz_N3 = hdul_N3[2].data, hdul_N3[3].data, hdul_N3[4].data
sigma_N3, dsigma_N3 = hdul_N3[5].data, hdul_N3[6].data
flux_OIII_N3 = hdul_N3[9].data
chisqr_N3, redchi_N3 = hdul_N3[15].data, hdul_N3[16].data


# plot the velocity field
x, y = np.meshgrid(np.arange(v_N1.shape[1]), np.arange(v_N1.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

#
hdr = hdr_N1
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
print(c2)
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
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

# Milky Way rotation disk
MW = gp.MilkyWayPotential(units=galactic)
R = np.zeros((3, 100))
R[0, :] = np.linspace(0, 20, 100)
MW_v = MW.circular_velocity(q=R * u.kpc)
ax.plot(R[0, :], MW_v * np.sin(i * np.pi / 180), 'k--', lw=1, label='MW disk')
ax.plot(-R[0, :], -MW_v * np.sin(i * np.pi / 180), 'k--', lw=1)

# Sombero galaxy
d_som = 11.219  # Mpc
# r_som_arcsec = np.array([0.15, 0.30, 0.52, 0.82, 1.2, 1.7, 2.4, 3.3, 4.57, 6.45, 8.77,
#                          11.4, 14.32, 17.7, 22.2, 28.58, 36.08, 44.4])
# r_som_kpc = r_som_arcsec * d_som * 1e3 * np.pi / 180 / 3600
# v_som = -1 * np.array([-36, -75, -112, -144, -172, -190, -208, -232, -236, -235, -189,
#                        -171, -187, -201, -228, -235, -285, -277])
r_som_kpc = np.array([0, 1.25, 2.5, 5, 10, 15])
v_som = np.array([0, 120, 230, 305, 332, 343])
ax.plot(r_som_kpc, v_som * np.sin(i * np.pi / 180), '--', color='red', lw=1, label='Sombrero')
ax.plot(-r_som_kpc, -v_som * np.sin(i * np.pi / 180), '--', color='red', lw=1)

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
# ax.plot(x * R_vir / kpc, v_c, '-r', label='NFW profile with i={} deg'.format(i))
# ax.plot(- x * R_vir / kpc, -v_c, '-r')


# Velocity profile
def v(index=2, v_max=800, i_deg=70, R_max=30):
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
# ax.plot(v()[0], v()[2], '-', linewidth=1, color='k')
# ax.plot(v()[1], v()[3], '-', linewidth=1, color='k')

def Bin(x, y, std, flux, bins=20):
    n, edges = np.histogram(x, bins=bins)
    y_mean = np.zeros(bins)
    y_std = np.zeros(bins)
    y_flux = np.zeros(bins)
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i in range(bins):
        mask = (x > edges[i]) * (x < edges[i + 1])
        y_mean[i] = y[mask].mean()
        y_std[i] = std[mask].mean()
        y_flux[i] = flux[mask].mean()
    return x_mean, y_mean, y_std, y_flux

# Components
v_N1_flatten = v_N1[0, :, :].flatten()
sigma_N1_flatten = sigma_N1[0, :, :].flatten()
flux_OIII_N1_flatten = flux_OIII_N1[0, :, :].flatten()
dis_blue_mean, v_N1_mean, sigma_N1_mean, flux_N1_mean = Bin(dis_blue, v_N1_flatten[mask][blue],
                                                            sigma_N1_flatten[mask][blue],
                                                            flux_OIII_N1_flatten[mask][blue], bins=20)
# ax.errorbar(dis_blue, v_N1_flatten[mask][blue], sigma_N1_flatten[mask][blue],
#             fmt='.k', capsize=1.5, elinewidth=0.7, mfc='C0', ms=8, markeredgewidth=0.7)
ax.errorbar(dis_blue_mean, v_N1_mean, sigma_N1_mean, fmt='.k', capsize=1.5, elinewidth=0.7, mfc='C0', ms=8,
            markeredgewidth=0.7)
# ax.plot(dis[mask][red], v_N1_flatten[mask][red], '.k')
print(v_N1_mean, flux_N1_mean)
v_array = np.linspace(-500, 500, 100)
# peak = flux_N1_mean / np.sqrt(2 * sigma_N1_mean ** 2 * np.pi)
# print(peak)
prob = np.exp(-(v_array[np.newaxis, :] - v_N1_mean[:, np.newaxis]) ** 2 / 2 / sigma_N1_mean[:, np.newaxis] ** 2)
print(prob[5, :])
prob = prob / np.nansum(prob, axis=1)[:, np.newaxis]
# prob = prob.T

# model = np.random.choice(v_array, size=(len(v_array), 20), p=prob)
# print(model[0, :])

for i in range(20):
    print(i, dis_blue_mean[i])
    prob_i = prob[i, :]
    if len(prob_i[np.isnan(prob_i)]) != 100:
        model = np.random.choice(v_array, size=len(v_array), p=prob_i)
        violin_parts_0 = ax.violinplot(model, positions=np.asarray([dis_blue_mean[i]]), points=50, widths=1.5, showmeans=False,
                                       showextrema=False, showmedians=False)
        violin_parts_0['bodies'][0].set_facecolor('C1')

#
v_N2_flatten_C1 = v_N2[0, :, :].flatten()
v_N2_flatten_C2 = v_N2[1, :, :].flatten()
v_N2_flatten_C1_sort = np.copy(v_N2_flatten_C1)
v_N2_flatten_C2_sort = np.copy(v_N2_flatten_C2)
sigma_N2_flatten_C1 = sigma_N2[0, :, :].flatten()
sigma_N2_flatten_C2 = sigma_N2[1, :, :].flatten()
sigma_N2_flatten_C1_sort = np.copy(sigma_N2_flatten_C1)
sigma_N2_flatten_C2_sort = np.copy(sigma_N2_flatten_C2)
flux_OIII_N2_flatten_C1 = flux_OIII_N2[0, :, :].flatten()
flux_OIII_N2_flatten_C2 = flux_OIII_N2[1, :, :].flatten()
flux_OIII_N2_flatten_C1_sort = np.copy(flux_OIII_N2_flatten_C1)
flux_OIII_N2_flatten_C2_sort = np.copy(flux_OIII_N2_flatten_C2)

#
v_sort = v_N2_flatten_C1 > v_N2_flatten_C2
v_N2_flatten_C1_sort[~v_sort] = v_N2_flatten_C2[~v_sort]
v_N2_flatten_C2_sort[~v_sort] = v_N2_flatten_C1[~v_sort]
sigma_N2_flatten_C1_sort[~v_sort] = sigma_N2_flatten_C2[~v_sort]
sigma_N2_flatten_C2_sort[~v_sort] = sigma_N2_flatten_C1[~v_sort]
flux_OIII_N2_flatten_C1_sort[~v_sort] = flux_OIII_N2_flatten_C2[~v_sort]
flux_OIII_N2_flatten_C2_sort[~v_sort] = flux_OIII_N2_flatten_C1[~v_sort]
v_N2_flatten_weight = (v_N2_flatten_C1_sort * flux_OIII_N2_flatten_C1_sort
                       + v_N2_flatten_C2_sort * flux_OIII_N2_flatten_C2_sort) / (flux_OIII_N2_flatten_C1_sort + flux_OIII_N2_flatten_C2_sort)
# ax.plot(dis_red, v_N2_flatten_C1[mask][red], '.r')
# ax.plot(dis_red, v_N2_flatten_C2[mask][red], '.r')
# dis_red_mean_C1, v_N2_mean_C1, sigma_N2_mean_C1 = Bin(dis_red, v_N2_flatten_C1_sort[mask][red],
#                                                       sigma_N2_flatten_C1[mask][red], bins=20)
# dis_red_mean_C2, v_N2_mean_C2, sigma_N2_mean_C2 = Bin(dis_red, v_N2_flatten_C2_sort[mask][red],
#                                                       sigma_N2_flatten_C2[mask][red], bins=20)
# ax.errorbar(dis_red, v_N2_flatten_C1[mask][red], sigma_N2_flatten_C1[mask][red],
#             fmt='.k', capsize=1.5, elinewidth=0.7, mfc='C1', ms=8, markeredgewidth=0.7)
# ax.errorbar(dis_red, v_N2_flatten_C2[mask][red], sigma_N2_flatten_C2[mask][red],
#             fmt='.k', capsize=1.5, elinewidth=0.7, mfc='C1', ms=8, markeredgewidth=0.7)
# ax.errorbar(dis_red_mean_C1, v_N2_mean_C1, sigma_N2_mean_C1, fmt='.k', capsize=1.5, elinewidth=0.7, mfc='C2', ms=8,
#             markeredgewidth=0.7)
# ax.errorbar(dis_red_mean_C2, v_N2_mean_C2, sigma_N2_mean_C2, fmt='.k', capsize=1.5, elinewidth=0.7, mfc='C1', ms=8,
#             markeredgewidth=0.7)
# ax.plot(dis[mask][red], v_N2_flatten_weight[mask][red], '.b', ms=10)
# ax.errorbar(dis[mask][red], v_N2_flatten_weight[mask][red], np.zeros_like(dis[mask][red]),
#             fmt='.k', capsize=0, elinewidth=0.7, mfc='C1', ms=8, markeredgewidth=0.7)

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


# raise ValueError('stop')

# Biconical outflow model
# Model Parameters
A = 0.00  # dust extinction level (0.0 - 1.0)
tau = 5.00  # shape of flux profile
D = 1.0  # length of bicone (arbitrary units)
fn = 1.0e3  # initial flux value at center

theta_in_deg = 10.0     # inner opening angle (degrees)
theta_out_deg = 50.0    # outer opening angle (degrees)

# Bicone inclination and PA
theta_B1_deg = 10   # rotation along x
theta_B2_deg = 60    # rotation along y
theta_B3_deg = 0     # rotation along z

# Dust plane inclination and PA
theta_D1_deg = 0.0    # rotation along x
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
                                                              sampling=sampling, plot=True, orientation=(azim, elev),
                                                              save_fig=True)

fmap, vmap, dmap, v_int, d_int = bicone.map_2d(xbgrid, ybgrid, zbgrid, fgrid, vgrid,
                                               D=D, sampling=sampling, interpolation=map_interpolation,
                                               plot=True, save_fig=True)


x, emline = bicone.emission_model(fgrid, vgrid, vmax=vmax, obs_res=obs_res, nbins=nbins, sampling=sampling,
                                  plot=True, save_fig=True)

# Compare with the data
# Load data and smoothing
if UseSmoothedCubes:
    cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
else:
    cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(
    cube_OIII.wave.coord())
flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori


# redchi_mean, redchi_median, redchi_std = stats.sigma_clipped_stats(redchi_N1[redchi_N1 != 0], sigma=3, maxiters=5)
# refit_seg = np.where((redchi_N1 > 1.0), redchi_N1, 0)
# v_N1_seg = refit_seg * v_N1[0, :, :]

# remap the vmap to the observed grid
coord_MUSE = (56, 83)

#
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
ax.imshow(v_N1[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
ax.imshow(np.flip(vmap, 1), cmap='coolwarm', extent=[c2[0] - 20, c2[0] + 20, c2[1] - 20, c2[1] + 20],
          vmin=-300, vmax=300, origin='lower')
ax.plot(c2[0], c2[1], '*', markersize=15)
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_sudo_cone.png', bbox_inches='tight')

def emission_pixel(fgrid, vgrid, vmax, coord_MUSE=None, nbins=25, sampling=100, z=z_qso):
    global c2
    # Having an odd sampling ensures that there is a value at (0,0,0)
    if int(sampling) % 2 == 0:
            sampling = int(sampling)+1

    X_sample, Y_sample = np.meshgrid(np.arange(sampling), np.arange(sampling))
    X_sample, Y_sample = X_sample.ravel(), Y_sample.ravel()
    pixcoord_sample = PixCoord(x=X_sample, y=Y_sample)

    #
    center_x, center_y = (sampling - 1) / 2, (sampling - 1) / 2
    pixel_scale = 40 / (sampling - 1)
    coord_sample = int(coord_MUSE[0] - c2[0]) / pixel_scale + center_x, \
                   int(coord_MUSE[1] - c2[1]) / pixel_scale + center_y
    rect_sample = RectanglePixelRegion(center=PixCoord(x=coord_sample[0], y=coord_sample[1]),
                                       width=1 / pixel_scale, height=1 / pixel_scale)
    mask_sample = rect_sample.contains(pixcoord_sample)

    # fig, ax = plt.subplots(1, 1)
    # # plt.imshow(v_N1[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
    # ax.imshow(np.flip(vmap, 1), cmap='coolwarm', vmin=-300, vmax=300, origin='lower')
    # patch = rect_sample.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
    # ax.plot(c2[0], c2[1], '*', markersize=15)
    # fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_cone_test.png', bbox_inches='tight')

    # Reshape grids into cubes
    mask_sample = mask_sample.reshape(sampling, sampling)
    fgrid = fgrid.reshape(sampling, sampling, sampling)
    vgrid = vgrid.reshape(sampling, sampling, sampling)
    bins = np.linspace(-vmax, vmax, nbins)

    # Convolve with seeing
    # kernel = Gaussian2DKernel(x_stddev=0.5 / pixel_scale, y_stddev=0.5 / pixel_scale,
    #                           x_size=9, y_size=9)
    # kernel_1 = Kernel(kernel.array[np.newaxis, :, :])
    # fgrid = convolve(fgrid, kernel_1)

    # find specific pixel
    v_xy = np.where(mask_sample[np.newaxis, :, :], np.flip(np.swapaxes(vgrid, 1, 2), 2), np.nan)
    f_xy = np.where(mask_sample[np.newaxis, :, :], np.flip(np.swapaxes(fgrid, 1, 2), 2), np.nan)

    # Remove NaNs
    v_xy = v_xy.ravel()
    f_xy = f_xy.ravel()
    v_xy = v_xy[f_xy > 0]
    f_xy = f_xy[f_xy > 0]
    print(v_xy, f_xy)

    v_hist, v_edges = np.histogram(v_xy, bins=bins, weights=f_xy)
    v_mid = (v_edges[1:] + v_edges[:-1]) / 2

    # convert x in km/s to angstroms
    c = 299792. # speed of light in km/s
    cw = 5008.239 * (1 + z)  # central wavelength; [OIII]5007 (SDSS)
    lambda_mid = cw + (v_mid * cw) / c
    return lambda_mid, v_hist / v_hist.max()


# Define the pixel coordinates
lambda_mid, v_hist = emission_pixel(fgrid, vgrid, coord_MUSE=coord_MUSE, vmax=vmax, nbins=nbins, sampling=sampling)


# plt.close('all')
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
ax.plot(wave_OIII_vac, flux_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max(),
        '-k', drawstyle='steps-mid')
ax.plot(wave_OIII_vac, flux_err_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max()
        , '-C0', drawstyle='steps-mid')
ax.plot(lambda_mid, v_hist, '-r', drawstyle='steps-mid')
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size = 20)
# ax.set_xlim(wave_OIII_vac.min(), wave_OIII_vac.max())
fig.savefig('/Users/lzq/Dropbox/MUSEQuBES+CUBS/fit_kin/3C57_cone_flux.png', bbox_inches='tight')
# plt.show()

# VVD
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

