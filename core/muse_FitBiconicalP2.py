import os
import aplpy
import lmfit
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from scipy import interpolate
from astropy.coordinates import SkyCoord
from regions import PixCoord
from astropy.coordinates import Angle
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from scipy.interpolate import interp1d
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from muse_FitBiconical import DrawBiconeModel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239

def likelihood(flux_OIII, flux_OIII_err, flux_OIII_mod):
    mask_v50 = ~np.isnan(v50)
    flux_OIII = np.where(mask_v50[np.newaxis, :, :], flux_OIII, np.nan)
    flux_OIII_err = np.where(mask_v50[np.newaxis, :, :], flux_OIII_err, np.nan)

    #
    flux_OIII_mod = np.where(flux_OIII_mod[10:-10, :, :] == 0, flux_OIII_mod[10:-10, :, :], 1)
    # flux_OIII_mod = flux_OIII * flux_OIII_mod

    chi2 = np.nansum((flux_OIII - flux_OIII_mod) ** 2 / flux_OIII_err ** 2, axis=0)
    chi2_all = np.nansum(chi2) / np.nansum(mask_v50) / len(flux_OIII)
               # / len(flux_OIII) / np.nansum(mask_v50)
    return chi2, chi2_all

# QSO information
cubename = '3C57'
str_zap = ''
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

# Measure kinematics
path_fit = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
hdul = fits.open(path_fit)
fs, hdr = hdul[1].data, hdul[2].header

# Hedaer information
path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
w = WCS(hdr_sub_gaia, naxis=2)
center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
c2 = w.world_to_pixel(center_qso)

# Path to the data
UseDataSeg = (1.5, 'gauss', None, None)
UseDetectionSeg = (1.5, 'gauss', 1.5, 'gauss')
line_OII, line_OIII = 'OII', 'OIII'
path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OIII, *UseDetectionSeg)
path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.format(cubename, str_zap, line_OII)
path_cube_smoothed_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                         '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.format(cubename, str_zap, line_OIII)
path_cube_smoothed_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                          '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDataSeg)
path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_plot.fits'
path_w80_plot = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_plot.fits'
hdul_v50_plot = fits.open(path_v50_plot)
hdul_w80_plot = fits.open(path_w80_plot)
v50, w80 = hdul_v50_plot[1].data, hdul_w80_plot[1].data

# Load data and smoothing
cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori
size = np.shape(flux_OIII)[1:]

# Create a bin according to the MUSE data
dlambda = wave_OIII_vac[1] - wave_OIII_vac[0]
wave_OIII_ext = np.zeros(len(wave_OIII_vac) + 20)
wave_OIII_ext[10:-10] = wave_OIII_vac
wave_OIII_ext[:10] = wave_OIII_vac[0] - np.flip(np.arange(1, 11)) * dlambda
wave_OIII_ext[-10:] = wave_OIII_vac[-1] + dlambda * np.arange(1, 11)
size_ext = len(wave_OIII_ext)

#
wave_bin = np.zeros(len(wave_OIII_ext) + 1)
wave_bin[:-1] = wave_OIII_ext - dlambda / 2
wave_bin[-1] = wave_OIII_ext[-1] + dlambda
wave_OIII_obs = wave_OIII5008_vac * (1 + z_qso)
v_bins = (wave_bin - wave_OIII_obs) / wave_OIII_obs * c_kms
mask = np.zeros(size)
mask[55:100, 54:100] = 1

# 764351.5794328616

# Define the pixel coordinates
ang_list = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
vel_list = np.array([1400, 1200, 1000, 800, 600])

# func = DrawBiconeModel(theta_B1_deg=30, vmax=1000, vtype='decreasing', bins=v_bins)
# func.Make2Dmap()
# flux_model_cube = func.emission_cube(pix_qso=c2, size_ext=size_ext, size=size, mask=mask)

# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 8))
# ax.imshow(v50, origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
# # ax.imshow(vmap_MUSE, cmap='coolwarm', origin='lower', vmin=-300, vmax=300)
# ax.imshow(np.flip(func.vmap, 1), cmap='coolwarm', extent=[c2[0] - 20, c2[0] + 20, c2[1] - 20, c2[1] + 20],
#           vmin=-300, vmax=300, origin='lower')
# ax.plot(c2[0], c2[1], '*', markersize=15)
# ax.set_xlim(0, 150)
# ax.set_ylim(0, 150)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# fig.savefig('../../MUSEQuBES+CUBS/fit_bic/3C57_cone_{}_{}.png'.format(), bbox_inches='tight')


# for i_ang in ang_list:
#     for i_vel in vel_list:
#         func = DrawBiconeModel(theta_B1_deg=i_ang, vmax=i_vel, vtype='decreasing', bins=v_bins)
#         flux_model_cube = func.emission_cube(pix_qso=c2, size_ext=size_ext, size=size, mask=mask)
#
#         # Save the results
#         path_test = '../../MUSEQuBES+CUBS/fit_bic/Biconical_cube_test_{}_{}.fits'.format(i_ang, i_vel)
#         hdul_test = fits.ImageHDU(flux_model_cube, header=None)
#         hdul_test.writeto(path_test, overwrite=True)

# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 8))
# for i_ang in ang_list:
#     for i_vel in vel_list:
#         path_test = '../../MUSEQuBES+CUBS/fit_bic/Biconical_cube_test_{}_{}.fits'.format(i_ang, i_vel)
#         hdul_test = fits.open(path_test)
#         flux_model_cube = hdul_test[1].data
#         chi2, chi2_all = likelihood(flux_OIII, flux_err_OIII, flux_model_cube)
#
#         print(i_ang, i_vel, chi2_all)
#         plt.scatter(i_ang, i_vel, c=np.log10(chi2_all), vmin=4.4, vmax=5.2)
#
#         # fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 8))
#         # ax.imshow(v50, origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
#         # # ax.imshow(vmap_MUSE, cmap='coolwarm', origin='lower', vmin=-300, vmax=300)
#         # ax.imshow(np.flip(func.vmap, 1), cmap='coolwarm', extent=[c2[0] - 20, c2[0] + 20, c2[1] - 20, c2[1] + 20],
#         #           vmin=-300, vmax=300, origin='lower')
#         # ax.plot(c2[0], c2[1], '*', markersize=15)
#         # ax.set_xlim(0, 150)
#         # ax.set_ylim(0, 150)
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         # ax.set_xticklabels([])
#         # ax.set_yticklabels([])
#         # fig.savefig('../../MUSEQuBES+CUBS/fit_bic/3C57_cone_{}_{}.png'.format(), bbox_inches='tight')
#         # print(chi2_all)
# ax.set_xlabel('rotation angle (deg)', size=20)
# ax.set_ylabel('Max velocity (km/s)', size=20)
# plt.colorbar()
# fig.savefig('../../MUSEQuBES+CUBS/fit_bic/3C57_cone_fit.png', bbox_inches='tight')

# Make a position velocity diagram
# func = DrawBiconeModel(theta_B1_deg=10, vmax=500, vtype='constant', bins=v_bins, theta_in_deg=0, theta_out_deg=40.0, tau=1)
# func.Make2Dmap()
# flux_model_cube = func.emission_cube(pix_qso=c2, size_ext=size_ext, size=size, mask=mask)

# Mask a slit
x, y = np.meshgrid(np.arange(101), np.arange(101))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

rectangle = RectanglePixelRegion(center=PixCoord(x=50, y=50), width=80, height=5, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - 50) ** 2 + (y - 50) ** 2) * 0.2 * 50 / 7 / 2.5
dis_mask = dis[mask]

red = ((x[mask] - 50) < 0) * ((y[mask] - 50) > 0)
blue = ~red
dis_red = dis_mask[red] * -1
dis_blue = dis_mask[blue]

# vmap_flatten = vmap.flatten()
# dmap_flatten = dmap.flatten()
# vmap_blue, vmap_red = vmap_flatten[mask][blue], vmap_flatten[mask][red]
# dmap_blue, dmap_red = dmap_flatten[mask][blue], dmap_flatten[mask][red]

# Create a interpolated model
vmap_red_array, vmap_blue_array = np.zeros((len(dis_red), 4, 3, 4)), np.zeros((len(dis_blue), 4, 3, 4))
dmap_red_array, dmap_blue_array = np.zeros((len(dis_red), 4, 3, 4)), np.zeros((len(dis_blue), 4, 3, 4))
B1_array = np.array([5, 10, 15, 20])
vmax_array = np.array([500, 750, 1000])
out_array = np.array([10, 20, 30, 40])
for i, i_B1 in enumerate(B1_array):
    for j, j_vmax in enumerate(vmax_array):
        for k, k_out in enumerate(out_array):
            path_2D = '../../MUSEQuBES+CUBS/fit_bic/fit_PV/Bicone_B1_{}_vmax_{}_out_{}.fits'.format(i_B1, j_vmax, k_out)


            if os.path.exists(path_2D):
                hdul_2D = fits.open(path_2D)
                vmap, dmap = hdul_2D[1].data, hdul_2D[2].data

            else:
                func = DrawBiconeModel(theta_B1_deg=i_B1, vmax=j_vmax, vtype='constant', bins=v_bins, theta_in_deg=0,
                                       theta_out_deg=k_out, tau=1, plot=False, save_fig=False)
                func.Make2Dmap()
                vmap = func.vmap
                dmap = func.dmap

                # Save the re_2D
                hdul_pri = fits.PrimaryHDU()
                hdul_vmap = fits.ImageHDU(func.vmap, header=None)
                hdul_dmap = fits.ImageHDU(func.dmap, header=None)
                hdul = fits.HDUList([hdul_pri, hdul_vmap, hdul_dmap])
                hdul.writeto(path_2D, overwrite=True)

            # Produce PV diagram
            vmap, dmap = (np.flip(vmap, 1), np.flip(dmap, 1))
            vmap_flatten = vmap.flatten()
            dmap_flatten = dmap.flatten()
            vmap_blue, vmap_red = vmap_flatten[mask][blue], vmap_flatten[mask][red]
            dmap_blue, dmap_red = dmap_flatten[mask][blue], dmap_flatten[mask][red]
            vmap_red_array[:, i, j, k] = vmap_red
            vmap_blue_array[:, i, j, k] = vmap_blue
            dmap_red_array[:, i, j, k] = dmap_red
            dmap_blue_array[:, i, j, k] = dmap_blue

# Interpolate the model
X_r, Y_r, Z_r, W_r = np.meshgrid(dis_red, B1_array, vmax_array, out_array, indexing='ij')
X_r, Y_r, Z_r, W_r = X_r.ravel(), Y_r.ravel(), Z_r.ravel(), W_r.ravel()
var_array_red = list(zip(X_r, Y_r, Z_r, W_r))
X_b, Y_b, Z_b, W_b = np.meshgrid(dis_blue, B1_array, vmax_array, out_array, indexing='ij')
X_b, Y_b, Z_b, W_b = X_b.ravel(), Y_b.ravel(), Z_b.ravel(), W_b.ravel()
var_array_blue = list(zip(X_b, Y_b, Z_b, W_b))

#
f_v_red = interpolate.LinearNDInterpolator(var_array_red, vmap_red_array.ravel(), fill_value=np.nan)
f_v_blue = interpolate.LinearNDInterpolator(var_array_blue, vmap_blue_array.ravel(), fill_value=np.nan)
f_d_red = interpolate.LinearNDInterpolator(var_array_red, dmap_red_array.ravel(), fill_value=np.nan)
f_d_blue = interpolate.LinearNDInterpolator(var_array_blue, dmap_blue_array.ravel(), fill_value=np.nan)


# Select MUSE V50 and W80
# Produce PV diagram
# x, y = np.meshgrid(np.arange(v50.shape[0]), np.arange(v50.shape[1]))
# x, y = x.flatten(), y.flatten()
# pixcoord = PixCoord(x=x, y=y)
#
# Mask the center
circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
center_mask_flatten = ~circle.contains(pixcoord)
center_mask = center_mask_flatten.reshape(v50.shape)
x, y = x[center_mask_flatten], y[center_mask_flatten]
pixcoord = pixcoord[center_mask_flatten]

# Mask a slit
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=5, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0]) ** 2 + (y - c2[1]) ** 2) * 0.2 * 50 / 7
dis_mask = dis[mask]

# Mask each side
red = ((x[mask] - c2[0]) < 0) * ((y[mask] - c2[1]) > 0)
blue = ~red
dis_red = dis_mask[red] * -1
dis_blue = dis_mask[blue]

v50_flatten = v50.flatten()[center_mask_flatten]
w80_flatten = w80.flatten()[center_mask_flatten]
v50_blue, v50_red = v50_flatten[mask][blue], v50_flatten[mask][red]
w80_blue, w80_red = w80_flatten[mask][blue], w80_flatten[mask][red]
#
# vmap_MUSE_flatten = vmap_MUSE.flatten()
# # dmap_MUSE_flatten = dmap_MUSE.flatten()
# vmap_MUSE_blue, vmap_MUSE_red = vmap_MUSE_flatten[mask][blue], vmap_MUSE_flatten[mask][red]
# # dmap_MUSE_blue, dmap_MUSE_red = dmap_MUSE_flatten[mask][blue], dmap_MUSE_flatten[mask][red]


# Save the results
# path_test = '../../MUSEQuBES+CUBS/fit_bic/Biconical_cube_test_checkVel.fits'.format(10, 1000)
# hdul_test = fits.ImageHDU(flux_model_cube, header=None)
# hdul_test.writeto(path_test, overwrite=True)


# vmap_MUSE = func.vmap_MUSE
# vmap, dmap = (np.flip(func.vmap, 1), np.flip(func.dmap, 1))

# # Mask a slit
# x, y = np.meshgrid(np.arange(vmap.shape[0]), np.arange(vmap.shape[1]))
# x, y = x.flatten(), y.flatten()
# pixcoord = PixCoord(x=x, y=y)
#
# rectangle = RectanglePixelRegion(center=PixCoord(x=50, y=50), width=80, height=5, angle=Angle(-30, 'deg'))
# mask = rectangle.contains(pixcoord)
# dis = np.sqrt((x - 50) ** 2 + (y - 50) ** 2) * 0.2 * 50 / 7 / 2.5
# dis_mask = dis[mask]
#
# red = ((x[mask] - 50) < 0) * ((y[mask] - 50) > 0)
# blue = ~red
# dis_red = dis_mask[red] * -1
# dis_blue = dis_mask[blue]
#
# vmap_flatten = vmap.flatten()
# dmap_flatten = dmap.flatten()
# vmap_blue, vmap_red = vmap_flatten[mask][blue], vmap_flatten[mask][red]
# dmap_blue, dmap_red = dmap_flatten[mask][blue], dmap_flatten[mask][red]

# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 8))
# ax.imshow(v50, origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
# ax.imshow(func.vmap_MUSE, cmap='coolwarm', origin='lower', vmin=-350, vmax=350)
# # ax.imshow(np.flip(func.vmap, 1), cmap='coolwarm', extent=[c2[0] - 20, c2[0] + 20, c2[1] - 20, c2[1] + 20],
# #           vmin=-300, vmax=300, origin='lower')
# # ax.imshow(np.flip(func.vmap, 1), cmap='coolwarm', vmin=-350, vmax=350, origin='lower')
# patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
# ax.plot(c2[0], c2[1], '*', markersize=15)
# ax.set_xlim(0, 150)
# ax.set_ylim(0, 150)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# fig.savefig('../../MUSEQuBES+CUBS/fit_bic/cone_model.png'.format(), bbox_inches='tight')

# MCMC over it
import emcee
import corner

def log_likelihood(x, f_v_red, f_v_blue, f_d_red, f_d_blue):
    b1, vmax, theta_out = x[0], x[1], x[2]
    v_red, v_blue = f_v_red(dis_red, *var_check), f_v_blue(dis_blue, *var_check)
    w80_red, w80_blue = f_d_red(dis_red, *var_check) * 2.563, f_d_blue(dis_blue, *var_check) * 2.563

    #
    chi2_v = np.sum((v_red - v50_red) ** 2) + np.sum((v_blue - v50_blue) ** 2)
    chi2_d = np.sum((v_red - w80_red) ** 2) + np.sum((v_blue - w80_blue) ** 2)
    return - 0.5 * np.nansum(sum_array[line_param[:, 1]])


plt.close('all')
fig, ax = plt.subplots(2, 1, figsize=(10, 7), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.0)

# Check
var_check = (7.5, 900, 23)
func = DrawBiconeModel(theta_B1_deg=var_check[0], vmax=var_check[1], vtype='constant', bins=v_bins, theta_in_deg=0,
                       theta_out_deg=var_check[2], tau=1, plot=False, save_fig=False)
func.Make2Dmap()
vmap = func.vmap
dmap = func.dmap
vmap, dmap = (np.flip(vmap, 1), np.flip(dmap, 1))
vmap_flatten = vmap.flatten()
dmap_flatten = dmap.flatten()
vmap_blue, vmap_red = vmap_flatten[mask][blue], vmap_flatten[mask][red]
dmap_blue, dmap_red = dmap_flatten[mask][blue], dmap_flatten[mask][red]

ax[0].scatter(dis_red, f_v_red(dis_red, *var_check), s=80, marker='o', edgecolors='k', linewidths=0.5, color='black',
              label=r'$\rm 3C\,57 \, northeast$')
ax[0].scatter(dis_blue, f_v_blue(dis_blue, *var_check), s=80, marker='o', edgecolors='k', linewidths=0.5, color='black',
              label=r'$\rm 3C\,57 \, northeast$')
ax[1].scatter(dis_red, f_d_red(dis_red, *var_check) * 2.563, s=80, marker='o', edgecolors='k', linewidths=0.5, color='black',
              label=r'$\rm 3C\,57 \, northeast$')
ax[1].scatter(dis_blue, f_d_blue(dis_blue, *var_check) * 2.563, s=80, marker='o', edgecolors='k', linewidths=0.5, color='black',
              label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_red, vmap_red, s=100, marker='*', edgecolors='purple', linewidths=0.5, color='black',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_red, vmap_red_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_blue, vmap_blue_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'$\rm 3C\,57 \, southwest$')
# ax[1].scatter(dis_red, dmap_red_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue, dmap_blue_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[0].scatter(dis_red, vmap_red, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
              label=r'$\rm 3C\,57 \, northeast$')
ax[0].scatter(dis_blue, vmap_blue, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
              label=r'$\rm 3C\,57 \, southwest$')
ax[1].scatter(dis_red, dmap_red * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
ax[1].scatter(dis_blue, dmap_blue * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
# ax[0].scatter(dis_red_MUSE, vmap_MUSE_red, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_blue_MUSE, vmap_MUSE_blue, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'$\rm 3C\,57 \, southwest$')
fig.savefig('../../MUSEQuBES+CUBS/fit_bic/PV_cone.png', bbox_inches='tight')
