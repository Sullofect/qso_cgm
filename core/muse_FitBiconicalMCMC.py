import os
import emcee
import corner
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

# Mask a slit
x, y = np.meshgrid(np.arange(101), np.arange(101))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)

rectangle = RectanglePixelRegion(center=PixCoord(x=50, y=50), width=80, height=5, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - 50) ** 2 + (y - 50) ** 2) * 0.2 * 50 / 7
dis_mask = dis[mask]

red = ((x[mask] - 50) < 0) * ((y[mask] - 50) > 0)
blue = ~red
dis_red = dis_mask[red] * -1
dis_blue = dis_mask[blue]

#
def Bin(x, y, bins=20):
    n, edges = np.histogram(x, bins=bins)
    y_mean, y_std = np.zeros(len(n)), np.zeros(len(n))
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i, i_val in enumerate(n):
        if n[i] == 0:
            y_mean[i], y_std[i] = np.nan, np.nan
        else:
            mask = (x > edges[i]) * (x < edges[i + 1])
            y_mean[i], y_std[i] = np.nanmean(y[mask]), np.nanstd(y[mask])
    return x_mean, y_mean, y_std

# Create a interpolated model
bin_red = np.flip(-np.arange(5, 50))
bin_blue = np.arange(5, 50)
dis_mid_red = (bin_red[1:] + bin_red[:-1]) / 2
dis_mid_blue = (bin_blue[1:] + bin_blue[:-1]) / 2
B1_array = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
vmax_array = np.array([350, 500, 750, 1000, 1250, 1500])
out_array = np.array([10, 20, 30, 40, 50])
vmap_red_array, vmap_blue_array = np.zeros((len(dis_mid_red), len(B1_array), len(vmax_array), len(out_array))), \
                                  np.zeros((len(dis_mid_blue), len(B1_array), len(vmax_array), len(out_array)))
dmap_red_array, dmap_blue_array = np.zeros((len(dis_mid_red), len(B1_array), len(vmax_array), len(out_array))), \
                                  np.zeros((len(dis_mid_blue), len(B1_array), len(vmax_array), len(out_array)))
for i, i_B1 in enumerate(B1_array):
    for j, j_vmax in enumerate(vmax_array):
        for k, k_out in enumerate(out_array):
            path_2D = '../../MUSEQuBES+CUBS/fit_bic/fit_PV/Bicone_B1_{}_vmax_{}_out_{}.fits'.format(i_B1, j_vmax, k_out)
            if os.path.exists(path_2D):
                hdul_2D = fits.open(path_2D)
                vmap, dmap = hdul_2D[1].data, hdul_2D[2].data

            else:
                func = DrawBiconeModel(theta_B1_deg=i_B1, theta_B2_deg=60, theta_B3_deg=45,
                                       vmax=j_vmax, vtype='constant', bins=v_bins, theta_in_deg=0,
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
            vmap_red_array[:, i, j, k] = Bin(dis_red, vmap_red, bins=bin_red)[1]
            vmap_blue_array[:, i, j, k] = Bin(dis_blue, vmap_blue, bins=bin_blue)[1]
            dmap_red_array[:, i, j, k] = Bin(dis_red, dmap_red, bins=bin_red)[1]
            dmap_blue_array[:, i, j, k] = Bin(dis_blue, dmap_blue, bins=bin_blue)[1]

            # vmap_red_array[:, i, j, k] = vmap_red
            # vmap_blue_array[:, i, j, k] = vmap_blue
            # dmap_red_array[:, i, j, k] = dmap_red
            # dmap_blue_array[:, i, j, k] = dmap_blue


# Interpolate the model
# X_r, Y_r, Z_r, W_r = np.meshgrid(dis_red, B1_array, vmax_array, out_array, indexing='ij')
# X_r, Y_r, Z_r, W_r = X_r.ravel(), Y_r.ravel(), Z_r.ravel(), W_r.ravel()
# var_array_red = list(zip(X_r, Y_r, Z_r, W_r))
# X_b, Y_b, Z_b, W_b = np.meshgrid(dis_blue, B1_array, vmax_array, out_array, indexing='ij')
# X_b, Y_b, Z_b, W_b = X_b.ravel(), Y_b.ravel(), Z_b.ravel(), W_b.ravel()
# var_array_blue = list(zip(X_b, Y_b, Z_b, W_b))
#
# # Interpolate the model
# f_v_red = interpolate.LinearNDInterpolator(var_array_red, vmap_red_array.ravel(), fill_value=np.nan)
# f_v_blue = interpolate.LinearNDInterpolator(var_array_blue, vmap_blue_array.ravel(), fill_value=np.nan)
# f_d_red = interpolate.LinearNDInterpolator(var_array_red, dmap_red_array.ravel(), fill_value=np.nan)
# f_d_blue = interpolate.LinearNDInterpolator(var_array_blue, dmap_blue_array.ravel(), fill_value=np.nan)

# Interpolate the model
var_array_red = (dis_mid_red, B1_array, vmax_array, out_array)
var_array_blue = (dis_mid_blue, B1_array, vmax_array, out_array)

# Interpolate the model
f_v_red = interpolate.RegularGridInterpolator(var_array_red, vmap_red_array, bounds_error=False, fill_value=np.nan)
f_v_blue = interpolate.RegularGridInterpolator(var_array_blue, vmap_blue_array, bounds_error=False, fill_value=np.nan)
f_d_red = interpolate.RegularGridInterpolator(var_array_red, dmap_red_array, bounds_error=False, fill_value=np.nan)
f_d_blue = interpolate.RegularGridInterpolator(var_array_blue, dmap_blue_array, bounds_error=False, fill_value=np.nan)

# Select MUSE V50 and W80
# Produce PV diagram
x, y = np.meshgrid(np.arange(v50.shape[0]), np.arange(v50.shape[1]))
x, y = x.flatten(), y.flatten()
pixcoord = PixCoord(x=x, y=y)
#
# Mask the center
circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
center_mask_flatten = ~circle.contains(pixcoord)
center_mask = center_mask_flatten.reshape(v50.shape)
x, y = x[center_mask_flatten], y[center_mask_flatten]
pixcoord = pixcoord[center_mask_flatten]

# Mask a slit
rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=5, angle=Angle(-30, 'deg'))
mask_muse = rectangle.contains(pixcoord)
dis = np.sqrt((x - c2[0]) ** 2 + (y - c2[1]) ** 2) * 0.2 * 50 / 7
dis_mask = dis[mask_muse]

# Mask each side
red_muse = ((x[mask_muse] - c2[0]) < 0) * ((y[mask_muse] - c2[1]) > 0)
blue_muse = ~red_muse
dis_red_muse = dis_mask[red_muse] * -1
dis_blue_muse = dis_mask[blue_muse]

v50_flatten = v50.flatten()[center_mask_flatten]
w80_flatten = w80.flatten()[center_mask_flatten]
v50_blue, v50_red = v50_flatten[mask_muse][blue_muse], v50_flatten[mask_muse][red_muse]
w80_blue, w80_red = w80_flatten[mask_muse][blue_muse], w80_flatten[mask_muse][red_muse]
dis_red_mean, v50_red_mean, v50_red_mean_err = Bin(dis_red_muse, v50_red, bins=20)
dis_blue_mean, v50_blue_mean, v50_blue_mean_err = Bin(dis_blue_muse, v50_blue, bins=20)
_, w80_red_mean, w80_red_mean_err = Bin(dis_red_muse, w80_red, bins=20)
_, w80_blue_mean, w80_blue_mean_err = Bin(dis_blue_muse, w80_blue, bins=20)
v50_red_mean_err, v50_blue_mean_err = np.sqrt(v50_red_mean_err ** 2 + 30 ** 2), np.sqrt(
    v50_blue_mean_err ** 2 + 30 ** 2)
w80_red_mean_err, w80_blue_mean_err = np.sqrt(w80_red_mean_err ** 2 + 30 ** 2), np.sqrt(
    w80_blue_mean_err ** 2 + 30 ** 2)

# raise ValueError('Stop here')
# MCMC over it
def log_prob(x, f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue, w80_red, w80_blue,
             dis_red_mean, dis_blue_mean, v50_red_mean, v50_red_mean_err, v50_blue_mean, v50_blue_mean_err,
             w80_red_mean, w80_red_mean_err, w80_blue_mean, w80_blue_mean_err):
    # beta_i, vmax_i, out_i, scale = x[:]
    beta_i, vmax_i, out_i = x[:]
    scale = 1.5
    if  beta_i < 5 or beta_i > 50:
        return -np.inf
    elif vmax_i < 350 or vmax_i > 1500:
        return -np.inf
    elif out_i < 10 or out_i > 40:
        return -np.inf
    # elif scale < 0.9 or scale > 1.8:
    #     return -np.inf
    # v_red, v_blue = f_v_red((dis_red_muse * scale, *x[:3])), f_v_blue((dis_blue_muse * scale, *x[:3]))
    # d_red, d_blue = f_d_red((dis_red_muse * scale, *x[:3])) * 2.563, f_d_blue((dis_blue_muse * scale, *x[:3])) * 2.563
    v_red, v_blue = f_v_red((dis_red_mean * scale, *x[:3])), f_v_blue((dis_blue_mean * scale, *x[:3]))
    d_red, d_blue = f_d_red((dis_red_mean * scale, *x[:3])) * 2.563, f_d_blue((dis_blue_mean * scale, *x[:3])) * 2.563

    # Compute chi square
    # chi2_red = ((v_red - v50_red) / 30) ** 2 + ((d_red - w80_red) / 30) ** 2
    # chi2_blue = ((v_blue - v50_blue) / 30) ** 2 + ((d_blue - w80_blue) / 30) ** 2
    chi2_red = ((v_red - v50_red_mean) / v50_red_mean_err) ** 2 + ((d_red - w80_red_mean) / w80_red_mean_err) ** 2
    chi2_blue = ((v_blue - v50_blue_mean) / v50_blue_mean_err) ** 2 + ((d_blue - w80_blue_mean) / w80_blue_mean_err) ** 2
    chi2_array = np.hstack([chi2_red, chi2_blue])
    return - 0.5 * np.nansum(chi2_array)

nums_chain, nums_disc = 5000, 1000
nwalkers = 40
ndim = 3
p0 = np.array([30, 700, 25]) + 0.1 * np.random.randn(nwalkers, ndim)
labels = [r"$\beta_{1}$", r"$v_{\rm max}$", r"$\theta_{open}$"]

# ndim = 4
# p0 = np.array([30, 700, 25, 1.5]) + 0.1 * np.random.randn(nwalkers, ndim)
# labels = [r"$\beta_{1}$", r"$v_{\rm max}$", r"$\theta_{open}$", r'scale']

filename = '../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_mean.h5'
figname_MCMC = '../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_mean.pdf'
backend = emcee.backends.HDFBackend(filename)


if os.path.exists(filename):
    samples = backend.get_chain(flat=True, discard=nums_disc)
    samples_corner = np.copy(samples)
else:
    # Run the MCMC
    print('Running MCMC')
    args = (f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue, w80_red, w80_blue,
            dis_red_mean, dis_blue_mean, v50_red_mean, v50_red_mean_err, v50_blue_mean, v50_blue_mean_err,
            w80_red_mean, w80_red_mean_err, w80_blue_mean, w80_blue_mean_err)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, backend=backend)
    state = sampler.run_mcmc(p0, nums_chain, progress=True)
    samples = sampler.get_chain(flat=True, discard=nums_disc)
    samples_corner = np.copy(samples)

median_values = np.median(samples, axis=0)
print('Median values:', median_values)
figure = corner.corner(samples_corner, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k',
                       title_kwargs={"fontsize": 13}, smooth=1., smooth1d=1., bins=25)
figure.savefig(figname_MCMC, bbox_inches='tight')


plt.close('all')
fig, ax = plt.subplots(2, 1, figsize=(10, 7), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.0)

# Check
var_check = (*median_values, 1.5)
# var_check = (40, 600, 29, 1.5)
func = DrawBiconeModel(theta_B1_deg=var_check[0], theta_B2_deg=60, theta_B3_deg=45, vmax=var_check[1],
                       vtype='constant', bins=v_bins, theta_in_deg=0,
                       theta_out_deg=var_check[2], tau=1, plot=True, save_fig=True)
func.Make2Dmap()
vmap = func.vmap
dmap = func.dmap
vmap, dmap = (np.flip(vmap, 1), np.flip(dmap, 1))
vmap_flatten = vmap.flatten()
dmap_flatten = dmap.flatten()
vmap_blue, vmap_red = vmap_flatten[mask][blue], vmap_flatten[mask][red]
dmap_blue, dmap_red = dmap_flatten[mask][blue], dmap_flatten[mask][red]

# MUSE
# ax[0].scatter(dis_red_muse, v50_red, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red')
# ax[0].scatter(dis_blue_muse, v50_blue, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue')
# ax[1].scatter(dis_red_muse, w80_red, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue_muse, w80_blue, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue')
ax[0].errorbar(dis_red_muse, v50_red, 30, fmt='o', color='g')
ax[0].errorbar(dis_blue_muse, v50_blue, 30, fmt='o', color='g')
ax[1].errorbar(dis_red_muse, w80_red, 30, fmt='o', color='g')
ax[1].errorbar(dis_blue_muse, w80_blue, 30, fmt='o', color='g')
ax[0].errorbar(dis_red_mean, v50_red_mean, yerr=v50_red_mean_err, fmt='o', color='red')
ax[0].errorbar(dis_blue_mean, v50_blue_mean, yerr=v50_blue_mean_err, fmt='o', color='blue')
ax[1].errorbar(dis_red_mean, w80_red_mean, yerr=w80_red_mean_err, fmt='o', color='red')
ax[1].errorbar(dis_blue_mean, w80_blue_mean, yerr=w80_blue_mean_err, fmt='o', color='blue')

# Interpolation
dis_red_sort = np.sort(dis_red)
dis_blue_sort = np.sort(dis_blue)
ax[0].plot(dis_red_sort / var_check[3], f_v_red((dis_red_sort, *var_check[:3])), lw=2, color='black')
ax[0].plot(dis_blue_sort / var_check[3], f_v_blue((dis_blue_sort, *var_check[:3])), lw=2, color='black')
ax[1].plot(dis_red_sort / var_check[3], f_d_red((dis_red_sort, *var_check[:3])) * 2.563, lw=2, color='black')
ax[1].plot(dis_blue_sort / var_check[3], f_d_blue((dis_blue_sort, *var_check[:3])) * 2.563, lw=2, color='black')
# ax[0].scatter(dis_red, vmap_red_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_blue, vmap_blue_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'$\rm 3C\,57 \, southwest$')
# ax[1].scatter(dis_red, dmap_red_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue, dmap_blue_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[0].scatter(dis_red / var_check[3], vmap_red, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
ax[0].scatter(dis_blue / var_check[3], vmap_blue, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[1].scatter(dis_red / var_check[3], dmap_red * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
ax[1].scatter(dis_blue / var_check[3], dmap_blue * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[0].set_xlim(-40, 40)
ax[0].set_ylim(-450, 450)
ax[1].set_ylim(0, 510)
ax[0].set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
ax[1].set_xlabel(r'$\rm Distance \, [kpc]$', size=25)
ax[1].set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25)
fig.savefig('../../MUSEQuBES+CUBS/fit_bic/PV_cone.png', bbox_inches='tight')
