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

# Hedaer information
path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
w = WCS(hdr_sub_gaia, naxis=2)
center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
c2 = w.world_to_pixel(center_qso)

# Path to the data
path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_plot.fits'
path_w80_plot = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_plot.fits'
hdul_v50_plot = fits.open(path_v50_plot)
hdul_w80_plot = fits.open(path_w80_plot)
v50, w80 = hdul_v50_plot[1].data, hdul_w80_plot[1].data

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
dis_red_sort = np.sort(dis_red)
dis_blue_sort = np.sort(dis_blue)

#
def Bin(x, y, bins=20):
    n, edges = np.histogram(x, bins=bins)
    y_mean, y_std, y_err = np.zeros(len(n)), np.zeros(len(n)), np.ones_like(y)
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i, i_val in enumerate(n):
        if n[i] == 0:
            y_mean[i], y_std[i] = np.nan, np.nan
        else:
            mask = (x > edges[i]) * (x <= edges[i + 1])
            y_mean[i], y_std[i] = np.nanmean(y[mask]), np.nanstd(y[mask])
            y_err[mask] = y_std[i]
    return x_mean, y_mean, y_std, y_err

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

# Mean
dis_red_mean, v50_red_mean, v50_red_mean_err, v50_red_err = Bin(dis_red_muse, v50_red, bins=20)
dis_blue_mean, v50_blue_mean, v50_blue_mean_err, v50_blue_err = Bin(dis_blue_muse, v50_blue, bins=20)
_, w80_red_mean, w80_red_mean_err, w80_red_err = Bin(dis_red_muse, w80_red, bins=20)
_, w80_blue_mean, w80_blue_mean_err, w80_blue_err = Bin(dis_blue_muse, w80_blue, bins=20)
v50_red_mean_err, v50_blue_mean_err = np.sqrt(v50_red_mean_err ** 2 + 30 ** 2), np.sqrt(
    v50_blue_mean_err ** 2 + 30 ** 2)
w80_red_mean_err, w80_blue_mean_err = np.sqrt(w80_red_mean_err ** 2 + 30 ** 2), np.sqrt(
    w80_blue_mean_err ** 2 + 30 ** 2)

# raise ValueError('Stop here')
# MCMC over it
def log_prob(x, f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue, w80_red, w80_blue,
             v50_red_err, v50_blue_err, w80_red_err, w80_blue_err,
             dis_red_mean, dis_blue_mean, v50_red_mean, v50_red_mean_err, v50_blue_mean, v50_blue_mean_err,
             w80_red_mean, w80_red_mean_err, w80_blue_mean, w80_blue_mean_err):
    # beta_i, vmax_i, out_i = x[:]
    # beta_i, vmax_i, out_i, scale = x[:]
    # beta_i, vmax_i, out_i, offset_i = x[:]
    # scale_i = 1.5
    beta_i, vmax_i, out_i, offset_i, scale_i = x[:]
    if  beta_i < 5 or beta_i > 50:
        return -np.inf
    elif vmax_i < 350 or vmax_i > 1500:
        return -np.inf
    elif out_i < 10 or out_i > 40:
        return -np.inf
    elif scale_i < 0.5 or scale_i > 2.5:
        return -np.inf
    v_red, v_blue = f_v_red((dis_red_muse * scale_i, *x[:3])), f_v_blue((dis_blue_muse * scale_i, *x[:3]))
    d_red, d_blue = f_d_red((dis_red_muse * scale_i, *x[:3])) * 2.563, f_d_blue((dis_blue_muse * scale_i, *x[:3])) * 2.563
    v_red[np.isnan(v_red)], v_blue[np.isnan(v_blue)] = 1000, -1000
    d_red[np.isnan(d_red)], d_blue[np.isnan(d_blue)] = 5000, 5000
    # v_red, v_blue = f_v_red((dis_red_mean * scale, *x[:3])), f_v_blue((dis_blue_mean * scale, *x[:3]))
    # d_red, d_blue = f_d_red((dis_red_mean * scale, *x[:3])) * 2.563, f_d_blue((dis_blue_mean * scale, *x[:3])) * 2.563

    # Compute chi square
    v_red, v_blue = v_red + offset_i, v_blue + offset_i  # Offset it
    chi2_red = ((v_red - v50_red) / v50_red_err) ** 2 + ((d_red - w80_red) / w80_red_err) ** 2
    chi2_blue = ((v_blue - v50_blue) / v50_blue_err) ** 2 + ((d_blue - w80_blue) / w80_blue_err) ** 2
    # chi2_red = ((v_red - v50_red_mean) / v50_red_mean_err) ** 2 + ((d_red - w80_red_mean) / w80_red_mean_err) ** 2
    # chi2_blue = ((v_blue - v50_blue_mean) / v50_blue_mean_err) ** 2 + ((d_blue - w80_blue_mean) / w80_blue_mean_err) ** 2
    chi2_array = np.hstack([chi2_red, chi2_blue])
    return - 0.5 * np.nansum(chi2_array)

nums_chain, nums_disc = 1000, 100
nwalkers = 40
ndim = 5
p0 = np.array([30, 700, 25, 30, 1.5]) + 0.1 * np.random.randn(nwalkers, ndim)
labels = [r"$\beta_{1}$", r"$v_{\rm max}$", r"$\theta_{open}$", r'offset', r'scale']

# ndim = 4
# p0 = np.array([30, 700, 25, 1.5]) + 0.1 * np.random.randn(nwalkers, ndim)
# labels = [r"$\beta_{1}$", r"$v_{\rm max}$", r"$\theta_{open}$", r'scale']

filename = '../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_para=5.h5'
figname_MCMC = '../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_para=5.pdf'
backend = emcee.backends.HDFBackend(filename)


if os.path.exists(filename):
    samples = backend.get_chain(flat=True, discard=nums_disc)
    samples_corner = np.copy(samples)
else:
    # Run the MCMC
    print('Running MCMC')
    args = (f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue, w80_red, w80_blue,
            v50_red_err, v50_blue_err, w80_red_err, w80_blue_err,
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
                       vtype='constant', bins=None, theta_in_deg=0,
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
ax[0].plot(dis_red_sort / var_check[4], f_v_red((dis_red_sort, *var_check[:3])) + var_check[3], lw=2, color='purple')
ax[0].plot(dis_blue_sort / var_check[4], f_v_blue((dis_blue_sort, *var_check[:3])) + var_check[3], lw=2, color='purple')
ax[1].plot(dis_red_sort / var_check[4], f_d_red((dis_red_sort, *var_check[:3])) * 2.563, lw=2, color='purple')
ax[1].plot(dis_blue_sort / var_check[4], f_d_blue((dis_blue_sort, *var_check[:3])) * 2.563, lw=2, color='purple')

# Make a group of points
draw = np.random.choice(len(samples), size=4000, replace=False)
samples_draw = samples[draw]
v_all_red = f_v_red((dis_red_sort[:, np.newaxis], samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2]))

ax[0].fill_between(dis_red_sort / var_check[4], np.min(v_all_red, axis=1) + var_check[3], np.max(v_all_red, axis=1) + var_check[3], color='purple', alpha=0.5)
ax[0].plot(dis_red_sort / var_check[4], np.max(v_all_red, axis=1) + var_check[3], lw=2, color='purple')
ax[0].plot(dis_red_sort / var_check[4], np.min(v_all_red, axis=1) + var_check[3], lw=2, color='purple')
# ax[0].plot(dis_blue_sort / var_check[3], f_v_blue((dis_blue_sort, *var_check[:3])), lw=2, color='black')
# ax[1].plot(dis_red_sort / var_check[3], f_d_red((dis_red_sort, *var_check[:3])) * 2.563, lw=2, color='black')
# ax[1].plot(dis_blue_sort / var_check[3], f_d_blue((dis_blue_sort, *var_check[:3])) * 2.563, lw=2, color='black')
# print(np.shape(f_v_red((dis_red_sort[:, np.newaxis], samples[:, 0], samples[:, 1], samples[:, 2]))))


# ax[0].scatter(dis_red, vmap_red_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_blue, vmap_blue_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'$\rm 3C\,57 \, southwest$')
# ax[1].scatter(dis_red, dmap_red_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue, dmap_blue_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
# ax[0].scatter(dis_red / var_check[3], vmap_red, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[0].scatter(dis_blue / var_check[3], vmap_blue, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
# ax[1].scatter(dis_red / var_check[3], dmap_red * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue / var_check[3], dmap_blue * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
ax[0].axhline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[1].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].set_xlim(-40, 40)
ax[0].set_ylim(-450, 450)
ax[1].set_ylim(0, 510)
ax[0].set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
ax[1].set_xlabel(r'$\rm Distance \, [kpc]$', size=25)
ax[1].set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25)
fig.savefig('../../MUSEQuBES+CUBS/fit_bic/PV_cone.png', bbox_inches='tight')
