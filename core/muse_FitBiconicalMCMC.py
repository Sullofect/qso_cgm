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
from palettable.cmocean.sequential import Dense_20_r
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

# Funcs
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

def Bin_maxmin(x, y, bins=20):
    n, edges = np.histogram(x, bins=bins)
    y_mean, y_std, y_max, y_min = np.zeros(len(n)), np.zeros(len(n)), np.zeros(len(n)), np.zeros(len(n))
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i, i_val in enumerate(n):
        if n[i] == 0:
            y_mean[i], y_std[i] = np.nan, np.nan
        else:
            mask = (x > edges[i]) * (x <= edges[i + 1])
            y_mean[i], y_std[i] = np.nanmean(y[mask]), np.nanstd(y[mask])
            y_max[i], y_min[i] = np.nanmax(y[mask]), np.nanmin(y[mask])
    return x_mean, y_mean, y_max, y_min

# Likelihood
factor = 0.2 * 50 / 7  # Covert MUSE pix to kpc
# scale = 50 / np.abs(np.min(dis_red_muse / factor))  # Scale the distance to 50 pixel in cone
# dis_red_mean, dis_blue_mean, v50_red_mean, v50_red_mean_err, v50_blue_mean, v50_blue_mean_err,
# w80_red_mean, w80_red_mean_err, w80_blue_mean, w80_blue_mean_err
def log_prob(x, f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue, w80_red, w80_blue,
             v50_red_err, v50_blue_err, w80_red_err, w80_blue_err):
    # beta_i, vmax_i, out_i, offset_i = x[:]
    # scale_i = scale
    beta_i, vmax_i, out_i, offset_i, scale_i = x[:]
    if beta_i < 5 or beta_i > 50:
        return -np.inf
    elif vmax_i < 200 or vmax_i > 1500:
        return -np.inf
    elif out_i < 10 or out_i > 50:
        return -np.inf

    # Redefine maximum scale
    v_red_range = f_v_red((dis_red_sort, *x[:3]))
    dis_red_max = np.abs(dis_red_sort[~np.isnan(v_red_range)][0])
    scale_max = dis_red_max / np.abs(np.min(dis_red_muse / factor))

    dis_red_min = np.abs(dis_red_sort[~np.isnan(v_red_range)][-1])
    scale_min = dis_red_min / np.abs(np.max(dis_red_muse / factor))
    # print(dis_red_sort[~np.isnan(v_red_range)])
    # print(beta_i, dis_red_max, scale_i)
    if scale_i < scale_min or scale_i > scale_max:
        return -np.inf
    dis_red_muse_n = dis_red_muse / factor * scale_i
    dis_blue_muse_n = dis_blue_muse / factor * scale_i

    v_red, v_blue = f_v_red((dis_red_muse_n, *x[:3])), f_v_blue((dis_blue_muse_n, *x[:3]))
    w_red, w_blue = f_d_red((dis_red_muse_n, *x[:3])) * 2.563, f_d_blue((dis_blue_muse_n, *x[:3])) * 2.563

    # Compute chi square
    # if np.min(dis_red_muse) * scale_i < -55 or np.max(dis_blue_muse) * scale_i > 55:
    #     return -np.inf
    # print(v_red[np.argsort(dis_red_muse)], v_blue[np.argsort(dis_blue_muse)])
    # print(v50_red, v50_blue, w80_red, w80_blue)
    # v_red[np.isnan(v_red)], v_blue[np.isnan(v_blue)] = 500, -500
    # w_red[np.isnan(w_red)], w_blue[np.isnan(w_blue)] = 1000, 1000


    # Compute chi square
    v_red, v_blue = v_red + offset_i, v_blue + offset_i  # Offset it
    # chi2_red = ((v_red - v50_red) / v50_red_err) ** 2 + ((d_red - w80_red) / w80_red_err) ** 2
    # chi2_blue = ((v_blue - v50_blue) / v50_blue_err) ** 2 + ((d_blue - w80_blue) / w80_blue_err) ** 2
    chi2_array = np.hstack([((v_red - v50_red) / v50_red_err) ** 2, ((w_red - w80_red) / w80_red_err) ** 2,
                            ((v_blue - v50_blue) / v50_blue_err) ** 2, ((w_blue - w80_blue) / w80_blue_err) ** 2])
    sigma2_array = np.hstack([v50_red_err ** 2, w80_red_err ** 2, v50_blue_err ** 2, w80_blue_err ** 2])
    # print(-0.5 * np.nansum(np.log(2 * np.pi * sigma2_array)))
    # return - 0.5 * np.nansum(chi2_array + np.log(2 * np.pi * sigma2_array))
    # return - 0.5 * np.nansum(chi2_array) + 0.5 * np.nansum(np.log(2 * np.pi * sigma2_array))
    return - 0.5 * np.nansum(chi2_array)

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

rectangle = RectanglePixelRegion(center=PixCoord(x=50, y=50), width=110, height=5, angle=Angle(-30, 'deg'))
mask = rectangle.contains(pixcoord)
dis = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
dis_mask = dis[mask]

red = ((x[mask] - 50) < 0) * ((y[mask] - 50) > 0)
blue = ~red
dis_red = dis_mask[red] * -1
dis_blue = dis_mask[blue]
dis_red_sort = np.sort(dis_red)
dis_blue_sort = np.sort(dis_blue)

# Create a interpolated model
bin_red = np.flip(-np.arange(0, int(dis_blue.max()) + 3))  # blue and red is the same in magnitude
bin_blue = np.arange(0, int(dis_blue.max()) + 3)
dis_mid_red = (bin_red[1:] + bin_red[:-1]) / 2
dis_mid_blue = (bin_blue[1:] + bin_blue[:-1]) / 2
B1_array = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
vmax_array = np.array([200, 350, 500, 750, 1000, 1250, 1500])
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
                                       vmax=j_vmax, vtype='constant', bins=None, theta_in_deg=0,
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
rectangle_muse = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=50, height=5, angle=Angle(-30, 'deg'))
mask_muse = rectangle_muse.contains(pixcoord)
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

# Plot
dis_red_mean_plot, v50_red_mean_plot, _, _ = Bin(dis_red_muse, v50_red, bins=20)
dis_blue_mean_plot, v50_blue_mean_plot, _, _ = Bin(dis_blue_muse, v50_blue, bins=20)
_, w80_red_mean_plot, _, _ = Bin(dis_red_muse, w80_red, bins=20)
_, w80_blue_mean_plot, _, _ = Bin(dis_blue_muse, w80_blue, bins=20)

# Remove nan
dis_red_muse, dis_blue_muse = dis_red_muse[~np.isnan(v50_red)], dis_blue_muse[~np.isnan(v50_blue)]
v50_blue, v50_red = v50_blue[~np.isnan(v50_blue)], v50_red[~np.isnan(v50_red)]
w80_blue, w80_red = w80_blue[~np.isnan(w80_blue)], w80_red[~np.isnan(w80_red)]

# Mean
dis_red_mean, v50_red_mean, v50_red_mean_err, v50_red_err = Bin(dis_red_muse, v50_red, bins=20)
dis_blue_mean, v50_blue_mean, v50_blue_mean_err, v50_blue_err = Bin(dis_blue_muse, v50_blue, bins=20)
_, w80_red_mean, w80_red_mean_err, w80_red_err = Bin(dis_red_muse, w80_red, bins=20)
_, w80_blue_mean, w80_blue_mean_err, w80_blue_err = Bin(dis_blue_muse, w80_blue, bins=20)
v50_red_mean_err, v50_blue_mean_err = np.sqrt(v50_red_mean_err ** 2 + 30 ** 2), np.sqrt(
    v50_blue_mean_err ** 2 + 30 ** 2)
w80_red_mean_err, w80_blue_mean_err = np.sqrt(w80_red_mean_err ** 2 + 30 ** 2), np.sqrt(
    w80_blue_mean_err ** 2 + 30 ** 2)
v50_red_err, v50_blue_err = np.sqrt(v50_red_err ** 2 + 20 ** 2), np.sqrt(v50_blue_err ** 2 + 20 ** 2)
w80_red_err, w80_blue_err = np.sqrt(w80_red_err ** 2 + 20 ** 2), np.sqrt(w80_blue_err ** 2 + 20 ** 2)


# print(log_prob([10, 960, 16, 75], f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue,
#                w80_red, w80_blue, v50_red_err, v50_blue_err, w80_red_err, w80_blue_err))
# print(log_prob([30, 600, 26, 75], f_v_red, f_v_blue, f_d_red, f_d_blue, dis_red_muse, dis_blue_muse, v50_red, v50_blue,
#                w80_red, w80_blue, v50_red_err, v50_blue_err, w80_red_err, w80_blue_err))
#
# raise ValueError('stop')
# MCMC over it
nums_chain, nums_disc = 10000, 2000
nwalkers = 40
# ndim = 4
# p0 = np.array([40, 700, 25, 50]) + 0.1 * np.random.randn(nwalkers, ndim)
# labels = [r"$\beta_{1}$", r"$v_{\rm max}$", r"$\theta_{open}$", r'offset']

ndim = 5
p0 = np.array([45, 700, 25, 50, 1.5]) + 0.1 * np.random.randn(nwalkers, ndim)
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
            v50_red_err, v50_blue_err, w80_red_err, w80_blue_err)
            # dis_red_mean, dis_blue_mean, v50_red_mean, v50_red_mean_err, v50_blue_mean, v50_blue_mean_err,
            # w80_red_mean, w80_red_mean_err, w80_blue_mean, w80_blue_mean_err)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, backend=backend)
    state = sampler.run_mcmc(p0, nums_chain, progress=True)
    samples = sampler.get_chain(flat=True, discard=nums_disc)
    samples_corner = np.copy(samples)

median_values = np.median(samples, axis=0)
print('Median values:', median_values)
figure = corner.corner(samples_corner, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k',
                       title_kwargs={"fontsize": 13}, smooth=1., smooth1d=1., bins=25)
figure.savefig(figname_MCMC, bbox_inches='tight')

# Check
# v_red_range = f_v_red((dis_red_sort, *median_values[:3]))
# dis_red_max = np.abs(dis_red_sort[~np.isnan(v_red_range)][0])
# scale_i = dis_red_max / np.abs(np.min(dis_red_muse / factor))
# var_check = (*median_values, 1.5)
# var_check = (40, 600, 29, 1.5)
# var_check = (*median_values, scale_i)
var_check = (median_values[:])
# var_check = (50, 700, 25, 50, 1.5)
func = DrawBiconeModel(theta_B1_deg=var_check[0], theta_B2_deg=60, theta_B3_deg=45, vmax=var_check[1],
                       vtype='constant', bins=None, theta_in_deg=0, theta_out_deg=var_check[2], tau=1,
                       plot=False, save_fig=False)
func.Make2Dmap()
vmap = func.vmap
dmap = func.dmap
vmap, dmap = np.flip(vmap, 1), np.flip(dmap, 1)
vmap_flatten = vmap.flatten()
dmap_flatten = dmap.flatten()
vmap_blue, vmap_red = vmap_flatten[mask][blue], vmap_flatten[mask][red]
dmap_blue, dmap_red = dmap_flatten[mask][blue], dmap_flatten[mask][red]


# v_1 = f_v_red((dis_mid_red, 5, 800, 30))
# v_2 = f_v_red((dis_mid_red, 50, 800, 30))
# print(dis_mid_red, v_1, v_2)
# print(dis_red_sort, vmap_red[np.argsort(dis_red)])
# print(f_v_red((dis_red_sort, *var_check[:3])))
# raise ValueError('Stop')

# Save cone result
extent = [median_values[4] * 75 / 2 - 50 / median_values[4], median_values[4] * 75 / 2 + 50 / median_values[4],
          median_values[4] * 75 / 2 - 50 / median_values[4], median_values[4] * 75 / 2 + 50 / median_values[4]]

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.imshow(vmap, origin='lower', cmap='coolwarm', vmin=-350, vmax=350, extent=extent)
# patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
ax.scatter(median_values[4] * 75 / 2, median_values[4] * 75 / 2, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=3000, zorder=100)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(0, median_values[4] * 75)
ax.set_ylim(0, median_values[4] * 75)
plt.savefig('../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_vmap.png', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.imshow(dmap * 2.563, origin='lower', vmin=0, vmax=800, cmap=Dense_20_r.mpl_colormap, extent=extent)
# patch = rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
ax.scatter(median_values[4] * 75 / 2, median_values[4] * 75 / 2, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=3000, zorder=100)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(0, median_values[4] * 75)
ax.set_ylim(0, median_values[4] * 75)
plt.savefig('../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_dmap.png', bbox_inches='tight')

# path_vmap_MCMC = '../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_vmap.fits'
# path_dmap_MCMC = '../../MUSEQuBES+CUBS/fit_bic/PV_MCMC_fit_dmap.fits'
# hdul_vmap_MCMC = fits.ImageHDU(vmap, header=None)
# hdul_vmap_MCMC.writeto(path_vmap_MCMC, overwrite=True)
# hdul_dmap_MCMC = fits.ImageHDU(dmap * 2.563, header=None)
# hdul_dmap_MCMC.writeto(path_dmap_MCMC, overwrite=True)


plt.close('all')
fig, ax = plt.subplots(2, 1, figsize=(10, 7), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.0)

# Interpolation and MCMC
# Make a group of points
dis_red_kpc, dis_blue_kpc = np.flip(-np.linspace(3, 37, 100)), np.linspace(3, 37, 100)
draw = np.random.choice(len(samples), size=4000, replace=False)
samples_draw = samples[draw]
dis_red_pix = dis_red_kpc[:, np.newaxis] / factor * samples_draw[:, 4]
dis_blue_pix = dis_blue_kpc[:, np.newaxis] / factor * samples_draw[:, 4]
v_all_red = f_v_red((dis_red_pix, samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2]))
v_all_blue = f_v_blue((dis_blue_pix, samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2]))
d_all_red = f_d_red((dis_red_pix, samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2])) * 2.563
d_all_blue = f_d_blue((dis_blue_pix, samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2])) * 2.563

#
dis_red_res, dis_blue_res = dis_red_pix / samples_draw[:, 4] * factor, dis_blue_pix / samples_draw[:, 4] * factor
dis_red_res_mean, _, v_red_max, v_red_min = Bin_maxmin(dis_red_res.ravel(), (v_all_red + samples_draw[:, 3]).ravel(), bins=20)
dis_blue_res_mean, _, v_blue_max, v_blue_min = Bin_maxmin(dis_blue_res.ravel(), (v_all_blue + samples_draw[:, 3]).ravel(), bins=20)
_, _, w_red_max, w_red_min = Bin_maxmin(dis_red_res.ravel(), d_all_red.ravel(), bins=20)
_, _, w_blue_max, w_blue_min = Bin_maxmin(dis_blue_res.ravel(), d_all_blue.ravel(), bins=20)

# dis_red_show, dis_blue_show = np.flip(-np.linspace(1, 60, 100)), np.linspace(1, 60, 100)
ax[0].plot(dis_red_res_mean, f_v_red((dis_red_res_mean / factor * var_check[4], *var_check[:3]))
           + var_check[3], lw=2, color='purple', alpha=0.8, label='Biconical model', zorder=-50)
ax[0].plot(dis_blue_res_mean, f_v_blue((dis_blue_res_mean / factor * var_check[4], *var_check[:3])) + var_check[3], lw=2,
           color='purple', alpha=0.8)
ax[1].plot(dis_red_res_mean, f_d_red((dis_red_res_mean / factor * var_check[4], *var_check[:3])) * 2.563, lw=2,
           color='purple', alpha=0.8)
ax[1].plot(dis_blue_res_mean, f_d_blue((dis_blue_res_mean / factor * var_check[4], *var_check[:3])) * 2.563, lw=2,
           color='purple', alpha=0.8)

ax[0].fill_between(dis_red_res_mean, v_red_min, v_red_max, color='purple', alpha=0.3)
ax[0].fill_between(dis_blue_res_mean, v_blue_min, v_blue_max, color='purple', alpha=0.3)
ax[1].fill_between(dis_red_res_mean, w_red_min, w_red_max, color='purple', alpha=0.3)
ax[1].fill_between(dis_blue_res_mean, w_blue_min, w_blue_max, color='purple', alpha=0.3)
# ax[0].fill_between(dis_red_show * factor / var_check[4], np.min(v_all_red, axis=1) + var_check[3],
#                    np.max(v_all_red, axis=1) + var_check[3], color='purple', alpha=0.3)
# ax[0].fill_between(dis_blue_show * factor / var_check[4], np.min(v_all_blue, axis=1) + var_check[3],
#                    np.max(v_all_blue, axis=1) + var_check[3], color='purple', alpha=0.3)
# ax[1].fill_between(dis_red_show * factor / var_check[4], np.min(d_all_red, axis=1), np.max(d_all_red, axis=1),
#                    color='purple', alpha=0.3)
# ax[1].fill_between(dis_blue_show * factor / var_check[4], np.min(d_all_blue, axis=1), np.max(d_all_blue, axis=1),
#                    color='purple', alpha=0.3)


# ax[0].scatter(dis_red, vmap_red_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
#               label=r'$\rm 3C\,57 \, northeast$')
# ax[0].scatter(dis_blue, vmap_blue_array[:, 1, 1, 1], s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
#               label=r'$\rm 3C\,57 \, southwest$')
# ax[1].scatter(dis_red, dmap_red_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue, dmap_blue_array[:, 1, 1, 1] * 2.563, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue')
# print(dis_red * factor / var_check[4])
# ax[0].scatter(dis_red * factor / var_check[4], vmap_red + var_check[3], s=50, marker='^', edgecolors='k', linewidths=0.5, color='red')
# ax[0].scatter(dis_blue * factor / var_check[4], vmap_blue + var_check[3], s=50, marker='^', edgecolors='k', linewidths=0.5, color='blue')
# ax[1].scatter(dis_red * factor / var_check[4], dmap_red * 2.563, s=50, marker='^', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue * factor / var_check[4], dmap_blue * 2.563, s=50, marker='^', edgecolors='k', linewidths=0.5, color='blue')
# ax[0].scatter(dis_red, vmap_red + var_check[3], s=50, marker='^', edgecolors='k', linewidths=0.5, color='red')
# ax[0].scatter(dis_blue, vmap_blue + var_check[3], s=50, marker='^', edgecolors='k', linewidths=0.5, color='blue')
# ax[1].scatter(dis_red, dmap_red * 2.563, s=50, marker='^', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue, dmap_blue * 2.563, s=50, marker='^', edgecolors='k', linewidths=0.5, color='blue')

# MUSE
# ax[0].scatter(dis_red_muse, v50_red, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red')
# ax[0].scatter(dis_blue_muse, v50_blue, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue')
# ax[1].scatter(dis_red_muse, w80_red, s=50, marker='*', edgecolors='k', linewidths=0.5, color='red')
# ax[1].scatter(dis_blue_muse, w80_blue, s=50, marker='*', edgecolors='k', linewidths=0.5, color='blue')
# ax[0].errorbar(dis_red_mean, v50_red_mean, yerr=v50_red_mean_err, fmt='o', color='red')
# ax[0].errorbar(dis_blue_mean, v50_blue_mean, yerr=v50_blue_mean_err, fmt='o', color='blue')
# ax[1].errorbar(dis_red_mean, w80_red_mean, yerr=w80_red_mean_err, fmt='o', color='red')
# ax[1].errorbar(dis_blue_mean, w80_blue_mean, yerr=w80_blue_mean_err, fmt='o', color='blue')
ax[0].scatter(dis_red_mean_plot, v50_red_mean_plot, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red',
              label=r'$\rm 3C\,57 \, northeast$', zorder=100)
ax[0].scatter(dis_blue_mean_plot, v50_blue_mean_plot, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue',
              label=r'$\rm 3C\,57 \, southwest$', zorder=100)
ax[1].scatter(dis_red_mean_plot, w80_red_mean_plot, s=50, marker='D', edgecolors='k', linewidths=0.5, color='red', zorder=100)
ax[1].scatter(dis_blue_mean_plot, w80_blue_mean_plot, s=50, marker='D', edgecolors='k', linewidths=0.5, color='blue', zorder=100)
# ax[0].errorbar(dis_red_muse, v50_red, v50_red_err, fmt='.r', capsize=0, capthick=1, mfc=None, ms=10, zorder=-100, alpha=0.2)
# ax[0].errorbar(dis_blue_muse, v50_blue, v50_blue_err, fmt='.b', capsize=0, capthick=1, mfc=None, ms=10, zorder=-100, alpha=0.2)
# ax[1].errorbar(dis_red_muse, w80_red, w80_red_err, fmt='.r', capsize=0, capthick=1, mfc=None, ms=10, zorder=-100, alpha=0.2)
# ax[1].errorbar(dis_blue_muse, w80_blue, w80_blue_err, fmt='.b', capsize=0, capthick=1, mfc=None, ms=10, zorder=-100, alpha=0.2)

ax[0].axhline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[1].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
ax[0].set_xlim(-40, 40)
ax[0].set_ylim(-450, 450)
ax[1].set_ylim(0, 520)
ax[0].set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
ax[1].set_xlabel(r'$\rm Distance \, [pkpc]$', size=25)
ax[1].set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25, labelpad=20)
ax[0].legend(loc='best', fontsize=15)
fig.savefig('../../MUSEQuBES+CUBS/fit_bic/PV_cone.png', bbox_inches='tight')
