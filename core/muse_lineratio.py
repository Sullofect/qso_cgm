import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import palettable.scientific.sequential as sequential_s
from matplotlib import rc
from matplotlib import cm
from PyAstronomy import pyasl
from astropy import units as u
from matplotlib.colors import ListedColormap
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)


def ConvertFits(filename=None, table=None):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_lineratio.fits', table, overwrite=True)
    data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_lineratio.fits', 0, header=True)
    hdr1['BITPIX'], hdr1['NAXIS'], hdr1['NAXIS1'], hdr1['NAXIS2'] = hdr['BITPIX'], hdr['NAXIS'], \
                                                                    hdr['NAXIS1'], hdr['NAXIS2']
    hdr1['CRPIX1'], hdr1['CRPIX2'], hdr1['CTYPE1'], hdr1['CTYPE2'] = hdr['CRPIX1'], hdr['CRPIX2'], \
                                                                     hdr['CTYPE1'], hdr['CTYPE2']
    hdr1['CRVAL1'], hdr1['CRVAL2'], hdr1['LONPOLE'], hdr1['LATPOLE'] = hdr['CRVAL1'], hdr['CRVAL2'], \
                                                                       hdr['LONPOLE'], hdr['LATPOLE']
    hdr1['CSYER1'], hdr1['CSYER2'], hdr1['MJDREF'], hdr1['RADESYS'] = hdr['CSYER1'], hdr['CSYER2'], \
                                                                      hdr['MJDREF'], hdr['RADESYS']
    hdr1['CD1_1'], hdr1['CD1_2'], hdr1['CD2_1'], hdr1['CD2_2'] = hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2']

    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_lineratio.fits', data1, hdr1, overwrite=True)


path_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
path_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_Hbeta_line_offset.fits')
path_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_4960_line_offset.fits')
path_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_5008_line_offset.fits')
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'gas_list_revised.reg')

# Sampled region
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

# Muse Cube
cube_OII = Cube(path_OII)
cube_Hbeta = Cube(path_Hbeta)
cube_OIII4960 = Cube(path_OIII4960)
cube_OIII5008 = Cube(path_OIII5008)
wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
wave_Hbeta_vac = pyasl.airtovac2(cube_Hbeta.wave.coord())
wave_OIII4960_vac = pyasl.airtovac2(cube_OIII4960.wave.coord())
wave_OIII5008_vac = pyasl.airtovac2(cube_OIII5008.wave.coord())
wave_vac_stack = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_all = np.array([wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)

flux_OII, flux_Hbeta = cube_OII.data * 1e-3, cube_Hbeta.data * 1e-3
flux_OIII4960, flux_OIII5008 = cube_OIII4960.data * 1e-3, cube_OIII5008.data * 1e-3
flux_OII_err, flux_Hbeta_err = np.sqrt(cube_OII.var) * 1e-3, np.sqrt(cube_Hbeta.var) * 1e-3
flux_OIII4960_err = np.sqrt(cube_OIII4960.var) * 1e-3
flux_OIII5008_err = np.sqrt(cube_OIII5008.var) * 1e-3

# Direct integration for every pixel
line_OII = 1.25 * integrate.simps(flux_OII, axis=0)
line_Hbeta = 1.25 * integrate.simps(flux_Hbeta, axis=0)
line_OIII4960 = 1.25 * integrate.simps(flux_OIII4960, axis=0)
line_OIII5008 = 1.25 * integrate.simps(flux_OIII5008, axis=0)

# Fig 1
ConvertFits(filename='image_OOHbeta_fitline', table=np.log10(line_OIII5008 / line_OII))
fig = plt.figure(figsize=(8, 8), dpi=300)
path_lr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_lineratio.fits')
gc = aplpy.FITSFigure(path_lr, figure=fig, north=True)
gc.set_system_latex(True)
gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_8.mpl_colormap)
gc.add_colorbar()
gc.ticks.set_length(30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='none', edgecolors='k',
                linewidths=0.5, s=250)
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.)
gc.colorbar.set_axis_label_text(r'$\mathrm{log[O \, III] / [O \, II]}$')
gc.colorbar.set_font(size=15)
gc.colorbar.set_axis_label_font(size=15)
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatioMap_OIII_OII.png', bbox_inches='tight')

# Fig 2
ConvertFits(filename='image_OOHbeta_fitline', table=np.log10(line_OIII5008 / line_Hbeta))
fig = plt.figure(figsize=(8, 8), dpi=300)
path_lr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_lineratio.fits')
gc = aplpy.FITSFigure(path_lr, figure=fig, north=True)
gc.set_system_latex(True)
gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_8.mpl_colormap)
gc.add_colorbar()
gc.ticks.set_length(30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='none', edgecolors='k',
                linewidths=0.5, s=250)
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.)
gc.colorbar.set_axis_label_text(r'$\mathrm{log[O \, III] / H \, \beta}$')
gc.colorbar.set_font(size=15)
gc.colorbar.set_axis_label_font(size=15)
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatioMap_OIII_Hbeta.png', bbox_inches='tight')

# Compared with the fitted result
line = 'OOHbeta'
path_fit_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                             'fit' + line + '_info_aperture_1.0.fits')
path_fit_info_err = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                                 'fit' + line + '_info_err_aperture_1.0.fits')
fit_info = fits.getdata(path_fit_info, 0, ignore_missing_end=True)
fit_info_err = fits.getdata(path_fit_info_err, 0, ignore_missing_end=True)


[z_fit, r_fit, fit_success, sigma_fit, flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII5008, a_fit_OII, a_fit_Hbeta,
 a_fit_OIII4960, a_fit_OIII5008, b_fit_OII, b_fit_Hbeta, b_fit_OIII4960, b_fit_OIII5008] = fit_info
[dz_fit, dr_fit, dsigma_fit, dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII5008, da_fit_OII, da_fit_Hbeta,
 da_fit_OIII4960, da_fit_OIII5008, db_fit_OII, db_fit_Hbeta, db_fit_OIII4960, db_fit_OIII5008] = fit_info_err

line_OII_fitted = 1.25 * flux_fit_OII
line_Hbeta_fitted = 1.25 * flux_fit_Hbeta
line_OIII4960_fitted = 1.25 * flux_fit_OIII5008 / 3
line_OIII5008_fitted = 1.25 * flux_fit_OIII5008

# Fig 3
ConvertFits(filename='image_OOHbeta_fitline', table=np.log10(line_OIII5008_fitted / line_OII_fitted))
fig = plt.figure(figsize=(8, 8), dpi=300)
path_lr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_lineratio.fits')
gc = aplpy.FITSFigure(path_lr, figure=fig, north=True)
gc.set_system_latex(True)
gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_8.mpl_colormap)
gc.add_colorbar()
gc.ticks.set_length(30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='none', edgecolors='k',
                linewidths=0.5, s=250)
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.)
gc.colorbar.set_axis_label_text(r'$\mathrm{log[O \, III] / [O \, II]}$')
gc.colorbar.set_font(size=15)
gc.colorbar.set_axis_label_font(size=15)
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatioMap_OIII_OII_fitted.png', bbox_inches='tight')

# Fig 4
ConvertFits(filename='image_OOHbeta_fitline', table=np.log10(line_OIII5008_fitted / line_Hbeta_fitted))
fig = plt.figure(figsize=(8, 8), dpi=300)
path_lr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_lineratio.fits')
gc = aplpy.FITSFigure(path_lr, figure=fig, north=True)
gc.set_system_latex(True)
gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_8.mpl_colormap)
gc.add_colorbar()
gc.ticks.set_length(30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='none', edgecolors='k',
                linewidths=0.5, s=250)
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.)
gc.colorbar.set_axis_label_text(r'$\mathrm{log[O \, III] / H \, \beta}$')
gc.colorbar.set_font(size=15)
gc.colorbar.set_axis_label_font(size=15)
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatioMap_OIII_Hbeta_fitted.png', bbox_inches='tight')


# Calculate line ratio in sample region
OIII_OII_array, OIII_OII_err_array = np.zeros(len(ra_array)), np.zeros(len(ra_array))
OIII_Hbeta_array, OIII_Hbeta_err_array = np.zeros(len(ra_array)), np.zeros(len(ra_array))
OIII_array, Hbeta_sigma_array = np.zeros(len(ra_array)), np.zeros(len(ra_array))
for i in range(len(ra_array)):
    spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
    spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    spe_OIII4960_i = cube_OIII4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    spe_OIII5008_i = cube_OIII5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)

    # Load the data
    flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
    flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
    flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
    flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3
    flux_all = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
    flux_err_all = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

    # Direct integrations
    line_OII_i = 1.25 * integrate.simps(flux_OII_i)
    line_Hbeta_i = 1.25 * integrate.simps(flux_Hbeta_i)
    line_OIII4960_i = 1.25 * integrate.simps(flux_OIII4960_i)
    line_OIII5008_i = 1.25 * integrate.simps(flux_OIII5008_i)
    OIII_array[i] = line_OIII5008_i
    # Error
    error_OII_i = np.sqrt(1.25 * integrate.simps(flux_OII_err_i ** 2))
    error_Hbeta_i = np.sqrt(1.25 * integrate.simps(flux_Hbeta_err_i ** 2))
    error_OIII4960_i = np.sqrt(1.25 * integrate.simps(flux_OIII4960_err_i ** 2))
    error_OIII5008_i = np.sqrt(1.25 * integrate.simps(flux_OIII5008_err_i ** 2))

    #
    Hbeta_sigma_array[i] = error_Hbeta_i
    OIII_OII_array[i] = line_OIII5008_i / line_OII_i
    OIII_OII_err_array[i] = (line_OIII5008_i / line_OII_i) *\
                            np.sqrt((error_OIII5008_i / line_OIII5008_i) ** 2 + (error_OII_i / line_OII_i) ** 2)
    OIII_Hbeta_array[i] = line_OIII5008_i / line_Hbeta_i
    OIII_Hbeta_err_array[i] = (line_OIII5008_i / line_Hbeta_i) *\
                            np.sqrt((error_OIII5008_i / line_OIII5008_i) ** 2 + (error_Hbeta_i / line_Hbeta_i) ** 2)
# Load fitted result
path_fit_info_sr= os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'line_profile_selected_region.fits')
data_fit_info_sr = fits.getdata(path_fit_info_sr, 0, ignore_missing_end=True)
flux_OII_sr, flux_Hbeta_sr, flux_OIII5008_sr = data_fit_info_sr[:, 0], data_fit_info_sr[:, 1], data_fit_info_sr[:, 2]
dflux_OII_sr, dflux_Hbeta_sr, dflux_OIII5008_sr = data_fit_info_sr[:, 3], data_fit_info_sr[:, 4], data_fit_info_sr[:, 5]

# Fitted result
OIII_Hbeta_fitted = flux_OIII5008_sr / flux_Hbeta_sr
OIII_Hbeta_err_fitted = (flux_OIII5008_sr / flux_Hbeta_sr) *\
                            np.sqrt((dflux_OIII5008_sr / flux_OIII5008_sr) ** 2 + (dflux_Hbeta_sr / flux_Hbeta_sr) ** 2)
OIII_Hbeta_err_fitted_log = OIII_Hbeta_err_fitted / np.log(10) / OIII_Hbeta_fitted

OIII_OII_fitted = flux_OIII5008_sr / flux_OII_sr
OIII_OII_err_fitted = (flux_OIII5008_sr / flux_OII_sr) *\
                            np.sqrt((dflux_OIII5008_sr / flux_OIII5008_sr) ** 2 + (dflux_OII_sr / flux_OII_sr) ** 2)
OIII_OII_err_fitted_log = OIII_OII_err_fitted / np.log(10) / OIII_OII_fitted

# Load grid
path_df = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                       'iteramodel_dustfreeAGN_Z1.0_n100.txt_grid1.txt')
path_dy = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                       'iteramodel_dustyAGN_Z1.0_n100.txt_grid1.txt')

grid_df = np.loadtxt(path_df, encoding="ISO-8859-1")
grid_dy = np.loadtxt(path_dy, encoding="ISO-8859-1")

alpha_df, alpha_dy = grid_df[:, 0], grid_dy[:, 0]
logu_df, logu_dy = grid_df[:, 1], grid_dy[:, 1]
OIII_Hbeta_df, OIII_Hbeta_dy = grid_df[:, 2], grid_dy[:, 2]
OIII_OII_df, OIII_OII_dy = grid_df[:, 4] / grid_df[:, 5], grid_dy[:, 4] / grid_df[:, 5]

OIII_Hbeta_df_mat, OIII_Hbeta_dy_mat = OIII_Hbeta_df.reshape((4, 13)), OIII_Hbeta_dy.reshape((4, 13))
OIII_OII_df_mat, OIII_OII_dy_mat = OIII_OII_df.reshape((4, 13)), OIII_OII_dy.reshape((4, 13))

fig, axarr = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)

# Fill between
x_1, x_2 = np.log10(OIII_Hbeta_df_mat), np.log10(OIII_Hbeta_dy_mat)
y_1, y_2 = np.log10(OIII_OII_df_mat), np.log10(OIII_OII_dy_mat)
y_1_sort, y_2_sort = np.sort(y_1, axis=1), np.sort(y_2, axis=1)
x_1_sort, x_2_sort = np.take_along_axis(x_1, np.argsort(y_1, axis=1), axis=1), \
                     np.take_along_axis(x_2, np.argsort(y_2, axis=1), axis=1)
axarr[0].fill(np.hstack((x_1_sort[0, :], x_1_sort[1, ::-1])), np.hstack((y_1_sort[0, :], y_1_sort[1, ::-1])),
              color='grey', alpha=0.2)
axarr[0].fill(np.hstack((x_1_sort[1, :], x_1_sort[2, ::-1])), np.hstack((y_1_sort[1, :], y_1_sort[2, ::-1])),
              color='grey', alpha=0.4)
axarr[0].fill(np.hstack((x_1_sort[2, 2:], x_1_sort[3, ::-1][2:])), np.hstack((y_1_sort[2, 2:], y_1_sort[3, ::-1][2:])),
              color='grey', alpha=0.6)
axarr[1].fill(np.hstack((x_2_sort[0, :], x_2_sort[1, ::-1])), np.hstack((y_2_sort[0, :], y_2_sort[1, ::-1])),
              color='red', alpha=0.2)
axarr[1].fill(np.hstack((x_2_sort[1, :], x_2_sort[2, ::-1])), np.hstack((y_2_sort[1, :], y_2_sort[2, ::-1])),
              color='red', alpha=0.4)
axarr[1].fill(np.hstack((x_2_sort[2, 2:], x_2_sort[3, ::-1][2:])), np.hstack((y_2_sort[2, 2:], y_2_sort[3, ::-1][2:])),
              color='red', alpha=0.6)

# Plot Data
OIII_Hbeta_err_array_log = OIII_Hbeta_err_array / np.log(10) / OIII_Hbeta_array
OIII_OII_err_array_log = OIII_OII_err_array / np.log(10) / OIII_OII_array
OIII_Hbeta_3sigma = OIII_array / (3 * Hbeta_sigma_array)
OIII_Hbeta_3sigma_fitted = flux_OIII5008_sr / (3 * dflux_Hbeta_sr)
# OIII_Hbeta_3sigma_err = (OIII_array / (3 * Hbeta_sigma_array)) - OIII_Hbeta_3sigma
# OIII_Hbeta_3sigma_err_log = OIII_Hbeta_3sigma_err / np.log(10) / OIII_Hbeta_3sigma
print(OIII_Hbeta_array)
for i, ival in enumerate(OIII_Hbeta_array):
    axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_array)[i] + 0.1, np.log10(OIII_OII_array)[i] - 0.8),
                      size=10, color='red', verticalalignment='top', horizontalalignment='right')
    axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_fitted)[i] + 0.1, np.log10(OIII_OII_fitted)[i] - 0.8),
                      size=10, color='orange', verticalalignment='top', horizontalalignment='right')
    if ival <= 0:
        axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_3sigma)[i] + 0.1, np.log10(OIII_OII_array)[i] - 0.8) ,
                          color='red', size=10, verticalalignment='top', horizontalalignment='right')
        axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_3sigma_fitted)[i] + 0.1,
                                             np.log10(OIII_OII_fitted)[i] - 0.8),
                          color='orange', size=10, verticalalignment='top', horizontalalignment='right')
        axarr[0].errorbar(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], fmt='.k', capsize=2,
                          elinewidth=1, mfc='red', xuplims=True, ms=10, yerr=OIII_OII_err_array_log[i],
                          xerr=[[0.0], [0]])
        axarr[0].errorbar(np.log10(OIII_Hbeta_3sigma_fitted)[i], np.log10(OIII_OII_fitted)[i], fmt='.k', capsize=2,
                          elinewidth=1, mfc='orange', xuplims=True, ms=10, yerr=OIII_OII_err_fitted_log[i],
                          xerr=[[0.0], [0]])
        axarr[0].arrow(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], dx=-0.5, dy=0,
                       facecolor='red', width=0.005, head_width=0.05, head_length=0.08)
        axarr[0].arrow(np.log10(OIII_Hbeta_3sigma_fitted)[i], np.log10(OIII_OII_fitted)[i], dx=-0.5, dy=0,
                       facecolor='orange', width=0.005, head_width=0.05, head_length=0.08)
        # axarr[1].errorbar(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], fmt='.k', capsize=2,
        #                   elinewidth=1, mfc='red', xuplims=True, ms=10, yerr=OIII_OII_err_array_log[i],
        #                   xerr=[[0.5], [0]])


axarr[0].errorbar(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), yerr=OIII_OII_err_array_log,
                  xerr=OIII_Hbeta_err_array_log, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
axarr[0].errorbar(np.log10(OIII_Hbeta_fitted), np.log10(OIII_OII_fitted), yerr=OIII_OII_err_fitted_log,
                  xerr=OIII_Hbeta_err_fitted_log, fmt='.k', capsize=2, elinewidth=1, mfc='orange', ms=10)


axarr[1].errorbar(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), yerr=OIII_OII_err_array_log,
                  xerr=OIII_Hbeta_err_array_log, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
axarr[1].errorbar(np.log10(OIII_Hbeta_fitted), np.log10(OIII_OII_fitted), yerr=OIII_OII_err_fitted_log,
                  xerr=OIII_Hbeta_err_fitted_log, fmt='.k', capsize=2, elinewidth=1, mfc='orange', ms=10)
# axarr[0].plot(np.log10(OIII_Hbeta_df), np.log10(OIII_OII_df), '.r', ms=3)
axarr[0].plot(np.log10(OIII_Hbeta_df_mat), np.log10(OIII_OII_df_mat), '-', color='grey', lw=1, alpha=0.3)
# axarr[1].plot(np.log10(OIII_Hbeta_dy), np.log10(OIII_OII_dy), '.r', ms=3)
axarr[1].plot(np.log10(OIII_Hbeta_dy_mat), np.log10(OIII_OII_dy_mat), '-', color='grey', lw=1, alpha=0.3)

fig.supxlabel(r'$\mathrm{log([O \, III]\lambda5008 / H\beta)}$', size=20, y=0.02)
fig.supylabel(r'$\mathrm{log([O \, III]\lambda5008  / [O \, II] \lambda \lambda 3727,29)}$', size=20, x=0.05)
axarr[0].minorticks_on()
axarr[1].minorticks_on()
axarr[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                     labelsize=20, size=5)
axarr[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                     size=3)
axarr[1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                     labelsize=20, size=5)
axarr[1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                     size=3)
# axarr[0].set_xlim(-0.2, 1.7)
# axarr[0].set_ylim(-1.2, 1.2)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatio_region.png', bbox_inches='tight')

