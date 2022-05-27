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
from astropy.io import ascii
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

#
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

# path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits_SUBTRACTED.fits')
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

ConvertFits(filename='image_OOHbeta_fitline', table=np.log10(line_OIII5008 / line_OII))

fig = plt.figure(figsize=(8, 8), dpi=300)
path_lr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_lineratio.fits')
gc = aplpy.FITSFigure(path_lr, figure=fig, north=True)
gc.set_system_latex(True)
gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_8.mpl_colormap)
gc.add_colorbar()
# gc.colorbar.set_box([0.1247, 0.0927, 0.7443, 0.03], box_orientation='horizontal')
gc.ticks.set_length(30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='none', edgecolors='k',
                linewidths=0.5, s=250)
# gc.show_markers(ra, dec, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
# gc.show_markers(ra, dec, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=-300, vmax=300, cmap='coolwarm')
# gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
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

ConvertFits(filename='image_OOHbeta_fitline', table=np.log10(line_OIII5008 / line_Hbeta))
fig = plt.figure(figsize=(8, 8), dpi=300)
path_lr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_lineratio.fits')
gc = aplpy.FITSFigure(path_lr, figure=fig, north=True)
gc.set_system_latex(True)
gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_8.mpl_colormap)
gc.add_colorbar()
# gc.colorbar.set_box([0.1247, 0.0927, 0.7443, 0.03], box_orientation='horizontal')
gc.ticks.set_length(30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='none', edgecolors='k',
                linewidths=0.5, s=250)
# gc.show_markers(ra, dec, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
# gc.show_markers(ra, dec, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=-300, vmax=300, cmap='coolwarm')
# gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
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
# axarr[0].plot(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), '.', color='blue', ms=5)
# axarr[1].plot(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), '.', color='blue', ms=5)
OIII_Hbeta_err_array_log = OIII_Hbeta_err_array / np.log(10) / OIII_Hbeta_array
OIII_OII_err_array_log = OIII_OII_err_array / np.log(10) / OIII_OII_array
OIII_Hbeta_sigma = OIII_array / Hbeta_sigma_array
OIII_Hbeta_3sigma_err = (OIII_array / (3 * Hbeta_sigma_array)) - OIII_Hbeta_sigma
OIII_Hbeta_3sigma_err_log = OIII_Hbeta_3sigma_err / np.log(10) / OIII_Hbeta_sigma
axarr[0].errorbar(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), yerr=OIII_OII_err_array_log,
                  xerr=OIII_Hbeta_err_array_log, fmt='.k', capsize=2, elinewidth=1, mfc='k', ms=10)
axarr[0].errorbar(np.log10(OIII_Hbeta_sigma)[-1], np.log10(OIII_OII_array)[-1],
                  yerr=OIII_OII_err_array_log[-1], xerr=OIII_Hbeta_3sigma_err_log[-1], fmt='.k',
                  capsize=2, elinewidth=1, mfc='red', ms=10)
axarr[1].errorbar(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), yerr=OIII_OII_err_array_log,
                  xerr=OIII_Hbeta_err_array_log, fmt='.k', capsize=2, elinewidth=1, mfc='k', ms=10)
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


