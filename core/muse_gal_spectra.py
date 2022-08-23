import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from astropy.table import Table
from PyAstronomy import pyasl
from muse_compare_z import compare_z
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)

def LoadGalData(row=None, qls=False):
    global ID_final, row_final, name_final
    row_sort = np.where(row_final == row)
    if qls is True:
        path_spe = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting',
                                'Subtracted_ESO_DEEP_offset_zapped_spec1D', str(row) + '_' + str(ID_final[row_sort][0])
                                + '_' + name_final[row_sort][0] + '_spec1D.fits')
    elif qls is False:
        path_spe = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting',
                                'ESO_DEEP_offset_zapped_spec1D', str(row) + '_' + str(ID_final[row_sort][0])
                                + '_' + name_final[row_sort][0] + '_spec1D.fits')
    spec = Table.read(path_spe)
    # spec = spec[spec['mask'] == 1]

    wave = spec['wave']  # in vacuum
    flux = spec['flux'] * 1e-3
    flux_err = spec['error'] * 1e-3
    model = spec['model'] * 1e-3
    return wave, flux, flux_err, model

# Example
# path_1 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
#                       '1_20002_J024034.53-185135.09_spec1D.fits')
# spec = Table.read(path_1)
# wave_1 = pyasl.vactoair2(spec['wave'])
# flux_1 = spec['flux'] * 1e-3
# model_1 = spec['model'] * 1e-3
# flux_err_1 = spec['error'] * 1e-3
#
# path_2 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
#                       '64_271_J024032.03-185139.81_spec1D.fits')
# spec = Table.read(path_2)
# wave_2 = pyasl.vactoair2(spec['wave'])
# flux_2 = spec['flux'] * 1e-3
# model_2 = spec['model'] * 1e-3
# flux_err_2 = spec['error'] * 1e-3
#
# path_3 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
#                       '88_301_J024030.95-185143.68_spec1D.fits')
# spec = Table.read(path_3)
# wave_3 = pyasl.vactoair2(spec['wave'])
# flux_3 = spec['flux'] * 1e-3
# model_3 = spec['model'] * 1e-3
# flux_err_3 = spec['error'] * 1e-3


def PlotGalSpectra(row_array=None, qls=False, figname='spectra_gal'):
    global z_final
    fig, axarr = plt.subplots(len(row_array), 1, figsize=(10, 2.5 * len(row_array)), sharex=True, dpi=300)
    plt.subplots_adjust(hspace=0.0)

    for i in range(len(row_array)):
        row_sort = np.where(row_final == row_array[i])
        z_i = z_final[row_sort]
        wave_i, flux_i, flux_err_i, model_i = LoadGalData(row=row_array[i], qls=qls)
        blue_mask, red_mask = np.where((wave_i < (1 + z_i) * 3950) * (wave_i > (1 + z_i) * 3850)), \
                              np.where((wave_i < (1 + z_i) * 4100) * (wave_i > (1 + z_i) * 4000))
        red = np.mean(flux_i[red_mask])  # Compute errors and report mean
        blue = np.mean(flux_i[blue_mask])
        print(red / blue)
        axarr[i].plot(wave_i, flux_i, color='k', drawstyle='steps-mid', lw=1)
        axarr[i].plot(wave_i, model_i, color='r', lw=1)
        axarr[i].plot(wave_i, flux_err_i, color='lightgrey', lw=1)
        axarr[i].set_title('G' + str(row_array[i]), x=0.1, y=0.80, size=20)
        axarr[0].annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.2, 0.65),
                          xycoords='axes fraction', size=20)
        axarr[0].annotate(text=r'$\mathrm{K, H, G}$', xy=(0.39, 0.65), xycoords='axes fraction', size=20)
        axarr[0].annotate(text=r'$\mathrm{H\beta}$', xy=(0.65, 0.65), xycoords='axes fraction', size=20)
        axarr[0].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.80, 0.65),
                          xycoords='axes fraction', size=20)
        lines = (1 + z_i) * np.array([3727.092, 3729.8754960, 3934.777, 3969.588, 4305.61, 4862.721, 4960.295, 5008.239])
        axarr[i].vlines(lines, lw=1,
                        ymin=[-5, -5, -5, -5, -5, -5, -5, -5],
                        ymax=[100, 100, 100, 100, 100, 100, 100, 100],
                        linestyles='dashed', colors='grey')
        axarr[i].set_xlim(4800, 9200)
        axarr[i].set_ylim(-0.1, flux_i.max() + 0.1)
        axarr[i].minorticks_on()
        axarr[i].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                             labelsize=20, size=5)
        axarr[i].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                             size=3)
    fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
    fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '.png', bbox_inches='tight')


# Load info
ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                     cat_will='ESO_DEEP_offset_zapped_objects.fits')
bins_final = ggp_info[0]
row_final = ggp_info[1]
ID_final = ggp_info[2]
z_final = ggp_info[3]
name_final = ggp_info[5]
ql_final = ggp_info[6]
ra_final = ggp_info[7]
dec_final = ggp_info[8]

col_ID = np.arange(len(row_final))
select_array = np.sort(np.array([1, 4, 5, 6, 7, 13, 20, 27, 35, 36, 57, 62, 64, 68, 72, 78, 80, 81, 82, 83, 88, 92,
                                 93, 120, 129, 134, 140, 141, 149, 162, 164, 179, 181, 182]))  # No row=11
select_gal = np.in1d(row_final, select_array)
row_final = row_final[select_gal]
ID_final = ID_final[select_gal]
name_final = name_final[select_gal]
z_final = z_final[select_gal]
ra_final = ra_final[select_gal]
dec_final = dec_final[select_gal]

PlotGalSpectra(row_array=[181, 182], qls=True)
#
# fig, axarr = plt.subplots(3, 1, figsize=(10, 15), sharex=True, dpi=300)
# plt.subplots_adjust(hspace=0.1)
# axarr[0].plot(wave_1, flux_1, color='k', lw=1)
# axarr[0].plot(wave_1, model_1, color='r', lw=1)
# axarr[0].plot(wave_1, flux_err_1, color='lightgrey')
# axarr[1].plot(wave_2, flux_2, color='k', lw=1)
# axarr[1].plot(wave_2, model_2, color='r', lw=1)
# axarr[1].plot(wave_2, flux_err_2, color='lightgrey')
# axarr[2].plot(wave_3, flux_3, color='k', lw=1)
# axarr[2].plot(wave_3, model_3, color='r', lw=1)
# axarr[2].plot(wave_3, flux_err_3, color='lightgrey')
# axarr[0].set_title('G1', x=0.1, y=0.85, size=20)
# axarr[1].set_title('G64', x=0.1, y=0.85, size=20)
# axarr[2].set_title('G88', x=0.1, y=0.85, size=20)
# axarr[0].minorticks_on()
# axarr[1].minorticks_on()
# axarr[2].minorticks_on()
# axarr[2].set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
# axarr[0].set_ylabel(r'${f}_{\lambda}$', size=20)
# axarr[1].set_ylabel(r'${f}_{\lambda}$', size=20)
# axarr[2].set_ylabel(r'${f}_{\lambda}$', size=20)
# axarr[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# axarr[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# axarr[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# axarr[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# axarr[2].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# axarr[2].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/QSP_lunch_talk.png', bbox_inches='tight')

#
# path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
# data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)
#
# fig = plt.figure(figsize=(8, 8), dpi=300)
# gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
# gc.set_xaxis_coord_type('scalar')
# gc.set_yaxis_coord_type('scalar')
# gc.recenter(40.1359, -18.8643, width=0.02, height=0.02) # 0.02 / 0.01
# gc.set_system_latex(True)
# gc.show_colorscale(cmap=newcmp)
# gc.ticks.set_length(30)
# gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
# gc.add_scalebar(length=15 * u.arcsecond)
# gc.scalebar.set_corner('top left')
# gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
# gc.scalebar.set_font_size(15)
# gc.ticks.hide()
# gc.tick_labels.hide()
# gc.axis_labels.hide()
# xw, yw = 40.125973, -18.858134
# gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
# gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
# gc.add_label(0.971, 0.87, r'N', size=15, relative=True)
# gc.add_label(0.912, 0.805, r'E', size=15, relative=True)
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/HST_lunch_talk.png', bbox_inches='tight')