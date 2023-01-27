import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from astropy.table import Table
from muse_RenameGal import ReturnGalLabel
from matplotlib.colors import ListedColormap
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)

# Load the data
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


# Plot the spectra
def PlotGalSpectra(row_array=None, qls=False, figname='spectra_gal'):
    global z_final, row_final, ID_sep_final
    fig, axarr = plt.subplots(len(row_array), 1, figsize=(10, 2.5 * len(row_array)), sharex=True, dpi=300)
    plt.subplots_adjust(hspace=0.0)

    # Iterate over galaxy
    for i in range(len(row_array)):
        row_sort = np.where(row_final == row_array[i])
        z_i = z_final[row_sort]
        ID_sep_final_i = ID_sep_final[row_sort]
        wave_i, flux_i, flux_err_i, model_i = LoadGalData(row=row_array[i], qls=qls)

        # Compute D4000 and its error
        blue_mask, red_mask = np.where((wave_i < (1 + z_i) * 3950) * (wave_i > (1 + z_i) * 3850)), \
                              np.where((wave_i < (1 + z_i) * 4100) * (wave_i > (1 + z_i) * 4000))
        blue = np.mean(flux_i[blue_mask])
        blue_error = np.sqrt(np.sum(flux_err_i[blue_mask] ** 2)) / len(flux_i[blue_mask])
        red = np.mean(flux_i[red_mask])
        red_error = np.sqrt(np.sum(flux_err_i[red_mask] ** 2)) / len(flux_i[red_mask])

        #
        D4000 = red / blue
        D4000_err = D4000 * np.sqrt((red_error / red) ** 2 + (blue_error / blue) ** 2)
        print(ID_sep_final_i, D4000, D4000_err)

        # Plot the spectrum
        if len(row_array) == 1:
            axarr_i = axarr
            axarr_0 = axarr
        else:
            axarr_i = axarr[i]
            axarr_0 = axarr[0]
        axarr_i.plot(wave_i, flux_i, color='k', drawstyle='steps-mid', lw=1)
        axarr_i.plot(wave_i, model_i, color='r', lw=1)
        axarr_i.plot(wave_i, flux_err_i, color='lightgrey', drawstyle='steps-mid', lw=1)
        axarr_i.set_title('G' + str(ID_sep_final_i[0]), x=0.1, y=0.75, size=20)
        axarr_0.annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.2, 0.72), xycoords='axes fraction', size=20)
        axarr_0.annotate(text=r'$\mathrm{K, H, G}$', xy=(0.39, 0.72), xycoords='axes fraction', size=20)
        axarr_0.annotate(text=r'$\mathrm{H\beta}$', xy=(0.65, 0.72), xycoords='axes fraction', size=20)
        axarr_0.annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.80, 0.72), xycoords='axes fraction', size=20)
        lines = (1 + z_i) * np.array([3727.092, 3729.8754960, 3934.777, 3969.588, 4305.61, 4862.721, 4960.295, 5008.239])
        axarr_i.vlines(lines, lw=1, ymin=-5 * np.ones_like(lines), ymax=100 * np.ones_like(lines),
                       linestyles='dashed', colors='grey')
        axarr_i.set_xlim(4800, 9200)
        axarr_i.set_ylim(-0.1, flux_i.max() + 0.2)
        axarr_i.minorticks_on()
        axarr_i.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                            labelsize=20, size=5)
        axarr_i.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                            size=3)
    if len(row_array) == 1:
        fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=15, y=-0.1)
        fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=15, x=0.06)
    else:
        fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
        fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/spectra_gal/' + figname + '.png', bbox_inches='tight')


# Load information
ra_final, dec_final, row_final, ID_final, z_final, name_final, ID_sep_final = ReturnGalLabel()

# for i in row_final:
#     mask = row_final == i
#     PlotGalSpectra(row_array=[i], qls=False, figname="spectra_gal_" + str(i) + '_' + str(ID_sep_final[mask][0]))
#
# for i in [5, 6, 7, 181, 182]:
#     mask = row_final == i
#     PlotGalSpectra(row_array=[i], qls=True, figname="spectra_gal_" + str(i) + '_' + str(ID_sep_final[mask][0]))

PlotGalSpectra(row_array=[64, 35, 1], qls=False, figname="spectra_gal_paper")