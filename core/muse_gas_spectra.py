import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.io import ascii
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'gas_list.reg')
# region = np.loadtxt(path_hb, usecols=[0, 1, 2])
ra_array = np.loadtxt(path_hb, usecols=[0, 1, 2])[:, 0]
dec_array = np.loadtxt(path_hb, usecols=[0, 1, 2])[:, 1]
radius_array = np.loadtxt(path_hb, usecols=[0, 1, 2])[:, 2]
text_array = np.loadtxt(path_hb, dtype=str, usecols=[3])

# path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits_SUBTRACTED.fits')
path_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
path_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_Hbeta_line_offset.fits')
path_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_4960_line_offset.fits')
path_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_5008_line_offset.fits')

cube_OII = Cube(path_OII)
cube_Hbeta = Cube(path_Hbeta)
cube_OIII4960 = Cube(path_OIII4960)
cube_OIII5008 = Cube(path_OIII5008)
wave_OII = pyasl.airtovac2(cube_OII.wave.coord())
wave_Hbeta = pyasl.airtovac2(cube_Hbeta.wave.coord())
wave_OIII4960 = pyasl.airtovac2(cube_OIII4960.wave.coord())
wave_OIII5008 = pyasl.airtovac2(cube_OIII5008.wave.coord())
wave_stack = np.hstack((wave_OII, wave_Hbeta, wave_OIII4960, wave_OIII5008))

fig, axarr = plt.subplots(len(ra_array), 2, figsize=(10, len(ra_array) * 5), sharey=False, dpi=300)
fig.subplots_adjust(hspace=0.1)
fig.subplots_adjust(wspace=0.18)
# axarr = axarr.ravel()
for i in range(len(ra_array)):
    spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
    spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    spe_OIII4960_i = cube_OIII4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    spe_OIII5008_i = cube_OIII5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)


    flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
    flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
    flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
    flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3
    flux_stack = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
    flux_err_stack = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))


    axarr[i, 0].plot(wave_stack, flux_stack, color='k', lw=1)
    axarr[i, 0].plot(wave_stack, flux_err_stack, color='lightgrey')
    axarr[i, 1].plot(wave_stack, flux_stack, color='k', lw=1)
    axarr[i, 1].plot(wave_stack, flux_err_stack, color='lightgrey')
    axarr[i, 0].set_title(text_array[i], x=0.2, y=0.85, size=20)
    axarr[i, 0].set_xlim(5900, 6300)
    axarr[i, 1].set_xlim(7800, 8200)
    axarr[i, 0].spines['right'].set_visible(False)
    axarr[i, 1].spines['left'].set_visible(False)

    # Mark line info
    # ymin, ymax = np.zeros(4), np.ones(4),
    # axarr[i].vlines([], )
    # axarr[i].set_ylim(flux_stack.min(), flux_stack.max())
    axarr[i, 0].annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.5, 0.65), xycoords='axes fraction', size=20)
    axarr[i, 1].annotate(text=r'$\mathrm{H\beta}$', xy=(0.1, 0.65), xycoords='axes fraction', size=20)
    axarr[i, 1].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.65, 0.65), xycoords='axes fraction', size=20)
    # axarr[i, 1].annotate(text=r'$\mathrm{[O \, III 5008]}$', xy=(0.5, 0.5), xycoords='figure fraction', size=20)

    # Make diagonal lines
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=axarr[i, 0].transAxes, color='k', clip_on=False)
    # axarr[i, 0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    # axarr[i, 0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    #
    # kwargs.update(transform=axarr[i, 1].transAxes)  # switch to the bottom axes
    # axarr[i, 1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    # axarr[i, 1].plot((-d, +d), (-d, +d), **kwargs)

    axarr[i, 0].minorticks_on()
    axarr[i, 1].minorticks_on()
    axarr[i, 0].set_ylabel(r'${f}_{\lambda}$', size=20)
    axarr[i, 0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right=False,
                            labelsize=20, size=5)
    axarr[i, 0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right=False, size=3)
    axarr[i, 0].tick_params(axis='y', which='both', right=False, labelright=False)
    axarr[i, 1].tick_params(axis='both', which='major', direction='in', bottom='on', left=False, right='on',
                            labelsize=20, size=5)
    axarr[i, 1].tick_params(axis='both', which='minor', direction='in', bottom='on', left=False, right='on', size=3)
    axarr[i, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
axarr[len(ra_array) - 1, 0].set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
axarr[len(ra_array) - 1, 0].xaxis.set_label_coords(1.15, -0.1)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/spectra_gas.png', bbox_inches='tight')