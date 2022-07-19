import os
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

# Read Data
path_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                        'CUBE_OII_line_offset_zapped.fits')
path_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                          'CUBE_Hbeta_line_offset_zapped.fits')
path_bet = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                        'CUBE_bet_Hbeta_OIII_line_offset_zapped.fits')
path_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                             'CUBE_OIII_4960_line_offset_zapped.fits')
path_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                             'CUBE_OIII_5008_line_offset_zapped.fits')
path_OII_conti = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                              'CUBE_OII_continuum_line_offset_zapped.fits')
path_ori = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data',
                        'ESO_DEEP_offset_zapped.fits_SUBTRACTED.fits')

cube_OII = Cube(path_OII)
cube_OII_conti = Cube(path_OII_conti)
cube_ori = Cube(path_ori)
cube_OII_ori = cube_ori.select_lambda(6020, 6120)
cube_Hbeta = Cube(path_Hbeta)
cube_bet = Cube(path_bet)
cube_OIII4960 = Cube(path_OIII4960)
cube_OIII5008 = Cube(path_OIII5008)
wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
wave_Hbeta_vac = pyasl.airtovac2(cube_Hbeta.wave.coord())
wave_bet_vac = pyasl.airtovac2(cube_bet.wave.coord())
wave_OIII4960_vac = pyasl.airtovac2(cube_OIII4960.wave.coord())
wave_OIII5008_vac = pyasl.airtovac2(cube_OIII5008.wave.coord())
wave_vac_stack = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_stack_plot = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_bet_vac, wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_all = np.array([wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)


# Def Plot function
def PlotGasSpectra(ra_array, dec_array, radius_array, text_array, figname='spectra_gas_1'):
    global cube_OII, cube_Hbeta, cube_OIII4960, cube_OIII5008, wave_vac_stack, wave_vac_all
    fig, axarr = plt.subplots(len(ra_array), 2, figsize=(10, len(ra_array) * 2.5),
                              gridspec_kw={'width_ratios': [1, 3]}, dpi=300)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.1)
    flux_info = np.zeros((len(ra_array), 6))
    # axarr = axarr.ravel()
    for i in range(len(ra_array)):
        spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
        spe_OII_conti_i = cube_OII_conti.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OII_ori_i = cube_OII_ori.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_bet_i = cube_bet.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII4960_i = cube_OIII4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
        spe_OIII5008_i = cube_OIII5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)

        # Second round continuum subtraction
        spe_OII_i.mask_region(6050, 6090)
        conti_OII = spe_OII_i.poly_spec(3, weight=True)
        spe_OII_i.unmask()
        spe_OII_i -= conti_OII

        flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
        flux_OII_conti_i, flux_OII_conti_err_i = spe_OII_conti_i.data * 1e-3, np.sqrt(spe_OII_conti_i.var) * 1e-3
        flux_OII_ori_i, flux_OII_ori_err_i = spe_OII_ori_i.data * 1e-3, np.sqrt(spe_OII_ori_i.var) * 1e-3
        flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
        flux_bet_i, flux_bet_err_i = spe_bet_i.data * 1e-3, np.sqrt(spe_bet_i.var) * 1e-3
        flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
        flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3
        flux_all = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
        flux_err_all = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

        # For plotting
        flux_all_plot = np.hstack((flux_OII_i, flux_Hbeta_i, flux_bet_i, flux_OIII4960_i, flux_OIII5008_i))
        flux_err_all_plot = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_bet_err_i, flux_OIII4960_err_i,
                                       flux_OIII5008_err_i))


        # Save the fitted result
        axarr[i, 0].plot(wave_OII_vac, flux_OII_ori_i, color='k', drawstyle='steps-mid', lw=1)
        axarr[i, 0].plot(wave_OII_vac, flux_OII_i, color='b', drawstyle='steps-mid', lw=1)
        axarr[i, 0].plot(wave_OII_vac, flux_OII_conti_i, color='r', drawstyle='steps-mid', lw=1)
        axarr[i, 0].plot(wave_OII_vac, conti_OII.data * 1e-3, color='orange', drawstyle='steps-mid', lw=1)
        axarr[i, 0].set_title(text_array[i], x=0.2, y=0.75, size=20)
        axarr[i, 0].set_xlim(6020, 6120)
        axarr[i, 1].set_xlim(7900, 8200)
        axarr[i, 0].spines['right'].set_visible(False)
        axarr[i, 1].spines['left'].set_visible(False)

        # Mark line info
        axarr[i, 0].set_ylim(flux_OII_conti_i.min() - 0.1, flux_OII_ori_i.max() + 0.1)
        axarr[i, 1].set_ylim(flux_OII_conti_i.min() - 0.1, flux_OII_ori_i.max() + 0.1)
        # if flux_Hbeta_i.max() > flux_OIII5008_i.max():
        #     axarr[i, 0].set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
        #     axarr[i, 1].set_ylim(flux_OII_i.min() - 0.1, flux_OII_i.max() + 0.1)
        axarr[0, 0].annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.55, 0.65), xycoords='axes fraction', size=20)
        axarr[0, 1].annotate(text=r'$\mathrm{H\beta}$', xy=(0.1, 0.65), xycoords='axes fraction', size=20)
        axarr[0, 1].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.45, 0.65), xycoords='axes fraction', size=20)
        axarr[0, 1].annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.7, 0.65), xycoords='axes fraction', size=20)

        axarr[i, 0].minorticks_on()
        axarr[i, 1].minorticks_on()
        axarr[i, 0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right=False,
                                labelsize=20, size=5)
        axarr[i, 0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right=False,
                                size=3)
        axarr[i, 0].tick_params(axis='y', which='both', right=False, labelright=False)
        axarr[i, 1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left=False, right='on',
                                labelsize=20, size=5)
        axarr[i, 1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left=False, right='on',
                                size=3)
        axarr[i, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
        if i != len(ra_array) - 1:
            axarr[i, 0].tick_params(axis='x', which='both', labelbottom=False)
            axarr[i, 1].tick_params(axis='x', which='both', labelbottom=False)
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/line_profile_selected_region.fits', flux_info, overwrite=True)
    fig.supxlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20, y=0.07)
    fig.supylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20, x=0.05)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + figname + '.png', bbox_inches='tight')


# Plot the data
# Read region file
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

PlotGasSpectra(ra_array[:7], dec_array[:7], radius_array[:7], text_array[:7], figname='spectra_gas_P1_test')
PlotGasSpectra(ra_array[7:], dec_array[7:], radius_array[7:], text_array[7:], figname='spectra_gas_P2_test')