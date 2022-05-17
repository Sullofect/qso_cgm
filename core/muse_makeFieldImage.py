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
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
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


path_1 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      '1_20002_J024034.53-185135.09_spec1D.fits')
spec = Table.read(path_1)
wave_1 = pyasl.vactoair2(spec['wave'])
flux_1 = spec['flux'] * 1e-3
model_1 = spec['model'] * 1e-3
flux_err_1 = spec['error'] * 1e-3

path_2 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      '64_271_J024032.03-185139.81_spec1D.fits')
spec = Table.read(path_2)
wave_2 = pyasl.vactoair2(spec['wave'])
flux_2 = spec['flux'] * 1e-3
model_2 = spec['model'] * 1e-3
flux_err_2 = spec['error'] * 1e-3

path_3 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      '88_301_J024030.95-185143.68_spec1D.fits')
spec = Table.read(path_3)
wave_3 = pyasl.vactoair2(spec['wave'])
flux_3 = spec['flux'] * 1e-3
model_3 = spec['model'] * 1e-3
flux_err_3 = spec['error'] * 1e-3

#
fig, axarr = plt.subplots(3, 1, figsize=(10, 15), sharex=True, dpi=300)
plt.subplots_adjust(hspace=0.1)
axarr[0].plot(wave_1, flux_1, color='k', lw=1)
axarr[0].plot(wave_1, model_1, color='r', lw=1)
axarr[0].plot(wave_1, flux_err_1, color='lightgrey')
axarr[1].plot(wave_2, flux_2, color='k', lw=1)
axarr[1].plot(wave_2, model_2, color='r', lw=1)
axarr[1].plot(wave_2, flux_err_2, color='lightgrey')
axarr[2].plot(wave_3, flux_3, color='k', lw=1)
axarr[2].plot(wave_3, model_3, color='r', lw=1)
axarr[2].plot(wave_3, flux_err_3, color='lightgrey')
axarr[0].set_title('G1', x=0.1, y=0.85, size=20)
axarr[1].set_title('G64', x=0.1, y=0.85, size=20)
axarr[2].set_title('G88', x=0.1, y=0.85, size=20)
axarr[0].minorticks_on()
axarr[1].minorticks_on()
axarr[2].minorticks_on()
axarr[2].set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
axarr[0].set_ylabel(r'${f}_{\lambda}$', size=20)
axarr[1].set_ylabel(r'${f}_{\lambda}$', size=20)
axarr[2].set_ylabel(r'${f}_{\lambda}$', size=20)
axarr[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
axarr[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
axarr[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
axarr[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
axarr[2].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
axarr[2].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/QSP_lunch_talk.png', bbox_inches='tight')

#
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc.recenter(40.1359, -18.8643, width=0.02, height=0.02) # 0.02 / 0.01
gc.set_system_latex(True)
gc.show_colorscale(cmap=newcmp)
gc.ticks.set_length(30)
gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
xw, yw = 40.125973, -18.858134
gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
gc.add_label(0.971, 0.87, r'N', size=15, relative=True)
gc.add_label(0.912, 0.805, r'E', size=15, relative=True)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/HST_lunch_talk.png', bbox_inches='tight')