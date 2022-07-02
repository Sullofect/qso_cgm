import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm
from astropy import units as u
from muse_compare_z import compare_z
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5

#
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

#
# Load galxies infomation
ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                     cat_will='ESO_DEEP_offset_zapped_objects.fits')
bins_final = ggp_info[0]
row_final = ggp_info[1]
ID_final = ggp_info[2]
z_final = ggp_info[3]
name_final = ggp_info[5]
ra_final = ggp_info[7]
dec_final = ggp_info[8]

col_ID = np.arange(len(row_final))
select_array = np.sort(np.array([1, 4, 5, 6, 7, 13, 20, 27, 35, 36, 57, 62, 64, 68, 72, 78, 80, 81, 82, 83, 88, 92,
                                 93, 120, 129, 134, 140, 141, 149, 162, 164, 179, 181, 182]))  # No row=11
select_gal = np.in1d(row_final, select_array)
row_final = row_final[select_gal]
z_final = z_final[select_gal]
ra_final = ra_final[select_gal]
dec_final = dec_final[select_gal]

# Change coordinate for galaxy with row = 1
row_1_sort = np.where(row_final == 1)
ra_final[row_1_sort] = 40.1440392
dec_final[row_1_sort] = -18.8597159

z_qso = 0.6282144177077355
v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)

# Calculate the offset between MUSE and HST
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_qso_hst, dec_qso_hst = 40.1359, -18.8643
# ra_final = ra_final - (ra_qso_hst - ra_qso_muse)  # Wrong!!!
# dec_final = dec_final - (dec_qso_hst - dec_qso_muse)  # Wrong!!!

# Plot
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc.recenter(40.1359, -18.8643, width=90/3600, height=90/3600)  # 0.02 / 0.01 40''
gc.show_rectangles(40.1359, -18.8643, width=40/3600, height=40/3600, color='k', linestyle='--')
gc.show_rectangles(40.1359, -18.8643, width=65/3600, height=65/3600, angle=60, color='k', linestyle='--')
gc.set_system_latex(True)
gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
gc.add_colorbar()
gc.colorbar.set_ticks([0])
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.0)
gc.colorbar.hide()
gc.ticks.set_length(30)
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
norm = mpl.colors.Normalize(vmin=-300, vmax=300)
gc.show_markers(ra_final, dec_final, marker='o', facecolor='none', c='none', edgecolors=plt.cm.coolwarm(norm(v_gal)),
                linewidths=1.2, s=80)
gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
# gc.show_markers(ra_final, dec_final, marker='o', c=v_gal, linewidths=0.5, alpha=1, facecolor='none', edgecolors=None,
#                 s=40, vmin=-300, vmax=300, cmap='coolwarm')
gc.add_label(0.87, 0.97, r'$\mathrm{ACS+F814W}$', color='k', size=15, relative=True)
xw, yw = 40.1231559, -18.8580071
gc.show_arrows(xw, yw, -0.0001 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.0001 * yw, color='k')
line = np.array([[40.1289217, 40.1429009], [-18.8576894, -18.8709014]])
gc.show_lines([line], color='k', alpha=0.3, linestyle='--')
gc.add_label(0.985, 0.85, r'N', size=15, relative=True)
gc.add_label(0.89, 0.748, r'E', size=15, relative=True)


# Determine whcih galaxy is below the line
# vector_ra = ra_final - line[0, 0]
# vector_dec = dec_final - line[1, 0]
# vector_line = np.array([line[0, 1] - line[0, 0], line[1, 1] - line[1, 0]])
# value = np.cross(np.vstack((vector_ra, vector_dec)).T,  vector_line)
# value_sort = value < 0
# row_above, row_below = row_final[value_sort], row_final[np.invert(value_sort)]
# v_above, v_below = v_gal[value_sort], v_gal[np.invert(value_sort)]
#
#
# # Plot the fitting
# mu_all, scale_all = norm.fit(v_gal)
# mu_above, scale_above = norm.fit(v_above)
# mu_below, scale_below = norm.fit(v_below)
#
# # Normalization
# nums, v_edge = np.histogram(v_gal, bins=bins_final)
# normalization_all = np.sum(nums) * 200
# nums, v_edge = np.histogram(v_above, bins=bins_final)
# normalization_above = np.sum(nums) * 200
# nums, v_edge = np.histogram(v_below, bins=bins_final)
# normalization_below = np.sum(nums) * 200

# Plot
# rv = np.linspace(-2000, 2000, 1000)
# axins = fig.add_axes([0.62, 0.17, 0.25, 0.15], zorder=1000)
# # plt.figure(figsize=(8, 5), dpi=300)
# axins.vlines(0, 0, 11, linestyles='--', color='k', label=r"$\mathrm{QSO's \; redshift}$")
# axins.hist(v_gal, bins=bins_final, color='k', histtype='step', label=r'$\mathrm{v_{all}}$')
# axins.hist(v_above, bins=bins_final, facecolor='red', histtype='stepfilled', alpha=0.5, label=r'$\mathrm{v_{above}}$')
# axins.hist(v_below, bins=bins_final, facecolor='blue', histtype='stepfilled', alpha=0.5, label=r'$\mathrm{v_{below}}$')
# axins.plot(rv, normalization_all * norm.pdf(rv, mu_all, scale_all), '-k', lw=1, alpha=0.5,
#            label=r'$\mu = \, $' + str("{0:.0f}".format(mu_all)) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $' +
#                  str("{0:.0f}".format(scale_all)) + r'$\mathrm{\, km/s}$')
# axins.plot(rv, normalization_above * norm.pdf(rv, mu_above, scale_above), '-r', lw=1, alpha=0.5,
#            label=r'$\mu = \, $' + str("{0:.0f}".format(mu_above)) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $' +
#                  str("{0:.0f}".format(scale_above)) + r'$\mathrm{\, km/s}$')
# axins.plot(rv, normalization_below * norm.pdf(rv, mu_below, scale_below), '-b', lw=1, alpha=0.5,
#            label=r'$\mu = \, $' + str("{0:.0f}".format(mu_below)) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $' +
#                  str("{0:.0f}".format(scale_below)) + r'$\mathrm{\, km/s}$')
# axins.set_xlim(-2000, 2000)
# axins.set_ylim(0, 11)
# # axins.minorticks_on()
# axins.set_xlabel(r'$\Delta v [\mathrm{km \; s^{-1}}]$', size=5, labelpad=0)
# axins.set_ylabel(r'$\mathrm{Numbers}$', size=5, labelpad=0)
# axins.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=3,
#                   labelsize=7)
# axins.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
# axins.legend(prop={'size': 5}, framealpha=0, loc=2, fontsize=15)

# # Second axis
# axins = fig.add_axes([0.165, 0.16, 0.15, 0.15], zorder=1000)
# axins.hist(v_above, bins=np.arange(-600, 1800, 200), facecolor='red', histtype='stepfilled', alpha=0.5)
# axins.set_xlim(-600, 1600)
# # axins.set_xticks(-600, )
#
# axins = fig.add_axes([0.72, 0.16, 0.15, 0.15], zorder=1000)
# # axins.hist(v_above, bins=bins_final, facecolor='red', histtype='stepfilled', alpha=0.5)
# axins.hist(v_below, bins=np.arange(-400, 400, 200), facecolor='blue', histtype='stepfilled', alpha=0.5)
# axins.set_xlim(-400, 200)
# axins.set_ylim(0, 10)
# axins.set_xticks([-400, -200, 0, 200])
# axins.set_yticks([0, 2, 4, 6, 8, 10])
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image_gal_dis.png', bbox_inches='tight')