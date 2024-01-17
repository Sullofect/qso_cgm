import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from regions import Regions
from scipy.stats import norm
from astropy import units as u
from scipy.optimize import minimize
from muse_compare_z import compare_z
from astropy.coordinates import SkyCoord
from muse_RenameGal import ReturnGalLabel
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12


# Minimize likelihood
def gauss(x, mu, sigma):
    return np.exp(- (x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)


def loglikelihood(x_0, x):
    mu1, sigma1, mu2, sigma2, p1 = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
    return -1 * np.sum(np.log(p1 * gauss(x, mu1, sigma1) + (1 - p1) * gauss(x, mu2, sigma2)))


# Load galxies infomation
row_final, ID_final, name_final, z_final, ra_hst, dec_hst, bins_final = ReturnGalLabel(sort_row=False, mode='initial',
                                                                                       return_HST=True,
                                                                                       return_bins=True)
z_qso = 0.6282144177077355
v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)
v_gal_qso = np.hstack((np.array([0]), v_gal))  # add the quasar

# Determine whcih galaxy is below the line
line = np.array([[40.1300330, 40.1417710], [-18.8587229, -18.8698312]])
vector_ra = ra_hst - line[0, 0]
vector_dec = dec_hst - line[1, 0]
vector_line = np.array([line[0, 1] - line[0, 0], line[1, 1] - line[1, 0]])
value = np.cross(np.vstack((vector_ra, vector_dec)).T,  vector_line)
value_sort = value < 0
row_above, row_below = row_final[value_sort], row_final[np.invert(value_sort)]
v_above, v_below = v_gal[value_sort], v_gal[np.invert(value_sort)]

# Normalization
nums, v_edge = np.histogram(v_gal_qso, bins=bins_final)
normalization_all = np.sum(nums) * 200

#
guesses = [-80, 200, 500, 500, 0.3]
bnds = ((-100, 0), (0, 500), (200, 1000), (0, 1000), (0, 1))
result = minimize(loglikelihood, guesses, (v_gal_qso), bounds=bnds, method="Powell")
print(result.x)

# Bootstrap
result_array = np.zeros((1000, 5))
for i in range(len(result_array)):
    v_gal_i = np.random.choice(v_gal_qso, replace=True, size=len(v_gal_qso))
    result_i = minimize(loglikelihood, guesses, (v_gal_i), bounds=bnds, method="Powell")
    result_array[i, :] = result_i.x
result_std = np.std(result_array, axis=0)
print(result_std)

# Test with "outliers" rejection
mu_test, scale_test = norm.fit(v_above[v_above > 0])
print(mu_test, scale_test)

# Compute group center
ID_ste = np.array([2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 19, 21, 22, 23, 24,
                   25, 26, 27, 28, 29, 30, 32, 33, 34])
M_ste = 10 ** np.array([9.5, 8.3, 7.4, 10.1, 9.3, 10.4, 9.7, 11.5, 10.6, 10.0, 9.0, 9.5, 10.1, 10.3, 10.3, 10.1, 9.5,
                        9.8, 8.8, 9.9, 9.5, 9.5, 8.3, 8.9, 9.0, 10.8, 11.2])
ID_sep_final = ReturnGalLabel(sort_row=True, mode='final')[6]
select_gal = np.in1d(ID_sep_final, ID_ste)
ID_sep_ste = ID_sep_final[select_gal]
ra_ste = ra_hst[select_gal]
dec_ste = dec_hst[select_gal]
v_gal_ste = v_gal[select_gal]

# All
ra_center_all, dec_center_all = np.sum(ra_ste * M_ste) / np.sum(M_ste), np.sum(dec_ste * M_ste) / np.sum(M_ste)
print('All stellar mass weighted are', ra_center_all, dec_center_all)

# Blue
P_all = (result.x[4] * gauss(v_gal, result.x[0], result.x[1])
         + (1 - result.x[4]) * gauss(v_gal, result.x[2], result.x[3]))
P_blue = result.x[4] * gauss(v_gal, result.x[0], result.x[1]) / P_all
norm_blue = np.sum(P_blue)
ra_center_blue, dec_center_blue = np.sum(ra_hst * P_blue) / norm_blue, \
                                  np.sum(dec_hst * P_blue) / norm_blue
print('Blue stellar mass weighted are', ra_center_blue, dec_center_blue)

# Red
P_red = (1 - result.x[4]) * gauss(v_gal, result.x[2], result.x[3]) / P_all
norm_red = np.sum(P_red)
ra_center_red, dec_center_red = np.sum(ra_hst * P_red) / norm_red, \
                                np.sum(dec_hst * P_red) / norm_red
print('Red Stellar mass weighted are', ra_center_red, dec_center_red)

# Plot
rv = np.linspace(-2000, 2000, 1000)
plt.figure(figsize=(8, 5), dpi=300)
# plt.vlines(0, 0, 15, linestyles='--', color='k', label=r"$\mathrm{QSO's \; redshift}$")
plt.hist(v_gal_qso, bins=bins_final, color='k', histtype='step', label=r'$v_{\rm all}$')
# plt.hist(v_above, bins=bins_final, facecolor='orange', histtype='stepfilled', alpha=0.5, label=r'$v_{\rm orange}$')
# plt.hist(v_below, bins=bins_final, facecolor='purple', histtype='stepfilled', alpha=0.5, label=r'$v_{\rm purple}$')
plt.plot(rv, result.x[4] * normalization_all * norm.pdf(rv, result.x[0], result.x[1]), '--', c='b', lw=1, alpha=1,
         label=r'$P_{1} = \,$' + str("{0:.2f}".format(result.x[4])) +
               '\n' + r'$\mu_{1} = \, $' + str("{0:.0f}".format(result.x[0]))
               + r'$\mathrm{\, km \, s^{-1}}$' + '\n' + r'$\sigma_{1} = \, $'
               + str("{0:.0f}".format(result.x[1])) + r'$\mathrm{\, km \, s^{-1}}$')
plt.plot(rv, (1 - result.x[4]) * normalization_all * norm.pdf(rv, result.x[2], result.x[3]), '--', c='red', lw=1,
         alpha=1, label=r'$P_{2} = \,$' + str("{0:.2f}".format(1 - result.x[4])) +
                        '\n' + r'$\mu_{2} = \, $' + str("{0:.0f}".format(result.x[2]))
                        + r'$\mathrm{\, km \, s^{-1}}$' + '\n' + r'$\sigma_{2} = \, $'
                        + str("{0:.0f}".format(result.x[3])) + r'$\mathrm{\, km \, s^{-1}}$')
plt.plot(rv, result.x[4] * normalization_all * norm.pdf(rv, result.x[0], result.x[1]) +
         (1 - result.x[4]) * normalization_all * norm.pdf(rv, result.x[2], result.x[3]), '-k',
         lw=1, alpha=1, label=r'$P_{1}N(\mu_{1}\mathrm{,} \, \sigma_{1}^2) + $'
                              + '\n' + r'$P_{2}N(\mu_{2}\mathrm{,} \, \sigma_{2}^2)$', zorder=-100)
plt.xlim(-2000, 2000)
plt.ylim(0, 13)
plt.yticks([2, 4, 6, 8, 10, 12])
plt.minorticks_on()
plt.xlabel(r'$\Delta v [\mathrm{km \; s^{-1}}]$', size=20)
plt.ylabel(r'$\mathrm{Numbers}$', size=20)
plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                labelsize=20)
plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
plt.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
plt.savefig(path_savefig + 'galaxy_velocity_all', bbox_inches='tight')


# Load the image
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

# Label the galaxy
path_label = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'galaxy_label_list.reg')
regions_label = Regions.read(path_label, format='ds9')

# Plot
fig = plt.figure(figsize=(8, 8), dpi=300)
gc1 = aplpy.FITSFigure(path_hb, figure=fig, north=True, hdu=1)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True, hdu=1)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc1.set_xaxis_coord_type('scalar')
gc1.set_yaxis_coord_type('scalar')

# Show polygons
# d = np.sqrt(2) * 65 / 2 / 3600
# angle = 75 * np.pi / 180
# N1 = [np.array([[40.1289217, 40.1429009, 40.1359 - d * np.cos(angle), 40.1359 - d * np.sin(angle)],
#                [-18.8576894, -18.8709014, -18.8643 - d * np.sin(angle), -18.8643 + d * np.cos(angle)]])]
N1 = [np.array([[40.1288438, 40.1428230, 40.1324189, 40.1228938],
                [-18.8577309, -18.8709429, -18.8766207, -18.8610104]])]
N2 = [np.array([[40.1289217, 40.1429009, 40.1489166, 40.1394084],
                [-18.8576894, -18.8709014, -18.8676171, -18.8519788]])]
# gc.show_polygons(N1, color='purple', linestyle='-', lw=2, alpha=0.5, zorder=1)
# gc.show_polygons(N2, color='orange', linestyle='-', lw=2, alpha=0.5, zorder=1)

#
gc.recenter(40.1359, -18.8643, width=90/3600, height=90/3600)
gc1.recenter(40.1359, -18.8643, width=40/3600, height=40/3600) # 0.02 / 0.01 40''
gc.show_rectangles(40.1344150, -18.8656933, width=30 / 3600, height=30 / 3600, color='k', linestyle='--')
gc.show_rectangles(40.1359, -18.8643, width=65/3600, height=65/3600, angle=60, color='k', linestyle='--')

# Label galaxies
# for i in range(len(row_final)):
#     x = regions_label[i].center.ra.degree
#     y = regions_label[i].center.dec.degree
#     text = 'G' + str(ID_sep_final[i])
#     gc.add_label(x, y, text, size=10)
# gc.show_arrows(40.1370596, -18.8662000, 40.1368331 - 40.1370596, -18.8658486 + 18.8662000, color='k')

#
gc.set_system_latex(True)
gc1.set_system_latex(True)
gc1.show_colorscale(cmap='coolwarm', vmin=-1000, vmax=1000)
gc1.hide_colorscale()
gc1.add_colorbar()
gc1.colorbar.set_box([0.15, 0.145, 0.38, 0.02], box_orientation='horizontal')
gc1.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
gc1.colorbar.set_axis_label_font(size=12)
gc1.colorbar.set_axis_label_pad(-40)
gc1.colorbar.set_location('bottom')
# gc1.colorbar.hide()
gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
gc.add_colorbar()
gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
gc.colorbar.hide()

# Scale bar
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)

# Hide ticks
gc.ticks.set_length(30)
gc1.ticks.set_length(30)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
gc1.ticks.hide()
gc1.tick_labels.hide()
gc1.axis_labels.hide()
norm = mpl.colors.Normalize(vmin=-1000, vmax=1000)

# Markers
# gc.show_markers(ra_center_all, dec_center_all, marker=(6, 2, 0), c='white', s=30)
# gc.show_markers(ra_center_red, dec_center_red, marker=(6, 2, 0), facecolor='r', s=30)
# gc.show_markers(ra_center_blue, dec_center_blue, marker=(6, 2, 0), facecolor='b', s=30)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                linewidths=0.5, s=400)
# gc.add_label(40.13564948691202 - 0.0015, -18.864301804042814, 'QSO', size=10)
# gc.show_markers(ra_hst, dec_hst, marker='o', facecolor='none', c='none', edgecolors=plt.cm.coolwarm(norm(v_gal)),
#                 linewidths=1.2, s=80)
# gc.show_markers(ra_hst, dec_hst, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
# line = np.array([[40.1289217, 40.1429009], [-18.8576894, -18.8709014]])
# gc.show_lines([line], color='k', alpha=0.3, linestyle='--')

# Contours
path_data = '/Users/lzq/Dropbox/Data/CGM/'
path_OII_SB = path_data + 'image_MakeMovie/OII_-100_100_contour_revised.fits'
path_OIII_SB = path_data + 'image_MakeMovie/OIII_-100_100_contour_revised.fits'
gc.show_contour(path_OII_SB, levels=[0.08, 0.3], layer='OII', kernel='gauss', colors='blue',
                linewidths=0.8, smooth=3)
gc.show_contour(path_OIII_SB, levels=[0.08, 0.3], layer='OIII', kernel='gauss', colors='red',
                linewidths=0.8, smooth=3)

# Labels
xw, yw = 40.1231559, -18.8580071
gc.show_arrows(xw, yw, -0.0001 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.0001 * yw, color='k')
gc.add_label(0.985, 0.85, r'N', size=15, relative=True)
gc.add_label(0.89, 0.748, r'E', size=15, relative=True)
gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
# gc.add_label(0.27, 0.86, r"$\rm MUSE \, 1'\times 1' \, FoV$", size=15, relative=True, rotation=60)
gc.add_label(0.47, 0.30, r"$\rm 30'' \times 30''$", size=15, relative=True)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image_gal_dis.png', bbox_inches='tight')