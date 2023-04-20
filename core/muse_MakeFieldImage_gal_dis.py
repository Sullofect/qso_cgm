import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from regions import Regions
from astropy import units as u
from muse_RenameGal import ReturnGalLabel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12

# Load the image
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

# Label the galaxy
path_label = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'galaxy_label_list.reg')
regions_label = Regions.read(path_label, format='ds9')

# Load galxies infomation
row_final, ID_final, name_final, z_final, ra_final, dec_final = ReturnGalLabel(sort_row=False, mode='initial')
ID_sep_final = ReturnGalLabel(sort_row=True, mode='final')[6]

# Change coordinate for galaxy with row = 1
row_1_sort = np.where(row_final == 1)
ra_final[row_1_sort] = 40.1440392
dec_final[row_1_sort] = -18.8597159

#
z_qso = 0.6282144177077355
v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)

# Plot
fig = plt.figure(figsize=(8, 8), dpi=300)
gc1 = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc1.set_xaxis_coord_type('scalar')
gc1.set_yaxis_coord_type('scalar')
# d = np.sqrt(2) * 65 / 2 / 3600
# angle = 75 * np.pi / 180
# N1 = [np.array([[40.1289217, 40.1429009, 40.1359 - d * np.cos(angle), 40.1359 - d * np.sin(angle)],
#                [-18.8576894, -18.8709014, -18.8643 - d * np.sin(angle), -18.8643 + d * np.cos(angle)]])]
N1 = [np.array([[40.1288438, 40.1428230, 40.1324189, 40.1228938],
                [-18.8577309, -18.8709429, -18.8766207, -18.8610104]])]
N2 = [np.array([[40.1289217, 40.1429009, 40.1489166, 40.1394084],
                [-18.8576894, -18.8709014, -18.8676171, -18.8519788]])]
gc.show_polygons(N1, color='purple', linestyle='-', lw=2, alpha=0.5, zorder=1)
gc.show_polygons(N2, color='orange', linestyle='-', lw=2, alpha=0.5, zorder=1)
gc.recenter(40.1359, -18.8643, width=90/3600, height=90/3600)  # 0.02 / 0.01 40''
gc.show_rectangles(40.1344150, -18.8656933, width=30 / 3600, height=30 / 3600, color='k', linestyle='--')
# gc.show_rectangles(40.1359, -18.8643, width=40/3600, height=40/3600, color='k', linestyle='--')
gc.show_rectangles(40.1359, -18.8643, width=65/3600, height=65/3600, angle=60, color='k', linestyle='--')
gc1.recenter(40.1359, -18.8643, width=40/3600, height=40/3600)  # 0.02 / 0.01 40''
for i in range(len(row_final)):
    x = regions_label[i].center.ra.degree
    y = regions_label[i].center.dec.degree
    text = 'G' + str(ID_sep_final[i])
    gc.add_label(x, y, text, size=10)
gc.show_arrows(40.1370596, -18.8662000, 40.1368331 - 40.1370596, -18.8658486 + 18.8662000, color='k')
gc.set_system_latex(True)
gc1.set_system_latex(True)
gc1.show_colorscale(cmap='coolwarm', vmin=-1000, vmax=1000)
gc1.add_colorbar()
gc1.hide_colorscale()
gc1.colorbar.set_box([0.15, 0.145, 0.38, 0.02], box_orientation='horizontal')
gc1.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
gc1.colorbar.set_axis_label_font(size=12)
gc1.colorbar.set_axis_label_pad(-40)
gc1.colorbar.set_location('bottom')
gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
gc.add_colorbar()
gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
gc.colorbar.hide()
gc.ticks.set_length(30)
gc1.ticks.set_length(30)
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
gc1.ticks.hide()
gc1.tick_labels.hide()
gc1.axis_labels.hide()
norm = mpl.colors.Normalize(vmin=-1000, vmax=1000)
gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                linewidths=0.5, s=400)
gc.add_label(40.13564948691202 - 0.0015, -18.864301804042814, 'QSO', size=10)
gc.show_markers(ra_final, dec_final, marker='o', facecolor='none', c='none',
                edgecolors=plt.cm.coolwarm(norm(v_gal)), linewidths=1.2, s=80)
gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
xw, yw = 40.1231559, -18.8580071
gc.show_arrows(xw, yw, -0.0001 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.0001 * yw, color='k')
line = np.array([[40.1289217, 40.1429009], [-18.8576894, -18.8709014]])
# gc.show_lines([line], color='k', alpha=0.3, linestyle='--')
gc.add_label(0.985, 0.85, r'N', size=15, relative=True)
gc.add_label(0.89, 0.748, r'E', size=15, relative=True)
gc.add_label(0.27, 0.86, r"$\rm MUSE \, 1'\times 1' \, FoV$", size=15, relative=True, rotation=60)
gc.add_label(0.47, 0.30, r"$\rm 30'' \times 30''$", size=15, relative=True)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image_gal_dis_ini.png', bbox_inches='tight')