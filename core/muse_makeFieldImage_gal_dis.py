import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy import units as u
from muse_compare_z import compare_z
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10

#
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

#
# Load galxies infomation
ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                     cat_will='ESO_DEEP_offset_zapped_objects.fits')
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
gc.recenter(40.1359, -18.8643, width=80/3600, height=80/3600)  # 0.02 / 0.01 40''
gc.set_system_latex(True)
gc.show_colorscale(cmap='Greys')
gc.add_colorbar()
gc.colorbar.set_ticks([0])
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.0)
# gc.colorbar.set_box([0.1247, 0.0927, 0.7443, 0.03], box_orientation='horizontal')
gc.colorbar.hide()
gc.ticks.set_length(30)
# gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('top left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(15)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
# gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', edgecolors=plt.cm.coolwarm(v_gal, vmin=-300, vmax=300),
#                 linewidths=0.8, s=100,
#                 vmin=-300, vmax=300, cmap='coolwarm')
gc.show_markers(ra_final, dec_final, marker='o', c=v_gal, linewidths=0.5, alpha=0.7, facecolor=None, edgecolors=None,
                s=40, vmin=-300, vmax=300, cmap='coolwarm')
gc.add_label(0.87, 0.97, r'$\mathrm{ACS+F814W}$', color='k', size=15, relative=True)
xw, yw = 40.1246000, -18.8589960
gc.show_arrows(xw, yw, -0.0001 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.0001 * yw, color='k')
gc.add_label(0.98, 0.85, r'N', size=15, relative=True)
gc.add_label(0.88, 0.734, r'E', size=15, relative=True)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image_gal_dis.png', bbox_inches='tight')