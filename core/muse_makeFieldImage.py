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

#
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)
path_OII_SB = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_OII_line_SB_offset_revised.fits')

fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc.show_contour(path_OII_SB, levels=[0.75, 1.5, 3], colors='red')
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
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image.png', bbox_inches='tight')