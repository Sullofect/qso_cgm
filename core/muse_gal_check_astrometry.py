import os
import aplpy
import matplotlib
import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from mpdaf.obj import WCS as mpdaf_WCS
from matplotlib import rc
from matplotlib import cm
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.colors import ListedColormap
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'

# Getting photometry zero point
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'HE0238-1904_sex.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)

path_image = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'check.fits')
data_image = fits.getdata(path_image, 1, ignore_missing_end=True)

path_gaia = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'Gaia_coor.fits')
data_gaia = fits.getdata(path_gaia, 1, ignore_missing_end=True)

ra_pho = data_pho['ALPHAWIN_J2000']
dec_pho = data_pho['DELTAWIN_J2000']

ra_gaia = data_gaia['ra']
dec_gaia = data_gaia['dec']

c_gaia = SkyCoord(ra_gaia, dec_gaia, unit="deg")
c_pho = SkyCoord(ra_pho, dec_pho, unit="deg")
idx, d2d, d3d = c_gaia.match_to_catalog_sky(c_pho)

max_sep = 1.0 * u.arcsec
sep_lim = d2d < max_sep

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)

fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_image, figure=fig, north=True)
gc.show_colorscale(vmin=0, vmax=3, cmap=newcmp)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc.recenter(40.1359, -18.8643, width=0.06, height=0.06)
gc.show_circles(ra_gaia[sep_lim], dec_gaia[sep_lim], 0.0002)
gc.show_circles(ra_pho[idx[sep_lim]], dec_pho[idx[sep_lim]], 0.0002, facecolor='red')
plt.savefig(path_savefig + 'HST_check', bbox_inches='tight')

print(d2d)
print(ra_pho[idx], dec_pho[idx])
print(ra_gaia, dec_gaia)
print(len(idx))