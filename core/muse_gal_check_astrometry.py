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

def offset():
    return np.mean(ra_pho[idx_total[sep_lim_total]] - ra_total[sep_lim_total]), \
           np.mean(dec_pho[idx_total[sep_lim_total]] - dec_total[sep_lim_total])

# Getting photometry zero point
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'HE0238-1904_sex.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)

path_image = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'check.fits')
data_image = fits.getdata(path_image, 1, ignore_missing_end=True)

path_gaia = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'Gaia_coor.fits')
data_gaia = fits.getdata(path_gaia, 1, ignore_missing_end=True)

path_gaia = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'Gaia_coor.fits')
data_gaia = fits.getdata(path_gaia, 1, ignore_missing_end=True)

path_0399m190 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'tractor-0399m190.fits')
data_0399m190 = fits.getdata(path_0399m190, 1, ignore_missing_end=True)

path_0402m190 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'tractor-0402m190.fits')
data_0402m190 = fits.getdata(path_0402m190, 1, ignore_missing_end=True)

path_0401m187 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'tractor-0401m187.fits')
data_0401m187 = fits.getdata(path_0401m187, 1, ignore_missing_end=True)

ra_0399m190, dec_0399m190 = data_0399m190['ra'], data_0399m190['dec']
ra_0402m190, dec_0402m190 = data_0402m190['ra'], data_0402m190['dec']
ra_0401m187, dec_0401m187 = data_0401m187['ra'], data_0401m187['dec']
flux_r_0399m190 = data_0399m190['flux_r']
flux_r_0402m190 = data_0402m190['flux_r']
flux_r_0401m187 = data_0401m187['flux_r']

ra_total = np.hstack([ra_0399m190, ra_0402m190, ra_0401m187])
dec_total = np.hstack([dec_0399m190, dec_0402m190, dec_0401m187])
mag_r_total = 22.5 - 2.5 * np.log10(np.hstack([flux_r_0399m190, flux_r_0402m190, flux_r_0401m187]))

mag_cut = np.where((mag_r_total < 22.5) & (mag_r_total > 18))
ra_total = ra_total[mag_cut]
dec_total = dec_total[mag_cut]

c_total = SkyCoord(ra_total, dec_total, unit="deg")

ra_pho = data_pho['ALPHAWIN_J2000']
dec_pho = data_pho['DELTAWIN_J2000']

ra_gaia = data_gaia['ra']
dec_gaia = data_gaia['dec']

c_gaia = SkyCoord(ra_gaia, dec_gaia, unit="deg")
c_pho = SkyCoord(ra_pho, dec_pho, unit="deg")

idx, d2d, d3d = c_gaia.match_to_catalog_sky(c_pho)
idx_total, d2d_total, d3d_total = c_total.match_to_catalog_sky(c_pho)

max_sep = 0.8 * u.arcsec
sep_lim_total = d2d_total < max_sep

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
gc.recenter(40.1359, -18.8643, width=0.08, height=0.12)
ra_offset, dec_offset = offset()
gc.show_circles(ra_total[sep_lim_total] + ra_offset, dec_total[sep_lim_total] + dec_offset, 0.0002, facecolor='blue')
gc.show_circles(ra_pho[idx_total[sep_lim_total]], dec_pho[idx_total[sep_lim_total]], 0.0002, facecolor='red')
plt.savefig(path_savefig + 'HST_check', bbox_inches='tight')

print(ra_pho[idx[sep_lim]] - ra_gaia[sep_lim])
print(dec_pho[idx[sep_lim]] - dec_gaia[sep_lim])

plt.figure(figsize=(8, 5))
plt.quiver(ra_pho[idx_total[sep_lim_total]], dec_pho[idx_total[sep_lim_total]], ra_pho[idx_total[sep_lim_total]] - ra_total[sep_lim_total],
           dec_pho[idx_total[sep_lim_total]] - dec_total[sep_lim_total])
# plt.plot(ra_gaia[sep_lim], ra_pho[idx[sep_lim]] - ra_gaia[sep_lim], '.')
plt.savefig(path_savefig + 'vector', bbox_inches='tight')
# plt.show()

f, axarr = plt.subplots(2, 2, figsize=(8, 5), dpi=300)
f.subplots_adjust(hspace=0.3)
f.subplots_adjust(wspace=0.3)
axarr[0, 0].plot(ra_pho[idx_total[sep_lim_total]], 3600 * (ra_pho[idx_total[sep_lim_total]] - ra_total[sep_lim_total]), '.')
axarr[0, 1].plot(ra_pho[idx_total[sep_lim_total]], 3600 * (dec_pho[idx_total[sep_lim_total]] - dec_total[sep_lim_total]), '.')
axarr[1, 0].plot(dec_pho[idx_total[sep_lim_total]], 3600 * (ra_pho[idx_total[sep_lim_total]] - ra_total[sep_lim_total]), '.')
axarr[1, 1].plot(dec_pho[idx_total[sep_lim_total]], 3600 * (dec_pho[idx_total[sep_lim_total]] - dec_total[sep_lim_total]), '.')
axarr[0, 0].set_xlabel('ra')
axarr[0, 0].set_ylabel(r'$\delta ra$')
axarr[0, 1].set_xlabel('ra')
axarr[0, 1].set_ylabel(r'$\delta dec$')
axarr[1, 0].set_xlabel('dec')
axarr[1, 0].set_ylabel(r'$\delta ra$')
axarr[1, 1].set_xlabel('dec')
axarr[1, 1].set_ylabel(r'$\delta dec$')
plt.savefig(path_savefig + 'coor_compare', bbox_inches='tight')
# plt.show()

# from astropy.table import Table
# data = Table()
# data["RA"] =
# data["DEC"] = dec_final
# data['ID'] = row_final
# #data["NAME"] =  np.core.defchararray.add(np.char.mod('%d', row_final), np.repeat(np.array(['o']), len(row_final)))
# #data["COLOR"] = np.repeat(np.array(['black']), len(row_final))
# #data["RADIUS"] = np.repeat(np.array(['1']), len(row_final))
# ascii.write(data, 'galaxys_list_xmatch.csv', format='csv', overwrite=True)