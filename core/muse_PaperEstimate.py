import os
import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate
from astropy.io import ascii
from astropy.table import Table
from matplotlib import rc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from regions import PixCoord
from regions import RectangleSkyRegion, RectanglePixelRegion
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


# Completness
path_s = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects_sean.fits')
data_s = fits.getdata(path_s, 1, ignore_missing_end=True)
path_w = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects.fits')
data_w = fits.getdata(path_w, 1, ignore_missing_end=True)

# Basic information in catalog
ra_w, dec_w = data_w['ra'], data_w['dec']
row_s, row_w = data_s['row'], data_w['row']
ID_s, ID_w = data_s['id'], data_w['id']
name_s, name_w = data_s['name'], data_w['name']
ql_s, ql_w = data_s['quality'], data_w['quality']
cl_s, cl_w = data_s['class'], data_w['class']
z_s, z_w = data_s['redshift'], data_w['redshift']
ct_s, ct_w = data_s['comment'], data_w['comment']
cl_s_num, cl_w_num = np.zeros_like(cl_s), np.zeros_like(cl_w)

# Only need
sort = ql_w != 0
ra_w, dec_w = ra_w[sort], dec_w[sort]
z_w = z_w[sort]
row_w = row_w[sort]

# Getting photometry zero point
path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'gal_all',
                        'HE0238-1904_sex_gal_all.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_drc_offset.fits')
wcs_hst = WCS(fits.open(path_hb)[1].header)

# match two catalog
catalog = SkyCoord(data_pho['AlPHAWIN_J2000'], data_pho['DELTAWIN_J2000'], unit="deg")
c = SkyCoord(ra_w, dec_w, unit="deg")
test = SkyCoord(40.1345022, -18.8509434, unit="deg")
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# Photometry
data_pho_red = data_pho[idx]
mag_auto, dmag_auto = data_pho['MAG_AUTO'], data_pho['MAGERR_AUTO']
center_sky = SkyCoord(40.1359, -18.8643, unit='deg', frame='fk5')
region_sky = RectangleSkyRegion(center=center_sky, width=65 / 3600 * u.deg, height=65 / 3600 * u.deg, angle=60 * u.deg)
mask = region_sky.contains(catalog, wcs_hst)
print(region_sky.contains(test, wcs_hst))
#
number_red, x_image_red, y_image_red = data_pho_red['NUMBER'], data_pho_red['X_IMAGE'], data_pho_red['Y_IMAGE']
mag_iso_red, dmag_iso_red = data_pho_red['MAG_ISO'], data_pho_red['MAGERR_ISO']
mag_isocor_red, dmag_isocor_red = data_pho_red['MAG_ISOCOR'], data_pho_red['MAGERR_ISOCOR']
mag_auto_red, dmag_auto_red = data_pho_red['MAG_AUTO'], data_pho_red['MAGERR_AUTO']
print(row_w[(mag_auto_red < 23) * (mag_auto_red > 22)])
print(mag_auto_red[(mag_auto_red < 23) * (mag_auto_red > 22)])
print(mag_auto[mask][(mag_auto[mask] < 23) * (mag_auto[mask] > 22)])
#
bins = np.linspace(18, 30, 13)
plt.figure(figsize=(5, 5))
plt.hist(mag_auto_red, bins=bins, color='red', histtype='step', alpha=0.5, label='Redshift survey')
plt.hist(mag_auto[mask], bins=bins, color='blue', histtype='step', alpha=0.5, label='Entire catalog')
plt.minorticks_on()
plt.xlabel('Redshift')
plt.ylabel('Number')
plt.legend()
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Mag_vs_redshift.png', bbox_inches='tight')





