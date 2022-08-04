import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from muse_compare_z import compare_z

#
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_drc_offset.fits')
data_hb = fits.open(path_hb)
wcs = WCS(data_hb[1].header)

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

# Calculate the offset between MUSE and gaia
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_qso_gaia, dec_qso_gaia = 40.13576715640353, -18.86426977828008
ra_final = ra_final - (ra_qso_gaia - ra_qso_muse)
dec_final = dec_final - (dec_qso_gaia - dec_qso_muse)


# Change coordinate for galaxy with row = 1
row_1_sort = np.where(row_final == 1)
ra_final[row_1_sort] = 40.1440392
dec_final[row_1_sort] = -18.8597159

z_qso = 0.6282144177077355
v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)

# Calculate the offset between MUSE and HST
# ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
# ra_qso_hst, dec_qso_hst = 40.1359, -18.8643
# ra_qso_gaia, dec_qso_gaia = 40.13576715640353, -18.86426977828008
# ra_final = ra_final - (ra_qso_hst - ra_qso_muse)  Wrong!!!
# dec_final = dec_final - (dec_qso_hst - dec_qso_muse)  # Wrong!!!

skycoord_host = SkyCoord(ra_qso_gaia, dec_qso_gaia, unit='deg', frame=FK5)
# print(skycoord_host.to_string('hmsdms', sep=':'))
skycoord = SkyCoord(ra_final, dec_final, unit='deg', frame=FK5)
sep = skycoord.separation(skycoord_host)
print(sep.arcsecond)


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.luminosity_distance(z_final).to(u.kpc)
d_l_host = cosmo.luminosity_distance(z_qso).to(u.kpc)
c1 = SkyCoord(ra_qso_gaia * u.deg, dec_qso_gaia * u.deg, distance=d_l_host, frame=FK5)
c2 = SkyCoord(ra_final * u.deg, dec_final * u.deg, distance=d_l, frame=FK5)
print(c1.separation_3d(c2))
print(v_gal)
# print(skycoord.to_string('hmsdms', sep=':'))
