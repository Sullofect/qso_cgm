import os
import glob
import aplpy
import coord
import shutil
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import rc
from astropy import stats
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.wcs import WCS
from mpdaf.obj import Cube, WaveCoord, Image
from astropy.coordinates import SkyCoord
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
from astropy.convolution import Kernel, convolve, Gaussian2DKernel

# IMACS object files
obj_3C57i1 = fits.open('../../MUSEQuBES+CUBS/IMACS/3C57/3C57i1_objects.fits')
obj_3C57i2 = fits.open('../../MUSEQuBES+CUBS/IMACS/3C57/3C57i2_objects.fits')
obj_3C57i3 = fits.open('../../MUSEQuBES+CUBS/IMACS/3C57/3C57i3_objects.fits')
dat_3C57 = ascii.read('../../MaskDesign/3C57_mask/3C57_@_ac.dat')
# obj_3C57i3 = fits.open('../../MUSEQuBES+CUBS/IMACS/3C57/3C57i3_objects.fits')

#
z_i1, z_i2, z_i3 = obj_3C57i1[1].data['redshift'], obj_3C57i2[1].data['redshift'], obj_3C57i3[1].data['redshift']
id_z1, id_z2, id_z3 = obj_3C57i1[1].data['id'], obj_3C57i2[1].data['id'], obj_3C57i3[1].data['id']
z_total = np.hstack((z_i1, z_i2, z_i3))
id_total = np.hstack((id_z1, id_z2, id_z3))
v_total = 3e5 * (z_total - 0.6718) / (1 + 0.6718)

#
id, ra, dec = dat_3C57['col1'], np.asarray(dat_3C57['col2']), np.asarray(dat_3C57['col3'])
mask = (v_total < 2000) * (v_total > -2000)
id_mask = id_total[mask]
id_mask = np.array(['@' + i for i in id_mask])
overlap = np.in1d(id, id_mask)
ra_overlap, dec_overlap = ra[overlap], dec[overlap]

coord = SkyCoord(ra_overlap, dec_overlap, unit=(u.hour, u.deg), frame='icrs')
print(coord.ra.deg, coord.dec.deg)
#
# plt.figure()
# plt.hist(z_total[mask])
# plt.show()

filename = '../../MUSEQuBES+CUBS/gal_info/3C57_gal_IMACS.fits'
t = Table()
t['ra'] = coord.ra.deg
t['dec'] = coord.dec.deg
t['z'] = z_total[mask]
t['v'] = v_total[mask]
t.write(filename, format='fits', overwrite=True)

cubename = '3C57'
path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
data_gal = fits.open(path_gal)[1].data
ra_gal, dec_gal, v_gal = data_gal['ra'], data_gal['dec'], data_gal['v']

# Check SDSS image
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure('../../MaskDesign/3C57/3C57_imagecut/3C57_r_coadd.fits', figure=fig, north=True)
gc.recenter(ra_gal[0], dec_gal[0], width=1200 / 3600, height=1200 / 3600)

gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
#
gc.set_system_latex(True)
gc.show_colorscale(cmap='Greys')

gc.add_colorbar()
gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
gc.colorbar.hide()

# Hide ticks
gc.ticks.set_length(30)
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()

# Markers
gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
gc.show_markers(t['ra'], t['dec'], facecolor='none', marker='o', c='none',
                edgecolors='r', linewidths=0.8, s=150)
path_savefig = '../../MUSEQuBES+CUBS/plots/{}_IMACS.png'.format(cubename)
fig.savefig(path_savefig, bbox_inches='tight')