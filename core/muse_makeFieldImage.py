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


# Convert Fits file into correct form
def ConvertFits(filename=None, smooth=True):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_narrow', filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
    if smooth is True:
        kernel = Gaussian2DKernel(x_stddev=10.0, x_size=3, y_size=3)
        data = convolve(data, kernel)
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits', data, overwrite=True)
    data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits', 0, header=True)
    hdr1['BITPIX'], hdr1['NAXIS'], hdr1['NAXIS1'], hdr1['NAXIS2'] = hdr['BITPIX'], hdr['NAXIS'], hdr['NAXIS1'], \
                                                                    hdr['NAXIS2']
    hdr1['CRPIX1'], hdr1['CRPIX2'], hdr1['CTYPE1'], hdr1['CTYPE2'] = hdr['CRPIX1'], hdr['CRPIX2'], hdr['CTYPE1'], \
                                                                     hdr['CTYPE2']
    hdr1['CRVAL1'], hdr1['CRVAL2'], hdr1['LONPOLE'], hdr1['LATPOLE'] = hdr['CRVAL1'], hdr['CRVAL2'], hdr['LONPOLE'], \
                                                                       hdr['LATPOLE']
    hdr1['CSYER1'], hdr1['CSYER2'], hdr1['MJDREF'], hdr1['RADESYS'] = hdr['CSYER1'], hdr['CSYER2'], hdr['MJDREF'], \
                                                                      hdr['RADESYS']
    hdr1['CD1_1'], hdr1['CD1_2'], hdr1['CD2_1'], hdr1['CD2_2'] =  hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2']

    # Rescale the data by 1e17
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits',
                 data1 * 1e17, hdr1, overwrite=True)


ConvertFits(filename='image_OII_line_SB_offset', smooth=False)
ConvertFits(filename='image_OIII_5008_line_SB_offset', smooth=False)

#
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)
path_OII_SB = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_narrow',
                           'image_OII_line_SB_offset_revised.fits')
path_OIII_SB = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_narrow',
                            'image_OIII_5008_line_SB_offset_revised.fits')

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
# select_array = np.sort(np.array([6, 7, 181, 182, 80, 81, 82, 83, 179, 4, 5, 64]))
select_array = np.sort(np.array([1, 4, 5, 6, 7, 13, 20, 27, 35, 36, 57, 62, 64, 68, 72, 78, 80, 81, 82, 83, 88, 92,
                                 93, 120, 129, 134, 140, 141, 149, 162, 164, 179, 181, 182]))
select_gal = np.in1d(row_final, select_array)
row_final = row_final[select_gal]
z_final = z_final[select_gal]
ra_final = ra_final[select_gal]
dec_final = dec_final[select_gal]

# Calculate the offset between MUSE and HST
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_qso_hst, dec_qso_hst = 40.1359, -18.8643  # Wrong!!!
# ra_final = ra_final - (ra_qso_hst - ra_qso_muse)  # Wrong!!!
# dec_final = dec_final - (dec_qso_hst - dec_qso_muse)  # Wrong!!!

# Plot
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc.show_contour(path_OII_SB, levels=[0.15, 0.3, 0.5, 2], colors='blue', linewidths=1, smooth=7)
gc.show_contour(path_OIII_SB, levels=[0.15, 0.3, 0.5, 2], colors='red', linewidths=1, smooth=7)
gc.recenter(40.1359, -18.8643, width=40 / 3600, height=40 / 3600)  # 0.02 / 0.01 40''
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
gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
gc.add_label(0.87, 0.93, r'MUSE [O II]', color='blue', size=15, relative=True)
gc.add_label(0.87, 0.89, r'MUSE [O III]', color='red', size=15, relative=True)
gc.add_label(0.87, 0.97, r'$\mathrm{ACS+F814W}$', color='k', size=15, relative=True)
xw, yw = 40.13029488729661, -18.861467749086557
gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
gc.add_label(0.9775, 0.85, r'N', size=15, relative=True)
gc.add_label(0.88, 0.75, r'E', size=15, relative=True)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image.png', bbox_inches='tight')