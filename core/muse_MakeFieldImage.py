import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from mpdaf.obj import Image
from astropy import units as u
from regions import Regions
from muse_compare_z import compare_z
from astropy.convolution import convolve
from muse_RenameGal import ReturnGalLabel
from astropy.convolution import Gaussian2DKernel
from regions import RectangleSkyRegion, PixCoord, RectanglePixelRegion
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10


# Convert Fits file into correct form
def ConvertFits(filename=None, smooth=True):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_narrow', filename + '.fits')
    image = Image(path)
    image.write('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits', savemask='nan')
    data, hdr = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits', 1, header=True)
    if smooth is True:
        kernel = Gaussian2DKernel(x_stddev=10.0, x_size=3, y_size=3)
        data = convolve(data, kernel)
    # Mask the data
    xx, yy = np.meshgrid(np.arange(len(image.data)), np.arange(len(image.data)))
    pixel_center = PixCoord(x=100, y=75)
    pixel_region = RectanglePixelRegion(center=pixel_center, width=90, height=90)
    pixel_data = PixCoord(x=xx, y=yy)
    mask = pixel_region.contains(pixel_data)
    data = np.where(mask, data, np.nan)
    # Rename
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits', data, overwrite=True)
    data_revised, hdr_revised = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits', 0, header=True)
    # Info
    hdr_revised['BITPIX'], hdr_revised['NAXIS'] = hdr['BITPIX'], hdr['NAXIS']
    hdr_revised['NAXIS1'], hdr_revised['NAXIS2'] = hdr['NAXIS1'], hdr['NAXIS2']
    hdr_revised['CRPIX1'], hdr_revised['CRPIX2'] = hdr['CRPIX1'], hdr['CRPIX2']
    hdr_revised['CTYPE1'], hdr_revised['CTYPE2'] = hdr['CTYPE1'], hdr['CTYPE2']
    hdr_revised['CRVAL1'], hdr_revised['CRVAL2'] = hdr['CRVAL1'], hdr['CRVAL2']
    hdr_revised['LONPOLE'], hdr_revised['LATPOLE'] = hdr['LONPOLE'], hdr['LATPOLE']
    hdr_revised['CSYER1'], hdr_revised['CSYER2'] = hdr['CSYER1'], hdr['CSYER2'],
    hdr_revised['MJDREF'], hdr_revised['RADESYS'] = hdr['MJDREF'], hdr['RADESYS']
    hdr_revised['CD1_1'], hdr_revised['CD1_2'] = hdr['CD1_1'], hdr['CD1_2']
    hdr_revised['CD2_1'], hdr_revised['CD2_2'] = hdr['CD2_1'], hdr['CD2_2']

    # Rescale the data by 1e17
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_narrow/' + filename + '_revised.fits',
                 data_revised * 1e17, hdr_revised, overwrite=True)


ConvertFits(filename='image_OII_line_SB_offset', smooth=True)
ConvertFits(filename='image_OIII_5008_line_SB_offset', smooth=True)

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

# Label the galaxy
path_label = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'galaxy_label_list.reg')
regions_label = Regions.read(path_label, format='ds9')

# Load galxies infomation
row_final, ID_final, name_final, z_final, ra_final, dec_final = ReturnGalLabel(sort_row=False, mode='initial')
ID_sep_final = ReturnGalLabel(sort_row=True, mode='final')[6]

# Calculate the offset between MUSE and HST
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_qso_hst, dec_qso_hst = 40.1359, -18.8643  # Wrong!!!

# Plot
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
gc.set_system_latex(True)
gc.recenter(40.1359, -18.8643, width=40 / 3600, height=40 / 3600)  # 0.02 / 0.01 40''
gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
# gc.show_colorscale(cmap='Greys', vmin=0, vmax=4.897e-2)
gc.show_contour(path_OII_SB, levels=[0.4, 2], colors='blue', linewidths=0.8, smooth=3)
gc.show_contour(path_OIII_SB, levels=[0.4, 2], colors='red', linewidths=0.8, smooth=3)
gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=1.5, s=330)
# for i in range(len(row_final)):
#     x = regions_label[i].center.ra.degree
#     y = regions_label[i].center.dec.degree
#     text = 'G' + str(ID_sep_final[i])
#     gc.add_label(x, y, text, size=20)

# Colorbar
gc.add_colorbar()
gc.colorbar.set_ticks([0])
gc.colorbar.set_location('bottom')
gc.colorbar.set_pad(0.0)
gc.colorbar.hide()

# Scalebar
gc.add_scalebar(length=15 * u.arcsecond)
gc.scalebar.set_corner('bottom left')
gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
gc.scalebar.set_font_size(20)

# Hide
gc.ticks.hide()
gc.tick_labels.hide()
gc.axis_labels.hide()
gc.ticks.set_length(30)

# Label
xw, yw = 40.13029488729661, -18.862067749086557
gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
gc.add_label(0.9778, 0.80, r'N', size=20, relative=True)
gc.add_label(0.88, 0.695, r'E', size=20, relative=True)
gc.add_label(0.85, 0.91, r'MUSE [O II]', color='blue', size=20, relative=True)
gc.add_label(0.85, 0.86, r'MUSE [O III]', color='red', size=20, relative=True)
gc.add_label(0.85, 0.96, r'$\mathrm{ACS+F814W}$', color='k', size=20, relative=True)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image.png', bbox_inches='tight')