import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from PyAstronomy import pyasl
from astropy import units as u
from muse_compare_z import compare_z
from astropy.convolution import convolve
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
from astropy.convolution import Gaussian2DKernel
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
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


# Convert Fits file into correct form
def ConvertFits(filename=None, smooth=True):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
    if smooth is True:
        kernel = Gaussian2DKernel(x_stddev=10.0, x_size=3, y_size=3)
        data = convolve(data, kernel)
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/' + filename + '_revised.fits', data, overwrite=True)
    data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/' + filename + '_revised.fits', 0, header=True)
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
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/' + filename + '_revised.fits', data1 * 1e17, hdr1, overwrite=True)


# Make movie OII
path_cube_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
cube_OII = Cube(path_cube_OII)

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
select_array = np.sort(np.array([6, 7, 181, 182, 80, 81, 82, 179, 4, 5, 64]))
select_gal = np.in1d(row_final, select_array)
row_final = row_final[select_gal]
z_final = z_final[select_gal]
ra_final = ra_final[select_gal]
dec_final = dec_final[select_gal]

# Calculate the offset between MUSE and HST
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
# ra_qso_hst, dec_qso_hst = 40.1359, -18.8643
# ra_final = ra_final - (ra_qso_hst - ra_qso_muse)  # Wrong!!!
# dec_final = dec_final - (dec_qso_hst - dec_qso_muse)  # Wrong!!!

for i in range(8):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    wave_i = 6050 + 5 * i
    wave_f = 6050 + 5 * (i + 1)
    wave_i_vac, wave_f_vac = pyasl.airtovac2(wave_i), pyasl.airtovac2(wave_f)
    z_qso = 0.6282144177077355
    dv_i, dv_f = 3e5 * ((wave_i_vac / 3727.092 - 1) - z_qso)/ (1 + z_qso), \
                 3e5 * ((wave_f_vac / 3727.092 - 1) - z_qso)/ (1 + z_qso)
    sub_cube = cube_OII.select_lambda(wave_i, wave_f)
    sub_cube = sub_cube.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2
    sub_cube.write('/Users/lzq/Dropbox/Data/CGM/image_make_OII_NB.fits')
    ConvertFits(filename='image_make_OII_NB')
    path_subcube = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_make_OII_NB_revised.fits')
    gc = aplpy.FITSFigure(path_subcube, figure=fig, north=True, animated=True)
    gc.set_system_latex(True)
    gc.show_colorscale(vmin=0, vmid=0.2, vmax=5.0, cmap=newcmp, stretch='arcsinh')
    gc.add_colorbar()
    gc.ticks.set_length(30)
    gc.show_markers(ra_qso_muse, dec_qso_muse, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=400)
    # gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    # gc.show_regions('/Users/lzq/Dropbox/Data/CGM/gas_list.reg')
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_ticks([1, 2, 3, 4])
    gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                    r's^{-1} \; arcsec^{-2}]}$')
    gc.colorbar.set_font(size=15)
    gc.colorbar.set_axis_label_font(size=15)
    # gc.colorbar.set_box([0.1247, 0.0927, 0.7443, 0.03], box_orientation='horizontal')
    gc.add_scalebar(length=15 * u.arcsecond)
    gc.scalebar.set_corner('top left')
    gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
    gc.scalebar.set_font_size(15)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.add_label(0.87, 0.97, r'MUSE [O II]', size=15, relative=True)
    gc.add_label(0.85, 0.92, r'$\mathrm{\lambda = \,}$' + str("{0:.0f}".format(wave_i_vac)) + ' to '
                 + str("{0:.0f}".format(wave_f_vac)) + r'$\mathrm{\AA}$', size=15,
                 relative=True)
    gc.add_label(0.81, 0.88, r'$\mathrm{\Delta} v \approx \,$' + str("{0:.0f}".format(dv_i)) + ' to '
                 + str("{0:.0f}".format(dv_f)) + r'$\mathrm{\, km \, s^{-1}}$', size=15,
                 relative=True)
    xw, yw = gc.pixel2world(195, 150)
    gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
    gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
    gc.add_label(0.9775, 0.85, r'N', size=15, relative=True)
    gc.add_label(0.88, 0.75, r'E', size=15, relative=True)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/NB_movie/image_OII_' + str("{0:.0f}".format(wave_i_vac)) + '_' +
                str("{0:.0f}".format(wave_f_vac)) + '.png', bbox_inches='tight')