import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from regions import Regions
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
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_MakeMovie',
                        filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
    if smooth is True:
        kernel = Gaussian2DKernel(x_stddev=10.0, x_size=3, y_size=3)
        data = convolve(data, kernel)
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_MakeMovie/' + filename + '_revised.fits', data, overwrite=True)
    data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_MakeMovie/' + filename + '_revised.fits',
                               0, header=True)
    hdr1['BITPIX'], hdr1['NAXIS'], hdr1['NAXIS1'], hdr1['NAXIS2'] = hdr['BITPIX'], hdr['NAXIS'], hdr['NAXIS1'], \
                                                                    hdr['NAXIS2']
    hdr1['CRPIX1'], hdr1['CRPIX2'], hdr1['CTYPE1'], hdr1['CTYPE2'] = hdr['CRPIX1'], hdr['CRPIX2'], hdr['CTYPE1'], \
                                                                     hdr['CTYPE2']
    hdr1['CRVAL1'], hdr1['CRVAL2'], hdr1['LONPOLE'], hdr1['LATPOLE'] = hdr['CRVAL1'], hdr['CRVAL2'], hdr['LONPOLE'], \
                                                                       hdr['LATPOLE']
    hdr1['CSYER1'], hdr1['CSYER2'], hdr1['MJDREF'], hdr1['RADESYS'] = hdr['CSYER1'], hdr['CSYER2'], hdr['MJDREF'], \
                                                                      hdr['RADESYS']
    hdr1['CD1_1'], hdr1['CD1_2'], hdr1['CD2_1'], hdr1['CD2_2'] = hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2']

    # Rescale the data by 1e17
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_MakeMovie/' + filename + '_revised.fits',
                 data1 * 1e17, hdr1, overwrite=True)

# OIII
path_cube_OIII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                              'CUBE_OIII_5008_line_offset.fits')
cube_OIII = Cube(path_cube_OIII)
# 8140
# 8180
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

# Plot the data
# Read region file
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

#
path_label = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_label_list.reg')
regions_label = Regions.read(path_label, format='ds9')


def MakeOIIIMap(gal=False, region=False, video=False):
    for i in range(6):
        fig = plt.figure(figsize=(8, 8), dpi=300)
        z_qso = 0.6282144177077355
        OIII_air = 5006.843

        # Split by velocity
        dv_i, dv_f = -500 + 200 * i, -500 + 200 * (i + 1)
        wave_i = OIII_air * (1 + z_qso) * (dv_i / 3e5 + 1)
        wave_f = OIII_air * (1 + z_qso) * (dv_f / 3e5 + 1)
        wave_i_vac, wave_f_vac = pyasl.airtovac2(wave_i), pyasl.airtovac2(wave_f)

        # Split by wavelength
        # wave_i = 8150 + 7 * i
        # wave_f = 8150 + 7 * (i + 1)
        # wave_i_vac, wave_f_vac = pyasl.airtovac2(wave_i), pyasl.airtovac2(wave_f)
        # dv_i, dv_f = 3e5 * ((wave_i_vac / 5008.239 - 1) - z_qso)/ (1 + z_qso), \
        #              3e5 * ((wave_f_vac / 5008.239 - 1) - z_qso)/ (1 + z_qso)

        # Slice the cube
        sub_cube = cube_OIII.select_lambda(wave_i, wave_f)
        sub_cube = sub_cube.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2
        sub_cube.write('/Users/lzq/Dropbox/Data/CGM/image_MakeMovie/image_make_OIII_NB.fits')
        ConvertFits(filename='image_make_OIII_NB')
        path_subcube = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_MakeMovie',
                                    'image_make_OIII_NB_revised.fits')

        # Plot
        gc = aplpy.FITSFigure(path_subcube, figure=fig, north=True, animated=True)
        gc.set_system_latex(True)
        gc.show_colorscale(vmin=0, vmid=0.2, vmax=15.0, cmap=newcmp, stretch='arcsinh')
        gc.show_markers(ra_qso_muse, dec_qso_muse, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                        linewidths=0.5, s=400, zorder=100)

        if gal:
            gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k',
                            linewidths=0.8, s=100)

        # Plot regions
        if region:
            gc.show_circles(ra_array, dec_array, radius_array / 3600, edgecolors='k', linestyles='--', linewidths=1,
                            alpha=0.3)
            for j in range(len(ra_array)):
                x = regions_label[j].center.ra.degree
                y = regions_label[j].center.dec.degree
                gc.add_label(x, y, text_array[j], size=20)
        else:
            gc.show_contour(path_subcube, levels=[0.3], colors='k', linewidths=0.8, smooth=3)

        # Colorbar
        gc.add_colorbar()
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_pad(0.0)
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_ticks([1, 5, 10])
        gc.colorbar.set_axis_label_font(size=20)
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')

        # Scalebar
        gc.add_scalebar(length=15 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
        gc.scalebar.set_font_size(20)

        # Hide
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()
        gc.ticks.set_length(30)

        # Label
        xw, yw = gc.pixel2world(195, 140)
        gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
        gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
        gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
        gc.add_label(0.88, 0.70, r'E', size=20, relative=True)
        gc.add_label(0.87, 0.97, r'MUSE [O III]', size=20, relative=True)
        gc.add_label(0.82, 0.91, r'$\mathrm{\lambda = \,}$' + str("{0:.0f}".format(wave_i_vac)) + ' to '
                     + str("{0:.0f}".format(wave_f_vac)) + r'$\mathrm{\AA}$', size=20, relative=True)
        gc.add_label(0.76, 0.85, r'$\mathrm{\Delta} v \approx \,$' + str("{0:.0f}".format(dv_i)) + ' to '
                     + str("{0:.0f}".format(dv_f)) + r'$\mathrm{\, km \, s^{-1}}$', size=20, relative=True)
        if region:
            fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/NB_movie/image_OIII_' + str("{0:.0f}".format(dv_i)) + '_' +
                        str("{0:.0f}".format(dv_f)) + '_region.png', bbox_inches='tight')
        else:
            fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/NB_movie/image_OIII_' + str("{0:.0f}".format(dv_i)) + '_' +
                        str("{0:.0f}".format(dv_f)) + '.png', bbox_inches='tight')
    if video:
        os.system('convert -delay 75 ~/dropbox/Data/CGM_plots/NB_movie/image_OIII_*.png '
                  '~/dropbox/Data/CGM_plots/NB_movie/OIII_movie.gif')


#
MakeOIIIMap(region=False)
MakeOIIIMap(region=True)