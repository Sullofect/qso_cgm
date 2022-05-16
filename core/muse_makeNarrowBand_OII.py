import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
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
def ConvertFits(filename='image_OIII_5008_line_SB_offset'):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
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


# Make movie
path_cube_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
cube_OII = Cube(path_cube_OII)
cube_OII = cube_OII[30:60, :, :]
path_cube_OIII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_5008_line_offset.fits')
cube_OIII = Cube(path_cube_OIII)
cube_OIII = cube_OIII[30:60, :, :]

# OII
for i in range(np.shape(cube_OII)[0]):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    wave_i = cube_OII.wave.coord()[i]
    sub_cube = cube_OII.select_lambda(wave_i)
    sub_cube = sub_cube * 1.25 * 1e-20 / 0.2 / 0.2
    sub_cube.write('/Users/lzq/Dropbox/Data/CGM/image_make_movie.fits')
    ConvertFits(filename='image_make_movie')
    path_subcube = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_make_movie_revised.fits')
    gc = aplpy.FITSFigure(path_subcube, figure=fig, north=True, animated=True)
    gc.set_system_latex(True)
    gc.show_colorscale(vmin=0, vmax=1.5, cmap=newcmp)
    gc.add_colorbar()
    gc.ticks.set_length(30)
    # gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                    r's^{-1} \; arcsec^{-2}]}$')
    gc.colorbar.set_font(size=15)
    gc.colorbar.set_axis_label_font(size=15)
    gc.add_scalebar(length=15 * u.arcsecond)
    gc.scalebar.set_corner('top left')
    gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
    gc.scalebar.set_font_size(15)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.add_label(0.95, 0.97, r'[O II]', size=15, relative=True)
    gc.add_label(0.88, 0.92, r'$\mathrm{\lambda_{air} = \,}$' + str("{0:.0f}".format(wave_i)) + r'$\mathrm{\AA}$', size=15,
                 relative=True)
    xw, yw = gc.pixel2world(195, 150)
    gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
    gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
    gc.add_label(0.9775, 0.85, r'N', size=15, relative=True)
    gc.add_label(0.88, 0.75, r'E', size=15, relative=True)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/NB_movie/image_OII_' + str("{0:.0f}".format(wave_i))
                + '.png', bbox_inches='tight')