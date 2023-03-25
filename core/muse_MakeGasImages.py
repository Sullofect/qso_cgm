import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import palettable.scientific.sequential as sequential_s
from matplotlib import rc
from regions import Regions
from PyAstronomy import pyasl
from astropy import units as u
from astropy.convolution import convolve
from muse_RenameGal import ReturnGalLabel
from astropy.convolution import Gaussian2DKernel, Box2DKernel
from mpdaf.obj import Image, Cube, WCS, WaveCoord, iter_spe, iter_ima
from regions import RectangleSkyRegion, PixCoord, RectanglePixelRegion
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
path_data = '/Users/lzq/Dropbox/Data/CGM/'


# Convert Fits file into correct form
def ConvertFits(table=None, type=None, mask=None, filename=None, smooth=True, smooth_val=3, contour=False):
    if type == 'NarrowBand':
        path = path_data + 'image_MakeMovie/' + filename + '.fits'
        if contour:
            filename = filename + '_contour'
        path_revised = path_data + 'image_MakeMovie/' + filename + '_revised.fits'
    elif type == 'FieldImage':
        path = path_data + 'image_narrow/' + filename + '.fits'
        path_revised = path_data + 'image_narrow/' + filename + '_revised.fits'
        image = Image(path)
        image.write(path_revised, savemask='nan')
    elif type == 'GasMap':
        path = path_data + 'image_plot/' + filename + '.fits'
        path_revised = path_data + 'image_plot/' + filename + '_revised.fits'
    else:
        print('Loading Data is wrong')

    data, hdr = fits.getdata(path, 1, header=True)
    if smooth:
        kernel = Box2DKernel(smooth_val)
        # kernel = Gaussian2DKernel(x_stddev=5.0, x_size=3, y_size=3)
        data = convolve(data, kernel)
    if table is not None:
        data = table
    if mask:
        # Mask the data
        xx, yy = np.meshgrid(np.arange(200), np.arange(200))
        pixel_center = PixCoord(x=100, y=75)
        pixel_region = RectanglePixelRegion(center=pixel_center, width=129, height=90)
        pixel_data = PixCoord(x=xx, y=yy)
        mask = pixel_region.contains(pixel_data)
        data = np.where(mask, data, np.nan)
    # Rename
    fits.writeto(path_revised, data, overwrite=True)
    data_revised, hdr_revised = fits.getdata(path_revised, 0, header=True)
    data_revised *= 1e17  # Rescale the data by 1e17
    if type == 'GasMap':
        data_revised /= 1e17  # no rescale for gas map

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
    fits.writeto(path_revised, data_revised, hdr_revised, overwrite=True)


def APLpyStyle(gc, type=None):
    gc.recenter(40.1344150, -18.8656933, width=30 / 3600, height=30 / 3600)
    gc.show_markers(ra_qso_muse, dec_qso_muse, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=600, zorder=100)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)
    if type == 'NarrowBand':
        gc.colorbar.set_ticks([1, 5, 10])
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        gc.colorbar.set_ticks([-200, -100, 0, 100, 200])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
    elif type == 'GasMap_sigma':
        gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
    else:
        gc.colorbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        gc.colorbar.set_axis_label_text(r'$\rm log([O \, III]/[O \, II])$')

    # Scale bar
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
    # xw, yw = gc.pixel2world(195, 140)  # original figure
    # xw, yw = gc.pixel2world(196, 105)
    xw, yw = 40.1302360960119, -18.863967747328896
    gc.show_arrows(xw, yw, -0.000035 * yw, 0, color='k')
    gc.show_arrows(xw, yw, 0, -0.000035 * yw, color='k')
    gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
    gc.add_label(0.88, 0.70, r'E', size=20, relative=True)


# Load galxies infomation
row_final, ID_final, name_final, z_final, ra_final, dec_final = ReturnGalLabel(sort_row=False, mode='initial')
ID_sep_final = ReturnGalLabel(sort_row=True, mode='final')[6]
col_ID = np.arange(len(row_final), dtype=int)
select_array = np.sort(np.array([1, 2, 3, 4, 5, 6, 7, 8, 13, 18, 20, 22]))
select_gal = np.in1d(ID_sep_final, select_array)
ID_sep_final = ID_sep_final[select_gal]
row_final = row_final[select_gal]
z_final = z_final[select_gal]
ra_final = ra_final[select_gal]
dec_final = dec_final[select_gal]

# Calculate the offset between MUSE and HST
z_qso = 0.6282144177077355
OII_air_1 = 3726.032
OII_air_2 = 3728.815
OIII_air = 5006.843
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814

# Plot the data
# Read region file
path_region = path_data + 'regions/gas_list_revised.reg'
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

#
path_gas_label = path_data + 'regions/gas_label_list.reg'
regions_label = Regions.read(path_gas_label, format='ds9')
path_gal_label = path_data + 'regions/galaxy_label_zoom_list.reg'
gal_labels = Regions.read(path_gal_label, format='ds9')


def MakeNarrowBands(gal=False, region=False, video=False, band='OII'):
    # Make movie OII
    if band == 'OII':
        path_cube = path_data + 'cube_narrow/CUBE_OII_line_offset.fits'
        wave_center = OII_air_2
    elif band == 'OIII':
        path_cube = path_data + 'cube_narrow/CUBE_OIII_5008_line_offset.fits'
        wave_center = OIII_air
    cube = Cube(path_cube)
    for i in range(6):
        fig = plt.figure(figsize=(8, 8), dpi=300)

        # Split by velocity
        dv_i, dv_f = -500 + 200 * i, -500 + 200 * (i + 1)
        wave_i = wave_center * (1 + z_qso) * (dv_i / 3e5 + 1)
        wave_f = wave_center * (1 + z_qso) * (dv_f / 3e5 + 1)
        wave_i_vac, wave_f_vac = pyasl.airtovac2(wave_i), pyasl.airtovac2(wave_f)

        # Slice the cube
        sub_cube = cube.select_lambda(wave_i, wave_f)
        sub_cube = sub_cube.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2
        path_image_make_NB = band + '_' + str(dv_i) + '_' + str(dv_f)
        sub_cube.write(path_data + 'image_MakeMovie/' + path_image_make_NB + '.fits')
        ConvertFits(type='NarrowBand', mask=False, filename=path_image_make_NB, smooth=True)
        ConvertFits(type='NarrowBand', mask=False, filename=path_image_make_NB, smooth=False, contour=True)
        path_subcube = path_data + 'image_MakeMovie/' + path_image_make_NB + '_revised.fits'
        path_contour = path_data + 'image_MakeMovie/' + path_image_make_NB + '_revised.fits'

        # Plot the data
        gc = aplpy.FITSFigure(path_subcube, figure=fig, north=True)
        gc.show_colorscale(vmin=0, vmid=0.2, vmax=15.0, cmap=plt.get_cmap('Blues'), stretch='arcsinh')
        # gc.show_colorscale(vmin=-0.2, vmid=0.2, vmax=7.0, cmap=newcmp, stretch='linear')
        if gal:
            gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none',
                            edgecolors='k', linewidths=0.8, s=100)

        # Plot regions
        if region:
            gc.show_circles(ra_array, dec_array, radius_array / 3600, edgecolors='k', linestyles='--', linewidths=1,
                            alpha=0.3)
            for j in range(len(ra_array)):
                x = regions_label[j].center.ra.degree
                y = regions_label[j].center.dec.degree
                gc.add_label(x, y, text_array[j], size=20)
        else:
            gc.show_contour(path_contour, levels=[0.08, 0.3], kernel='gauss', colors='k', linewidths=0.8, smooth=3)

        APLpyStyle(gc, type='NarrowBand')
        gc.add_label(0.82, 0.91, r'$\mathrm{\lambda = \,}$' + str("{0:.0f}".format(wave_i_vac)) + ' to '
                     + str("{0:.0f}".format(wave_f_vac)) + r'$\mathrm{\AA}$', size=20, relative=True)
        gc.add_label(0.76, 0.85, r'$\mathrm{\Delta} v \approx \,$' + str("{0:.0f}".format(dv_i)) + ' to '
                     + str("{0:.0f}".format(dv_f)) + r'$\mathrm{\, km \, s^{-1}}$', size=20, relative=True)
        figname = '/Users/lzq/Dropbox/Data/CGM_plots/NB_movie/'
        if band == 'OII':
            gc.add_label(0.87, 0.97, r'MUSE [O II]', size=20, relative=True)
            figname += 'image_OII_' + str("{0:.0f}".format(dv_i)) + '_' + str("{0:.0f}".format(dv_f))
        elif band == 'OIII':
            gc.add_label(0.87, 0.97, r'MUSE [O III]', size=20, relative=True)
            figname += 'image_OIII_' + str("{0:.0f}".format(dv_i)) + '_' + str("{0:.0f}".format(dv_f))
        if region:
            figname += '_region'
        fig.savefig(figname + '.png', bbox_inches='tight')
    if video:
        if band == 'OII':
            os.system('convert -delay 75 ~/dropbox/Data/CGM_plots/NB_movie/image_OII_*.png '
                      '~/dropbox/Data/CGM_plots/NB_movie/OII_movie.gif')
        elif band == 'OII':
            os.system('convert -delay 75 ~/dropbox/Data/CGM_plots/NB_movie/image_OIII_*.png '
                      '~/dropbox/Data/CGM_plots/NB_movie/OIII_movie.gif')


# Make field image
def MakeFieldImage(label_gal=False):
    # Load the contour
    ConvertFits(type='FieldImage', mask=True, filename='image_OII_line_SB_offset', smooth=True)
    ConvertFits(type='FieldImage', mask=True, filename='image_OIII_5008_line_SB_offset', smooth=True)

    #
    path_hb = path_data + 'raw_data/HE0238-1904_drc_offset.fits'
    path_OII_SB = path_data + 'image_MakeMovie/OII_-100_100_contour_revised.fits'
    path_OIII_SB = path_data + 'image_MakeMovie/OIII_-100_100_contour_revised.fits'

    # Plot
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
    gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    # gc.show_colorscale(cmap='Greys', vmin=0, vmax=4.897e-2)
    gc.show_contour(path_OII_SB, levels=[0.08, 0.3], kernel='gauss', colors='blue', linewidths=0.8, smooth=3)
    gc.show_contour(path_OIII_SB, levels=[0.08, 0.3], kernel='gauss', colors='red', linewidths=0.8, smooth=3)
    gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=1.5, s=330)
    if label_gal:
        gc.show_arrows(40.1371817, -18.8663804, 40.1366338 - 40.1371817, -18.8656749 + 18.8663804, color='k')
        gc.show_arrows(40.1366225, -18.8668026, 40.1364435 - 40.1366225, -18.8660348 + 18.8668026, color='k')
        for i, ix in enumerate(col_ID[select_gal]):
            x = gal_labels[ix].center.ra.degree
            y = gal_labels[ix].center.dec.degree
            text = 'G' + str(ID_sep_final[i])
            gc.add_label(x, y, text, size=20)

    APLpyStyle(gc, type='FieldImage')
    gc.add_label(0.85, 0.91, r'MUSE [O II]', color='blue', size=20, relative=True)
    gc.add_label(0.85, 0.86, r'MUSE [O III]', color='red', size=20, relative=True)
    gc.add_label(0.85, 0.96, r'$\mathrm{ACS+F814W}$', color='k', size=20, relative=True)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Field_Image.png', bbox_inches='tight')


# Plot Kinematics maps
def MakeGasMap(line='OIII', method='pixel', method_spe=None, check=False, test=True, snr_thr=3, v_thr=300):
    global path_data
    # Load OIII
    path_data_line = path_data + 'fit_' + line
    if test is True:
        path_fit_info = path_data_line + '/fit' + line + '_info_test.fits'
        path_fit_info_err = path_data_line + '/fit' + line + '_info_err_test.fits'
    else:
        path_fit_info = path_data_line + '/fit' + line + '_info_' + method + '_' + method_spe + '.fits'
        path_fit_info_err = path_data_line + '/fit' + line + '_info_err_' + method + '_' + method_spe + '.fits'

    fit_info = fits.getdata(path_fit_info, 0, ignore_missing_end=True)
    fit_info_err = fits.getdata(path_fit_info_err, 0, ignore_missing_end=True)

    if line == 'OOHbeta':
        [z_fit, r_fit, fit_success, sigma_fit, flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII5008, a_fit_OII, a_fit_Hbeta,
         a_fit_OIII4960, a_fit_OIII5008, b_fit_OII, b_fit_Hbeta, b_fit_OIII4960, b_fit_OIII5008] = fit_info
        [dz_fit, dr_fit, dsigma_fit, dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII5008, da_fit_OII, da_fit_Hbeta,
         da_fit_OIII4960, da_fit_OIII5008, db_fit_OII, db_fit_Hbeta, db_fit_OIII4960, db_fit_OIII5008] = fit_info_err
        print(flux_fit_Hbeta)
    elif line == 'OII':
        [z_fit, sigma_fit, flux_fit, fit_success, r_fit, a_fit, b_fit] = fit_info
        [dz_fit, dsigma_fit, dflux_fit, dr_fit, da_fit, db_fit] = fit_info_err
    else:
        # Load data
        [z_fit, sigma_fit, flux_fit, fit_success, a_fit, b_fit] = fit_info
        [dz_fit, dsigma_fit, dflux_fit, da_fit, db_fit] = fit_info_err

    v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)
    v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)
    log_OIII_OII_fit = np.log10(flux_fit_OIII5008 / flux_fit_OII)

    # Check consistency
    if check:
        plt.figure(figsize=(8, 8), dpi=300)
        plt.imshow(v_fit, cmap='coolwarm', vmin=-300, vmax=300, origin='lower')
        plt.show()

    if line == 'OOHbeta':
        if method_spe == '1.0_zapped_NoHbeta':
            flux_stack = np.stack((flux_fit_OII, flux_fit_OIII5008), axis=0)
            dflux_stack = np.stack((dflux_fit_OII, dflux_fit_OIII5008), axis=0)
        else:
            flux_stack = np.stack((flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII5008), axis=0)
            dflux_stack = np.stack((dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII5008), axis=0)
        print(len(flux_stack[np.isnan(flux_stack)]))
        print(len(dflux_stack[np.isnan(dflux_stack)]))

        v_fit = np.where(fit_success == 1, v_fit, np.nan)
        sigma_fit = np.where(fit_success == 1, sigma_fit, np.nan)
        log_OIII_OII_fit = np.where(fit_success == 1, log_OIII_OII_fit, np.nan)

        #
        fit_max = np.nanmax(flux_stack / dflux_stack, axis=0)
        log_OIII_OII_fit = np.where((fit_max > snr_thr), log_OIII_OII_fit, np.nan)
        v_fit = np.where((fit_max > snr_thr), v_fit, np.nan)
        sigma_fit = np.where((fit_max > snr_thr), sigma_fit, np.nan)
        log_OIII_OII_fit = np.where((v_fit <= v_thr), log_OIII_OII_fit, np.nan)
        sigma_fit = np.where((v_fit <= v_thr), sigma_fit, np.nan)
        v_fit = np.where((v_fit <= v_thr), v_fit, np.nan)

    else:
        # Final data
        v_fit = np.where((flux_fit / dflux_fit > snr_thr), v_fit, np.nan)
        sigma_fit = np.where((flux_fit / dflux_fit > snr_thr), sigma_fit, np.nan)
        sigma_fit = np.where((v_fit < v_thr), sigma_fit, np.nan)
        v_fit = np.where((v_fit < v_thr), v_fit, np.nan)
        log_OIII_OII_fit = np.log10(flux_fit_OIII5008 / flux_fit_OII)

    ConvertFits(table=v_fit, type='GasMap', mask=True, smooth=False, filename='image_' + line + '_fitline')

    # Plot velocity map
    path_OII_SB = path_data + 'image_MakeMovie/OII_-100_100_contour_revised.fits'
    path_OIII_SB = path_data + 'image_MakeMovie/OIII_-100_100_contour_revised.fits'
    fig = plt.figure(figsize=(8, 8), dpi=300)
    path_dv = path_data + 'image_plot/image_' + line + '_fitline_revised.fits'
    gc = aplpy.FITSFigure(path_dv, figure=fig, north=True)
    gc.show_colorscale(vmin=-300, vmax=300, cmap='coolwarm')
    # gc.show_contour(path_OII_SB, levels=[-np.inf, 0.1], filled=False, kernel='gauss', colors='black', linewidths=0.8, smooth=3)
    # gc.show_contour(path_OIII_SB, levels=[0.1], filled=False, kernel='gauss', colors='red', linewidths=0.8, smooth=3)
    gc.show_markers(ra_final, dec_final, facecolor='white', marker='o', c='white',
                    edgecolors='none', linewidths=0.8, s=100)
    gc.show_markers(ra_final, dec_final, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra_final, dec_final, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=-300, vmax=300,
                    cmap='coolwarm')
    APLpyStyle(gc, type='GasMap')
    # Label
    if line == 'OIII':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{[O \, III]}} - v_{\mathrm{qso}}$', size=20, relative=True)
    elif line == 'OII':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{[O \, II]}} - v_{\mathrm{qso}}$', size=20, relative=True)
    elif line == 'OOHbeta':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{lines}} - v_{\mathrm{qso}}$', size=20, relative=True)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + line + '_dv_map_' + method + '_' + method_spe + '.png',
                bbox_inches='tight')

    # Plot sigma map
    ConvertFits(table=sigma_fit, type='GasMap', mask=True, smooth=False, filename='image_' + line + '_fitline')
    fig = plt.figure(figsize=(8, 8), dpi=300)
    path_sigma_v = path_data + 'image_plot/image_' + line + '_fitline_revised.fits'
    gc = aplpy.FITSFigure(path_sigma_v, figure=fig, north=True)
    gc.show_colorscale(vmin=0, vmax=200, cmap=sequential_s.Acton_6.mpl_colormap)
    APLpyStyle(gc, type='GasMap_sigma')
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + line + '_sigma_v_map_' + method + '_' + method_spe
                + '.png', bbox_inches='tight')

    # Plot line ratio map
    ConvertFits(table=log_OIII_OII_fit, type='GasMap', mask=True, smooth=False, filename='image_' + line + '_fitline')

    fig = plt.figure(figsize=(8, 8), dpi=300)
    path_sigma_v = path_data + 'image_plot/image_' + line + '_fitline_revised.fits'
    gc = aplpy.FITSFigure(path_sigma_v, figure=fig, north=True)
    gc.show_colorscale(vmin=-1, vmax=2, cmap=sequential_s.Buda_20.mpl_colormap)
    APLpyStyle(gc, type='else')
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + line + '_logOIII_OII_map_' + method + '_' + method_spe
                + '.png', bbox_inches='tight')




#
# MakeNarrowBands(region=False)
# MakeNarrowBands(region=True)
# MakeNarrowBands(region=False, band='OIII')
# MakeNarrowBands(region=True, band='OIII')
MakeFieldImage(label_gal=True)
# MakeGasMap(line='OOHbeta', method='aperture', method_spe='1.0_zapped', test=False, snr_thr=8, v_thr=np.inf, check=False)