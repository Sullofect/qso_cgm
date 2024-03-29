import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from astropy import units as u
from muse_compare_z import compare_z
from matplotlib.colors import ListedColormap
import palettable.colorbrewer.sequential as sequential
import palettable.scientific.sequential as sequential_s
import palettable.cmocean.sequential as sequential_c
import palettable.colorbrewer.diverging as diverging
from regions import RectangleSkyRegion, PixCoord, RectanglePixelRegion
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10

# Cmap
Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)


#
def ConvertFits(filename=None, table=None, sigma_v=False):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_plot', filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
    if sigma_v is True:
        fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_plot/' + filename + '_sigma_v_revised.fits',
                     table, overwrite=True)
        data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_plot/' + filename + '_sigma_v_revised.fits',
                                   0, header=True)
    elif sigma_v is False:
        fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_plot/' + filename + '_revised.fits', table, overwrite=True)
        data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/image_plot/' + filename + '_revised.fits',
                                   0, header=True)
    hdr1['BITPIX'], hdr1['NAXIS'], hdr1['NAXIS1'], hdr1['NAXIS2'] = hdr['BITPIX'], hdr['NAXIS'], \
                                                                    hdr['NAXIS1'], hdr['NAXIS2']
    hdr1['CRPIX1'], hdr1['CRPIX2'], hdr1['CTYPE1'], hdr1['CTYPE2'] = hdr['CRPIX1'], hdr['CRPIX2'], \
                                                                     hdr['CTYPE1'], hdr['CTYPE2']
    hdr1['CRVAL1'], hdr1['CRVAL2'], hdr1['LONPOLE'], hdr1['LATPOLE'] = hdr['CRVAL1'], hdr['CRVAL2'], \
                                                                       hdr['LONPOLE'], hdr['LATPOLE']
    hdr1['CSYER1'], hdr1['CSYER2'], hdr1['MJDREF'], hdr1['RADESYS'] = hdr['CSYER1'], hdr['CSYER2'], \
                                                                      hdr['MJDREF'], hdr['RADESYS']
    hdr1['CD1_1'], hdr1['CD1_2'], hdr1['CD2_1'], hdr1['CD2_2'] = hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2']
    if sigma_v is True:
        fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_plot/' + filename + '_sigma_v_revised.fits', data1, hdr1,
                     overwrite=True)
    elif sigma_v is False:
        fits.writeto('/Users/lzq/Dropbox/Data/CGM/image_plot/' + filename + '_revised.fits', data1, hdr1,
                     overwrite=True)


#
def PlotMap(line='OIII', method='pixel', method_spe=None, check=False, test=True, snr_thr=3, v_thr=300, row=None, z=None,
            ra=None, dec=None):
    # Load OIII
    if test is True:
        path_fit_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit_' + line,
                                     'fit' + line + '_info_test.fits')
        path_fit_info_err = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit_' + line,
                                         'fit' + line + '_info_err_test.fits')
    else:
        path_fit_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit_' + line,
                                     'fit' + line + '_info_' + method + '_' + method_spe + '.fits')
        path_fit_info_err = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit_' + line,
                                         'fit' + line + '_info_err_' + method + '_' + method_spe + '.fits')
    fit_info = fits.getdata(path_fit_info, 0, ignore_missing_end=True)
    fit_info_err = fits.getdata(path_fit_info_err, 0, ignore_missing_end=True)

    if line == 'OOHbeta':
        # [z_fit, r_fit, sigma_fit_OII, sigma_fit_Hbeta, sigma_fit_OIII4960, sigma_fit_OIII5008,
        #  flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII4960, flux_fit_OIII5008, a_fit_OII, a_fit_Hbeta,
        #  a_fit_OIII4960, a_fit_OIII5008, b_fit_OII, b_fit_Hbeta, b_fit_OIII4960, b_fit_OIII5008] = fit_info
        # [dz_fit, dr_fit, dsigma_fit_OII, dsigma_fit_Hbeta, dsigma_fit_OIII4960, dsigma_fit_OIII5008,
        #  dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII4960, dflux_fit_OIII5008, da_fit_OII, da_fit_Hbeta,
        #  da_fit_OIII4960, da_fit_OIII5008, db_fit_OII, db_fit_Hbeta, db_fit_OIII4960, db_fit_OIII5008] = fit_info_err

        [z_fit, r_fit, fit_success, sigma_fit, flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII5008, a_fit_OII, a_fit_Hbeta,
         a_fit_OIII4960, a_fit_OIII5008, b_fit_OII, b_fit_Hbeta, b_fit_OIII4960, b_fit_OIII5008] = fit_info
        [dz_fit, dr_fit, dsigma_fit, dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII5008, da_fit_OII, da_fit_Hbeta,
         da_fit_OIII4960, da_fit_OIII5008, db_fit_OII, db_fit_Hbeta, db_fit_OIII4960, db_fit_OIII5008] = fit_info_err
    elif line == 'OII':
        [z_fit, sigma_fit, flux_fit, fit_success, r_fit, a_fit, b_fit] = fit_info
        [dz_fit, dsigma_fit, dflux_fit, dr_fit, da_fit, db_fit] = fit_info_err
    else:
        # Load data
        [z_fit, sigma_fit, flux_fit, fit_success, a_fit, b_fit] = fit_info
        [dz_fit, dsigma_fit, dflux_fit, da_fit, db_fit] = fit_info_err

    z_qso = 0.6282144177077355
    v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)
    v_gal = 3e5 * (z - z_qso) / (1 + z_qso)

    # Select region
    xx, yy = np.meshgrid(np.arange(200), np.arange(200))
    pixel_center = PixCoord(x=100, y=75)
    pixel_region = RectanglePixelRegion(center=pixel_center, width=100, height=100)
    pixel_data = PixCoord(x=xx, y=yy)
    mask = pixel_region.contains(pixel_data)
    v_fit = np.where(mask, v_fit, np.nan)
    sigma_fit = np.where(mask, sigma_fit, np.nan)

    # Check consistency
    if check is True:
        plt.figure(figsize=(8, 8), dpi=300)
        plt.imshow(v_fit, cmap='coolwarm', vmin=-300, vmax=300, origin='lower')
        plt.show()

    if line == 'OOHbeta':
        flux_stack = np.stack((flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII5008), axis=0)
        dflux_stack = np.stack((dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII5008), axis=0)
        fit_max = np.amax(flux_stack / dflux_stack, axis=0)
        v_fit = np.where((fit_max > snr_thr), v_fit, np.nan)
        sigma_fit = np.where((fit_max > snr_thr), sigma_fit, np.nan)
        sigma_fit = np.where((v_fit < v_thr), sigma_fit, np.nan)
        v_fit = np.where((v_fit < v_thr), v_fit, np.nan)
        ConvertFits(filename='image_' + line + '_fitline', table=v_fit)
        ConvertFits(filename='image_' + line + '_fitline', table=sigma_fit, sigma_v=True)

    else:
        # Final data
        v_fit = np.where((flux_fit / dflux_fit > snr_thr), v_fit, np.nan)
        sigma_fit = np.where((flux_fit / dflux_fit > snr_thr), sigma_fit, np.nan)
        sigma_fit = np.where((v_fit < v_thr), sigma_fit, np.nan)
        v_fit = np.where((v_fit < v_thr), v_fit, np.nan)
        ConvertFits(filename='image_' + line + '_fitline', table=v_fit)
        ConvertFits(filename='image_' + line + '_fitline', table=sigma_fit, sigma_v=True)

    # Plot velocity map
    fig = plt.figure(figsize=(8, 8), dpi=300)
    path_dv = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_plot',
                           'image_' + line + '_fitline_revised.fits')
    gc = aplpy.FITSFigure(path_dv, figure=fig, north=True)
    gc.set_system_latex(True)
    gc.show_colorscale(vmin=-300, vmax=300, cmap='coolwarm')
    gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='lightgrey',
                    edgecolors='k', linewidths=0.5, s=400)
    gc.show_markers(ra, dec, facecolor='white', marker='o', c='white', edgecolors='none', linewidths=0.8, s=100)
    gc.show_markers(ra, dec, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra, dec, marker='o', c=v_gal, linewidths=0.5, s=40, vmin=-300, vmax=300, cmap='coolwarm')

    # colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_ticks([-200, -100, 0, 100, 200])
    gc.colorbar.set_pad(0.)
    gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)

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
    if line == 'OIII':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{[O \, III]}} - v_{\mathrm{qso}}$', size=15, relative=True)
    elif line == 'OII':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{[O \, II]}} - v_{\mathrm{qso}}$', size=15, relative=True)
    elif line == 'OOHbeta':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{lines}} - v_{\mathrm{qso}}$', size=20, relative=True)
    xw, yw = gc.pixel2world(195, 140)
    gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
    gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
    gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
    gc.add_label(0.88, 0.70, r'E', size=20, relative=True)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + line + '_dv_map_' + method + '_' + method_spe + '.png',
                bbox_inches='tight')

    # Plot sigma map
    fig = plt.figure(figsize=(8, 8), dpi=300)
    path_sigma_v = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_plot', 'image_' + line
                                + '_fitline_sigma_v_revised.fits')
    gc = aplpy.FITSFigure(path_sigma_v, figure=fig, north=True)
    gc.set_system_latex(True)
    gc.show_colorscale(vmin=0, vmax=200, cmap=sequential_s.Acton_6.mpl_colormap)
    gc.show_markers(40.13564948691202, -18.864301804042814, facecolors='none', marker='*', c='lightgrey',
                    edgecolors='k', linewidths=0.5, s=400)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.)
    gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
    gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)

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
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + line + '_sigma_v_map_' + method + '_' + method_spe
                + '.png', bbox_inches='tight')


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
# ra_qso_hst, dec_qso_hst = 40.1359, -18.8643

# run
# PlotMap(line='OII', check=False, snr_thr=2.5, row=row_final, z=z_final, ra=ra_final, dec=dec_final)
# PlotMap(line='OIII', snr_thr=3, row=row_final, z=z_final, ra=ra_final, dec=dec_final)
PlotMap(line='OOHbeta', method='aperture', method_spe='1.0', test=False, snr_thr=5, v_thr=np.inf, check=False,
        row=row_final, z=z_final, ra=ra_final, dec=dec_final)