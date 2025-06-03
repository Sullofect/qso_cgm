import os
import aplpy
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Constants
c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239


def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None):
    # 3C57 paper
    # gc.recenter(ra_qso, dec_qso, width=15 / 3600, height=15 / 3600)

    # All
    gc.recenter(ra_qso, dec_qso, width=30 / 3600, height=30 / 3600)

    # For JWST proposal
    # gc.recenter(ra_qso, dec_qso, width=20 / 3600, height=20 / 3600)
    # if cubename == 'Q1354+048':
    #     gc.recenter(ra_qso - 0.0005, dec_qso, width=30 / 3600, height=30 / 3600)

    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=1000, zorder=100)
    gc.set_system_latex(True)

    # calculate angular diameter distance
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
    arcsec_str = '{:.0f}'.format(((50 / d_A_kpc) * 206265))
    gc.add_scalebar(length=float(arcsec_str) * u.arcsecond)
    gc.scalebar.set_corner('top left')
    gc.scalebar.set_label(arcsec_str + r"$'' \approx 50 \mathrm{\; kpc}$")
    gc.scalebar.set_font_size(30)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=30)
    gc.colorbar.set_axis_label_font(size=30)
    # gc.colorbar.set_font(size=20)
    # gc.colorbar.set_axis_label_font(size=20)
    if type == 'HST':
        gc.colorbar.hide()
    elif type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=30)
        gc.colorbar.set_axis_label_text(r'$\mathrm{SB \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        gc.colorbar.set_axis_label_font(size=30)
    elif type == 'GasMap':
        # gc.colorbar.set_ticks([-300, -150, 0, 150, 300])
        gc.colorbar.set_ticks([-300, -200, -100, 0, 100, 200, 300])
        # gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        # gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v_{50} / v_{max} \mathrm{\; [km \, s^{-1}]}$')
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v_{50} / v_{\rm max}$')
        gc.colorbar.hide()
    elif type == 'GasMap_sigma':
        gc.colorbar.set_ticks([0, 150, 300, 450, 600, 750])
        # gc.colorbar.set_axis_label_text(r'$\rm \sigma_{80} \mathrm{\; [km \, s^{-1}]}$')
        gc.colorbar.set_axis_label_text(r'$\rm \sigma_{80} / \sigma_{\rm max}$')
        gc.colorbar.hide()

    # Cubename and redshift
    if cubename == 'LBQS1435-0134':
        cubename = 'Q1435-0134'
    elif cubename == 'TEX0206-048':
        cubename = 'TXS0206-048'
    elif cubename == 'PB6291':
        cubename = 'Q0107-025'

    if type == 'HST':
        if cubename == '3C57':
            gc.add_label(0.98, 0.95, r'$\rm 3C\,57$', size=35, relative=True, horizontalalignment='right')
        else:
            try:
                split = '-'
                split_lax = r'$-$'
                name_1, name_2 = cubename.split(split)
            except ValueError:
                split = '+'
                split_lax = r'$+$'
                name_1, name_2 = cubename.split(split)


            for i in range(len(cubename)):
                if not cubename[i].isalpha():
                    break

            if i == 1:
                cubename_label = name_1 + split_lax + name_2
            else:
                cubename_label = cubename[:i] + r'$\,$' + name_1[i:] + split_lax + name_2
            gc.add_label(0.98, 0.95, cubename_label, size=35, relative=True, horizontalalignment='right')
        gc.add_label(0.98, 0.87, r'$z=$' + ' {:.4f}'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    # Hide
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.ticks.set_length(30)


def MakeV50W80(cubename=None, v_max=300, sigma_max=300, contour_level_OII=0.2, contour_level_OIII=0.2,
               nums_seg_OII=[], rmbkgResidue=False, nums_seg_OIII=[], HSTcentroid=False):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Load data
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    UseDataSeg=(1.5, 'gauss', None, None)
    line = 'OII+OIII'
    line_OII, line_OIII = 'OII', 'OIII'
    figurename_V50 = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_{}_{}_{}_{}_{}_{}_{}.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    figurename_S80 = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_{}_{}_{}_{}_{}_{}_{}.png'. \
        format(cubename, line, True, 3728, *UseDataSeg)
    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        v_gal = data_gal['v']
        if HSTcentroid:
            ra_gal, dec_gal, type = data_gal['ra_HST'], data_gal['dec_HST'], data_gal['type']
        else:
            ra_gal, dec_gal, type = data_gal['ra'], data_gal['dec'], data_gal['type']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal, ra_hst, dec_hst = [], [], [], [], []

    # V50, S80
    path_v50 = '../../MUSEQuBES+CUBS/fit_kin/{}_V50.fits'.format(cubename)
    path_w80 = '../../MUSEQuBES+CUBS/fit_kin/{}_W80.fits'.format(cubename)
    hdul_v50 = fits.open(path_v50)
    hdul_s80 = fits.open(path_w80)
    hdul_s80[1].data = hdul_s80[1].data / 2.563
    hdr = hdul_v50[1].header

    # Replace coordinate to Gaia
    path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}_WCS_subcube.fits'.format(cubename)
    if os.path.exists(path_sub_white_gaia):
        print('correcting to gaia')
        hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
        w = WCS(hdr_sub_gaia, naxis=2)

        hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
        hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
        hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
        hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
        hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
        hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
        hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
        hdr['CD2_2'] = hdr_sub_gaia['CD2_2']
    else:
        w = WCS(hdr, naxis=2)
        print('No gaia correction info')

    center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
    c2 = w.world_to_pixel(center_qso)

    #
    path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
    path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)

    # Plot the velocity field
    x, y = np.meshgrid(np.arange(hdul_v50[1].data.shape[0]), np.arange(hdul_v50[1].data.shape[1]))
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)

    # Mask the center
    circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
    center_mask_flatten = ~circle.contains(pixcoord)
    center_mask = center_mask_flatten.reshape(hdul_v50[1].data.shape)

    hdul_v50[1].data = np.where(center_mask, hdul_v50[1].data, np.nan)
    hdul_s80[1].data = np.where(center_mask, hdul_s80[1].data, np.nan)

    hdul_v50[1].header = hdr
    hdul_s80[1].header = hdr
    hdul_v50.writeto(path_v50_plot, overwrite=True)
    hdul_s80.writeto(path_s80_plot, overwrite=True)

    # OII SBs
    if cubename == 'TEX0206-048':
        str_zap = '_zapped'
    else:
        str_zap = ''

    path_SB_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    path_SB_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OIII, *UseSeg)
    path_SB_OII_kin = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    path_SB_OIII_kin = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OIII, *UseSeg)
    path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OIII, *UseSeg)
    figurename_SB_OII = '../../MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    figurename_SB_OIII = '../../MUSEQuBES+CUBS/fit_kin/{}{}_SB_{}_{}_{}_{}_{}.png'. \
        format(cubename, str_zap, line_OIII, *UseSeg)

    # Special cases due to sky line
    if cubename == 'PKS0552-640':
        path_SB_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
    elif cubename == 'HE0226-4110':
        path_SB_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)
        path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)

    # Load segmentation
    seg_OII = fits.open(path_3Dseg_OII)[1].data
    seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)

    # Fix v50 and s80
    hdul_v50[1].data = np.where(seg_OII_mask != -1, hdul_v50[1].data, np.nan)
    hdul_s80[1].data = np.where(seg_OII_mask != -1, hdul_s80[1].data, np.nan)

    # OII SB
    path_OII_contour = path_SB_OII_kin
    SB_OII = fits.open(path_SB_OII)[1].data
    bkgrd_OII = np.where(seg_OII_mask == 0, SB_OII, np.nan)
    np.random.seed(1)
    bkgrd_OII_random = np.random.choice(bkgrd_OII.flatten()[~np.isnan(bkgrd_OII.flatten())],
                                        bkgrd_OII.shape, replace=True).reshape(bkgrd_OII.shape)
    SB_OII = np.where(seg_OII_mask != -1, SB_OII, bkgrd_OII_random)
    if rmbkgResidue:
        path_SB_OII_kin_2lev = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}_2lev.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)
        bkgrd_OII_2lev = np.where((bkgrd_OII <= 1.8 * contour_level_OII), bkgrd_OII, bkgrd_OII_random)
        SB_OII_2lev = np.where(seg_OII_mask != 0, SB_OII, bkgrd_OII_2lev)
        hdul_SB_OII_kin_2lev = fits.ImageHDU(SB_OII_2lev, header=hdr)
        hdul_SB_OII_kin_2lev.writeto(path_SB_OII_kin_2lev, overwrite=True)
        path_OII_contour = path_SB_OII_kin_2lev
    hdul_SB_OII_kin = fits.ImageHDU(SB_OII, header=hdr)
    hdul_SB_OII_kin.writeto(path_SB_OII_kin, overwrite=True)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_SB_OII_kin, figure=fig, hdu=1)
    gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
    gc.show_contour(path_OII_contour, levels=[contour_level_OII], colors='black', linewidths=2,
                    smooth=5, kernel='box', hdu=1)
    APLpyStyle(gc, type='NarrowBand', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
    if cubename == 'HE0435-5304':
        gc.add_label(0.03, 0.08, r'$\rm [O\,II]$', color='black', size=40, relative=True, horizontalalignment='left')
    if cubename != 'PKS0232-04' and cubename != 'TEX0206-048' and cubename != 'PKS0355-483' \
            and cubename != 'HE0246-4101' and cubename != '3C57':
        gc.colorbar.hide()
    # gc.add_label(0.08, 0.08, '(b)', color='k', size=40, relative=True)

    # For JWST proposal
    # if cubename == 'PKS0232-04':
    #     gc.show_rectangles(38.7804671, -4.0357864, 3 / 3600, 3 / 3600, edgecolor='black', linewidth=2)
    # elif cubename == 'PKS2242-498':
    #     gc.show_rectangles(341.2493718, -49.5295075, 3 / 3600, 3 / 3600, edgecolor='black', linewidth=2)
    # elif cubename == 'Q0107-0235':
    #     gc.show_rectangles(17.5556650, -2.3308936, 3 / 3600, 3 / 3600, edgecolor='black', linewidth=2)
    # elif cubename == 'Q1354+048':
    #     gc.show_rectangles(209.3582986, 4.5953355, 3 / 3600, 3 / 3600, edgecolor='black', linewidth=2)
    fig.savefig(figurename_SB_OII, bbox_inches='tight')


    # OIII SB
    if os.path.exists(path_SB_OIII):
        # Load segmentation
        seg_OIII = fits.open(path_3Dseg_OIII)[1].data
        seg_OIII_mask = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, -1)

        # Fix v50 and s80
        hdul_v50[1].data = np.where(seg_OIII_mask != -1, hdul_v50[1].data, np.nan)
        hdul_s80[1].data = np.where(seg_OIII_mask != -1, hdul_s80[1].data, np.nan)

        #
        path_OIII_contour = path_SB_OIII_kin
        SB_OIII = fits.open(path_SB_OIII)[1].data
        bkgrd_OIII = np.where(seg_OIII_mask == 0, SB_OIII, np.nan)
        bkgrd_OIII_random = np.random.choice(bkgrd_OIII.flatten()[~np.isnan(bkgrd_OIII.flatten())],
                                             bkgrd_OIII.shape, replace=True).reshape(bkgrd_OIII.shape)
        SB_OIII = np.where(seg_OIII_mask != -1, SB_OIII, bkgrd_OIII_random)
        if rmbkgResidue:
            path_SB_OIII_kin_2lev = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}_2lev.fits'. \
                format(cubename, str_zap, line_OIII, *UseSeg)
            bkgrd_OIII_2lev = np.where((bkgrd_OIII <= 1.8 * contour_level_OIII), bkgrd_OIII, bkgrd_OIII_random)
            SB_OIII_2lev = np.where(seg_OIII_mask != 0, SB_OIII, bkgrd_OIII_2lev)
            hdul_SB_OIII_kin_2lev = fits.ImageHDU(SB_OIII_2lev, header=hdr)
            hdul_SB_OIII_kin_2lev.writeto(path_SB_OIII_kin_2lev, overwrite=True)
            path_OIII_contour = path_SB_OIII_kin_2lev
        hdul_SB_OIII_kin = fits.ImageHDU(SB_OIII, header=hdr)
        hdul_SB_OIII_kin.writeto(path_SB_OIII_kin, overwrite=True)

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_SB_OIII_kin, figure=fig, hdu=1)
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        gc.show_contour(path_OIII_contour, levels=[contour_level_OIII], colors='black', linewidths=2,
                        smooth=5, kernel='box', hdu=1)
        APLpyStyle(gc, type='NarrowBand', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
        if cubename == 'HE0435-5304':
            gc.add_label(0.03, 0.08, r'$\rm [O\,III]$', color='black', size=40, relative=True, horizontalalignment='left')
        if cubename != 'PKS0232-04' and cubename != 'TEX0206-048' and cubename != 'PKS0355-483' \
                and cubename != 'HE0246-4101' and cubename != '3C57':
            gc.colorbar.hide()
        # gc.add_label(0.08, 0.08, '(c)', color='k', size=40, relative=True)
        fig.savefig(figurename_SB_OIII, bbox_inches='tight')

    # V50 and S80
    hdul_v50[1].header = hdr
    hdul_s80[1].header = hdr
    hdul_v50.writeto(path_v50_plot, overwrite=True)
    hdul_s80.writeto(path_s80_plot, overwrite=True)

    # V50
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_v50_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=-v_max, vmax=v_max, cmap='coolwarm')
    APLpyStyle(gc, type='GasMap', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
    if cubename == 'PKS0232-04' or cubename == 'TEX0206-048' or cubename == 'PKS0355-483' \
            or cubename == 'HE0246-4101' or cubename == '3C57':
        gc.colorbar.set_ticks([-v_max, -v_max / 2, 0, v_max / 2, v_max])
        # gc.colorbar._colorbar.set_ticklabels([r'$-v_{\rm max}$', r'$-\frac{1}{2}v_{\rm max}$',
        #                                       0, r'$\frac{1}{2}v_{\rm max}$', r'$v_{\rm max}$'])
        gc.colorbar._colorbar.set_ticklabels([r'$-1$', r'$-\frac{1}{2}$', 0, r'$\frac{1}{2}$', r'$1$'])
        tick_labels = gc.colorbar._colorbar.ax.get_xticklabels()
        tick_labels[0].set_ha('left')

        # For HST proposal
        # gc.colorbar.set_ticks([-600, -400, -200, 0, 200, 400, 600])


    gc.add_label(0.05, 0.08, '[{}, {}]'.format(-v_max, v_max), size=30, relative=True, horizontalalignment='left')
    ra_emi, dec_emi, v_emi = ra_gal[type == 'emi'], dec_gal[type == 'emi'], v_gal[type == 'emi']
    ra_abs, dec_abs, v_abs = ra_gal[type != 'emi'], dec_gal[type != 'emi'], v_gal[type != 'emi']
    gc.show_markers(ra_emi, dec_emi, facecolor='white', marker='D', c='white', edgecolors='none', linewidths=0.8, s=80)
    gc.show_markers(ra_emi, dec_emi, facecolor='none', marker='D', c='none', edgecolors='k', linewidths=0.8, s=80)
    gc.show_markers(ra_emi, dec_emi, marker='D', c=v_emi, linewidths=0.5, s=30, vmin=-v_max, vmax=v_max, cmap='coolwarm')
    gc.show_markers(ra_abs, dec_abs, facecolor='white', marker='o', c='white', edgecolors='none', linewidths=0.8, s=100)
    gc.show_markers(ra_abs, dec_abs, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra_abs, dec_abs, marker='o', c=v_abs, linewidths=0.5, s=40, vmin=-v_max, vmax=v_max, cmap='coolwarm')
    fig.savefig(figurename_V50, bbox_inches='tight')

    # S80 map converted from W80 to sigma
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_s80_plot, figure=fig, hdu=1)
    gc.show_colorscale(vmin=0, vmax=sigma_max, cmap=Dense_20_r.mpl_colormap)
    gc.add_label(0.05, 0.08, '[{}, {}]'.format(0, sigma_max), size=30, relative=True, horizontalalignment='left')
    APLpyStyle(gc, type='GasMap_sigma', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
    if cubename == 'PKS0232-04' or cubename == 'TEX0206-048' or cubename == 'PKS0355-483' \
            or cubename == 'HE0246-4101' or cubename == '3C57':
        gc.colorbar.set_ticks([0, sigma_max / 4, sigma_max / 2,  3 * sigma_max / 4, sigma_max])
        # gc.colorbar._colorbar.set_ticklabels([r'$0$', r'$\frac{1}{4}\sigma_{\rm max}$',
        #                                       r'$\frac{1}{2}\sigma_{\rm max}$', r'$\frac{3}{4}\sigma_{\rm max}$',
        #                                       r'$\sigma_{\rm max}$'])
        gc.colorbar._colorbar.set_ticklabels([r'$0$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$\frac{3}{4}$', r'$1$'])
        tick_labels = gc.colorbar._colorbar.ax.get_xticklabels()
        tick_labels[0].set_ha('left')
    fig.savefig(figurename_S80, bbox_inches='tight')

    # HST image with MUSE field of view
    path_hb = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)
    if os.path.exists(path_hb):
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_hb, figure=fig, north=True, hdu=1)
        gc.recenter(ra_qso, dec_qso, width=30 / 3600, height=30 / 3600)
        gc.show_colorscale(cmap='Greys', vmin=-0.005, vmax=1, vmid=-0.001, stretch='arcsinh')
        APLpyStyle(gc, type='HST', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
        gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=530)

        # Draw contours
        gc.show_contour(path_OII_contour, levels=[contour_level_OII], colors='blue', linewidths=2, smooth=5,
                        kernel='box', hdu=1)
        if os.path.exists(path_SB_OIII_kin):
            gc.show_contour(path_OIII_contour, levels=[contour_level_OIII], colors='red', linewidths=2,
                            smooth=5, kernel='box', hdu=1)

        # labels
        if cubename == 'HE0435-5304':
            gc.add_label(0.62, 0.13, r'$\rm MUSE \, [O\,II]$', color='blue', size=30,
                         relative=True, horizontalalignment='left')
            gc.add_label(0.62, 0.05, r'$\rm MUSE \, [O\,III]$', color='red', size=30,
                         relative=True, horizontalalignment='left')
            gc.add_label(0.02, 0.05, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=40,
                         relative=True, horizontalalignment='left')

        # Labels
        path_savefig_mini = '../../MUSEQuBES+CUBS/plots/{}_mini_gaia_25.png'.format(cubename)
        fig.savefig(path_savefig_mini, bbox_inches='tight')
    else:
        # Plot MUSE white light image
        path_white_gaia = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_F814W_gaia.fits'.format(cubename)

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_white_gaia, figure=fig, hdu=1)
        gc.recenter(ra_qso, dec_qso, width=30 / 3600, height=30 / 3600)
        gc.show_colorscale(cmap='Greys', vmin=-0.2, vmid=0.2, vmax=20, stretch='arcsinh')
        APLpyStyle(gc, type='HST', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso)
        gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=530)

        # Draw contours
        gc.show_contour(path_OII_contour, levels=[contour_level_OII], colors='blue', linewidths=2,
                        smooth=5, kernel='box', hdu=1)
        if os.path.exists(path_SB_OIII_kin):
            gc.show_contour(path_OIII_contour, levels=[contour_level_OIII], colors='red', linewidths=2,
                            smooth=5, kernel='box', hdu=1)
        path_savefig_mini = '../../MUSEQuBES+CUBS/plots/{}_mini_gaia_MUSE.png'.format(cubename)
        fig.savefig(path_savefig_mini, bbox_inches='tight')



# MakeV50W80(cubename='HE0435-5304', v_max=100, sigma_max=300, HSTcentroid=True, rmbkgResidue=True)
# MakeV50W80(cubename='HE0153-4520', v_max=300, sigma_max=300, contour_level_OII=0.5, contour_level_OIII=1.0)
# MakeV50W80(cubename='HE0226-4110', v_max=300, sigma_max=300, nums_seg_OII=[12, 14, 15, 16, 17, 20],
#            nums_seg_OIII=[5, 11, 16, 19], contour_level_OII=0.1, contour_level_OIII=0.1, rmbkgResidue=True, HSTcentroid=True)
# MakeV50W80(cubename='PKS0405-123', v_max=800, sigma_max=300, contour_level_OIII=0.5, nums_seg_OII=[5],
#            nums_seg_OIII=[15], HSTcentroid=True)
# MakeV50W80(cubename='HE0238-1904', v_max=300, sigma_max=300, HSTcentroid=True, rmbkgResidue=False)
# MakeV50W80(cubename='3C57', v_max=350, sigma_max=300, HSTcentroid=False, rmbkgResidue=True)
# MakeV50W80(cubename='PKS0552-640', v_max=300, sigma_max=300, contour_level_OII=0.2, contour_level_OIII=0.2,
#            nums_seg_OII=[2, 6, 7, 9, 10, 14, 18], nums_seg_OIII=[7, 9, 12, 19, 20], rmbkgResidue=True)
# MakeV50W80(cubename='J0110-1648', v_max=300, sigma_max=300, rmbkgResidue=True)
# MakeV50W80(cubename='J0454-6116', v_max=500, sigma_max=400, nums_seg_OII=[2, 6, 8, 13, 17, 18],
#            nums_seg_OIII=[2, 9, 10, 18])
# MakeV50W80(cubename='J2135-5316', v_max=300, sigma_max=300, contour_level_OII=0.3,
#            nums_seg_OII=[3, 4, 10, 12, 13, 14, 16, 17, 18, 19],
#            nums_seg_OIII=[4, 12, 13, 14, 15, 17, 19, 20], rmbkgResidue=True) # Double component # SNR=5
# MakeV50W80(cubename='J0119-2010', v_max=500, sigma_max=300, contour_level_OIII=0.5,
#            nums_seg_OII=[4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20], nums_seg_OIII=[7, 9, 11, 12, 14, 16, 17, 18],
#            rmbkgResidue=True)  # Double component
# MakeV50W80(cubename='HE0246-4101', v_max=300, sigma_max=300)
# MakeV50W80(cubename='J0028-3305', v_max=300, sigma_max=300, contour_level_OII=0.3)
# MakeV50W80(cubename='HE0419-5657', v_max=400, sigma_max=300, nums_seg_OII=[1], rmbkgResidue=True)
# MakeV50W80(cubename='PB6291', v_max=400, sigma_max=300, nums_seg_OII=[3, 5], HSTcentroid=True)
# MakeV50W80(cubename='Q0107-0235', v_max=400, sigma_max=300, nums_seg_OII=[7], rmbkgResidue=True, HSTcentroid=True)
# MakeV50W80(cubename='PKS2242-498', v_max=400, sigma_max=300, contour_level_OII=0.3)
# MakeV50W80(cubename='PKS0355-483', v_max=300, sigma_max=300, nums_seg_OII=[6, 14], rmbkgResidue=True)
# MakeV50W80(cubename='HE0112-4145', v_max=300, sigma_max=300, rmbkgResidue=True)
# MakeV50W80(cubename='HE0439-5254', v_max=500, sigma_max=300, HSTcentroid=True)
# MakeV50W80(cubename='HE2305-5315', v_max=500, sigma_max=300, nums_seg_OII=[5, 6, 7, 8])
# MakeV50W80(cubename='HE1003+0149', v_max=300, sigma_max=300, contour_level_OII=0.3, HSTcentroid=True)
# MakeV50W80(cubename='HE0331-4112', v_max=500, sigma_max=300, nums_seg_OII=[1, 2], rmbkgResidue=True)
# MakeV50W80(cubename='TEX0206-048', v_max=600, sigma_max=400,
#            nums_seg_OII=np.setdiff1d(np.arange(1, 1001),
#                                      [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 22, 23, 26, 27, 28, 34, 57,
#                                       60, 79, 81, 101, 107, 108, 114, 118, 317, 547, 552]), HSTcentroid=True) # SNR=20
# MakeV50W80(cubename='Q1354+048', v_max=400, sigma_max=300, rmbkgResidue=True, HSTcentroid=True)
# MakeV50W80(cubename='J0154-0712', v_max=300, sigma_max=300)
MakeV50W80(cubename='LBQS1435-0134', v_max=400, sigma_max=400, HSTcentroid=True)
# MakeV50W80(cubename='PG1522+101', v_max=300, sigma_max=300, nums_seg_OII=[6, 12], HSTcentroid=True)
# MakeV50W80(cubename='HE2336-5540', v_max=300, sigma_max=300, nums_seg_OII=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# MakeV50W80(cubename='PKS0232-04', v_max=400, sigma_max=300, nums_seg_OII=[4, 5, 7], HSTcentroid=True)