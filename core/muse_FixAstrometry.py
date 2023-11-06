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
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
from astropy.convolution import Kernel, convolve, Gaussian2DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12


def FixCubeHeader(cubename=None):
    path_muse_white = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/test/{}_ESO-DEEP_ZAP_WHITE.fits'.format(cubename)
    hdul_muse_white = fits.open(path_muse_white)
    hdul_muse_white[1].header.remove('CRDER3')
    hdul_muse_white[2].header.remove('CRDER3')
    hdul_muse_white.writeto(path_muse_white, overwrite=True)


def FixAstrometry(cubename=None, useGAIA=False, thr_hst=3, thr_muse=3, deblend_hst=False, deblend_muse=True,
                  checkHST=False, checkMUSE=False, update_offset=True):
    path_savefig = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_offset_gaia.png'.format(cubename)
    path_savefig_MUSE = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_WHITE_gaia.png'.format(cubename)
    path_savefig_vect = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_vector.png'.format(cubename)
    path_savefig_vect_rot = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_vector_rot.png'.format(cubename)

    # Load info
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    c_qso = SkyCoord(ra=ra_qso * u.degree, dec=dec_qso * u.degree, frame='icrs')

    #
    path_gal = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/{}_gal_info.fits'.format(cubename)
    data_gal = fits.getdata(path_gal, 1, ignore_missing_end=True)
    row_gal, ID_gal, z_gal, v_gal = data_gal['row'], data_gal['ID'], data_gal['z'], data_gal['v']
    name_gal, ql_gal, ra_gal, dec_gal = data_gal['name'], data_gal['ql'], data_gal['ra'], data_gal['dec']

    # Load the image
    path_hb = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_drc_offset.fits'.format(cubename)
    hdul_hb = fits.open(path_hb)
    data_hb = hdul_hb[1].data

    # Gaia
    path_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/GAIA_{}.txt'.format(cubename)
    data_gaia = ascii.read(path_gaia)
    ra_gaia, dec_gaia = np.asarray(data_gaia['ra']), np.asarray(data_gaia['dec'])
    c1 = SkyCoord(ra=ra_gaia * u.degree, dec=dec_gaia * u.degree, frame='icrs')

    # Segmentation for HST
    bkg_estimator = MedianBackground()
    bkg = Background2D(data_hb, (200, 200), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg = data_hb - bkg.background
    threshold = thr_hst * bkg.background_rms
    kernel = Gaussian2DKernel(1)
    convolved_data = convolve(data_bkg, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    if deblend_hst:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=32, contrast=0.001)
    cat_hb = SourceCatalog(convolved_data, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid
    w = WCS(hdul_hb[1].header)
    c2_hst = w.pixel_to_world(x_cen, y_cen)

    # Matching
    if useGAIA:
        idx, d2d, d3d = c1.match_to_catalog_sky(c2_hst)
        offset_ra_hst, offset_dec_hst = c1.ra - c2_hst.ra[idx], c1.dec - c2_hst.dec[idx]
    else:
        idx_qso, d2d_qso, d3d_qso = c_qso.match_to_catalog_sky(c2_hst)
        offset_ra_hst, offset_dec_hst = (c_qso.ra - c2_hst.ra[idx_qso]).value, (c_qso.dec - c2_hst.dec[idx_qso]).value

    # Save offsets
    path_offset = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/offsets.dat'
    data_offset = ascii.read(path_offset, format='fixed_width')
    data_offset['offset_ra_hst'][data_offset['name'] == cubename] = offset_ra_hst
    data_offset['offset_dec_hst'][data_offset['name'] == cubename] = offset_dec_hst

    path_hb_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_drc_offset_gaia.fits'.format(cubename)
    hdul_hb[1].header['CRVAL1'] += np.median(offset_ra_hst)
    hdul_hb[1].header['CRVAL2'] += np.median(offset_dec_hst)
    hdul_hb.writeto(path_hb_gaia, overwrite=True)

    # Fix MUSE cube and whitelight image
    path_muse = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP.fits'.format(cubename)
    path_muse_white = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_WHITE.fits'.format(cubename)
    hdul_muse = fits.open(path_muse)
    hdul_muse_white = fits.open(path_muse_white)
    if cubename == 'HE0153-4520' or cubename == '3C57':
        data_muse = hdul_muse_white[0].data
    else:
        data_muse = hdul_muse_white[1].data

    # Segmentation for MUSE
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(data_muse, 50, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    except ValueError:
        bkg = Background2D(data_muse, 51, filter_size=(3, 3), bkg_estimator=bkg_estimator, exclude_percentile=12)
    data_muse_bkg = data_muse - bkg.background
    threshold = thr_muse * bkg.background_rms
    kernel = Gaussian2DKernel(1.5)  # corresponding to a 0.7 FWHM
    convolved_data_muse = convolve(data_muse_bkg, kernel)
    segment_map = detect_sources(convolved_data_muse, threshold, npixels=10)
    if deblend_muse:
        segment_map = deblend_sources(convolved_data_muse, segment_map, npixels=10, nlevels=32, contrast=0.001)
    cat_muse = SourceCatalog(convolved_data_muse, segment_map)
    x_cen, y_cen = cat_muse.xcentroid, cat_muse.ycentroid
    x_cen, y_cen = x_cen[~np.isnan(x_cen)], y_cen[~np.isnan(y_cen)]
    w = WCS(hdul_muse[1].header, naxis=2)
    c2_muse = w.pixel_to_world(x_cen, y_cen)

    idx_qso, d2d_qso, d3d_qso = c_qso.match_to_catalog_sky(c2_muse)
    offset_ra_muse, offset_dec_muse = (c_qso.ra - c2_muse.ra[idx_qso]).value, (c_qso.dec - c2_muse.dec[idx_qso]).value

    # Save MUSE offset
    data_offset['offset_ra_muse'][data_offset['name'] == cubename] = offset_ra_muse
    data_offset['offset_dec_muse'][data_offset['name'] == cubename] = offset_dec_muse
    if update_offset:
        ascii.write(data_offset, path_offset, overwrite=True, format='fixed_width')


    path_muse_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia.fits'.format(cubename)
    path_muse_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.format(cubename)
    hdul_muse[0].header['RA'] += np.median(offset_ra_muse)
    hdul_muse[0].header['DEC'] += np.median(offset_dec_muse)
    hdul_muse[1].header['CRVAL1'] += np.median(offset_ra_muse)
    hdul_muse[1].header['CRVAL2'] += np.median(offset_dec_muse)
    hdul_muse[2].header['CRVAL1'] += np.median(offset_ra_muse)
    hdul_muse[2].header['CRVAL2'] += np.median(offset_dec_muse)
    hdul_muse.writeto(path_muse_gaia, overwrite=True)

    # Problem with HE0153-4520 whitelight image header
    if cubename == 'HE0153-4520' or cubename == '3C57':
        hdul_2 = fits.ImageHDU(hdul_muse_white[0].data)
        hdul_2.header = hdul_muse[1].header
        hdul_2.header.remove('CTYPE3')
        hdul_2.header.remove('CUNIT3')
        hdul_2.header.remove('CD3_3')
        hdul_2.header.remove('CRPIX3')
        hdul_2.header.remove('CRVAL3')
        hdul_2.header.remove('CRDER3')
        hdul_2.header.remove('CD1_3')
        hdul_2.header.remove('CD2_3')
        hdul_2.header.remove('CD3_1')
        hdul_2.header.remove('CD3_2')
        hdul_muse_white = fits.HDUList([hdul_muse[0], hdul_2, hdul_2])
    else:
        hdul_muse_white[0].header['RA'] += np.median(offset_ra_muse)
        hdul_muse_white[0].header['DEC'] += np.median(offset_dec_muse)
        hdul_muse_white[1].header['CRVAL1'] += np.median(offset_ra_muse)
        hdul_muse_white[1].header['CRVAL2'] += np.median(offset_dec_muse)
        hdul_muse_white[2].header['CRVAL1'] += np.median(offset_ra_muse)
        hdul_muse_white[2].header['CRVAL2'] += np.median(offset_dec_muse)
        hdul_muse_white[1].header.remove('CRDER3')
        hdul_muse_white[2].header.remove('CRDER3')
    hdul_muse_white.writeto(path_muse_white_gaia, overwrite=True)

    # LS
    path_ls = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/LS_{}.txt'.format(cubename)
    data_ls = ascii.read(path_ls)
    ra_ls, dec_ls = np.asarray(data_ls['ra']), np.asarray(data_ls['dec'])
    c_ls = SkyCoord(ra=ra_ls * u.degree, dec=dec_ls * u.degree, frame='icrs')

    # correct muse coordinate
    c2_muse_c = SkyCoord(ra=c2_muse.ra + (offset_ra_muse * u.degree),
                         dec=c2_muse.dec + (offset_dec_muse * u.degree), frame='icrs')
    idx_ls, d2d_ls, d3d_ls = c2_muse_c.match_to_catalog_sky(c_ls)
    max_sep = 0.5 * u.arcsec
    sep_constraint = d2d_ls < max_sep
    c2_muse_m = c2_muse_c[sep_constraint]
    c_ls_m = c_ls[idx_ls[sep_constraint]]
    print(len(c_ls_m))
    offset_ra_muse_gal, offset_dec_muse_gal = np.median(c_ls_m.ra.value - c2_muse_m.ra.value), \
                                              np.median(c_ls_m.dec.value - c2_muse_m.dec.value)
    # idx_ls, d2d_ls, d3d_ls = c2_hst.match_to_catalog_sky(c_ls)
    # max_sep = 0.5 * u.arcsec
    # sep_constraint = d2d_ls < max_sep
    # c2_hst_m = c2_hst[sep_constraint]
    # c_ls_m = c_ls[idx_ls[sep_constraint]]

    if checkMUSE:
        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.quiver(c_ls_m.ra.value, c_ls_m.dec.value,
                   c_ls_m.ra.value - c2_muse_m.ra.value,
                   c_ls_m.dec.value - c2_muse_m.dec.value,
                   angles='xy', scale_units='xy')
        # plt.plot(c_ls_m.ra.value, c_ls_m.dec.value, '.k')
        # plt.plot(c2_muse_m.ra.value, c2_muse_m.dec.value, '.r')
        plt.plot(c_qso.ra.value, c_qso.dec.value, '.r', ms=10)
        plt.plot(c2_muse_c.ra[idx_qso], c2_muse_c.dec[idx_qso], '.b', ms=10)
        fig.savefig(path_savefig_vect, bbox_inches='tight')

        # Figure
        path_muse_white_gaia_test = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/test/{}_ESO-DEEP_ZAP_gaia_WHITE_astro.fits'.format(cubename)
        hdul_muse_white_gaia_test = fits.open(path_muse_white_gaia_test)
        data_muse_test = hdul_muse_white_gaia_test[1].data
        bkg_estimator = MedianBackground()
        bkg = Background2D(data_muse_test, 51, filter_size=(3, 3), bkg_estimator=bkg_estimator, exclude_percentile=12)
        data_muse_bkg_test = data_muse_test - bkg.background
        threshold = thr_muse * bkg.background_rms
        kernel = Gaussian2DKernel(1.5)  # corresponding to a 0.7 FWHM
        convolved_data_muse_test = convolve(data_muse_bkg_test, kernel)
        segment_map = detect_sources(convolved_data_muse_test, threshold, npixels=10)
        if deblend_muse:
            segment_map = deblend_sources(convolved_data_muse_test, segment_map, npixels=10, nlevels=32, contrast=0.001)
        cat_muse = SourceCatalog(convolved_data_muse_test, segment_map)
        x_cen, y_cen = cat_muse.xcentroid, cat_muse.ycentroid
        x_cen, y_cen = x_cen[~np.isnan(x_cen)], y_cen[~np.isnan(y_cen)]
        w = WCS(hdul_muse_white_gaia_test[1].header, naxis=2)
        c2_muse = w.pixel_to_world(x_cen, y_cen)

        #
        idx_ls, d2d_ls, d3d_ls = c2_muse.match_to_catalog_sky(c_ls)
        max_sep = 0.5 * u.arcsec
        sep_constraint = d2d_ls < max_sep
        c2_muse_m = c2_muse[sep_constraint]
        c_ls_m = c_ls[idx_ls[sep_constraint]]

        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.quiver(c_ls_m.ra.value, c_ls_m.dec.value,
                   c_ls_m.ra.value - c2_muse_m.ra.value,
                   c_ls_m.dec.value - c2_muse_m.dec.value,
                   angles='xy', scale_units='xy')
        # plt.plot(c_ls_m.ra.value, c_ls_m.dec.value, '.k')
        # plt.plot(c2_muse_m.ra.value, c2_muse_m.dec.value, '.r')
        plt.plot(c_qso.ra.value, c_qso.dec.value, '.r', ms=10)
        # plt.plot(c2_muse_c.ra[idx_qso], c2_muse_c.dec[idx_qso], '.b', ms=10)
        fig.savefig(path_savefig_vect_rot, bbox_inches='tight')

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_muse_white_gaia, figure=fig, hdu=1, north=True)

        #
        gc.set_system_latex(True)
        gc.show_colorscale(cmap='Greys', pmin=2.0, pmax=98.0)
        # gc.add_colorbar()
        # gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
        # gc.colorbar.hide()

        # Hide ticks
        gc.ticks.set_length(30)
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()


        # Markers
        gc.show_markers(ra_ls, dec_ls, facecolors='none', marker='o', c='none',
                        edgecolors='blue', linewidths=0.8, s=150)
        gc.show_markers(ra_gaia, dec_gaia, facecolors='none', marker='o', c='none',
                        edgecolors='red', linewidths=0.8, s=140)
        gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='o', c='none',
                        edgecolors='white', linewidths=0.8, s=120)
        gc.show_markers(c2_muse.ra.value, c2_muse.dec.value,
                        facecolors='none', marker='o', c='none', edgecolors='white', linewidths=0.8, s=100)
        # gc.show_markers(c2_muse.ra.value + np.median(offset_ra_muse), c2_muse.dec.value + np.median(offset_dec_muse),
        #                 facecolors='none', marker='o', c='none', edgecolors='white', linewidths=0.8, s=100)
        gc.show_markers(ra_gal + np.median(offset_ra_muse), dec_gal + np.median(offset_dec_muse),
                        facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)

        # Labels
        gc.add_label(0.87, 0.97, r'$\mathrm{MUSE}$', color='k', size=15, relative=True)
        fig.savefig(path_savefig_MUSE, bbox_inches='tight')

    # Figure
    if checkHST:
        path_hb_gaia_test = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/test/{}_drc_offset_astro.fits'.format(cubename)
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_hb_gaia_test, figure=fig, north=True)
        gc.set_xaxis_coord_type('scalar')
        gc.set_yaxis_coord_type('scalar')

        #
        gc.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)

        #
        gc.set_system_latex(True)
        gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
        gc.add_colorbar()
        gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
        gc.colorbar.hide()

        # Hide ticks
        gc.ticks.set_length(30)
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()

        # Markers
        gc.show_markers(ra_gaia, dec_gaia, facecolors='none', marker='o', c='none',
                        edgecolors='red', linewidths=0.8, s=140)
        gc.show_markers(ra_ls, dec_ls, facecolors='none', marker='o', c='none',
                        edgecolors='blue', linewidths=0.8, s=130)
        gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='o', c='none',
                        edgecolors='white', linewidths=0.8, s=120)
        gc.show_markers(ra_gal + np.median(offset_ra_hst), dec_gal + np.median(offset_dec_hst),
                        facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
        gc.show_markers(c2_hst.ra.value + np.median(offset_ra_hst), c2_hst.dec.value + np.median(offset_dec_hst),
                        facecolors='none', marker='o', c='none', edgecolors='white', linewidths=0.8, s=100)

        # Labels
        gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
        fig.savefig(path_savefig, bbox_inches='tight')


def FixDat(cubename=None):
    #
    path_offset = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/offsets.dat'
    data_offset = ascii.read(path_offset, format='fixed_width')
    data_offset = data_offset[data_offset['name'] == cubename]
    offset_ra_muse, offset_dec_muse = data_offset['offset_ra_muse'], data_offset['offset_dec_muse']

    #
    path_dat = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/{}_ESO-DEEP_ZAP.dat'.format(cubename)
    path_dat_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/{}_ESO-DEEP_ZAP_gaia.dat'.format(cubename)
    path_reg_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/{}_ESO-DEEP_ZAP_gaia.reg'.format(cubename)
    dat = ascii.read(path_dat, format='fixed_width')
    dat['ra'] = np.round(dat['ra'] + offset_ra_muse, 6)
    dat['dec'] = np.round(dat['dec'] + offset_dec_muse, 6)
    dat['name'] = np.asarray(coord.coordstring(dat['ra'], dat['dec']))[2]
    ascii.write(dat, path_dat_gaia, overwrite=True, format='fixed_width')

    # Create regions files
    reg_pre = np.array(['# Region file format: DS9 version 4.1',
                        'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
                        'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                        'fk5'])
    array_like = np.asarray(dat['ra'])
    reg_main = list(map(''.join, zip(np.full_like(array_like, 'circle(', dtype='<U15'),
                                     np.asarray(dat['ra'], dtype=str), np.full_like(array_like, ', ', dtype=str),
                                     np.asarray(dat['dec'], dtype=str), np.full_like(array_like, ', ', dtype=str),
                                     np.asarray(dat['radius'], dtype=str), np.full_like(array_like, '") # text={', dtype='<U15'),
                                     np.asarray(dat['id'], dtype=str), np.full_like(array_like, '}', dtype='<U30'))))
    reg_post = np.hstack((reg_pre, reg_main))
    np.savetxt(path_reg_gaia, reg_post, fmt="%s")

def CopyCurrentObj(cubename=None):
    # Object files
    path_obj = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_spec1D'.format(cubename)
    path_obj_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia_spec1D'.format(cubename)

    if not os.path.exists(path_obj_gaia):
        shutil.copytree(path_obj, path_obj_gaia)
    else:
        raise ValueError('Folder exists')

    path_dat = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP.dat'.format(cubename)
    path_dat_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia.dat'.format(cubename)
    dat = ascii.read(path_dat, format='fixed_width')
    dat_gaia = ascii.read(path_dat_gaia, format='fixed_width')

    # Rename obj and obj.bkp files
    path_obj_obj = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects.fits'.format(cubename))
    path_obj_bkp = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects_bkp.fits'.format(cubename))
    path_obj_obj_gaia = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_gaia_objects.fits'.format(cubename))
    path_obj_bkp_gaia = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_gaia_objects_bkp.fits'.format(cubename))

    hdul_obj_obj = fits.open(path_obj_obj)
    hdul_obj_bkp = fits.open(path_obj_bkp)
    hdul_obj_obj[1].data['name'] = dat_gaia['name']
    hdul_obj_obj[1].data['ra'] = dat_gaia['ra']
    hdul_obj_obj[1].data['dec'] = dat_gaia['dec']
    hdul_obj_bkp[1].data['name'] = dat_gaia['name']
    hdul_obj_bkp[1].data['ra'] = dat_gaia['ra']
    hdul_obj_bkp[1].data['dec'] = dat_gaia['dec']
    hdul_obj_obj.writeto(path_obj_obj_gaia, overwrite=True)
    hdul_obj_bkp.writeto(path_obj_bkp_gaia, overwrite=True)
    os.remove(os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects.fits'.format(cubename)))
    os.remove(os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects_bkp.fits'.format(cubename)))

    # Rename each file
    for i in range(len(dat)):
        dat_i, dat_gaia_i = dat[i], dat_gaia[i]
        redshift_i = '{}_{}_{}_redshift.fits'.format(dat_i['row'], dat_i['id'], dat_i['name'])
        spec1D_i = '{}_{}_{}_spec1D.fits'.format(dat_i['row'], dat_i['id'], dat_i['name'])
        spec2D_i = '{}_{}_{}_spec2D.fits'.format(dat_i['row'], dat_i['id'], dat_i['name'])
        redshift_f = '{}_{}_{}_redshift.fits'.format(dat_gaia_i['row'], dat_gaia_i['id'], dat_gaia_i['name'])
        spec1D_f = '{}_{}_{}_spec1D.fits'.format(dat_gaia_i['row'], dat_gaia_i['id'], dat_gaia_i['name'])
        spec2D_f = '{}_{}_{}_spec2D.fits'.format(dat_gaia_i['row'], dat_gaia_i['id'], dat_gaia_i['name'])
        os.rename(os.path.join(path_obj_gaia, redshift_i), os.path.join(path_obj_gaia, redshift_f))
        os.rename(os.path.join(path_obj_gaia, spec1D_i), os.path.join(path_obj_gaia, spec1D_f))
        os.rename(os.path.join(path_obj_gaia, spec2D_i), os.path.join(path_obj_gaia, spec2D_f))


def FixCube(cubename=None):
    path_muse_white_gaia_test = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/test/{}_ESO-DEEP_ZAP_gaia_WHITE_astro.fits'.format(
        cubename)
    path_muse_white_gaia_test2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/test/{}_ESO-DEEP_ZAP_gaia_WHITE_astro_2.fits'.format(
        cubename)
    path_muse_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.format(cubename)

    hdul_muse = fits.open(path_muse_white_gaia_test)
    hdul_muse_gaia = fits.open(path_muse_white_gaia)

    hdul_muse[1].header['CRPIX1'] = hdul_muse_gaia[1].header['CRPIX1']
    hdul_muse[1].header['CRPIX2'] = hdul_muse_gaia[1].header['CRPIX2']
    hdul_muse.writeto(path_muse_white_gaia_test2, overwrite=True)


def FixWithAstrometry(cubename=None, FixHST=True, CheckHST=False, CheckMUSE=False, thr_hst=3, thr_muse=3,
                      deblend_hst=False, deblend_muse=True):
    path_savefig = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_offset_gaia.png'.format(cubename)
    path_savefig_MUSE = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_WHITE_gaia.png'.format(cubename)
    path_savefig_vect = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_vector.png'.format(cubename)
    path_savefig_vect_rot = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_vector_rot.png'.format(cubename)

    path_muse = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia.fits'.format(cubename)
    path_muse_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia.fits'.format(cubename)
    path_muse_white = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.format(cubename)
    path_muse_white_astro = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia_WHITE_astro.fits'.format(cubename)
    path_muse_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.format(cubename)

    #
    path_hst = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_drc_offset_gaia.fits'.format(cubename)
    path_hst_astro = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_drc_offset_gaia_astro.fits'.format(cubename)
    path_hst_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)
    path_ls = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/LS_{}.txt'.format(cubename)

    # Convert MUSE white light image and HST image
    os.system('astrometry {} -c {} -hdul_idx 1 --sigma_threshold_for_source_detection 20'.format(path_muse_white,
                                                                                                 path_ls))
    os.rename(path_muse_white_astro, path_muse_white_gaia)

    if FixHST:
        os.system('astrometry {} -c {} -hdul_idx 1 '
                  '--sigma_threshold_for_source_detection 30 -high_res True'.format(path_hst, path_ls))
    os.rename(path_hst_astro, path_hst_gaia)

    # Copy White light image header and fix MUSE header
    hdul_muse_white = fits.open(path_muse_white_gaia)
    hdul_muse_white[1].header.append('CD1_1')
    hdul_muse_white[1].header.append('CD1_2')
    hdul_muse_white[1].header.append('CD2_1')
    hdul_muse_white[1].header.append('CD2_2')
    hdul_muse_white[1].header['CD1_1'] = hdul_muse_white[1].header['PC1_1'] * hdul_muse_white[1].header['CDELT1']
    hdul_muse_white[1].header['CD2_1'] = hdul_muse_white[1].header['PC2_1'] * hdul_muse_white[1].header['CDELT2']
    hdul_muse_white[1].header['CD1_2'] = hdul_muse_white[1].header['PC1_2'] * hdul_muse_white[1].header['CDELT1']
    hdul_muse_white[1].header['CD2_2'] = hdul_muse_white[1].header['PC2_2'] * hdul_muse_white[1].header['CDELT2']
    hdul_muse_white[1].header.remove('PC1_1')
    hdul_muse_white[1].header.remove('PC1_2')
    hdul_muse_white[1].header.remove('PC2_1')
    hdul_muse_white[1].header.remove('PC2_2')
    hdul_muse_white[1].header.remove('CDELT1')
    hdul_muse_white[1].header.remove('CDELT2')
    hdul_muse_white.writeto(path_muse_white_gaia, overwrite=True)
    hdul_muse_white = fits.open(path_muse_white_gaia)
    hdr_muse_white = hdul_muse_white[1].header
    hdul_muse = fits.open(path_muse)
    hdul_muse[1].header['CD1_1'] = hdr_muse_white['CD1_1']
    hdul_muse[1].header['CD2_1'] = hdr_muse_white['CD2_1']
    hdul_muse[1].header['CD1_2'] = hdr_muse_white['CD1_2']
    hdul_muse[1].header['CD2_2'] = hdr_muse_white['CD2_2']
    hdul_muse[1].header['CRVAL1'] = hdr_muse_white['CRVAL1']
    hdul_muse[1].header['CRVAL2'] = hdr_muse_white['CRVAL2']
    hdul_muse[1].header['CRPIX1'] = hdr_muse_white['CRPIX1']
    hdul_muse[1].header['CRPIX2'] = hdr_muse_white['CRPIX2']
    hdul_muse.writeto(path_muse_gaia, overwrite=True)

    # Fix HST header
    hdul_hst_gaia = fits.open(path_hst_gaia)
    hdul_hst_gaia[1].header.append('CD1_1')
    hdul_hst_gaia[1].header.append('CD1_2')
    hdul_hst_gaia[1].header.append('CD2_1')
    hdul_hst_gaia[1].header.append('CD2_2')
    hdul_hst_gaia[1].header['CD1_1'] = hdul_hst_gaia[1].header['PC1_1'] * hdul_hst_gaia[1].header['CDELT1']
    hdul_hst_gaia[1].header['CD2_1'] = hdul_hst_gaia[1].header['PC2_1'] * hdul_hst_gaia[1].header['CDELT2']
    hdul_hst_gaia[1].header['CD1_2'] = hdul_hst_gaia[1].header['PC1_2'] * hdul_hst_gaia[1].header['CDELT1']
    hdul_hst_gaia[1].header['CD2_2'] = hdul_hst_gaia[1].header['PC2_2'] * hdul_hst_gaia[1].header['CDELT2']
    hdul_hst_gaia[1].header.remove('PC1_1')
    hdul_hst_gaia[1].header.remove('PC1_2')
    hdul_hst_gaia[1].header.remove('PC2_1')
    hdul_hst_gaia[1].header.remove('PC2_2')
    hdul_hst_gaia[1].header.remove('CDELT1')
    hdul_hst_gaia[1].header.remove('CDELT2')
    hdul_hst_gaia.writeto(path_hst_gaia, overwrite=True)

    # Load info
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    c_qso = SkyCoord(ra=ra_qso * u.degree, dec=dec_qso * u.degree, frame='icrs')

    # Gaia and Legacy Survey
    path_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/GAIA_{}.txt'.format(cubename)
    data_gaia = ascii.read(path_gaia)
    ra_gaia, dec_gaia = np.asarray(data_gaia['ra']), np.asarray(data_gaia['dec'])
    c_gaia = SkyCoord(ra=ra_gaia * u.degree, dec=dec_gaia * u.degree, frame='icrs')

    path_ls = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/LS_{}.txt'.format(cubename)
    data_ls = ascii.read(path_ls)
    ra_ls, dec_ls = np.asarray(data_ls['ra']), np.asarray(data_ls['dec'])
    c_ls = SkyCoord(ra=ra_ls * u.degree, dec=dec_ls * u.degree, frame='icrs')


    if CheckMUSE:
        data_muse_white = hdul_muse_white[1].data
        # fig = plt.figure(figsize=(5, 5), dpi=300)
        # plt.quiver(c_ls_m.ra.value, c_ls_m.dec.value,
        #            c_ls_m.ra.value - c2_muse_m.ra.value,
        #            c_ls_m.dec.value - c2_muse_m.dec.value,
        #            angles='xy', scale_units='xy')
        # # plt.plot(c_ls_m.ra.value, c_ls_m.dec.value, '.k')
        # # plt.plot(c2_muse_m.ra.value, c2_muse_m.dec.value, '.r')
        # plt.plot(c_qso.ra.value, c_qso.dec.value, '.r', ms=10)
        # plt.plot(c2_muse_c.ra[idx_qso], c2_muse_c.dec[idx_qso], '.b', ms=10)
        # fig.savefig(path_savefig_vect, bbox_inches='tight')

        # Figure
        bkg_estimator = MedianBackground()
        bkg = Background2D(data_muse_white, 51, filter_size=(3, 3), bkg_estimator=bkg_estimator, exclude_percentile=12)
        data_muse_bkg_test = data_muse_white - bkg.background
        threshold = thr_muse * bkg.background_rms
        kernel = Gaussian2DKernel(1.5)  # corresponding to a 0.7 FWHM
        convolved_data_muse_test = convolve(data_muse_bkg_test, kernel)
        segment_map = detect_sources(convolved_data_muse_test, threshold, npixels=10)
        if deblend_muse:
            segment_map = deblend_sources(convolved_data_muse_test, segment_map, npixels=10, nlevels=32, contrast=0.001)
        cat_muse = SourceCatalog(convolved_data_muse_test, segment_map)
        x_cen, y_cen = cat_muse.xcentroid, cat_muse.ycentroid
        x_cen, y_cen = x_cen[~np.isnan(x_cen)], y_cen[~np.isnan(y_cen)]
        w = WCS(hdul_muse_white[1].header, naxis=2)
        c2_muse = w.pixel_to_world(x_cen, y_cen)

        #
        # idx_ls, d2d_ls, d3d_ls = c2_muse.match_to_catalog_sky(c_ls)
        # max_sep = 0.5 * u.arcsec
        # sep_constraint = d2d_ls < max_sep
        # c2_muse_m = c2_muse[sep_constraint]
        # c_ls_m = c_ls[idx_ls[sep_constraint]]

        # fig = plt.figure(figsize=(5, 5), dpi=300)
        # plt.quiver(c_ls_m.ra.value, c_ls_m.dec.value,
        #            c_ls_m.ra.value - c2_muse_m.ra.value,
        #            c_ls_m.dec.value - c2_muse_m.dec.value,
        #            angles='xy', scale_units='xy')
        # # plt.plot(c_ls_m.ra.value, c_ls_m.dec.value, '.k')
        # # plt.plot(c2_muse_m.ra.value, c2_muse_m.dec.value, '.r')
        # plt.plot(c_qso.ra.value, c_qso.dec.value, '.r', ms=10)
        # # plt.plot(c2_muse_c.ra[idx_qso], c2_muse_c.dec[idx_qso], '.b', ms=10)
        # fig.savefig(path_savefig_vect_rot, bbox_inches='tight')

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_muse_white_gaia, figure=fig, hdu=1, north=True)

        #
        gc.set_system_latex(True)
        gc.show_colorscale(cmap='Greys', pmin=2.0, pmax=98.0)

        # Hide ticks
        gc.ticks.set_length(30)
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()


        # Markers
        gc.show_markers(ra_ls, dec_ls, facecolors='none', marker='o', c='none',
                        edgecolors='blue', linewidths=0.8, s=150)
        gc.show_markers(ra_gaia, dec_gaia, facecolors='none', marker='o', c='none',
                        edgecolors='red', linewidths=0.8, s=140)
        gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='o', c='none',
                        edgecolors='white', linewidths=0.8, s=120)
        gc.show_markers(c2_muse.ra.value, c2_muse.dec.value,
                        facecolors='none', marker='o', c='none', edgecolors='white', linewidths=0.8, s=100)

        # Labels
        gc.add_label(0.87, 0.97, r'$\mathrm{MUSE}$', color='k', size=15, relative=True)
        fig.savefig(path_savefig_MUSE, bbox_inches='tight')


    if CheckHST:
        hdul_hst_gaia = fits.open(path_hst_gaia)
        data_hst_gaia = hdul_hst_gaia[1].data

        # Segmentation for HST
        bkg_estimator = MedianBackground()
        bkg = Background2D(data_hst_gaia, (200, 200), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        data_bkg = data_hst_gaia - bkg.background
        threshold = thr_hst * bkg.background_rms
        kernel = Gaussian2DKernel(1)
        convolved_data = convolve(data_bkg, kernel)
        segment_map = detect_sources(convolved_data, threshold, npixels=10)
        if deblend_hst:
            segment_map = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=32, contrast=0.001)
        cat_hb = SourceCatalog(convolved_data, segment_map)
        x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid
        w = WCS(hdul_hst_gaia[1].header)
        c2_hst = w.pixel_to_world(x_cen, y_cen)

        idx_hst_ls, d2d_hst_ls, d3d_hst_ls = c2_hst.match_to_catalog_sky(c_ls)
        sep_constraint = d2d_hst_ls < (0.5 * u.arcsec)
        c2_hst_ls = c2_hst[sep_constraint]
        c_ls_hst = c_ls[idx_hst_ls[sep_constraint]]

        f, ax = plt.subplots(2, 2, figsize=(8, 5), dpi=300)
        f.subplots_adjust(hspace=0.3)
        f.subplots_adjust(wspace=0.3)
        ax[0, 0].plot(c2_hst_ls.ra.value, 3600 * (ra_pho[idx_total[sep_lim_total]] - ra_total[sep_lim_total]), '.')
        ax[0, 1].plot(c2_hst_ls.ra.value, 3600 * (dec_pho[idx_total[sep_lim_total]] - dec_total[sep_lim_total]), '.')
        ax[1, 0].plot(c2_hst_ls.dec.value,
                         3600 * (ra_pho[idx_total[sep_lim_total]] - ra_total[sep_lim_total]), '.')
        ax[1, 1].plot(c2_hst_ls.dec.value,
                         3600 * (dec_pho[idx_total[sep_lim_total]] - dec_total[sep_lim_total]), '.')
        ax[0, 0].set_xlabel('ra')
        ax[0, 0].set_ylabel(r'$\delta ra$')
        ax[0, 1].set_xlabel('ra')
        ax[0, 1].set_ylabel(r'$\delta dec$')
        ax[1, 0].set_xlabel('dec')
        ax[1, 0].set_ylabel(r'$\delta ra$')
        ax[1, 1].set_xlabel('dec')
        ax[1, 1].set_ylabel(r'$\delta dec$')
        plt.savefig(path_savefig + 'coor_compare', bbox_inches='tight')


        # Field Image
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_hst_gaia, figure=fig, north=True)
        gc.set_xaxis_coord_type('scalar')
        gc.set_yaxis_coord_type('scalar')

        #
        gc.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)

        #
        gc.set_system_latex(True)
        gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
        gc.add_colorbar()
        gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
        gc.colorbar.hide()

        # Hide ticks
        gc.ticks.set_length(30)
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()

        # Markers
        gc.show_markers(c2_hst.ra.value, c2_hst.dec.value,
                        facecolors='none', marker='o', c='none', edgecolors='white', linewidths=0.8, s=100)
        gc.show_markers(ra_gaia, dec_gaia, facecolors='none', marker='o', c='none',
                        edgecolors='red', linewidths=0.8, s=140)
        gc.show_markers(ra_ls, dec_ls, facecolors='none', marker='o', c='none',
                        edgecolors='blue', linewidths=0.8, s=130)
        gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='o', c='none',
                        edgecolors='white', linewidths=0.8, s=120)

        # Labels
        gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
        fig.savefig(path_savefig, bbox_inches='tight')




def FixDatWithAstrometry(cubename=None):
    path_muse_white = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_WHITE.fits'.format(cubename)
    path_muse_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.format(cubename)


    #
    path_dat = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/{}_ESO-DEEP_ZAP.dat'.format(cubename)
    path_dat_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia.dat'.format(cubename)
    path_reg_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia.reg'.format(cubename)
    hdr = fits.open(path_muse_white)[1].header
    hdr_gaia = fits.open(path_muse_white_gaia)[1].header

    #
    dat = ascii.read(path_dat, format='fixed_width')
    ra, dec = dat['ra'], dat['dec']
    sky = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    w = WCS(hdr, naxis=2)
    w_gaia = WCS(hdr_gaia, naxis=2)
    x, y = w.world_to_pixel(sky)
    sky_gaia = w_gaia.pixel_to_world(x, y)
    dat['ra'] = np.round(sky_gaia.ra.value, 6)
    dat['dec'] = np.round(sky_gaia.dec.value, 6)
    dat['name'] = np.asarray(coord.coordstring(dat['ra'], dat['dec']))[2]
    ascii.write(dat, path_dat_gaia, overwrite=True, format='fixed_width')

    # Create regions files
    reg_pre = np.array(['# Region file format: DS9 version 4.1',
                        'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
                        'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                        'fk5'])
    array_like = np.asarray(dat['ra'])
    reg_main = list(map(''.join, zip(np.full_like(array_like, 'circle(', dtype='<U15'),
                                     np.asarray(dat['ra'], dtype=str), np.full_like(array_like, ', ', dtype=str),
                                     np.asarray(dat['dec'], dtype=str), np.full_like(array_like, ', ', dtype=str),
                                     np.asarray(dat['radius'], dtype=str), np.full_like(array_like, '") # text={', dtype='<U15'),
                                     np.asarray(dat['id'], dtype=str), np.full_like(array_like, '}', dtype='<U30'))))
    reg_post = np.hstack((reg_pre, reg_main))
    np.savetxt(path_reg_gaia, reg_post, fmt="%s")

def CopyCurrentObjWithAstrometry(cubename=None):
    # Object files
    path_obj = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_spec1D'.format(cubename)
    path_obj_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia_spec1D'.format(cubename)

    if not os.path.exists(path_obj_gaia):
        shutil.copytree(path_obj, path_obj_gaia)
    else:
        raise ValueError('Folder exists')

    path_dat = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP.dat'.format(cubename)
    path_dat_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia.dat'.format(cubename)
    dat = ascii.read(path_dat, format='fixed_width')
    dat_gaia = ascii.read(path_dat_gaia, format='fixed_width')

    # Rename obj and obj.bkp files
    path_obj_obj = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects.fits'.format(cubename))
    path_obj_bkp = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects_bkp.fits'.format(cubename))
    path_obj_obj_gaia = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_gaia_objects.fits'.format(cubename))
    path_obj_bkp_gaia = os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_gaia_objects_bkp.fits'.format(cubename))

    hdul_obj_obj = fits.open(path_obj_obj)
    hdul_obj_bkp = fits.open(path_obj_bkp)
    hdul_obj_obj[1].data['name'] = dat_gaia['name']
    hdul_obj_obj[1].data['ra'] = dat_gaia['ra']
    hdul_obj_obj[1].data['dec'] = dat_gaia['dec']
    hdul_obj_bkp[1].data['name'] = dat_gaia['name']
    hdul_obj_bkp[1].data['ra'] = dat_gaia['ra']
    hdul_obj_bkp[1].data['dec'] = dat_gaia['dec']
    hdul_obj_obj.writeto(path_obj_obj_gaia, overwrite=True)
    hdul_obj_bkp.writeto(path_obj_bkp_gaia, overwrite=True)
    os.remove(os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects.fits'.format(cubename)))
    os.remove(os.path.join(path_obj_gaia, '{}_ESO-DEEP_ZAP_objects_bkp.fits'.format(cubename)))

    # Rename each file
    for i in range(len(dat)):
        dat_i, dat_gaia_i = dat[i], dat_gaia[i]
        redshift_i = '{}_{}_{}_redshift.fits'.format(dat_i['row'], dat_i['id'], dat_i['name'])
        spec1D_i = '{}_{}_{}_spec1D.fits'.format(dat_i['row'], dat_i['id'], dat_i['name'])
        spec2D_i = '{}_{}_{}_spec2D.fits'.format(dat_i['row'], dat_i['id'], dat_i['name'])
        redshift_f = '{}_{}_{}_redshift.fits'.format(dat_gaia_i['row'], dat_gaia_i['id'], dat_gaia_i['name'])
        spec1D_f = '{}_{}_{}_spec1D.fits'.format(dat_gaia_i['row'], dat_gaia_i['id'], dat_gaia_i['name'])
        spec2D_f = '{}_{}_{}_spec2D.fits'.format(dat_gaia_i['row'], dat_gaia_i['id'], dat_gaia_i['name'])
        os.rename(os.path.join(path_obj_gaia, redshift_i), os.path.join(path_obj_gaia, redshift_f))
        os.rename(os.path.join(path_obj_gaia, spec1D_i), os.path.join(path_obj_gaia, spec1D_f))
        os.rename(os.path.join(path_obj_gaia, spec2D_i), os.path.join(path_obj_gaia, spec2D_f))

#
# FixCubeHeader(cubename='PKS0552-640')


# FixCube(cubename='PKS0552-640')
# FixCube(cubename='HE0439-5254')

# FixAstrometry(cubename='Q0107-0235', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='PB6291', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='HE0153-4520', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='3C57', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='TEX0206-048', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='HE0226-4110', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='PKS0232-04', thr_hst=10, checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='HE0435-5304', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='HE0439-5254', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='PKS0552-640', checkMUSE=True, checkHST=True, update_offset=False)
# FixAstrometry(cubename='Q1354+048', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='LBQS1435-0134', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='PG1522+101', deblend_hst=True, checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='HE1003+0149', checkMUSE=True, checkHST=False)
# FixAstrometry(cubename='PKS0405-123', checkMUSE=True, checkHST=False)


# Fix Dat
# FixDat(cubename='Q0107-0235')
# FixDat(cubename='PB6291')
# FixDat(cubename='HE0153-4520')
# FixDat(cubename='3C57')
# FixDat(cubename='TEX0206-048')
# FixDat(cubename='HE0226-4110')
# FixDat(cubename='PKS0232-04')
# FixDat(cubename='HE0435-5304')
# FixDat(cubename='HE0439-5254')
# FixDat(cubename='PKS0552-640')
# FixDat(cubename='Q1354+048')
# FixDat(cubename='LBQS1435-0134')
# FixDat(cubename='PG1522+101')
# FixDat(cubename='HE1003+0149')
# FixDat(cubename='PKS0405-123')

# Copy
# CopyCurrentObj(cubename='Q0107-0235')


# FixWithAstrometry(cubename='PKS0552-640', FixHST=True, CheckHST=True, CheckMUSE=True, thr_hst=3, thr_muse=3,
#                   deblend_hst=False, deblend_muse=True)
# FixDatWithAstrometry(cubename='PKS0552-640')
CopyCurrentObjWithAstrometry(cubename='PKS0552-640')