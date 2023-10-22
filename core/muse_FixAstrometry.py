import aplpy
import coord
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

def FixAstrometry(cubename=None, useGAIA=False, thr=1.5, deblend=False, checkHST=False, checkMUSE=False):
    path_savefig = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_offset_gaia.png'.format(cubename)
    path_savefig_MUSE = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_MUSE_WHITE_gaia.png'.format(cubename)

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

    # Segmentation for HST
    bkg_estimator = MedianBackground()
    bkg = Background2D(data_hb, (200, 200), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg = data_hb - bkg.background
    threshold = thr * bkg.background_rms
    kernel = Gaussian2DKernel(1)
    convolved_data = convolve(data_bkg, kernel)
    segment_map = detect_sources(data_bkg, threshold, npixels=10)
    if deblend:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=32, contrast=0.001)
    cat_hb = SourceCatalog(data_bkg, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid
    w = WCS(hdul_hb[1].header)
    c2 = w.pixel_to_world(x_cen, y_cen)

    # Matching
    if useGAIA:
        path_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/GAIA_{}.txt'.format(cubename)
        data_gaia = ascii.read(path_gaia)
        ra_gaia, dec_gaia = np.asarray(data_gaia['ra']), np.asarray(data_gaia['dec'])
        c1 = SkyCoord(ra=ra_gaia * u.degree, dec=dec_gaia * u.degree, frame='icrs')

        idx, d2d, d3d = c1.match_to_catalog_sky(c2)
        offset_ra, offset_dec = c1.ra - c2.ra[idx], c1.dec - c2.dec[idx]
    else:
        idx_qso, d2d_qso, d3d_qso = c_qso.match_to_catalog_sky(c2)
        offset_ra, offset_dec = (c_qso.ra - c2.ra[idx_qso]).value, \
                                (c_qso.dec - c2.dec[idx_qso]).value

    # Save offsets
    path_offset = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/offsets.dat'
    data_offset = ascii.read(path_offset, format='fixed_width')
    data_offset['offset_ra_hst'][data_offset['name'] == cubename] = offset_ra
    data_offset['offset_dec_hst'][data_offset['name'] == cubename] = offset_dec

    path_hb_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_drc_offset_gaia.fits'.format(cubename)
    hdul_hb[1].header['CRVAL1'] += np.median(offset_ra)
    hdul_hb[1].header['CRVAL2'] += np.median(offset_dec)
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
        bkg = Background2D(data_muse, 51, filter_size=(3, 3), bkg_estimator=bkg_estimator, exclude_percentile=20)
    data_muse_bkg = data_muse - bkg.background
    threshold = thr * bkg.background_rms
    kernel = Gaussian2DKernel(1)
    convolved_data = convolve(data_muse_bkg, kernel)
    segment_map = detect_sources(data_muse_bkg, threshold, npixels=10)
    if deblend:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=32, contrast=0.001)
    cat_muse = SourceCatalog(data_muse_bkg, segment_map)
    x_cen, y_cen = cat_muse.xcentroid, cat_muse.ycentroid
    x_cen, y_cen = x_cen[~np.isnan(x_cen)], y_cen[~np.isnan(y_cen)]
    w = WCS(hdul_muse[1].header, naxis=2)
    c2_muse = w.pixel_to_world(x_cen, y_cen)

    idx_qso, d2d_qso, d3d_qso = c_qso.match_to_catalog_sky(c2_muse)
    offset_ra, offset_dec = (c_qso.ra - c2_muse.ra[idx_qso]).value, (c_qso.dec - c2_muse.dec[idx_qso]).value
    # Save MUSE offset
    data_offset['offset_ra_muse'][data_offset['name'] == cubename] = offset_ra
    data_offset['offset_dec_muse'][data_offset['name'] == cubename] = offset_dec
    ascii.write(data_offset, path_offset, overwrite=True, format='fixed_width')


    path_muse_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia.fits'.format(cubename)
    path_muse_white_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_ESO-DEEP_ZAP_gaia_WHITE.fits'.format(cubename)
    hdul_muse[0].header['RA'] += np.median(offset_ra)
    hdul_muse[0].header['DEC'] += np.median(offset_dec)
    hdul_muse[1].header['CRVAL1'] += np.median(offset_ra)
    hdul_muse[1].header['CRVAL2'] += np.median(offset_dec)
    hdul_muse[2].header['CRVAL1'] += np.median(offset_ra)
    hdul_muse[2].header['CRVAL2'] += np.median(offset_dec)
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
        hdul_muse_white[0].header['RA'] += np.median(offset_ra)
        hdul_muse_white[0].header['DEC'] += np.median(offset_dec)
        hdul_muse_white[1].header['CRVAL1'] += np.median(offset_ra)
        hdul_muse_white[1].header['CRVAL2'] += np.median(offset_dec)
        hdul_muse_white[2].header['CRVAL1'] += np.median(offset_ra)
        hdul_muse_white[2].header['CRVAL2'] += np.median(offset_dec)
        hdul_muse_white[1].header.remove('CRDER3')
        hdul_muse_white[2].header.remove('CRDER3')
    hdul_muse_white.writeto(path_muse_white_gaia, overwrite=True)

    # LS
    path_ls = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/LS_{}.txt'.format(cubename)
    data_ls = ascii.read(path_ls)
    ra_ls, dec_ls = np.asarray(data_ls['ra']), np.asarray(data_ls['dec'])

    if checkMUSE:
        # Figure
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
                        edgecolors='blue', linewidths=0.8, s=130)
        gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='o', c='none',
                        edgecolors='white', linewidths=0.8, s=120)
        gc.show_markers(ra_gal + np.median(offset_ra), dec_gal + np.median(offset_dec),
                        facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)

        # Labels
        gc.add_label(0.87, 0.97, r'$\mathrm{MUSE}$', color='k', size=15, relative=True)
        fig.savefig(path_savefig_MUSE, bbox_inches='tight')

    # Figure
    if checkHST:
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc1 = aplpy.FITSFigure(path_hb_gaia, figure=fig, north=True)
        gc = aplpy.FITSFigure(path_hb_gaia, figure=fig, north=True)
        gc.set_xaxis_coord_type('scalar')
        gc.set_yaxis_coord_type('scalar')
        gc1.set_xaxis_coord_type('scalar')
        gc1.set_yaxis_coord_type('scalar')

        #
        gc.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)
        gc1.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)  # 0.02 / 0.01 40''

        #
        gc.set_system_latex(True)
        gc1.set_system_latex(True)
        gc1.show_colorscale(cmap='coolwarm', vmin=-1000, vmax=1000)
        gc1.hide_colorscale()
        gc1.add_colorbar()
        gc1.colorbar.set_box([0.15, 0.145, 0.38, 0.02], box_orientation='horizontal')
        gc1.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        gc1.colorbar.set_axis_label_font(size=12)
        gc1.colorbar.set_axis_label_pad(-40)
        gc1.colorbar.set_location('bottom')
        gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
        gc.add_colorbar()
        gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
        gc.colorbar.hide()

        # Hide ticks
        gc.ticks.set_length(30)
        gc1.ticks.set_length(30)
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()
        gc1.ticks.hide()
        gc1.tick_labels.hide()
        gc1.axis_labels.hide()
        norm = mpl.colors.Normalize(vmin=-1000, vmax=1000)

        # Markers
        gc.show_markers(ra_ls, dec_ls, facecolors='none', marker='o', c='none',
                        edgecolors='blue', linewidths=0.8, s=130)
        gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='o', c='none',
                        edgecolors='white', linewidths=0.8, s=120)
        gc.add_label(ra_qso - 0.0015, dec_qso, 'QSO', size=10)
        gc.show_markers(ra_gal + np.median(offset_ra), dec_gal + np.median(offset_dec), marker='o',
                        facecolor='none', c='none',
                        edgecolors=plt.cm.coolwarm(norm(v_gal)),
                        linewidths=1.2, s=80)
        gc.show_markers(ra_gal + np.median(offset_ra), dec_gal + np.median(offset_dec),
                        facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)

        # Labels
        # xw, yw = 40.1231559, -18.8580071
        # gc.show_arrows(xw, yw, -0.0001 * yw, 0, color='k')
        # gc.show_arrows(xw, yw, 0, -0.0001 * yw, color='k')
        # gc.add_label(0.985, 0.85, r'N', size=15, relative=True)
        # gc.add_label(0.89, 0.748, r'E', size=15, relative=True)
        gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
        # gc.add_label(0.27, 0.86, r"$\rm MUSE \, 1'\times 1' \, FoV$", size=15, relative=True, rotation=60)
        # gc.add_label(0.47, 0.30, r"$\rm 30'' \times 30''$", size=15, relative=True)
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

def FixCurrentObj(cubename=None):
    #
    path_obj =

#
# FixAstrometry(cubename='Q0107-0235', checkMUSE=True)
# FixAstrometry(cubename='PB6291', checkMUSE=True)
# FixAstrometry(cubename='HE0153-4520', checkMUSE=True)
# FixAstrometry(cubename='3C57', checkMUSE=True)
# FixAstrometry(cubename='TEX0206-048', checkMUSE=True)
# FixAstrometry(cubename='HE0226-4110', checkMUSE=True)
# FixAstrometry(cubename='PKS0232-04', thr=10, checkMUSE=True)
# FixAstrometry(cubename='HE0435-5304', checkMUSE=True)
# FixAstrometry(cubename='HE0439-5254', checkMUSE=True)
# FixAstrometry(cubename='PKS0552-640', checkMUSE=True)
# FixAstrometry(cubename='Q1354+048', checkMUSE=True)
# FixAstrometry(cubename='LBQS1435-0134', checkMUSE=True)
# FixAstrometry(cubename='PG1522+101', deblend=True, checkMUSE=True)
# FixAstrometry(cubename='HE1003+0149', checkMUSE=True)
# FixAstrometry(cubename='PKS0405-123', checkMUSE=True)


# Fix Dat
FixDat(cubename='Q0107-0235')
FixDat(cubename='PB6291')
FixDat(cubename='HE0153-4520')
FixDat(cubename='3C57')
FixDat(cubename='TEX0206-048')
FixDat(cubename='HE0226-4110')
FixDat(cubename='PKS0232-04')
FixDat(cubename='HE0435-5304')
FixDat(cubename='HE0439-5254')
FixDat(cubename='PKS0552-640')
FixDat(cubename='Q1354+048')
FixDat(cubename='LBQS1435-0134')
FixDat(cubename='PG1522+101')
FixDat(cubename='HE1003+0149')
FixDat(cubename='PKS0405-123')
