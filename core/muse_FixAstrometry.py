import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog
from astropy.convolution import Kernel, convolve, Gaussian2DKernel

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12



def FixAstrometry(cubename=None):
    path_savefig = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}.png'.format(cubename)

    # Load info
    path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    #
    path_gal = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/{}_gal_info.fits'.format(cubename)
    data_gal = fits.getdata(path_gal, 1, ignore_missing_end=True)
    row_gal, ID_gal, z_gal, v_gal = data_gal['row'], data_gal['ID'], data_gal['z'] , data_gal['v']
    name_gal, ql_gal, ra_gal, dec_gal = data_gal['name'], data_gal['ql'], data_gal['ra'], data_gal['dec']

    # Load the image
    path_hb = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/{}_drc_offset.fits'.format(cubename)
    data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

    path_gaia = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/astrometry/GAIA_{}.txt'.format(cubename)
    data_gaia = ascii.read(path_gaia)
    ra_gaia, dec_gaia = np.asarray(data_gaia['ra']), np.asarray(data_gaia['dec'])
    c1 = SkyCoord(ra=ra_gaia * u.degree, dec=dec_gaia * u.degree, frame='icrs')

    bkg_estimator = MedianBackground()
    bkg = Background2D(data_hb, (200, 200), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg = data_hb - bkg.background
    threshold = 1.5 * bkg.background_rms

    kernel = Gaussian2DKernel(1)
    # make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data_bkg, kernel)

    segment_map = detect_sources(data_bkg, threshold, npixels=10)
    cat_hb = SourceCatalog(data_bkg, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid
    header = fits.getheader(path_hb, ext=1)
    w = WCS(header)
    c2 = w.pixel_to_world(x_cen, y_cen)

    # Matching
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)
    print(idx)
    offset_ra, offset_dec = c1.ra - c2.ra[idx], c1.dec - c2.dec[idx]
    plt.figure()
    plt.plot(offset_ra, offset_dec, '.')
    plt.show()


    # Figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc1 = aplpy.FITSFigure(path_hb, figure=fig, north=True)
    gc = aplpy.FITSFigure(path_hb, figure=fig, north=True)
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
    gc.show_markers(ra_qso - np.median(offset_ra.value), dec_qso - np.median(offset_dec.value), facecolors='none', marker='o', c='none',
                    edgecolors='k', linewidths=0.8, s=120)
    gc.show_markers(c1.ra.value, c1.dec.value, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
    gc.add_label(ra_qso - 0.0015, dec_qso, 'QSO', size=10)
    gc.show_markers(ra_gal, dec_gal, marker='o', facecolor='none', c='none',
                    edgecolors=plt.cm.coolwarm(norm(v_gal)),
                    linewidths=1.2, s=80)
    gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)

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



FixAstrometry(cubename='Q0107-0235')
