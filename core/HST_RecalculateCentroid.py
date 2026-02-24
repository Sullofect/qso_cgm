import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import lacosmic
from matplotlib import rc
from astropy.wcs import WCS
from scipy.stats import norm
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
from scipy.optimize import minimize
from muse_compare_z import compare_z
from muse_RenameGal import ReturnGalLabel
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

def RemoveCosmicRays(cubename=None):
    # Load HST
    path_hst_gaia = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)
    path_hst_gaia_cosmic = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia_nocr.fits'.format(cubename)
    hdul_hst_gaia = fits.open(path_hst_gaia)
    data_hst_gaia = hdul_hst_gaia[1].data
    error_hst_gaia = 1 / np.sqrt(hdul_hst_gaia[2].data)

    # Remove cosmic rays
    # data_cosmic = lacosmic.lacosmic(data_hst_gaia, 1, 1.5, 0.1, error=error_hst_gaia)
    data_cosmic = lacosmic.lacosmic(data_hst_gaia, 5, 2, 1, effective_gain=1200, readnoise=5, maxiter=2)

    # data_bkg = data_cosmic - bkg.background
    # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax[0].imshow(data_cosmic[0], cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    # ax[1].imshow(data_cosmic[1], cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    # ax[1].imshow(data_hst_gaia / error_hst_gaia, cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    # plt.show()
    # raise ValueError("Check the cosmic ray removal")

    hdul_hst_gaia[1].data = data_cosmic[0]
    hdul_hst_gaia.writeto(path_hst_gaia_cosmic, overwrite=True)

def RecalculateCentroid(cubename=None, deblend_hst=False, thr_hst=3, WhichHST='ori'):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Load the galaxy catalog
    filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    t = Table.read(filename)
    ra_muse = t['ra']
    dec_muse = t['dec']
    c_muse = SkyCoord(ra=ra_muse * u.degree, dec=dec_muse * u.degree, frame='icrs')

    # Load HST
    if WhichHST == 'ori':
        path_hst_gaia = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)
        hdul_hst_gaia = fits.open(path_hst_gaia)
        data_hst_gaia = hdul_hst_gaia[1].data
    elif WhichHST == 'nocr':
        path_hst_gaia = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia_nocr.fits'.format(cubename)
        hdul_hst_gaia = fits.open(path_hst_gaia)
        data_hst_gaia = hdul_hst_gaia[1].data

    # Make a figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_hst_gaia, figure=fig, north=True)
    gc.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)
    gc.set_system_latex(True)
    gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    gc.add_colorbar()
    gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
    gc.colorbar.hide()
    gc.ticks.set_length(30)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()

    # Save the result
    filename_txt = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.dat'.format(cubename)
    if os.path.exists(filename_txt):
        # Rewrite the .fits file to keep it consistent
        t = Table.read(filename_txt, format='ascii.fixed_width')
        t.write(filename, format='fits', overwrite=True)
        gc.show_markers(t['ra'], t['dec'], facecolors='none', marker='o', c='none',
                        edgecolors='black', linewidths=0.8, s=140)
        gc.show_markers(t['ra_HST'], t['dec_HST'], facecolors='none', marker='o', c='none',
                        edgecolors='purple', linewidths=0.8, s=100)
    else:
        print('No galaxy catalog found for {}. Please run the script to create it.'.format(cubename))
        # Segmentation for HST
        bkg_estimator = MedianBackground()
        bkg = Background2D(data_hst_gaia, (200, 200), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        data_bkg = data_hst_gaia - bkg.background
        threshold = thr_hst * bkg.background_rms

        kernel = Gaussian2DKernel(3)
        convolved_data = convolve(data_bkg, kernel)
        segment_map = detect_sources(convolved_data, threshold, npixels=3)
        if deblend_hst:
            segment_map = deblend_sources(convolved_data, segment_map, npixels=3, nlevels=32, contrast=0.001)
        cat_hb = SourceCatalog(convolved_data, segment_map)
        x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid
        w = WCS(hdul_hst_gaia[1].header)
        c_hst = w.pixel_to_world(x_cen, y_cen)

        idx_muse_hst, d2d_muse_hst, _ = c_muse.match_to_catalog_sky(c_hst)
        sep_constraint = d2d_muse_hst < (0.5 * u.arcsec)
        c_muse_hst = c_muse[sep_constraint]
        c_hst_muse = c_hst[idx_muse_hst[sep_constraint]]
        c_combined = c_muse.copy()
        c_combined[sep_constraint] = c_hst_muse
        print(len(ra_muse), len(c_hst_muse.ra))
        t['ra_HST'] = c_combined.ra
        t['dec_HST'] = c_combined.dec
        t.write(filename, format='fits', overwrite=True)
        t.write(filename_txt, format='ascii.fixed_width', overwrite=True)
        gc.show_markers(ra_muse, dec_muse, facecolors='none', marker='o', c='none',
                        edgecolors='red', linewidths=0.8, s=140)
        gc.show_markers(c_hst_muse.ra.value, c_hst_muse.dec.value,
                        facecolors='none', marker='o', c='none', edgecolors='white', linewidths=0.8, s=100)
        gc.show_markers(c_combined.ra.value, c_combined.dec.value,
                        facecolors='none', marker='o', c='none', edgecolors='blue', linewidths=0.8, linestyle='--', s=100)
    for i in range(len(t['ra'])):
        gc.add_label(t['ra'][i], t['dec'][i], t['row'][i], color='red', size=12, relative=False)
    gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
    path_savefig = '../../MUSEQuBES+CUBS/plots/{}_HST_FixCentroid.png'.format(cubename)
    fig.savefig(path_savefig, bbox_inches='tight')


def CentroidHE0238(cubename='HE0238-1904'):
    # Load the MUSE catalog
    _, _, _, _, ra_hst, dec_hst, _ = ReturnGalLabel(sort_row=False, mode='initial', return_HST=True, return_bins=True)
    path_hst = '../../MUSEQuBES+CUBS/datacubes/{}_drc_offset.fits'.format(cubename)
    path_hst_gaia = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)

    c = SkyCoord(ra=ra_hst * u.deg, dec=dec_hst * u.deg, frame='icrs')
    w = WCS(fits.open(path_hst)[1].header, naxis=2)
    w_gaia = WCS(fits.open(path_hst_gaia)[1].header, naxis=2)
    x, y = w.world_to_pixel(c)
    c_gaia = w_gaia.pixel_to_world(x, y)

    # Load the galaxy catalog
    filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    filename_txt = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.dat'.format(cubename)
    t = Table.read(filename)
    t['ra_HST'] = c_gaia.ra
    t['dec_HST'] = c_gaia.dec
    if not os.path.exists(filename_text):
        t.write(filename, format='fits', overwrite=True)
        t.write(filename_txt, format='ascii.fixed_width', overwrite=True)

# Remove cosmic rays
# RemoveCosmicRays(cubename='HE0435-5304')
# RemoveCosmicRays(cubename='HE0153-4520')
# RemoveCosmicRays(cubename='HE0226-4110')
# RemoveCosmicRays(cubename='PKS0405-123')
# RemoveCosmicRays(cubename='HE0226-4110')
# RemoveCosmicRays(cubename='LBQS1435-0134')

# RecalculateCentroid(cubename='HE0435-5304', WhichHST='nocr', thr_hst=0.9) # Done, check emi/abs done 1. both 2. emi
# RecalculateCentroid(cubename='HE0153-4520') # no group member, check emi/abs done
# RecalculateCentroid(cubename='HE0226-4110', deblend_hst=True, thr_hst=0.9) # check emi/abs done
# RecalculateCentroid(cubename='PKS0405-123', deblend_hst=True, thr_hst=0.1) # check emi/abs done
# CentroidHE0238()
# RecalculateCentroid(cubename='HE0238-1904', deblend_hst=True, thr_hst=0.9) # check emi/abs done
# RecalculateCentroid(cubename='3C57', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='PKS0552-640', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='PB6291', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='Q0107-0235', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE0439-5254', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE1003+0149', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='TEX0206-048', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='Q1354+048', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='LBQS1435-0134', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='PG1522+101', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='PKS0232-04', deblend_hst=True, thr_hst=1.5) # check emi/abs done


# CUBS
# RecalculateCentroid(cubename='J0454-6116', deblend_hst=True, thr_hst=0.9) # check emi/abs done
# RecalculateCentroid(cubename='J2135-5316', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE0246-4101', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='J0028-3305', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE0419-5657', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='PKS2242-498', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE2305-5315', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE0331-4112', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='J0154-0712', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='J0110-1648', deblend_hst=True, thr_hst=1.5) # check emi/abs done


# Auto correction from STScI
# RecalculateCentroid(cubename='HE2336-5540', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='J0119-2010', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='PKS0355-483', deblend_hst=True, thr_hst=1.5) # check emi/abs done
# RecalculateCentroid(cubename='HE0112-4145', deblend_hst=True, thr_hst=1.5) # check emi/abs done

