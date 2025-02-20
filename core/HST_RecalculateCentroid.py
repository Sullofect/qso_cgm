import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from regions import Regions
from scipy.stats import norm
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
from scipy.optimize import minimize
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


def RecalculateCentroid(cubename=None, deblend_hst=False, thr_hst=3):
    # Load the galaxy catalog
    filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    t = Table.read(filename)
    ra_muse = t['ra']
    dec_muse = t['dec']
    c_muse = SkyCoord(ra=ra_muse * u.degree, dec=dec_muse * u.degree, frame='icrs')

    # Load HST
    path_hst_gaia = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)
    hdul_hst_gaia = fits.open(path_hst_gaia)
    data_hst_gaia = hdul_hst_gaia[1].data

    # Segmentation for HST
    bkg_estimator = MedianBackground()
    bkg = Background2D(data_hst_gaia, (200, 200), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg = data_hst_gaia - bkg.background
    threshold = thr_hst * bkg.background_rms
    kernel = Gaussian2DKernel(3)
    convolved_data = convolve(data_bkg, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=5)
    if deblend_hst:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=5, nlevels=32, contrast=0.001)
    cat_hb = SourceCatalog(convolved_data, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid
    w = WCS(hdul_hst_gaia[1].header)
    c_hst = w.pixel_to_world(x_cen, y_cen)

    # idx_hst_muse, d2d_hst_muse, d3d_hst_muse = c2_hst.match_to_catalog_sky(c_muse)
    # sep_constraint = d2d_hst_muse < (0.5 * u.arcsec)
    # c2_hst_muse = c2_hst[sep_constraint]
    # c_muse_hst = c_muse[idx_hst_muse[sep_constraint]]

    idx_muse_hst, d2d_muse_hst, _ = c_muse.match_to_catalog_sky(c_hst)
    sep_constraint = d2d_muse_hst < (1.0 * u.arcsec)
    c_muse_hst = c_muse[sep_constraint]
    c_hst_muse = c_hst[idx_muse_hst[sep_constraint]]

    print(len(ra_muse), len(c_hst_muse.ra))


    t['ra_HST'] = c_hst_muse.ra
    t['dec_HST'] = c_hst_muse.dec

    t.write(filename, format='fits', overwrite=True)

# RecalculateCentroid(cubename='HE0435-5304') # Done
# RecalculateCentroid(cubename='PKS0232-04', deblend_hst=True, thr_hst=1.5) # MUSE centroid is better
RecalculateCentroid(cubename='Q1354+048', deblend_hst=True, thr_hst=1.5)
# RecalculateCentroid(cubename='PKS0405-123', deblend_hst=True, thr_hst=0.1) # not done
# RecalculateCentroid(cubename='PKS0552-640')  # not done
# MakeFieldImage(cubename='Q0107-0235')
# MakeFieldImage(cubename='PB6291')
# MakeFieldImage(cubename='HE0153-4520')
# MakeFieldImage(cubename='3C57')
# MakeFieldImage(cubename='TEX0206-048')
# MakeFieldImage(cubename='HE0226-4110')
# MakeFieldImage(cubename='PKS0232-04')
# MakeFieldImage(cubename='HE0439-5254')
# MakeFieldImage(cubename='LBQS1435-0134')
# MakeFieldImage(cubename='PG1522+101')
# MakeFieldImage(cubename='HE1003+0149')
# MakeFieldImage(cubename='HE0238-1904')