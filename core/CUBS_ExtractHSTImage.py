import os
import aplpy
import coord
import shutil
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import rc
from astropy import stats
from scipy import interpolate
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.wcs import WCS
from mpdaf.obj import Cube, WaveCoord, Image
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

def ConvertObjtoDat(cubename=None):
    # Only need to run once
    path_obj = '../../MUSEQuBES+CUBS/CUBS_dats/{}_COMBINED_CUBE_MED_FINAL_vac_objects.fits'.format(cubename)
    path_cat_muse = '../../MUSEQuBES+CUBS/CUBS_dats/{}_muse_cat.dat'.format(cubename)

    # Write to .dat
    hdul_obj = fits.open(path_obj)
    data_obj = hdul_obj[1].data
    table = Table()
    table['row'] = data_obj['row']
    table['id'] = data_obj['id']
    table['name'] = data_obj['name']
    table['ra'] = data_obj['ra']
    table['dec'] = data_obj['dec']
    table['radius'] = data_obj['radius']
    table.write(path_cat_muse, format='ascii.fixed_width', overwrite=True)

# Only need to run once and never again.
# ConvertObjtoDat(cubename='J0154-0712')
# ConvertObjtoDat(cubename='J0357-4812')
# ConvertObjtoDat(cubename='J0333-4102')
# ConvertObjtoDat(cubename='J2308-5258')
# ConvertObjtoDat(cubename='J0114-4129')
# ConvertObjtoDat(cubename='J2245-4931')
# ConvertObjtoDat(cubename='J0420-5650')
# ConvertObjtoDat(cubename='J0248-4048')


def extract_hst_image(cubename=None, deblend_hst=True, thr_hst=1):
    # Load HST_MUSE_which should be sci
    path_hst_gaia = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci.fits'.format(cubename)
    hdul_hst_gaia = fits.open(path_hst_gaia)
    data_hst_gaia = hdul_hst_gaia[1].data

    # Make a figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_hst_gaia, figure=fig, north=True, hdu=1)
    gc.set_system_latex(True)
    gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    gc.add_colorbar()
    gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
    gc.colorbar.hide()
    gc.ticks.set_length(30)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()

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
    gc.show_markers(c_hst.ra.value, c_hst.dec.value, facecolors='none', marker='o', c='none', edgecolors='red',
                    linewidths=0.8, s=140)
    path_savefig = '../../MUSEQuBES+CUBS/plots/{}_HST_Continuum.png'.format(cubename)
    fig.savefig(path_savefig, bbox_inches='tight')

    # Save the catalog
    path_cat_hst = '../../MUSEQuBES+CUBS/CUBS_dats/{}_hst_cat.dat'.format(cubename)
    names = ['ID', 'RA', 'DEC']
    ids = np.array([f'HST_{i}' for i in range(len(c_hst))])
    table_hst = Table([ids, c_hst.ra.value, c_hst.dec.value], names=names)
    table_hst.write(path_cat_hst, format='ascii.fixed_width', overwrite=True)


def MergeObjectFiles(cubename=None):
    # Name changes
    if cubename == 'J0110-1648':
        cubename_load = 'Q0110-1648'
    elif cubename == 'J2135-5316':
        cubename_load = 'Q2135-5316'
    elif cubename == 'J0119-2010':
        cubename_load = 'Q0119-2010'
    elif cubename == 'HE0246-4101':
        cubename_load = 'Q0248-4048'
    elif cubename == 'J0028-3305':
        cubename_load = 'Q0028-3305'
    elif cubename == 'HE0419-5657':
        cubename_load = 'Q0420-5650'
    elif cubename == 'PKS2242-498':
        cubename_load = 'Q2245-4931'
    elif cubename == 'PKS0355-483':
        cubename_load = 'Q0357-4812'
    elif cubename == 'HE0112-4145':
        cubename_load = 'Q0114-4129'
    elif cubename == 'J0111-0316':
        cubename_load = 'Q0111-0316'
    elif cubename == 'HE2336-5540':
        cubename_load = 'Q2339-5523'
    elif cubename == 'HE2305-5315':
        cubename_load = 'Q2308-5258'
    elif cubename == 'J0454-6116':
        cubename_load = 'Q0454-6116'
    elif cubename == 'HE0331-4112':
        cubename_load = 'Q0333-4102'
    elif cubename == 'J0154-0712':
        cubename_load = 'Q0154-0712'

    #
    path_cat_hst = '../../MUSEQuBES+CUBS/CUBS_dats/{}_hst_cat.dat'.format(cubename)
    cat_hst = Table.read(path_cat_hst, format='ascii.fixed_width')
    c_hst = SkyCoord(ra=cat_hst['RA'] * u.degree, dec=cat_hst['DEC'] * u.degree, frame='icrs')

    path_cat_muse = '../../MUSEQuBES+CUBS/CUBS_dats/{}_muse_cat.dat'.format(cubename_load)
    cat_muse = Table.read(path_cat_muse, format='ascii.fixed_width')
    row_muse, id_muse, name_muse = cat_muse['row'], cat_muse['id'], cat_muse['name']
    c_muse = SkyCoord(ra=cat_muse['ra'] * u.degree, dec=cat_muse['dec'] * u.degree, frame='icrs')

    # Compare two catalogs and find the common objects within a certain radius (e.g., 1 arcsec)
    idx_muse_hst, d2d_muse_hst, _ = c_muse.match_to_catalog_sky(c_hst)
    sep_constraint = d2d_muse_hst < (0.7 * u.arcsec)

    # HST indices that have a MUSE match
    matched_hst_indices = np.unique(idx_muse_hst[sep_constraint])

    # Build mask in HST space
    hst_mask = np.ones(len(c_hst), dtype=bool)
    hst_mask[matched_hst_indices] = False

    # Unique HST sources (not matched by MUSE)
    c_unique_hst = c_hst[hst_mask]

    # Get the name for the unique hst objects
    ra_str = c_unique_hst.ra.to_string(unit=u.hour, sep='', precision=2, pad=True)
    dec_str = c_unique_hst.dec.to_string(unit=u.deg, sep='', precision=2, alwayssign=True, pad=True)
    name_str = np.array([f"J{r}{d}" for r, d in zip(ra_str, dec_str)])

    # Combine them to a new catalog and save it
    c_combined = SkyCoord(ra=np.concatenate((c_muse.ra.value, c_unique_hst.ra.value)) * u.degree,
                          dec=np.concatenate((c_muse.dec.value, c_unique_hst.dec.value)) * u.degree, frame='icrs')

    # Save the result to a new catalog
    path_cat_combined = '../../MUSEQuBES+CUBS/CUBS_dats/{}_combined_cat.dat'.format(cubename)
    rows = np.concatenate((row_muse, len(row_muse) + 1 + np.arange(len(c_unique_hst))))
    ids = np.concatenate((id_muse, np.array([f'{i}' for i in range(50000, 50000 + len(c_unique_hst))])))
    names = np.concatenate((name_muse, name_str))
    ras, decs = np.round(c_combined.ra.value, 6), np.round(c_combined.dec.value, 6)
    radii = np.full(len(c_combined), 0.6)
    table_combined = Table([rows, ids, names, ras, decs, radii], names=['row', 'id', 'name', 'ra', 'dec', 'radius'])
    table_combined.write(path_cat_combined, format='ascii.fixed_width', overwrite=True)

    # Also save as a region file for visualization in DS9
    path_cat_combined_reg = '../../MUSEQuBES+CUBS/CUBS_dats/{}_combined_cat.reg'.format(cubename)
    with open(path_cat_combined_reg, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 '
                'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write('fk5\n')
        i = 0

        # MUSE objects are marked with green and HST unique objects are marked with red
        # Text their row number in the combined catalog for more info
        for row, ra, dec, radius in zip(rows, ras, decs, radii):
            color = 'green' if i < len(c_unique_hst) else 'red'
            f.write(f'circle({ra}, {dec}, {radius}") # color={color} text={{{row}}}\n')
            i += 1









# def ConvertRegion

# Continuum detections on HST images for CUBS fields besides J0119-2010
# extract_hst_image(cubename='J0110-1648')
# extract_hst_image(cubename='J2135-5316')
# extract_hst_image(cubename='HE0246-4101')
# extract_hst_image(cubename='J0028-3305')
# extract_hst_image(cubename='HE0419-5657')
# extract_hst_image(cubename='PKS2242-498')
# extract_hst_image(cubename='PKS0355-483')
# extract_hst_image(cubename='HE0112-4145')
# extract_hst_image(cubename='J0111-0316')
# extract_hst_image(cubename='HE2336-5540')
# extract_hst_image(cubename='HE2305-5315')
# extract_hst_image(cubename='J0454-6116')
# extract_hst_image(cubename='J0154-0712')
# extract_hst_image(cubename='HE0331-4112')
# extract_hst_image(cubename='J0119-2010') # have no sci exposure

# Merge the HST and MUSE catalogs for CUBS fields besides J0119-2010
# MergeObjectFiles(cubename='J0110-1648')
MergeObjectFiles(cubename='J2135-5316')
MergeObjectFiles(cubename='HE0246-4101')
MergeObjectFiles(cubename='J0028-3305')
MergeObjectFiles(cubename='HE0419-5657')
MergeObjectFiles(cubename='PKS2242-498')
MergeObjectFiles(cubename='PKS0355-483')
MergeObjectFiles(cubename='HE0112-4145')
MergeObjectFiles(cubename='J0111-0316')
MergeObjectFiles(cubename='HE2336-5540')
MergeObjectFiles(cubename='HE2305-5315')
MergeObjectFiles(cubename='J0454-6116')
MergeObjectFiles(cubename='J0154-0712')
MergeObjectFiles(cubename='HE0331-4112')
# extract_hst_image(cubename='J0119-2010') # have no sci exposure


# Regenerate .dat files after visually inspecting the combined catalogs and removing some spurious sources in DS9
