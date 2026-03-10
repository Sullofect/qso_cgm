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
from astropy.nddata import Cutout2D
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

def FixHeader(cubename='J0454-6116'):
    #
    path_hst_gaia_astro = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci_astro.fits'.format(cubename)
    hdul_hst_gaia_astro = fits.open(path_hst_gaia_astro)

    path_hst_gaia = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci.fits'.format(cubename)
    path_hst_gaia_new = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci_new.fits'.format(cubename)
    hdul_hst_gaia = fits.open(path_hst_gaia)


    try:
        print(hdul_hst_gaia_astro[1].header['PC2_1'])
    except KeyError:
        print('no rotation copying PC1_1 to PC2_2')
        hdul_hst_gaia_astro[1].header.append('PC2_1', 'PC1_2', 'PC2_2')
        hdul_hst_gaia_astro[1].header['PC2_1'] = 0
        hdul_hst_gaia_astro[1].header['PC1_2'] = 0
        hdul_hst_gaia_astro[1].header['PC2_2'] = -1 * hdul_hst_gaia_astro[1].header['PC1_1']

    hdul_hst_gaia[1].header['CD1_1'] = hdul_hst_gaia_astro[1].header['PC1_1'] * hdul_hst_gaia_astro[1].header['CDELT1']
    hdul_hst_gaia[1].header['CD2_1'] = hdul_hst_gaia_astro[1].header['PC2_1'] * hdul_hst_gaia_astro[1].header['CDELT2']
    hdul_hst_gaia[1].header['CD1_2'] = hdul_hst_gaia_astro[1].header['PC1_2'] * hdul_hst_gaia_astro[1].header['CDELT1']
    hdul_hst_gaia[1].header['CD2_2'] = hdul_hst_gaia_astro[1].header['PC2_2'] * hdul_hst_gaia_astro[1].header['CDELT2']

    hdul_hst_gaia[1].header['CRVAL1'] = hdul_hst_gaia_astro[1].header['CRVAL1']
    hdul_hst_gaia[1].header['CRVAL2'] = hdul_hst_gaia_astro[1].header['CRVAL2']
    hdul_hst_gaia[1].header['CRPIX1'] = hdul_hst_gaia_astro[1].header['CRPIX1']
    hdul_hst_gaia[1].header['CRPIX2'] = hdul_hst_gaia_astro[1].header['CRPIX2']

    # Save the result
    hdul_hst_gaia.writeto(path_hst_gaia_new, overwrite=True)

# Only need to use once
# FixHeader(cubename='J0454-6116')

def extract_hst_image(cubename=None, deblend_hst=True, thr_hst=1):
    # Load HST_MUSE_which should be sci
    if cubename == "J0119-2010":
        path_hst_gaia = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia.fits'.format(cubename)
    else:
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

    if cubename == "J0119-2010":
        # Select range 1743: 3321, 1409: 3017
        mask = (y_cen > 1743) & (y_cen < 3321) & (x_cen > 1409) & (x_cen < 3017)
        x_cen = x_cen[mask]
        y_cen = y_cen[mask]

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
            color = 'green' if i < len(c_muse) else 'red'
            f.write(f'circle({ra}, {dec}, {radius}") # color={color} text={{{row}}}\n')
            i += 1


def UpdateDatFromRegion(cubename=None):
    path_cat_reg_ac = '../../MUSEQuBES+CUBS/CUBS_dats/{}_combined_cat_ac.reg'.format(cubename)
    regions = Regions.read(path_cat_reg_ac, format='ds9')
    data = []
    for r in regions:
        ra = r.center.ra.deg
        dec = r.center.dec.deg
        radius = r.radius.to("arcsec").value
        label = int(r.meta.get("text"))
        data.append([label, ra, dec, radius])
    data = np.array(data)

    # catalog path
    path_cat = '../../MUSEQuBES+CUBS/CUBS_dats/{}_combined_cat.dat'.format(cubename)
    cat = Table.read(path_cat, format='ascii.fixed_width')
    row, id, name, radius = cat['row'], cat['id'], cat['name'], cat['radius']

    # match with the same row number and use updated positions and radii from the region file
    idx_cat = np.isin(row, data[:, 0])
    row_updated = 1 + np.arange(len(row[idx_cat]))
    id_updated = id[idx_cat]
    ra_updated = np.round(data[:, 1], 6)
    dec_updated = np.round(data[:, 2], 6)
    radius_updated = radius[idx_cat]

    # Recalculate name given the coordinates
    c_updated = SkyCoord(ra=ra_updated * u.degree, dec=dec_updated * u.degree, frame='icrs')
    ra_str = c_updated.ra.to_string(unit=u.hour, sep='', precision=2, pad=True)
    dec_str = c_updated.dec.to_string(unit=u.deg, sep='', precision=2, alwayssign=True, pad=True)
    name_updated = np.array([f"J{r}{d}" for r, d in zip(ra_str, dec_str)])

    # Determine which name to use

    if cubename == 'J2135-5316':
        cubename_save = 'Q2135-5316'
    elif cubename == 'J0454-6116':
        cubename_save = 'Q0454-6116'
    elif cubename == 'J0119-2010':
        cubename_save = 'Q0119-2010'
    elif cubename == 'HE0246-4101':
        cubename_save = 'Q0248-4048'
    elif cubename == 'HE2336-5540':
        cubename_save = 'Q2339-5523'
    else:
        cubename_save = cubename

    # Generate .dat files
    path_cat_updated = '../../MUSEQuBES+CUBS/CUBS_dats/{}_eso_coadd_nosky_sub_ZAP.dat'.format(cubename_save)
    table_updated = Table([row_updated, id_updated, name_updated, ra_updated, dec_updated, radius_updated],
                            names=['row', 'id', 'name', 'ra', 'dec', 'radius'])
    table_updated.write(path_cat_updated, format='ascii.fixed_width', overwrite=True)


def CopyObjectFile(cubename=None):
    # Copy the object file to the CUBS_dats folder for easier access
    path_obj_src = '../../MUSEQuBES+CUBS/CUBS_redshifting_galaxies/{}/{}_COMBINED_CUBE_MED_FINAL_vac_spec1D' \
                   '/{}_COMBINED_CUBE_MED_FINAL_vac_objects.fits'.format(cubename, cubename, cubename)

    path_obj_dst = '../../MUSEQuBES+CUBS/CUBS_cubes/{}_eso_coadd_nosky_sub_ZAP_spec1D/' \
                   '{}_eso_coadd_nosky_sub_ZAP_objects.fits'.format(cubename, cubename)

    # path_obj_test = '../../MUSEQuBES+CUBS/CUBS_cubes/{}_eso_coadd_nosky_sub_ZAP_spec1D/' \
    #                '{}_eso_coadd_nosky_sub_ZAP_objects_test.fits'.format(cubename, cubename)

    # Load two .fits file
    hdul_src = fits.open(path_obj_src)
    hdul_dst = fits.open(path_obj_dst)

    # find the rows with the same id and same name
    id_src = hdul_src[1].data['id']
    name_src = hdul_src[1].data['name']
    id_dst = hdul_dst[1].data['id']
    name_dst = hdul_dst[1].data['name']

    # build keys (pairing id+name row-by-row)
    key_src = np.array([f"{i}||{n}" for i, n in zip(id_src, name_src)])
    key_dst = np.array([f"{i}||{n}" for i, n in zip(id_dst, name_dst)])

    # aligned matches
    # row numbers (aligned)
    _, idx_src, idx_dst = np.intersect1d(key_src, key_dst, return_indices=True)

    # copy rows (row-by-row avoids the FITS_rec bulk-assignment TypeError)
    for s, d in zip(idx_src, idx_dst):
        hdul_dst[1].data[d] = hdul_src[1].data[s]

    # Save the updated destination file
    hdul_dst.writeto(path_obj, overwrite=True)






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
# extract_hst_image(cubename='J0119-2010', thr_hst=1) # have no sci exposure

# Merge the HST and MUSE catalogs for CUBS fields besides J0119-2010
# MergeObjectFiles(cubename='J0110-1648')
# MergeObjectFiles(cubename='J2135-5316')
# MergeObjectFiles(cubename='HE0246-4101')
# MergeObjectFiles(cubename='J0028-3305')
# MergeObjectFiles(cubename='HE0419-5657')
# MergeObjectFiles(cubename='PKS2242-498')
# MergeObjectFiles(cubename='PKS0355-483')
# MergeObjectFiles(cubename='HE0112-4145')
# MergeObjectFiles(cubename='J0111-0316')
# MergeObjectFiles(cubename='HE2336-5540')
# MergeObjectFiles(cubename='HE2305-5315')
# MergeObjectFiles(cubename='J0454-6116')
# MergeObjectFiles(cubename='J0154-0712')
# MergeObjectFiles(cubename='HE0331-4112')
# MergeObjectFiles(cubename='J0119-2010') # have no sci exposure

# Regenerate .dat files after visually inspecting the combined catalogs and removing some spurious sources in DS9
# Only need to run one time after the visual inspection and cleaning
# UpdateDatFromRegion(cubename='J0110-1648')

#
# UpdateDatFromRegion(cubename='J2135-5316')
# UpdateDatFromRegion(cubename='HE0246-4101')
# UpdateDatFromRegion(cubename='J0028-3305')
# UpdateDatFromRegion(cubename='HE0419-5657')
# UpdateDatFromRegion(cubename='PKS2242-498')
# UpdateDatFromRegion(cubename='PKS0355-483')
# UpdateDatFromRegion(cubename='HE0112-4145')
# UpdateDatFromRegion(cubename='J0111-0316')
# UpdateDatFromRegion(cubename='HE2336-5540')
# UpdateDatFromRegion(cubename='HE2305-5315')
# UpdateDatFromRegion(cubename='J0454-6116')
# UpdateDatFromRegion(cubename='J0154-0712')
# UpdateDatFromRegion(cubename='HE0331-4112')
# UpdateDatFromRegion(cubename='J0119-2010')


# Copy the object file to the CUBS_dats folder for easier access
# Only need to run one time
# CopyObjectFile(cubename='J0110-1648')