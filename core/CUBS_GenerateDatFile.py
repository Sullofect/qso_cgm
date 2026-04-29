import os
import aplpy
import coord
import shutil
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy import stats
from astropy.wcs import WCS
from regions import Regions
from astropy.io import ascii
from scipy import interpolate
from astropy import units as u
from astropy.table import Table
from astropy.nddata import Cutout2D
from mpdaf.obj import Cube, WaveCoord, Image
from astropy.coordinates import SkyCoord
from astropy.table import Column
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
from astropy.convolution import Kernel, convolve, Gaussian2DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12

object_aliases = {"J0110-1648": ["Q0110-1648"],
                  "J0454-6116": ["Q0454-6116"],
                  "J2135-5316": ["Q2135-5316"],
                  "J0119-2010": ["Q0119-2010"],
                  "HE0246-4101": ["Q0248-4048", "J0248-4048"],
                  "J0028-3305": ["Q0028-3305"],
                  "HE0419-5657": ["Q0420-5650", "J0420-5650"],
                  "PKS2242-498": ["Q2245-4931", "J2245-4931"],
                  "PKS0355-483": ["Q0357-4812", "J0357-4812"],
                  "HE0112-4145": ["Q0114-4129", "J0114-4129"],
                  "HE2305-5315": ["Q2308-5258", "J2308-5258"],
                  "HE0331-4112": ["Q0333-4102", "J0333-4102"],
                  "J0111-0316": ["Q0111-0316"],
                  "J0154-0712": ["Q0154-0712"],
                  "HE2336-5540": ["Q2339-5523", "J2339-5523"],
}

def find_existing_file(candidates, template):
    for name in candidates:
        path = template.format(name)
        if os.path.exists(path):
            return path
    return None

def extract_hst_image(cubename=None, deblend_hst=True, thr_hst=1):
    # Load HST_MUSE_which should be sci
    if cubename == "J0119-2010":
        path_hst_gaia = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_astro.fits'.format(cubename)
    else:
        path_hst_gaia = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci_astro.fits'.format(cubename)
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
    print('shape {}'.format(np.shape(bkg.background_rms)))
    print('5-sigma threshold: {} in count/s'.format(5 * np.median(bkg.background_rms)))
    kernel = Gaussian2DKernel(3)
    convolved_data = convolve(data_bkg, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=3)
    if deblend_hst:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=3, nlevels=32, contrast=0.001)

    # Use original image for source detection
    cat_hb = SourceCatalog(data_hst_gaia, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid

    # Calculate total counts for each source and magnitude
    segment_flux = cat_hb.segment_flux
    ZP = -2.5 * np.log10(hdul_hst_gaia[1].header['PHOTFLAM']) - 21.10 \
         - 5 * np.log10(hdul_hst_gaia[1].header['PHOTPLAM']) + 18.6921
    segment_mag = -2.5 * np.log10(segment_flux) + ZP

    # Select range 1743: 3321, 1409: 3017
    if cubename == "J0119-2010":
        mask = (y_cen > 1743) & (y_cen < 3321) & (x_cen > 1409) & (x_cen < 3017)
        x_cen = x_cen[mask]
        y_cen = y_cen[mask]
        segment_flux = segment_flux[mask]
        segment_mag = segment_mag[mask]

    w = WCS(hdul_hst_gaia[1].header)
    c_hst = w.pixel_to_world(x_cen, y_cen)
    gc.show_markers(c_hst.ra.value, c_hst.dec.value, facecolors='none', marker='o', c='none', edgecolors='red',
                    linewidths=0.8, s=140)
    path_savefig = '../../MUSEQuBES+CUBS/plots/{}_HST_Continuum.png'.format(cubename)
    fig.savefig(path_savefig, bbox_inches='tight')

    # Save the catalog
    path_cat_hst = '../../MUSEQuBES+CUBS/CUBS_dats/{}_hst_cat_astro.dat'.format(cubename)
    names = ['ID', 'RA', 'DEC', 'TOTAL_COUNTS', 'MAG']
    ids = np.array([f'HST_{i}' for i in range(len(c_hst))])
    table_hst = Table([ids, c_hst.ra.value, c_hst.dec.value, segment_flux, segment_mag], names=names)
    table_hst.write(path_cat_hst, format='ascii.fixed_width', overwrite=True)

def extract_muse_image(cubename=None, deblend_muse=True, thr_muse=1.25):
    name_candidates = [cubename] + [a for a in object_aliases.get(cubename, []) if a != cubename]
    print(name_candidates)

    # Path MUSE
    if cubename == 'J0454-6116':
        path_muse_gaia = "../../MUSEQuBES+CUBS/CUBS_cubes_gaia/Q0454-6116_eso_coadd_nc_nosky_sub_ZAP_WHITE_astro.fits"
    else:
        path_muse_gaia = find_existing_file(name_candidates,
                                            '../../MUSEQuBES+CUBS/CUBS_cubes_gaia/{}_eso_coadd_nosky_sub_ZAP_WHITE_astro.fits')
    hdul_muse_gaia = fits.open(path_muse_gaia)
    data_muse_gaia = hdul_muse_gaia[1].data

    # Make a figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc = aplpy.FITSFigure(path_muse_gaia, figure=fig, north=True, hdu=1)
    gc.set_system_latex(True)
    gc.show_colorscale(cmap='Greys', vmin=np.nanpercentile(data_muse_gaia, 5),
                       vmax=np.nanpercentile(data_muse_gaia, 98))
    gc.add_colorbar()
    gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
    gc.colorbar.hide()
    gc.ticks.set_length(30)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()

    # Segmentation for HST
    bkg_estimator = MedianBackground()
    bkg = Background2D(data_muse_gaia, (64, 64), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg = data_muse_gaia - bkg.background
    threshold = thr_muse * bkg.background_rms
    print('shape {}'.format(np.shape(bkg.background_rms)))
    print('5-sigma threshold: {} in count/s'.format(5 * np.median(bkg.background_rms)))

    kernel = Gaussian2DKernel(1)
    convolved_data = convolve(data_bkg, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=5)
    if deblend_muse:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=3, nlevels=32, contrast=0.001)

    cat_hb = SourceCatalog(data_muse_gaia, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid

    #
    w = WCS(hdul_muse_gaia[1].header)
    c_muse = w.pixel_to_world(x_cen, y_cen)
    gc.show_markers(c_muse.ra.value, c_muse.dec.value, facecolors='none', marker='o', c='none', edgecolors='blue',
                    linewidths=0.8, s=140)
    path_savefig = '../../MUSEQuBES+CUBS/plots/{}_MUSE_Continuum.png'.format(cubename)
    fig.savefig(path_savefig, bbox_inches='tight')

    # Save the catalog
    path_cat_muse = '../../MUSEQuBES+CUBS/CUBS_dats/{}_muse_cat_astro.dat'.format(cubename)
    names = ['ID', 'RA', 'DEC']
    ids = np.array([f'MUSE_{i}' for i in range(len(c_muse))])
    table_hst = Table([ids, c_muse.ra.value, c_muse.dec.value], names=names)
    table_hst.write(path_cat_muse, format='ascii.fixed_width', overwrite=True)

def MergeDatFiles(cubename=None):
    name_candidates = [cubename] + [a for a in object_aliases.get(cubename, []) if a != cubename]
    print(name_candidates)

    # Path MUSE
    if cubename == 'J0454-6116':
        path_dat = "../../MUSEQuBES+CUBS/CUBS_cubes/Q0454-6116_eso_coadd_nc_nosky_sub_ZAP.dat"
    else:
        path_dat = find_existing_file(name_candidates,
                                      '../../MUSEQuBES+CUBS/CUBS_cubes/{}_eso_coadd_nosky_sub_ZAP.dat')
    path_cat_hst = '../../MUSEQuBES+CUBS/CUBS_dats/{}_hst_cat_astro.dat'.format(cubename)
    path_cat_muse = '../../MUSEQuBES+CUBS/CUBS_dats/{}_muse_cat_astro.dat'.format(cubename)

    # Read
    table_dat = Table.read(path_dat, format='ascii.fixed_width')
    table_hst = Table.read(path_cat_hst, format='ascii.fixed_width')
    table_muse = Table.read(path_cat_muse, format='ascii.fixed_width')

    # Build coordinates for updated catalogs
    coord = SkyCoord(table_dat['ra'], table_dat['dec'], unit='deg')
    coord_hst = SkyCoord(table_hst['RA'], table_hst['DEC'], unit='deg')
    coord_muse = SkyCoord(table_muse['RA'], table_muse['DEC'], unit='deg')

    # Copy old catalog so row number and id are preserved
    table_new = table_dat.copy()

    # Matching radius
    max_sep = 2.0 * u.arcsec
    n_hst = 0
    n_muse = 0
    n_unmatched = 0

    # match with HST and MUSE
    for i, entry in enumerate(table_dat):
        if str(entry['id']).startswith('500'):
            idx, sep2d, _ = coord[i].match_to_catalog_sky(coord_hst)
            if sep2d < max_sep:
                table_new['ra'][i] = np.round(table_hst['RA'][idx], 6)
                table_new['dec'][i] = np.round(table_hst['DEC'][idx], 6)
                n_hst += 1
            else:
                n_unmatched += 1
        else:
            idx, sep2d, _ = coord[i].match_to_catalog_sky(coord_muse)
            if sep2d < max_sep:
                table_new['ra'][i] = np.round(table_muse['RA'][idx], 6)
                table_new['dec'][i] = np.round(table_muse['DEC'][idx], 6)
                n_muse += 1
            else:
                n_unmatched += 1

    # Update names for all matches
    c_macthed = SkyCoord(table_new['ra'], table_new['dec'], unit='deg')
    ra_str = c_macthed.ra.to_string(unit=u.hour, sep='', precision=2, pad=True)
    dec_str = c_macthed.dec.to_string(unit=u.deg, sep='', precision=2, alwayssign=True, pad=True)
    name_str = np.array([f"J{r}{d}" for r, d in zip(ra_str, dec_str)])
    table_new['name'] = name_str

    print(f"HST matches: {n_hst}")
    print(f"MUSE matches: {n_muse}")
    print(f"Unmatched: {n_unmatched}")
    print(f"Total: {len(table_dat)}")

    # Add a new column called F814W mag
    table_new.add_column(Column(np.full(len(table_new), -99.9), name='mag_F814W'))

    # Match all sources in table_new to the HST catalog
    idx, sep2d, _ = coord.match_to_catalog_sky(coord_hst)
    mask = sep2d < max_sep

    # Fill matched magnitudes
    table_new['mag_F814W'][mask] = table_hst['MAG'][idx[mask]]

    # Save
    outpath = path_dat.replace('.dat', '_gai_bc.dat')
    table_new.write(outpath, format='ascii.fixed_width', overwrite=True)

    # Also save as a Region file
    region_outpath = outpath.replace('.dat', '.reg')
    with open(region_outpath, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 "
                "fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        f.write("fk5\n")
        for entry in table_new:
            f.write(f"circle({entry['ra']},{entry['dec']}, 0.6\") # text={{{entry['row']}}} \n")


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
    data = np.asarray(data)
    data_row = data[:, 0].astype(int)

    # Catalog path
    path_cat = '../../MUSEQuBES+CUBS/CUBS_dats/{}_combined_cat.dat'.format(cubename)
    cat = Table.read(path_cat, format='ascii.fixed_width')
    row, id, name, radius = cat['row'], cat['id'], cat['name'], cat['radius']

    # match with the same row number and use updated positions and radii from the region file
    row_updated = 1 + np.arange(len(data))
    ra_updated = np.round(data[:, 1], 6)
    dec_updated = np.round(data[:, 2], 6)
    radius_updated = data[:, 3]

    # Match to original catalog by row number
    idx_cat = np.isin(data_row, row)
    row_to_id = dict(zip(row, id))

    # Updated IDs
    id_updated = np.empty(len(data), dtype=object)
    id_updated[idx_cat] = [row_to_id[r] for r in data_row[idx_cat]]
    id_updated[~idx_cat] = [60000 + i for i in range(np.sum(~idx_cat))]  # Some manually added sources

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
    elif cubename == 'J0454-6116':
        cubename_save = 'Q0454-6116'
    elif cubename == 'HE0419-5657':
        cubename_save = 'J0420-5650'
    elif cubename == 'PKS2242-498':
        cubename_save = 'J2245-4931'
    elif cubename == 'PKS0355-483':
        cubename_save = 'J0357-4812'
    elif cubename == 'HE0112-4145':
        cubename_save = 'J0114-4129'
    elif cubename == 'HE2305-5315':
        cubename_save = 'J2308-5258'
    elif cubename == 'HE0331-4112':
        cubename_save = 'J0333-4102'
    else:
        cubename_save = cubename

    # Generate .dat files
    path_cat_updated = '../../MUSEQuBES+CUBS/CUBS_dats/{}_eso_coadd_nosky_sub_ZAP.dat'.format(cubename_save)
    table_updated = Table([row_updated, id_updated, name_updated, ra_updated, dec_updated, radius_updated],
                          names=['row', 'id', 'name', 'ra', 'dec', 'radius'])
    table_updated.write(path_cat_updated, format='ascii.fixed_width', overwrite=True)



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
# extract_hst_image(cubename='J0119-2010', thr_hst=1) # have no sci exposure
# extract_hst_image(cubename='HE0331-4112')

# Extract MUSE images
# extract_muse_image(cubename='J0110-1648')
# extract_muse_image(cubename='J2135-5316')
# extract_muse_image(cubename='HE0246-4101')
# extract_muse_image(cubename='J0028-3305')
# extract_muse_image(cubename='HE0419-5657')
# extract_muse_image(cubename='PKS2242-498')
# extract_muse_image(cubename='PKS0355-483')
# extract_muse_image(cubename='HE0112-4145')
# extract_muse_image(cubename='J0111-0316')
# extract_muse_image(cubename='HE2336-5540')
# extract_muse_image(cubename='HE2305-5315')
# extract_muse_image(cubename='J0454-6116')
# extract_muse_image(cubename='J0154-0712')
# extract_muse_image(cubename='J0119-2010') # have no sci exposure
# extract_muse_image(cubename='HE0331-4112')


# Merge Dat files
MergeDatFiles(cubename='J0110-1648')
MergeDatFiles(cubename='J2135-5316')
MergeDatFiles(cubename='HE0246-4101')
MergeDatFiles(cubename='J0028-3305')
MergeDatFiles(cubename='HE0419-5657')
MergeDatFiles(cubename='PKS2242-498')
MergeDatFiles(cubename='PKS0355-483')
MergeDatFiles(cubename='HE0112-4145')
MergeDatFiles(cubename='J0111-0316')
MergeDatFiles(cubename='HE2336-5540')
MergeDatFiles(cubename='HE2305-5315')
MergeDatFiles(cubename='J0454-6116')
MergeDatFiles(cubename='J0154-0712')
MergeDatFiles(cubename='J0119-2010')
MergeDatFiles(cubename='HE0331-4112')


# Regenerate .dat files after visually inspecting the combined catalogs and removing some spurious sources in DS9
# Only need to run one time after the visual inspection and cleaning