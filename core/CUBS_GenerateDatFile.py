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
    cat_hb = SourceCatalog(convolved_data, segment_map)
    x_cen, y_cen = cat_hb.xcentroid, cat_hb.ycentroid

    # Calculate total counts for each source and magnitude
    segment_flux = cat_hb.segment_flux
    segment_mag = -2.5 * np.log10(segment_flux) + hdul_hst_gaia[1].header['PHOTZPT']

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

def extract_muse_image(cubename=None, deblend_muse=True, thr_muse=1.5):
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
    segment_map = detect_sources(convolved_data, threshold, npixels=3)
    if deblend_muse:
        segment_map = deblend_sources(convolved_data, segment_map, npixels=2, nlevels=32, contrast=0.001)
    cat_hb = SourceCatalog(convolved_data, segment_map)
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
    max_sep = 1.0 * u.arcsec

    n_hst = 0
    n_muse = 0
    n_unmatched = 0

    for i, entry in enumerate(table_dat):
        if entry['id'] == '5000X':
            # match with HST
            idx, sep2d, _ = coord[i].match_to_catalog_sky(coord_hst)
            if sep2d < max_sep:
                table_new['ra'][i] = table_hst['ra'][idx]
                table_new['dec'][i] = table_hst['dec'][idx]
                table_new['name'][i] =
                n_hst += 1
            else:
                n_unmatched += 1

        else:
            # match with MUSE
            idx, sep2d, _ = coord[i].match_to_catalog_sky(coord_muse)

            if sep2d < max_sep:
                table_new[ra_col][i] = table_muse[ra_col][idx]
                table_new[dec_col][i] = table_muse[dec_col][idx]
                n_muse += 1
            else:
                n_unmatched += 1

    print(f"HST matches: {n_hst}")
    print(f"MUSE matches: {n_muse}")
    print(f"Unmatched: {n_unmatched}")
    print(f"Total: {len(table_dat)}")

    # Save
    outpath = path_dat.replace('.dat', '_gaia.dat')
    table_new.write(outpath, format='ascii.fixed_width', overwrite=True)



# Continuum detections on HST images for CUBS fields besides J0119-2010
extract_hst_image(cubename='J0110-1648')
extract_hst_image(cubename='J2135-5316')
extract_hst_image(cubename='HE0246-4101')
extract_hst_image(cubename='J0028-3305')
extract_hst_image(cubename='HE0419-5657')
extract_hst_image(cubename='PKS2242-498')
extract_hst_image(cubename='PKS0355-483')
extract_hst_image(cubename='HE0112-4145')
extract_hst_image(cubename='J0111-0316')
extract_hst_image(cubename='HE2336-5540')
extract_hst_image(cubename='HE2305-5315')
extract_hst_image(cubename='J0454-6116')
extract_hst_image(cubename='J0154-0712')
extract_hst_image(cubename='J0119-2010', thr_hst=1) # have no sci exposure
extract_hst_image(cubename='HE0331-4112')



# Extract MUSE images
extract_muse_image(cubename='J0110-1648')
extract_muse_image(cubename='J2135-5316')
extract_muse_image(cubename='HE0246-4101')
extract_muse_image(cubename='J0028-3305')
extract_muse_image(cubename='HE0419-5657')
extract_muse_image(cubename='PKS2242-498')
extract_muse_image(cubename='PKS0355-483')
extract_muse_image(cubename='HE0112-4145')
extract_muse_image(cubename='J0111-0316')
extract_muse_image(cubename='HE2336-5540')
extract_muse_image(cubename='HE2305-5315')
extract_muse_image(cubename='J0454-6116')
extract_muse_image(cubename='J0154-0712')
extract_muse_image(cubename='J0119-2010') # have no sci exposure
extract_muse_image(cubename='HE0331-4112')
