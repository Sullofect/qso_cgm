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
from astropy import stats
from scipy import interpolate
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from mpdaf.obj import Cube, WaveCoord, Image
from astropy.coordinates import SkyCoord
from photutils.background import Background2D, MedianBackground
from photutils.centroids import centroid_sources, centroid_com
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog, deblend_sources
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



def FixCubeHeader(cubename=None):
    path_muse_white = '../../MUSEQuBES+CUBS/CUBS/{}_ESO-DEEP_WHITE.fits'.format(cubename)
    hdul_muse_white = fits.open(path_muse_white)
    hdul_muse_white[1].header.remove('CRDER3')
    hdul_muse_white[2].header.remove('CRDER3')
    hdul_muse_white.writeto(path_muse_white, overwrite=True)


def FixAstrometry(cubename, str_zap=''):
    # Will be replaced by a table
    path_subcube = '../../MUSEQuBES+CUBS/gal_info/subcubes.dat'
    data_subcube = ascii.read(path_subcube, format='fixed_width')
    data_subcube = data_subcube[data_subcube['name'] == cubename]
    ra_muse, dec_muse, radius = data_subcube['ra_center'][0], data_subcube['dec_center'][0], data_subcube['radius'][0]
    c_muse = SkyCoord(ra=ra_muse * u.degree, dec=dec_muse * u.degree, frame='icrs')


    path_muse_white = '../../MUSEQuBES+CUBS/CUBS/{}_ESO-DEEP_white.fits'. \
        format(cubename)
    path_muse_white_gaia = '../../MUSEQuBES+CUBS/CUBS/{}_ESO-DEEP_white_gaia.fits'. \
        format(cubename)
    hdul_muse_white = fits.open(path_muse_white)
    hdul_muse_white_gaia = fits.open(path_muse_white_gaia)

    try:
        print(hdul_muse_white_gaia[1].header['PC2_1'])
    except KeyError:
        print('no rotation copying PC1_1 to PC2_2')
        hdul_muse_white_gaia[1].header.append('PC2_1', 'PC1_2', 'PC2_2')
        hdul_muse_white_gaia[1].header['PC2_1'] = 0
        hdul_muse_white_gaia[1].header['PC1_2'] = 0
        hdul_muse_white_gaia[1].header['PC2_2'] = -1 * hdul_muse_white_gaia[1].header['PC1_1']
    hdul_muse_white_gaia[1].header.append('CD1_1')
    hdul_muse_white_gaia[1].header.append('CD1_2')
    hdul_muse_white_gaia[1].header.append('CD2_1')
    hdul_muse_white_gaia[1].header.append('CD2_2')
    hdul_muse_white_gaia[1].header['CD1_1'] = hdul_muse_white_gaia[1].header['PC1_1'] * hdul_muse_white_gaia[1].header[
        'CDELT1']
    hdul_muse_white_gaia[1].header['CD2_1'] = hdul_muse_white_gaia[1].header['PC2_1'] * hdul_muse_white_gaia[1].header[
        'CDELT2']
    hdul_muse_white_gaia[1].header['CD1_2'] = hdul_muse_white_gaia[1].header['PC1_2'] * hdul_muse_white_gaia[1].header[
        'CDELT1']
    hdul_muse_white_gaia[1].header['CD2_2'] = hdul_muse_white_gaia[1].header['PC2_2'] * hdul_muse_white_gaia[1].header[
        'CDELT2']
    hdul_muse_white_gaia[1].header.remove('PC1_1')
    hdul_muse_white_gaia[1].header.remove('PC1_2')
    hdul_muse_white_gaia[1].header.remove('PC2_1')
    hdul_muse_white_gaia[1].header.remove('PC2_2')
    hdul_muse_white_gaia[1].header.remove('CDELT1')
    hdul_muse_white_gaia[1].header.remove('CDELT2')
    try:
        hdul_muse_white_gaia[2].header['CD1_1'] = hdul_muse_white_gaia[1].header['CD1_1']
        hdul_muse_white_gaia[2].header['CD2_1'] = hdul_muse_white_gaia[1].header['CD2_1']
        hdul_muse_white_gaia[2].header['CD1_2'] = hdul_muse_white_gaia[1].header['CD1_2']
        hdul_muse_white_gaia[2].header['CD2_2'] = hdul_muse_white_gaia[1].header['CD2_2']
        hdul_muse_white_gaia[2].header['CRVAL1'] = hdul_muse_white_gaia[1].header['CRVAL1']
        hdul_muse_white_gaia[2].header['CRVAL2'] = hdul_muse_white_gaia[1].header['CRVAL2']
        hdul_muse_white_gaia[2].header['CRPIX1'] = hdul_muse_white_gaia[1].header['CRPIX1']
        hdul_muse_white_gaia[2].header['CRPIX2'] = hdul_muse_white_gaia[1].header['CRPIX2']
    except IndexError:
        print('no second extension')
    hdul_muse_white_gaia.writeto(path_muse_white_gaia, overwrite=True)

    # if cubename == 'HE0153-4520' or cubename == '3C57':
    #     w = WCS(hdul_muse_white[0].header, naxis=2)
    # else:
    w = WCS(hdul_muse_white[1].header, naxis=2)
    w_gaia = WCS(hdul_muse_white_gaia[1].header, naxis=2)
    x, y = w.world_to_pixel(c_muse)
    c_muse_gaia = w_gaia.pixel_to_world(x, y)
    muse_white_gaia = Image(path_muse_white_gaia)
    sub_muse_white_gaia = muse_white_gaia.subimage(center=(c_muse_gaia.dec.value, c_muse_gaia.ra.value), size=30)

    path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
    sub_muse_white_gaia.write(path_sub_white_gaia)


def FixGalaxyCatalog(cubename=None):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    c_kms = 2.998e5

    try:
        name_1, name_2 = cubename.split('-')
    except ValueError:
        name_1, name_2 = cubename.split('+')

    if name_1 == 'HE0112':
        name_1 = 'J0114'
    elif name_1 == 'HE0246':
        name_1 = 'J0248'
    elif name_1 == 'HE0331':
        name_1 = 'J0333'
    elif name_1 == 'PKS0355':
        name_1 = 'J0357'
    elif name_1 == 'HE0419':
        name_1 = 'J0420'
    elif name_1 == 'PKS2242':
        name_1 = 'J2245'
    elif name_1 == 'HE2305':
        name_1 = 'J2308'
    elif name_1 == 'HE2336':
        name_1 = 'J2339'

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

    path_group = '../../MUSEQuBES+CUBS/group/{}_group.txt'.format(name_1)
    data_group = ascii.read(path_group)
    ra, dec, z, path = data_group['col2'], data_group['col3'], data_group['col4'], data_group['col5']
    v = c_kms * (z - z_qso) / (1 + z_qso)

    row_array, ID_array, name_array = [], [], []
    for path_i in path:
        last = path_i.split('/')[-1]
        row, ID, name = int(last.split('_')[0]), int(last.split('_')[1]), last.split('_')[2]
        row_array.append(row)
        ID_array.append(ID)
        name_array.append(name)

    # Load MUSE white header
    if cubename_load == 'Q2135-5316':
        path_muse_white_ori = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_FINAL_vac_WHITE.fits'.format(
            cubename_load)
    else:
        path_muse_white_ori = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_WHITE.fits'.format(cubename_load)
    path_muse_white_gaia = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_WHITE_gaia.fits'.format(cubename)
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    w = WCS(fits.open(path_muse_white_ori)[1].header, naxis=2)
    w_gaia = WCS(fits.open(path_muse_white_gaia)[1].header, naxis=2)
    x, y = w.world_to_pixel(c)
    c_gaia = w_gaia.pixel_to_world(x, y)

    filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    filename_txt = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.dat'.format(cubename)

    # Rewrite the .fits file to keep it consistent
    if os.path.exists(filename_txt):
        t = Table.read(filename_txt, format='ascii.fixed_width')
        t.write(filename, format='fits', overwrite=True)
    else:
        t = Table()
        t['row'] = row_array
        t['ID'] = ID_array
        t['z'] = z
        t['v'] = v
        t['name'] = name_array
        t['ql'] = np.zeros(len(name_array), dtype=int)
        t['ra'] = ra
        t['dec'] = dec
        t['ra_cor'] = c_gaia.ra
        t['dec_cor'] = c_gaia.dec
        t.write(filename, format='fits', overwrite=True)
        t.write(filename_txt, format='ascii.fixed_width', overwrite=True)


def FixAstrometrySeb(cubename):

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

    # Will be replaced by a table
    path_muse_white_gaia = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_WHITE_gaia.fits'. \
        format(cubename_load)
    path_muse_white_gaia_save = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_WHITE_gaia.fits'. \
        format(cubename)
    hdul_muse_white_gaia = fits.open(path_muse_white_gaia)

    try:
        print(hdul_muse_white_gaia[1].header['PC2_1'])
    except KeyError:
        print('no rotation copying PC1_1 to PC2_2')
        hdul_muse_white_gaia[1].header.append('PC2_1', 'PC1_2', 'PC2_2')
        hdul_muse_white_gaia[1].header['PC2_1'] = 0
        hdul_muse_white_gaia[1].header['PC1_2'] = 0
        hdul_muse_white_gaia[1].header['PC2_2'] = -1 * hdul_muse_white_gaia[1].header['PC1_1']
    hdul_muse_white_gaia[1].header.append('CD1_1')
    hdul_muse_white_gaia[1].header.append('CD1_2')
    hdul_muse_white_gaia[1].header.append('CD2_1')
    hdul_muse_white_gaia[1].header.append('CD2_2')
    hdul_muse_white_gaia[1].header['CD1_1'] = hdul_muse_white_gaia[1].header['PC1_1'] * hdul_muse_white_gaia[1].header[
        'CDELT1']
    hdul_muse_white_gaia[1].header['CD2_1'] = hdul_muse_white_gaia[1].header['PC2_1'] * hdul_muse_white_gaia[1].header[
        'CDELT2']
    hdul_muse_white_gaia[1].header['CD1_2'] = hdul_muse_white_gaia[1].header['PC1_2'] * hdul_muse_white_gaia[1].header[
        'CDELT1']
    hdul_muse_white_gaia[1].header['CD2_2'] = hdul_muse_white_gaia[1].header['PC2_2'] * hdul_muse_white_gaia[1].header[
        'CDELT2']
    hdul_muse_white_gaia[1].header.remove('PC1_1')
    hdul_muse_white_gaia[1].header.remove('PC1_2')
    hdul_muse_white_gaia[1].header.remove('PC2_1')
    hdul_muse_white_gaia[1].header.remove('PC2_2')
    hdul_muse_white_gaia[1].header.remove('CDELT1')
    hdul_muse_white_gaia[1].header.remove('CDELT2')
    try:
        hdul_muse_white_gaia[2].header['CD1_1'] = hdul_muse_white_gaia[1].header['CD1_1']
        hdul_muse_white_gaia[2].header['CD2_1'] = hdul_muse_white_gaia[1].header['CD2_1']
        hdul_muse_white_gaia[2].header['CD1_2'] = hdul_muse_white_gaia[1].header['CD1_2']
        hdul_muse_white_gaia[2].header['CD2_2'] = hdul_muse_white_gaia[1].header['CD2_2']
        hdul_muse_white_gaia[2].header['CRVAL1'] = hdul_muse_white_gaia[1].header['CRVAL1']
        hdul_muse_white_gaia[2].header['CRVAL2'] = hdul_muse_white_gaia[1].header['CRVAL2']
        hdul_muse_white_gaia[2].header['CRPIX1'] = hdul_muse_white_gaia[1].header['CRPIX1']
        hdul_muse_white_gaia[2].header['CRPIX2'] = hdul_muse_white_gaia[1].header['CRPIX2']
    except IndexError:
        print('no second extension')

    hdul_muse_white_gaia.writeto(path_muse_white_gaia_save, overwrite=True)


def GenerateF814WImage(cubename):
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
    else:
        cubename_load = cubename

    # Load MUSE cube
    if cubename_load == 'Q2135-5316':
        path_muse = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_FINAL_vac.fits'.format(cubename_load)
    elif cubename_load == 'HE0226-4110':
        path_muse = '../../MUSEQuBES+CUBS/CUBS/DATACUBE-RXSJ02282-4057-v01-PROPVAR-ZAP.fits'
        # DATACUBE-RXSJ02282-4057-v01-PROPVAR.fits
    else:
        path_muse = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac.fits'.format(cubename_load)


    cube = Cube(path_muse)
    wave_vac = cube.wave.coord()  # Already in vacuum wavelength
    flux = cube.data  # Keep the initial unit 1e-20 erg/s/cm2/Ang

    # Filter
    F814W = '../../pyobs/data/kcorrect/filters/ACS_F814W.fits'
    F814W = fits.open(F814W)[1].data
    wave, trans = F814W['wave'], F814W['transmission']
    wave_mask = (wave > wave_vac[0]) & (wave < wave_vac[-1])
    wave = wave[wave_mask]
    trans = trans[wave_mask]
    f = interpolate.interp1d(wave, trans, kind='linear', bounds_error=False)

    # Check passed!
    # plt.figure()
    # plt.plot(wave, trans, '-k')
    # plt.plot(wave_vac, f(wave_vac), '--r')
    # plt.show()

    # i band
    path_muse_white_gaia = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_WHITE_gaia.fits'. \
        format(cubename)
    hdul_muse_white_gaia = fits.open(path_muse_white_gaia)
    path_muse_F814W_band_gaia = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac_F814W_gaia.fits'.format(cubename)

    # Save result with gaia header
    i_band = np.nansum(flux * f(wave_vac)[:, np.newaxis, np.newaxis], axis=0) / np.nansum(f(wave_vac))
    hdul_muse_white_gaia[1].data = i_band.data
    hdul_muse_white_gaia.writeto(path_muse_F814W_band_gaia, overwrite=True)


def load_2d_image_and_wcs(path, ext=0):
    hdul = fits.open(path)
    data = hdul[ext].data
    hdr = hdul[ext].header

    # If image is not already 2D, squeeze it
    data = np.squeeze(data)
    wcs = WCS(hdr).celestial

    return hdul, data, hdr, wcs


def detect_bright_continuum_source(data, npixels=10, nsigma=2.5):
    """
    Detect the brightest segmented source and centroid it.
    Returns xcen, ycen in pixel coordinates.
    """
    # Clean bad pixels
    img = np.array(data, dtype=float)
    bad = ~np.isfinite(img)
    if np.any(bad):
        med = np.nanmedian(img)
        img[bad] = med

    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    img_sub = img - median

    threshold = detect_threshold(img_sub, nsigma=nsigma)
    segm = detect_sources(img_sub, threshold, npixels=npixels)

    if segm is None:
        raise RuntimeError("No sources detected.")

    cat = SourceCatalog(img_sub, segm)

    xcen = np.array(cat.xcentroid)
    ycen = np.array(cat.ycentroid)
    labels = np.array(cat.labels)

    return xcen, ycen, segm


def pixel_to_sky(x, y, wcs):
    return wcs.pixel_to_world(x, y)


def print_offset(label1, c1, label2, c2):
    dra = (c2.ra - c1.ra).to(u.arcsec)
    ddec = (c2.dec - c1.dec).to(u.arcsec)
    sep = c1.separation(c2).to(u.arcsec)

    print(f'{label1}: RA={c1.ra.deg:.8f}, Dec={c1.dec.deg:.8f}')
    print(f'{label2}: RA={c2.ra.deg:.8f}, Dec={c2.dec.deg:.8f}')
    print(f'Offset ({label2} - {label1}): dRA={dra.value:.3f}\"  dDec={ddec.value:.3f}\"')
    print(f'Total separation: {sep.value:.3f}\"')


def find_existing_file(candidates, template):
    for name in candidates:
        path = template.format(name)
        if os.path.exists(path):
            return path
    return None

def FixAstrometryESO_SDJ(cubename):
    # Try canonical first, then aliases
    name_candidates = [cubename] + [a for a in object_aliases.get(cubename, []) if a != cubename]
    print(name_candidates)

    # Path HST
    if cubename == "J0119-2010":
        path_HST = '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_astro.fits'.format(cubename)
    else:
        path_HST = find_existing_file(
            name_candidates,
            '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci_astro.fits')

    # Path MUSE
    if cubename == 'J0454-6116':
        path_MUSE = "../../MUSEQuBES+CUBS/CUBS_cubes_gaia/Q0454-6116_eso_coadd_nc_nosky_sub_ZAP_WHITE_astro.fits"
    else:
        path_MUSE = find_existing_file(
            name_candidates,
            '../../MUSEQuBES+CUBS/CUBS_cubes_gaia/{}_eso_coadd_nosky_sub_ZAP_WHITE_astro.fits')

    # Path LS catalog
    path_LS = find_existing_file(
        name_candidates,
        '../../MUSEQuBES+CUBS/astrometry/LS_{}.txt')

    hdul_hst, data_hst, hdr_hst, wcs_hst = load_2d_image_and_wcs(path_HST, ext=1)
    hdul_muse, data_muse, hdr_muse, wcs_muse = load_2d_image_and_wcs(path_MUSE, ext=1)

    x_hst, y_hst, segm_hst = detect_bright_continuum_source(data_hst, npixels=10, nsigma=5)
    x_muse, y_muse, segm_muse = detect_bright_continuum_source(data_muse, npixels=10, nsigma=1)

    sky_hst = pixel_to_sky(x_hst, y_hst, wcs_hst)
    sky_muse = pixel_to_sky(x_muse, y_muse, wcs_muse)

    ls = ascii.read(path_LS)

    # Adjust these column names to match your file
    # Common possibilities: ra/dec, RA/DEC, ra_1/dec_1
    if 'ra' in ls.colnames and 'dec' in ls.colnames:
        ra_ls = np.array(ls['ra'], dtype=float)
        dec_ls = np.array(ls['dec'], dtype=float)
    elif 'RA' in ls.colnames and 'DEC' in ls.colnames:
        ra_ls = np.array(ls['RA'], dtype=float)
        dec_ls = np.array(ls['DEC'], dtype=float)
    else:
        raise ValueError(f"Could not find RA/Dec columns in {path_LS}. Columns are: {ls.colnames}")

    coords_ls = SkyCoord(ra_ls * u.deg, dec_ls * u.deg)

    # =========================
    # 6. Optional: compare closest LS source to HST/MUSE centroid
    # =========================
    idx_hst, sep2d_hst, _ = sky_hst.match_to_catalog_sky(coords_ls)
    idx_muse, sep2d_muse, _ = sky_muse.match_to_catalog_sky(coords_ls)

    max_sep = 1 * u.arcsec
    sep_constraint_hst = sep2d_hst < max_sep
    sep_constraint_muse = sep2d_muse < max_sep

    print('\n=== Nearest LS source ===')
    print(f'HST -> LS nearest separation:  {np.mean(sep2d_hst.arcsec[sep_constraint_hst])}"')
    print(f'MUSE -> LS nearest separation: {np.mean(sep2d_muse.arcsec[sep_constraint_muse])}"')

    # =========================
    # HST image
    # =========================
    fig_hst = plt.figure(figsize=(8, 8), dpi=300)
    f_hst = aplpy.FITSFigure(path_HST, figure=fig_hst, north=True, hdu=1)  # use the HST FITS file path
    f_hst.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    f_hst.ticks.set_length(30)
    f_hst.ticks.hide()
    f_hst.tick_labels.hide()
    f_hst.axis_labels.hide()

    # overlay circles (pixel coordinates)
    f_hst.show_markers(
        sky_hst.ra, sky_hst.dec,
        edgecolor='red',
        facecolor='none',
        marker='o',
        s=140,
        alpha=1.0,
        linewidth=0.8,
    )

    f_hst.show_markers(
        ra_ls, dec_ls,
        edgecolor='blue',
        facecolor='none',
        marker='o',
        s=120,
        linewidth=0.8,
    )
    f_hst.save('../../MUSEQuBES+CUBS/plots/astrometry_diagnostic_hst_{}.png'.format(cubename), dpi=300)
    plt.close(fig_hst)

    # =========================
    # MUSE image
    # =========================
    fig_muse = plt.figure(figsize=(8, 8), dpi=300)
    f_muse = aplpy.FITSFigure(path_MUSE, figure=fig_muse, north=True, hdu=1)  # use the MUSE FITS file path
    f_muse.show_colorscale(cmap='Greys', vmin=np.nanpercentile(data_muse, 5), vmax=np.nanpercentile(data_muse, 98))
    f_muse.ticks.set_length(30)
    f_muse.ticks.hide()
    f_muse.tick_labels.hide()
    f_muse.axis_labels.hide()


    # overlay circles (pixel coordinates)
    f_muse.show_markers(
        sky_muse.ra, sky_muse.dec,
        edgecolor='red',
        facecolor='none',
        marker='o',
        s=140,
        linewidth=0.8,
    )

    f_muse.show_markers(
        ra_ls, dec_ls,
        edgecolor='blue',
        facecolor='none',
        marker='o',
        s=120,
        linewidth=0.8,
    )

    f_muse.save('../../MUSEQuBES+CUBS/plots/astrometry_diagnostic_muse_{}.png'.format(cubename), dpi=300)
    plt.close(fig_muse)

def FixDrizzleHSTImages(cubename):
    # Try canonical first, then aliases
    name_candidates = [cubename] + [a for a in object_aliases.get(cubename, []) if a != cubename]

    # Path LS catalog
    path_LS = find_existing_file(
        name_candidates,
        '../../MUSEQuBES+CUBS/astrometry/LS_{}.txt')

    # Read LS catalog
    ls_cat = Table.read(path_LS, format='ascii')

    # Create a circular cut around the quasar position to limit the number of LS sources
    # used for astrometric matching
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    qso_coord = SkyCoord(ra=ra_qso * u.degree, dec=dec_qso * u.degree, frame='icrs')
    search_radius = 100 * u.arcsec  # adjust as needed

    ls_coords = SkyCoord(ls_cat['ra'] * u.deg, ls_cat['dec'] * u.deg)
    sep = qso_coord.separation(ls_coords)

    # Filter catalog to sources inside the circular region
    ls_cat_cut = ls_cat[sep < search_radius]

    # Save the filtered catalog
    path_LS_cut = '../../MUSEQuBES+CUBS/astrometry/LS_{}_cut.txt'.format(cubename)
    ls_cat_cut.write(path_LS_cut, format='csv', overwrite=True)

    # # Path HST
    path_HST = find_existing_file(
        name_candidates,
        '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia.fits')

    os.system('astrometry {} -c {} -hdul_idx 1 '
              '-sigma_threshold_for_source_detection {} '
              '-vignette_rec 0.9 -rot_scale 0 -xy_trafo 0'.format(path_HST, path_LS, 5))

    # if cubename != "J0119-2010":
    #     path_HST_sci = find_existing_file(
    #         name_candidates,
    #         '../../MUSEQuBES+CUBS/HST_drizzles/{}_drc_offset_gaia_sci.fits')
    #
    #     os.system('astrometry {} -c {} -hdul_idx 1 '
    #               '-sigma_threshold_for_source_detection {}'.format(path_HST_sci, path_LS_cut, 10))


# CUBS
# FixCubeHeader(cubename='J0110-1648')
# FixCubeHeader(cubename='J2135-5316')
# FixCubeHeader(cubename='J0119-2010')
# FixCubeHeader(cubename='HE0246-4101')
# FixCubeHeader(cubename='J0028-3305')
# FixCubeHeader(cubename='HE0419-5657')
# FixCubeHeader(cubename='PKS2242-498')
# FixCubeHeader(cubename='PKS0355-483')
# FixCubeHeader(cubename='HE0112-4145')
# FixCubeHeader(cubename='J0111-0316')
# FixCubeHeader(cubename='HE2336-5540')
# FixCubeHeader(cubename='HE2305-5315')
# FixCubeHeader(cubename='J0454-6116')  # need to generate from cube
# FixCubeHeader(cubename='HE0331-4112')
# FixCubeHeader(cubename='J0154-0712')

# CUBS
# FixAstrometry(cubename='J0110-1648')
# FixAstrometry(cubename='J2135-5316')
# FixAstrometry(cubename='J0119-2010')
# FixAstrometry(cubename='HE0246-4101')
# FixAstrometry(cubename='J0028-3305')
# FixAstrometry(cubename='HE0419-5657')
# FixAstrometry(cubename='PKS2242-498')
# FixAstrometry(cubename='PKS0355-483')
# FixAstrometry(cubename='HE0112-4145')
# FixAstrometry(cubename='J0111-0316')
# FixAstrometry(cubename='HE2336-5540')
# FixAstrometry(cubename='HE2305-5315')
# FixAstrometry(cubename='J0454-6116')  # need to generate from cube
# FixAstrometry(cubename='J0154-0712')
# FixAstrometry(cubename='HE0331-4112')

# Get galaxy catalog
# FixGalaxyCatalog(cubename='J0110-1648')
# FixGalaxyCatalog(cubename='J2135-5316')
# FixGalaxyCatalog(cubename='J0119-2010')
# FixGalaxyCatalog(cubename='HE0246-4101')
# FixGalaxyCatalog(cubename='J0028-3305')
# FixGalaxyCatalog(cubename='HE0419-5657')
# FixGalaxyCatalog(cubename='PKS2242-498')
# FixGalaxyCatalog(cubename='PKS0355-483')
# FixGalaxyCatalog(cubename='HE0112-4145')
# FixGalaxyCatalog(cubename='J0111-0316')
# FixGalaxyCatalog(cubename='HE2336-5540')
# FixGalaxyCatalog(cubename='HE2305-5315')
# FixGalaxyCatalog(cubename='J0454-6116')
# FixGalaxyCatalog(cubename='J0154-0712')
# FixGalaxyCatalog(cubename='HE0331-4112')

# Fix Seb+Mandy white light image
# FixAstrometrySeb(cubename='J0110-1648')
# FixAstrometrySeb(cubename='J2135-5316')
# FixAstrometrySeb(cubename='J0119-2010')
# FixAstrometrySeb(cubename='HE0246-4101')
# FixAstrometrySeb(cubename='J0028-3305')
# FixAstrometrySeb(cubename='HE0419-5657')
# FixAstrometrySeb(cubename='PKS2242-498')
# FixAstrometrySeb(cubename='PKS0355-483')
# FixAstrometrySeb(cubename='HE0112-4145')
# FixAstrometrySeb(cubename='J0111-0316')
# FixAstrometrySeb(cubename='HE2336-5540')
# FixAstrometrySeb(cubename='HE2305-5315')
# FixAstrometrySeb(cubename='J0454-6116')
# FixAstrometrySeb(cubename='J0154-0712')
# FixAstrometrySeb(cubename='HE0331-4112')

# Generate i-band image
# GenerateF814WImage(cubename='J0110-1648')
# GenerateF814WImage(cubename='J2135-5316')
# GenerateF814WImage(cubename='J0119-2010')
# GenerateF814WImage(cubename='HE0246-4101')
# GenerateF814WImage(cubename='J0028-3305')
# GenerateF814WImage(cubename='HE0419-5657')
# GenerateF814WImage(cubename='PKS2242-498')
# GenerateF814WImage(cubename='PKS0355-483')
# GenerateF814WImage(cubename='HE0112-4145')
# GenerateF814WImage(cubename='J0111-0316')
# GenerateF814WImage(cubename='HE2336-5540')
# GenerateF814WImage(cubename='HE2305-5315')
# GenerateF814WImage(cubename='J0454-6116')
# GenerateF814WImage(cubename='J0154-0712')
# GenerateF814WImage(cubename='HE0331-4112')

# Reveal tidal tail for HE0226-4110 which is a MUSE CUBE
# GenerateF814WImage(cubename='HE0226-4110')


# Redo the astrometry for Drizzles HST Images
# FixDrizzleHSTImages(cubename='J0110-1648')
# FixDrizzleHSTImages(cubename='J2135-5316') # done
# FixDrizzleHSTImages(cubename='J0119-2010') # done
# FixDrizzleHSTImages(cubename='HE0246-4101') # done
# FixDrizzleHSTImages(cubename='J0028-3305')
# FixDrizzleHSTImages(cubename='HE0419-5657') # done
FixDrizzleHSTImages(cubename='PKS2242-498') # done
FixDrizzleHSTImages(cubename='PKS0355-483')
FixDrizzleHSTImages(cubename='HE0112-4145')
FixDrizzleHSTImages(cubename='J0111-0316')
FixDrizzleHSTImages(cubename='HE2336-5540') # done
FixDrizzleHSTImages(cubename='HE2305-5315') # done
FixDrizzleHSTImages(cubename='J0454-6116')
FixDrizzleHSTImages(cubename='J0154-0712')
FixDrizzleHSTImages(cubename='HE0331-4112')

# Check and Fix the astrometry for the MUSE cubes and the HST/MUSE centroids from ESO SDJ
# FixAstrometryESO_SDJ(cubename='J0110-1648')
# FixAstrometryESO_SDJ(cubename='J2135-5316')
# FixAstrometryESO_SDJ(cubename='J0119-2010')
# FixAstrometryESO_SDJ(cubename='HE0246-4101')
# FixAstrometryESO_SDJ(cubename='J0028-3305')
# FixAstrometryESO_SDJ(cubename='HE0419-5657')
# FixAstrometryESO_SDJ(cubename='PKS2242-498')
# FixAstrometryESO_SDJ(cubename='PKS0355-483')
# FixAstrometryESO_SDJ(cubename='HE0112-4145')
# FixAstrometryESO_SDJ(cubename='J0111-0316')
# FixAstrometryESO_SDJ(cubename='HE2336-5540')
# FixAstrometryESO_SDJ(cubename='HE2305-5315')
# FixAstrometryESO_SDJ(cubename='J0454-6116')
# FixAstrometryESO_SDJ(cubename='J0154-0712')
# FixAstrometryESO_SDJ(cubename='HE0331-4112')