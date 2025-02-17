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

    path_muse = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac.fits'.format(cubename)
    # Q0119 - 2010
    # _COMBINED_CUBE_MED_FINAL_vac.fits


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

    path_group = '../../MUSEQuBES+CUBS/group/{}_group.txt'.format(name_1)
    data_group = ascii.read(path_group)
    ra, dec, z = data_group['col2'], data_group['col3'], data_group['col4']
    v = c_kms * (z - z_qso) / (1 + z_qso)

    filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    # if os.path.isfile(filename) is not True:
    t = Table()
    # t['row'] = row_ggp
    t['ra'] = ra
    t['dec'] = dec
    # t['ID'] = ID_ggp
    t['z'] = z
    t['v'] = v
    # t['name'] = name_ggp
    # t['ql'] = ql_ggp
    t.write(filename, format='fits', overwrite=True)






    # if cubename == 'HE0238-1904':
    #     path_group = ''


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

    # Load MUSE cube
    if cubename_load == 'Q2135-5316':
        path_muse = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_FINAL_vac.fits'.format(cubename_load)
    else:
        path_muse = '../../MUSEQuBES+CUBS/CUBS/{}_COMBINED_CUBE_MED_FINAL_vac.fits'.format(cubename_load)

    cube = Cube(path_muse)
    wave_vac = cube.wave.coord()  # Already in vacuum wavelength
    flux = cube.data * 1e-3

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
GenerateF814WImage(cubename='J0110-1648')
GenerateF814WImage(cubename='J2135-5316')
GenerateF814WImage(cubename='J0119-2010')
GenerateF814WImage(cubename='HE0246-4101')
GenerateF814WImage(cubename='J0028-3305')
GenerateF814WImage(cubename='HE0419-5657')
GenerateF814WImage(cubename='PKS2242-498')
GenerateF814WImage(cubename='PKS0355-483')
GenerateF814WImage(cubename='HE0112-4145')
GenerateF814WImage(cubename='J0111-0316')
GenerateF814WImage(cubename='HE2336-5540')
GenerateF814WImage(cubename='HE2305-5315')
GenerateF814WImage(cubename='J0454-6116')
GenerateF814WImage(cubename='J0154-0712')
GenerateF814WImage(cubename='HE0331-4112')
