import os
import aplpy
import statmorph
import numpy as np
from statmorph_ZQL import source_morphology
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from photutils.segmentation import detect_threshold, detect_sources
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
# from statmorph.utils.image_diagnostics import make_figure
from image_diagnostics import make_figure
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Constants
c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239

def AnalyzeMorphology(cubename=None, threshold=1.5, HSTcentroid=False):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    #
    path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}_WCS_subcube.fits'.format(cubename)
    hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
    w = WCS(hdr_sub_gaia, naxis=2)
    center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
    c2 = w.world_to_pixel(center_qso)

    # Load data
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    UseDataSeg=(1.5, 'gauss', None, None)
    line = 'OII+OIII'
    line_OII, line_OIII = 'OII', 'OIII'
    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        v_gal = data_gal['v']
        if HSTcentroid:
            ra_gal, dec_gal = data_gal['ra_HST'], data_gal['dec_HST']
        else:
            ra_gal, dec_gal = data_gal['ra'], data_gal['dec']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal, ra_hst, dec_hst = [], [], [], [], []

    # OII SBs
    if cubename == 'TEX0206-048':
        str_zap = '_zapped'
    else:
        str_zap = ''

    path_SB_OII_kin = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    path_SB_OIII_kin = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OIII, *UseSeg)
    path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OIII, *UseSeg)
    path_savefig_OII_morph = '../../MUSEQuBES+CUBS/plots/{}_{}_morph.png'.format(cubename, line_OII)

    # Special cases due to sky line
    if cubename == 'PKS0552-640':
        path_SB_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        # path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
        #     format(cubename, str_zap, line_OIII, *UseSeg)
    elif cubename == 'HE0226-4110':
        path_SB_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)

    # Analyze asymetry and kinematics
    SB_OII = fits.open(path_SB_OII_kin)[1].data


    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)
    threshold = detect_threshold(SB_OII, 1.5)
    npixels = 5  # minimum number of connected pixels
    convolved_image = convolve(SB_OII, kernel)
    kernel.normalize()
    psf = kernel.array
    segmap = detect_sources(convolved_image, threshold, npixels)

    # Only select the largest in size
    areas = segmap.areas
    labels = segmap.labels
    segmap.data = np.where(segmap.data == labels[np.argmax(areas)], segmap.data, 0)

    # plt.figure()
    # plt.imshow(segmap, origin='lower', cmap='gray')
    # plt.show()
    # raise Exception('segmap')

    source_morphs = source_morphology(SB_OII, segmap, gain=1e5, psf=psf, x_qso=c2[0], y_qso=c2[1],)
    morph = source_morphs[0]
    print(morph.xc_centroid, morph.yc_centroid)

    # OIII SB
    # if os.path.exists(path_SB_OIII):

    # sm = statmorph.SourceMorphology(data=morph)
    fig = make_figure(morph)
    plt.savefig(path_savefig_OII_morph, dpi=300, bbox_inches='tight')

# AnalyzeMorphology(cubename='HE0435-5304')
# AnalyzeMorphology(cubename='HE0153-4520')
# AnalyzeMorphology(cubename='HE0226-4110')
# AnalyzeMorphology(cubename='PKS0405-123')
# AnalyzeMorphology(cubename='HE0238-1904', threshold=1.0)
# AnalyzeMorphology(cubename='3C57')
# AnalyzeMorphology(cubename='PKS0552-640')
# AnalyzeMorphology(cubename='J0110-1648')
# AnalyzeMorphology(cubename='J0454-6116')
# AnalyzeMorphology(cubename='J2135-5316')
# AnalyzeMorphology(cubename='J0119-2010')
# AnalyzeMorphology(cubename='HE0246-4101')
# AnalyzeMorphology(cubename='J0028-3305')
# AnalyzeMorphology(cubename='HE0419-5657')
# AnalyzeMorphology(cubename='PB6291')
# AnalyzeMorphology(cubename='Q0107-0235')
# AnalyzeMorphology(cubename='PKS2242-498')
# AnalyzeMorphology(cubename='PKS0355-483')
# AnalyzeMorphology(cubename='HE0112-4145')
# AnalyzeMorphology(cubename='HE0439-5254')
# AnalyzeMorphology(cubename='HE2305-5315')
# AnalyzeMorphology(cubename='HE1003+0149')
# AnalyzeMorphology(cubename='TEX0206-048')
# AnalyzeMorphology(cubename='Q1354+048')
# AnalyzeMorphology(cubename='J0154-0712')
# AnalyzeMorphology(cubename='LBQS1435-0134')
# AnalyzeMorphology(cubename='PG1522+101')
# AnalyzeMorphology(cubename='HE2336-5540')
AnalyzeMorphology(cubename='PKS0232-04')