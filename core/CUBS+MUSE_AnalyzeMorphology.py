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

    # Load data
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    line_OII, line_OIII = 'OII', 'OIII'

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
    path_savefig_OIII_morph = '../../MUSEQuBES+CUBS/plots/{}_{}_morph.png'.format(cubename, line_OIII)

    # Analyze asymetry and kinematics
    SB_OII = fits.open(path_SB_OII_kin)[1].data
    w = WCS(fits.open(path_SB_OII_kin)[1].header, naxis=2)  # OII_kin is already in gaia coordinate
    center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
    c2 = w.world_to_pixel(center_qso)

    # Mask the centroid
    x, y = np.meshgrid(np.arange(SB_OII.shape[0]), np.arange(SB_OII.shape[1]))
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)
    circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
    center_mask_flatten = ~circle.contains(pixcoord)
    center_mask = center_mask_flatten.reshape(SB_OII.shape)
    # SB_OII = np.where(center_mask, SB_OII, np.nan)

    seg_OII = fits.open(path_3Dseg_OII)[1].data
    seg_OII = np.where(seg_OII == 0 , seg_OII, 1)
    # seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)
    threshold = detect_threshold(SB_OII, 0.8)
    npixels = 20  # minimum number of connected pixels
    convolved_image = convolve(SB_OII, kernel)
    kernel.normalize()
    psf = kernel.array
    convolved_image = np.where(center_mask, convolved_image, np.nan)

    segmap = detect_sources(convolved_image, threshold, npixels)

    # Only select the largest in size
    areas = segmap.areas
    labels = segmap.labels
    segmap.data = np.where(segmap.data == labels[np.argmax(areas)], segmap.data, 0)

    # plt.figure()
    # plt.imshow(seg_OII, origin='lower', cmap='gray')
    # plt.show()
    # raise Exception('segmap')

    source_morphs = source_morphology(SB_OII, seg_OII, gain=1e5, psf=psf, x_qso=c2[0], y_qso=c2[1])
    morph = source_morphs[0]

    print('A =', morph.asymmetry)
    print('A_shape=', morph.shape_asymmetry)
    fig = make_figure(morph)
    plt.savefig(path_savefig_OII_morph, dpi=300, bbox_inches='tight')

    # OIII SB
    # if os.path.exists(path_SB_OIII):


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