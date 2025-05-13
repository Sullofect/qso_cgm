import os
import aplpy
import statmorph
import numpy as np
import skimage.measure
import skimage.transform
import skimage.feature
import skimage.segmentation
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
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                EllipticalAperture, EllipticalAnnulus)

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


def CalculateAsymmetry(image=None, mask=None, center=None, sky_asymmetry=None, type='shape'):
    # Rotate around given center
    image_180 = skimage.transform.rotate(image, 180.0, center=center)

    # Apply symmetric mask
    mask_180 = skimage.transform.rotate(mask, 180.0, center=center)
    mask_180 = mask_180 >= 0.5  # convert back to bool
    mask_symmetric = mask | mask_180
    image = np.where(~mask_symmetric, image, 0.0)
    image_180 = np.where(~mask_symmetric, image_180, 0.0)

    # Testing
    # plt.figure()
    # plt.imshow(np.abs(image), origin='lower', cmap='gray')
    # plt.show()
    # raise ValueError('Debugging: Check the image and mask')

    ap_abs_sum = np.nansum(np.abs(image))
    ap_abs_diff = np.nansum(np.abs(image_180 - image))

    # Difference between this function and the original one
    # image_2 = np.where(image, 1.0, 0.0)
    # ap = CircularAperture(center, 27.145397887497694)
    # ap_abs_sum = ap.do_photometry(np.abs(image), method='exact')[0][0]
    # ap_abs_diff = ap.do_photometry(np.abs(image_180 - image), method='exact')[0][0]
    # print('Asymmetry: ', ap_abs_diff, ap_abs_sum)

    if type == 'shape':
        return ap_abs_diff / ap_abs_sum
    elif type == 'standard':
        return (ap_abs_diff - np.nansum(mask) * sky_asymmetry) / ap_abs_sum

def AnalyzeMorphology(cubename=None, nums_seg_OII=[], select_seg=False):
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
    SB_OII = np.where(center_mask, SB_OII, np.nan)

    seg_OII = fits.open(path_3Dseg_OII)[1].data
    seg_OII = np.where(center_mask, seg_OII, 0)
    if select_seg:
        nums_seg_OII = np.setdiff1d(np.arange(1, np.max(seg_OII) + 1), nums_seg_OII)
    seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)
    seg_OII = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, 0)
    seg_OII = np.where(seg_OII == 0 , seg_OII, 1)

    bkgrd_OII = np.where(seg_OII_mask == 0, SB_OII, np.nan)
    bkgrd_OII_random = np.random.choice(bkgrd_OII.flatten()[~np.isnan(bkgrd_OII.flatten())],
                                        bkgrd_OII.shape, replace=True).reshape(bkgrd_OII.shape)
    SB_OII = np.where(seg_OII_mask != -1, SB_OII, bkgrd_OII_random)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)
    kernel.normalize()
    psf = kernel.array
    # threshold = detect_threshold(SB_OII, 0.8)
    # npixels = 20  # minimum number of connected pixels
    # convolved_image = convolve(SB_OII, kernel)
    # convolved_image = np.where(center_mask, convolved_image, np.nan)
    # segmap = detect_sources(convolved_image, threshold, npixels)

    # Only select the largest in size
    # areas = segmap.areas
    # labels = segmap.labels
    # segmap.data = np.where(segmap.data == labels[np.argmax(areas)], segmap.data, 0)

    # Test with a circular region
    # circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=30)
    # center_mask_flatten = ~circle.contains(pixcoord)
    # center_mask = center_mask_flatten.reshape(SB_OII.shape)
    # SB_OII = np.where(center_mask, SB_OII, 10)
    # seg_OII = np.where(center_mask, seg_OII, 1)

    # plt.figure()
    # plt.imshow(seg_OII, origin='lower', cmap='gray')
    # plt.show()
    # raise Exception('segmap')
    source_morphs = source_morphology(SB_OII, seg_OII, mask=np.isnan(SB_OII),gain=1e5, psf=psf,
                                      x_qso=c2[0], y_qso=c2[1], annulus_width=2.5, skybox_size=32, petro_extent_cas=1.5)
    morph = source_morphs[0]
    # plt.figure()
    # plt.imshow(morph._segmap_shape_asym, origin='lower', cmap='gray')
    # plt.show()
    # raise Exception('segmap')


    A_ZQL = CalculateAsymmetry(image=morph._segmap_shape_asym, mask=morph._mask_stamp, center=c2, type='shape')
    A_ZQL_2 = CalculateAsymmetry(image=morph._cutout_stamp_maskzeroed_no_bg, mask=morph._mask_stamp, center=c2,
                                 sky_asymmetry=morph._sky_asymmetry, type='standard')
    seg_OII_cutout = np.where(morph._cutout_stamp_maskzeroed_no_bg == 0, morph._cutout_stamp_maskzeroed_no_bg, 1)
    A_ZQL_3 = CalculateAsymmetry(image=seg_OII_cutout, mask=morph._mask_stamp, center=c2, type='shape')

    # plt.figure()
    # plt.imshow(seg_OII_cutout, origin='lower', cmap='gray')
    # plt.show()
    # raise Exception('segmap')

    # Print the asymmetry values
    print('A_ZQL_given seg', A_ZQL)
    print('A_ZQL_standard', A_ZQL_2)
    print('A_ZQL_my own seg', A_ZQL_3)
    print('A =', morph.asymmetry)
    print('A_rms =', morph.rms_asymmetry2)
    print('A_outer =', morph.outer_asymmetry)
    print('A_shape=', morph.shape_asymmetry)

    fig = make_figure(morph)
    plt.savefig(path_savefig_OII_morph, dpi=300, bbox_inches='tight')

    # OIII SB
    # if os.path.exists(path_SB_OIII):


# AnalyzeMorphology(cubename='HE0435-5304', nums_seg_OII=[1])
# AnalyzeMorphology(cubename='HE0153-4520')
# AnalyzeMorphology(cubename='HE0226-4110', nums_seg_OII=[14, 15, 16, 17, 20])
# AnalyzeMorphology(cubename='PKS0405-123')
# AnalyzeMorphology(cubename='HE0238-1904')
AnalyzeMorphology(cubename='3C57', nums_seg_OII=[2])
# AnalyzeMorphology(cubename='PKS0552-640', nums_seg_OII=[2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
# AnalyzeMorphology(cubename='J0110-1648', nums_seg_OII=[1])
# AnalyzeMorphology(cubename='J0454-6116', nums_seg_OII=[2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 17, 18])
# AnalyzeMorphology(cubename='J2135-5316', nums_seg_OII=[2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19])
# AnalyzeMorphology(cubename='J0119-2010', nums_seg_OII=[3, 4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20])
# AnalyzeMorphology(cubename='HE0246-4101', nums_seg_OII=[1], select_seg=True) # 0.7 difference between A and A_shape
# AnalyzeMorphology(cubename='J0028-3305', nums_seg_OII=[2], select_seg=True) # 0.7 difference between A and A_shape
# AnalyzeMorphology(cubename='HE0419-5657', nums_seg_OII=[2, 4, 5], select_seg=True) # 0.7 difference between A and A_shape
# AnalyzeMorphology(cubename='PB6291', nums_seg_OII=[2, 6, 7], select_seg=True) # 0.9 difference between A and A_shape
# AnalyzeMorphology(cubename='Q0107-0235', nums_seg_OII=[1, 4, 5, 6], select_seg=True) # 0.2 difference between A and A_shape
# AnalyzeMorphology(cubename='PKS2242-498', nums_seg_OII=[1, 2], select_seg=True) # 0.6 difference between A and A_shape
# AnalyzeMorphology(cubename='PKS0355-483', nums_seg_OII=[2, 3, 4, 8, 9, 10, 11],
#                   select_seg=True) # 0.3 difference between A and A_shape
# AnalyzeMorphology(cubename='HE0112-4145') # 0.25 difference between A and A_shape
# AnalyzeMorphology(cubename='HE0439-5254') # 0.5 difference between A and A_shape
# AnalyzeMorphology(cubename='HE2305-5315')
# AnalyzeMorphology(cubename='HE1003+0149') # 0.5 difference between A and A_shape
# AnalyzeMorphology(cubename='HE0331-4112', nums_seg_OII=[6], select_seg=True) # 0.2 difference between A and A_shape
# AnalyzeMorphology(cubename='TEX0206-048', nums_seg_OII=[1, 8, 12, 13, 15, 20, 23, 26, 27, 28, 34, 57, 60, 79, 81,
#                                                         101, 107, 108, 114, 118, 317, 547, 552],
#                                                         select_seg=True) # 0.1 difference between A and A_shape
# AnalyzeMorphology(cubename='Q1354+048', nums_seg_OII=[1, 2])  # large difference between A and A_shape
# AnalyzeMorphology(cubename='J0154-0712', nums_seg_OII=[5])  # large difference between A and A_shape
# AnalyzeMorphology(cubename='LBQS1435-0134', nums_seg_OII=[1, 3, 7], select_seg=True)
# AnalyzeMorphology(cubename='PG1522+101', nums_seg_OII=[2, 3, 8, 11], select_seg=True)
# AnalyzeMorphology(cubename='HE2336-5540')
# AnalyzeMorphology(cubename='PKS0232-04', nums_seg_OII=[2, 4, 5, 7])