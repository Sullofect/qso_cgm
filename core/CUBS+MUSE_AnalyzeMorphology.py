import os
import numpy as np
import skimage.measure
import skimage.transform
import skimage.feature
import skimage.segmentation
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from statmorph_ZQL import source_morphology
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.table import Table
from mpdaf.obj import Image, Cube
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from photutils.segmentation import detect_threshold, detect_sources
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from image_diagnostics import make_figure

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
np.random.seed(1)

def CalculateAsymmetry(image=None, mask=None, center=None, sky_asymmetry=None, type='shape'):
    # Rotate around given center
    image_180 = skimage.transform.rotate(image, 180.0, center=center)

    # Apply symmetric mask
    mask_180 = skimage.transform.rotate(mask, 180.0, center=center)
    mask_180 = mask_180 >= 0.5  # convert back to bool
    mask_symmetric = mask | mask_180
    image = np.where(~mask_symmetric, image, 0.0)
    image_180 = np.where(~mask_symmetric, image_180, 0.0)

    # Debugging
    # plt.figure()
    # plt.imshow(np.abs(image_180), origin='lower', cmap='gray')
    # plt.plot(center[0], center[1], 'r+')
    # plt.show()
    # raise ValueError('Debugging: Check the image and mask')

    ap_abs_sum = np.nansum(np.abs(image))
    ap_abs_diff = np.nansum(np.abs(image_180 - image))

    if type == 'shape':
        return ap_abs_diff / ap_abs_sum
    elif type == 'standard':
        return (ap_abs_diff - np.nansum(mask) * sky_asymmetry) / ap_abs_sum

def ComputeGini(image, segmap):
    image = image.flatten()
    segmap = np.asarray(segmap.flatten(), dtype=bool)
    sorted_pixelvals = np.sort(np.abs(image[segmap]))
    n = len(sorted_pixelvals)
    indices = np.arange(1, n+1)  # start at i=1
    gini = (np.sum((2*indices-n-1) * sorted_pixelvals) /
            (float(n-1) * np.sum(sorted_pixelvals)))
    return gini

def AnalyzeMorphology(cubename=None, nums_seg_OII=[], nums_seg_OIII=[], select_seg_OII=False, select_seg_OIII=False,
                      savefig=False, addon=''):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Load data
    UseDataSeg = (1.5, 'gauss', None, None)
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
    path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_{}_{}.fits'.\
        format(cubename, str_zap, line_OII, *UseSeg)
    path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_{}_{}.fits'.\
        format(cubename, str_zap, line_OIII, *UseSeg)
    path_cube_OII_ori = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.format(cubename, str_zap, line_OII)
    path_cube_OIII_ori = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'.format(cubename, str_zap, line_OIII)
    path_savefig_OII_morph = '../../MUSEQuBES+CUBS/plots/{}_{}_morph{}.png'.format(cubename, line_OII, addon)
    path_savefig_OIII_morph = '../../MUSEQuBES+CUBS/plots/{}_{}_morph{}.png'.format(cubename, line_OIII, addon)

    # Special cases due to sky line
    if cubename == 'PKS0552-640':
        path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
    elif cubename == 'HE0226-4110':
        path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)
        path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
            format(cubename, str_zap, line_OII, *UseSeg)

    # Calculate smoothing
    # Assuming Gaussian and size_max corresponds to FWHM in kpc
    size_max = 7.3  # kpc
    smooth_std = size_max / (cosmo.angular_diameter_distance(z_qso).value * 1e3) * 206265 / 0.2 / 2.3548  # to pixel
    smooth_std = np.round(smooth_std, 1)  # round to 2 decimal places
    kernel = Gaussian2DKernel(x_stddev=smooth_std, y_stddev=smooth_std)
    kernel_1 = Kernel(kernel.array[np.newaxis, :, :])

    cube_OII = Cube(path_cube_OII)
    cube_OII_ori = Cube(path_cube_OII_ori)
    cube_OII_ori = cube_OII_ori.select_lambda(np.min(cube_OII.wave.coord()), np.max(cube_OII.wave.coord()))
    flux_OII = convolve(cube_OII_ori.data * 1e-3, kernel_1)
    assert len(cube_OII.wave.coord()) == len(cube_OII_ori.wave.coord())

    # Analyze asymmetry and kinematics
    seg_OII_3D, seg_OII = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OII)[1].data
    SB_OII = fits.open(path_SB_OII_kin)[1].data
    # SB_OII_smoothed = np.nansum(np.where(fits.open(path_3Dseg_OII)[0].data != 0,
    #                                      fits.open(path_cube_OII)[1].data, np.nan), axis=0) * 1.25 / 0.2 / 0.2 * 1e-3
    SB_OII_smoothed = np.nansum(np.where(seg_OII_3D != 0, flux_OII, np.nan), axis=0)
    seg_OII_3D = np.where(seg_OII[np.newaxis, :, :] == 1, seg_OII_3D, 0)
    idx = np.argmax(np.sum(seg_OII_3D, axis=(1, 2)), axis=0)
    SB_OII_smoothed = np.where(SB_OII_smoothed != 0, SB_OII_smoothed, 3 * flux_OII[idx, :, :])
    SB_OII_smoothed *= 1.25 / 0.2 / 0.2

    # Calcuate the centroid
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
    # SB_OII_smoothed = np.where(center_mask, SB_OII_smoothed, np.nan)

    # Mask the centroid and then remove other objects
    seg_OII = np.where(center_mask, seg_OII, 0)
    if select_seg_OII:
        nums_seg_OII = np.setdiff1d(np.arange(1, np.max(seg_OII) + 1), nums_seg_OII)
    seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)
    seg_OII = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, 0)
    seg_OII = np.where(seg_OII == 0 , seg_OII, 1)

    # bkgrd_OII = np.where(seg_OII_mask == 0, SB_OII, np.nan)
    # bkgrd_OII_random = np.random.choice(bkgrd_OII.flatten()[~np.isnan(bkgrd_OII.flatten())],
    #                                     bkgrd_OII.shape, replace=True).reshape(bkgrd_OII.shape)
    # SB_OII = np.where(seg_OII_mask != -1, SB_OII, bkgrd_OII_random)

    # bkgrd_OII_smoothed = np.where(seg_OII_mask == 0, SB_OII_smoothed, np.nan)
    # bkgrd_OII_smoothed_random = np.random.choice(bkgrd_OII_smoothed.flatten()[~np.isnan(bkgrd_OII_smoothed.flatten())],
    #                                     bkgrd_OII_smoothed.shape, replace=True).reshape(bkgrd_OII_smoothed.shape)
    # SB_OII_smoothed = np.where(seg_OII_mask != -1, SB_OII_smoothed, bkgrd_OII_smoothed_random)

    # PSF and gain do not matter for asymmetry
    # kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)
    kernel.normalize()
    psf = kernel.array

    # Test with a circular region at the center and away from center, A_circular = 0, A_side = 2
    # circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=20)
    # circle = CirclePixelRegion(center=PixCoord(x=c2[0]+8, y=c2[1]), radius=4)
    # center_mask_flatten = ~circle.contains(pixcoord)
    # center_mask = center_mask_flatten.reshape(SB_OII.shape)
    # SB_OII[:, :] = bkgrd_OII_random
    # seg_OII[:, :] = 0
    # SB_OII = np.where(center_mask, SB_OII, 3)
    # seg_OII = np.where(center_mask, seg_OII, 1)

    source_morphs = source_morphology(SB_OII, seg_OII, mask=np.isnan(SB_OII),gain=1e5, psf=psf,
                                      x_qso=c2[0], y_qso=c2[1], annulus_width=2.5, skybox_size=32, petro_extent_cas=1.5)
    morph = source_morphs[0]
    # source_morphs = source_morphology(SB_OII_smoothed, seg_OII, mask=np.isnan(SB_OII_smoothed),gain=1e5, psf=psf,
    #                                   x_qso=c2[0], y_qso=c2[1], annulus_width=2.5, skybox_size=32, petro_extent_cas=1.5)
    # morph_smoothed = source_morphs[0]

    A_ZQL = CalculateAsymmetry(image=morph._segmap_shape_asym, mask=morph._mask_stamp,
                               center=morph._asymmetry_center, type='shape')
    A_ZQL_2 = CalculateAsymmetry(image=morph._cutout_stamp_maskzeroed_no_bg, mask=morph._mask_stamp,
                                 center=morph._asymmetry_center, sky_asymmetry=morph._sky_asymmetry, type='standard')
    seg_OII_cutout = np.where(morph._cutout_stamp_maskzeroed_no_bg == 0, morph._cutout_stamp_maskzeroed_no_bg, 1)
    A_ZQL_3 = CalculateAsymmetry(image=seg_OII_cutout, mask=morph._mask_stamp, center=morph._asymmetry_center, type='shape')

    # Print the asymmetry values
    # print('OII A_ZQL_shape_ori', A_ZQL)
    # print('A_ZQL_standard', A_ZQL_2)
    print('A_ZQL_shape', A_ZQL_3)
    # print('A =', morph.asymmetry)
    # print('A_rms =', morph.rms_asymmetry2)
    # print('A_outer =', morph.outer_asymmetry)
    # print('A_shape=', morph.shape_asymmetry)

    # Gini map
    Gini = ComputeGini(morph._cutout_stamp_maskzeroed_no_bg, seg_OII_cutout)
    Gini_smoothed = ComputeGini(SB_OII_smoothed, seg_OII)
    print('OII Gini index is', np.round(Gini, 5))
    print('OII Gini index smoothed is', Gini_smoothed)

    # Save the asymmetry values
    path_OII_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OII_asymmetry{}.txt'.format(addon)

    if os.path.exists(path_OII_asymmetry):
        t = Table.read(path_OII_asymmetry, format='ascii.fixed_width')
    else:
        t = Table(names=('cubename', 'A', 'A_rms', 'A_outer', 'A_shape', 'A_ZQL',
                         'A_shape_ZQL', 'Gini', 'Gini_smoothed'),
                  dtype=('S15', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    if cubename in t['cubename']:
        index = np.where(t['cubename'] == cubename)[0][0]
        t['A'][index] = np.round(morph.asymmetry, 2)
        t['A_rms'][index] = np.round(morph.rms_asymmetry2, 2)
        t['A_outer'][index] = np.round(morph.outer_asymmetry, 2)
        t['A_shape'][index] = np.round(morph.shape_asymmetry, 2)
        t['A_ZQL'][index] = np.round(A_ZQL_2, 2)
        t['A_shape_ZQL'][index] = np.round(A_ZQL_3, 2)
        t['Gini'][index] = np.round(Gini, 2)
        t['Gini_smoothed'][index] = np.round(Gini_smoothed, 2)
    else:
        t.add_row((cubename, np.round(morph.asymmetry, 2), np.round(morph.rms_asymmetry2, 2),
                   np.round(morph.outer_asymmetry, 2), np.round(morph.shape_asymmetry, 2),
                   np.round(A_ZQL_2, 2), np.round(A_ZQL_3, 2), np.round(Gini, 2), np.round(Gini_smoothed, 2)))
    t.write(path_OII_asymmetry, format='ascii.fixed_width', overwrite=True)

    if savefig:
        fig = make_figure(morph)
        fig.savefig(path_savefig_OII_morph, dpi=300, bbox_inches='tight')

    # OIII
    if os.path.exists(path_SB_OIII_kin):
        # Cubes
        cube_OIII = Cube(path_cube_OIII)
        cube_OIII_ori = Cube(path_cube_OIII_ori)
        cube_OIII_ori = cube_OIII_ori.select_lambda(np.min(cube_OIII.wave.coord()), np.max(cube_OIII.wave.coord()))
        flux_OIII = convolve(cube_OIII_ori.data * 1e-3, kernel_1)
        assert len(cube_OIII.wave.coord()) == len(cube_OIII_ori.wave.coord())

        # Analyze asymmetry and kinematics
        seg_OIII_3D, seg_OIII = fits.open(path_3Dseg_OIII)[0].data, fits.open(path_3Dseg_OIII)[1].data
        SB_OIII_smoothed = np.nansum(np.where(seg_OIII_3D != 0, flux_OIII, np.nan), axis=0)
        seg_OIII_3D = np.where(seg_OIII[np.newaxis, :, :] == 1, seg_OIII_3D, 0)
        idx = np.argmax(np.sum(seg_OIII_3D, axis=(1, 2)), axis=0)
        SB_OIII_smoothed = np.where(SB_OIII_smoothed != 0, SB_OIII_smoothed, 3 * flux_OIII[idx, :, :])
        SB_OIII_smoothed *= 1.25 / 0.2 / 0.2

        # Analyze asymetry and kinematics
        SB_OIII = fits.open(path_SB_OIII_kin)[1].data
        SB_OIII = np.where(center_mask, SB_OIII, np.nan)
        seg_OIII = np.where(center_mask, seg_OIII, 0)
        if select_seg_OIII:
            nums_seg_OIII = np.setdiff1d(np.arange(1, np.max(seg_OIII) + 1), nums_seg_OIII)
        seg_OIII_mask = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, -1)
        seg_OIII = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, 0)
        seg_OIII = np.where(seg_OIII == 0, seg_OIII, 1)

        # bkgrd_OIII = np.where(seg_OIII_mask == 0, SB_OIII, np.nan)
        # bkgrd_OIII_random = np.random.choice(bkgrd_OIII.flatten()[~np.isnan(bkgrd_OIII.flatten())],
        #                                     bkgrd_OIII.shape, replace=True).reshape(bkgrd_OIII.shape)
        # SB_OIII = np.where(seg_OIII_mask != -1, SB_OIII, bkgrd_OIII_random)

        c3 = w.world_to_pixel(center_qso)  # Sometimes the center is changed after running source_morphology
        source_morphs = source_morphology(SB_OIII, seg_OIII, mask=np.isnan(SB_OIII), gain=1e5, psf=psf,
                                          x_qso=c3[0], y_qso=c3[1], annulus_width=2.5, skybox_size=32,
                                          petro_extent_cas=1.5)
        morph = source_morphs[0]

        A_ZQL = CalculateAsymmetry(image=morph._segmap_shape_asym, mask=morph._mask_stamp,
                                   center=morph._asymmetry_center, type='shape')
        A_ZQL_2 = CalculateAsymmetry(image=morph._cutout_stamp_maskzeroed_no_bg, mask=morph._mask_stamp,
                                     center=morph._asymmetry_center, sky_asymmetry=morph._sky_asymmetry,
                                     type='standard')
        seg_OIII_cutout = np.where(morph._cutout_stamp_maskzeroed_no_bg == 0, morph._cutout_stamp_maskzeroed_no_bg, 1)
        A_ZQL_3 = CalculateAsymmetry(image=seg_OIII_cutout, mask=morph._mask_stamp, center=morph._asymmetry_center,
                                     type='shape')

        # plt.figure()
        # plt.imshow(morph._cutout_stamp_maskzeroed_no_bg, origin='lower', cmap='gray')
        # plt.imshow(SB_OIII, origin='lower', cmap='gray')
        # plt.plot(morph._asymmetry_center[0], morph._asymmetry_center[1], 'ro', markersize=5)
        # print('QSO_centroid', c3)
        # plt.plot(c2[0], c2[1], 'ro', markersize=5)
        # # plt.plot(c3[0], c3[1], 'go', markersize=10)
        # plt.show()
        # raise Exception('segmap')

        # Print the asymmetry values
        # print('OIII A_ZQL_shape_ori', A_ZQL)
        # print('A_ZQL_standard', A_ZQL_2)
        print('A_ZQL_shape', A_ZQL_3)
        # print('A =', morph.asymmetry)
        # print('A_rms =', morph.rms_asymmetry2)
        # print('A_outer =', morph.outer_asymmetry)
        # print('A_shape=', morph.shape_asymmetry)

        # Gini map
        Gini = ComputeGini(morph._cutout_stamp_maskzeroed_no_bg, seg_OIII_cutout)
        Gini_smoothed = ComputeGini(SB_OIII_smoothed, seg_OIII)
        print('OIII Gini index is', Gini)
        print('OIII Gini index smoothed is', Gini_smoothed)

        # Save the asymmetry values
        path_OIII_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OIII_asymmetry{}.txt'.format(addon)

        if os.path.exists(path_OIII_asymmetry):
            t = Table.read(path_OIII_asymmetry, format='ascii.fixed_width')
        else:
            t = Table(names=('cubename', 'A', 'A_rms', 'A_outer', 'A_shape', 'A_ZQL',
                             'A_shape_ZQL', 'Gini', 'Gini_smoothed'),
                      dtype=('S15', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

        if cubename in t['cubename']:
            index = np.where(t['cubename'] == cubename)[0][0]
            t['A'][index] = np.round(morph.asymmetry, 2)
            t['A_rms'][index] = np.round(morph.rms_asymmetry2, 2)
            t['A_outer'][index] = np.round(morph.outer_asymmetry, 2)
            t['A_shape'][index] = np.round(morph.shape_asymmetry, 2)
            t['A_ZQL'][index] = np.round(A_ZQL_2, 2)
            t['A_shape_ZQL'][index] = np.round(A_ZQL_3, 2)
            t['Gini'][index] = np.round(Gini, 2)
            t['Gini_smoothed'][index] = np.round(Gini_smoothed, 2)
        else:
            t.add_row((cubename, np.round(morph.asymmetry, 2), np.round(morph.rms_asymmetry2, 2),
                       np.round(morph.outer_asymmetry, 2), np.round(morph.shape_asymmetry, 2),
                       np.round(A_ZQL_2, 2), np.round(A_ZQL_3, 2), np.round(Gini, 2), np.round(Gini_smoothed, 2)))
        t.write(path_OIII_asymmetry, format='ascii.fixed_width', overwrite=True)

        if savefig:
            fig = make_figure(morph)
            fig.savefig(path_savefig_OIII_morph, dpi=300, bbox_inches='tight')

def Analyze21cmMorphology(gals, dis):
    path_table_gals = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/table_gals.fits'
    table_gals = fits.open(path_table_gals)[1].data

    for ind, igal in enumerate(gals):
        if igal == 'NGC2594':
            igal_cube = 'NGC2592'
        elif igal == 'NGC3619':
            igal_cube = 'NGC3613'
        else:
            igal_cube = igal

        # Load the distances and angles
        dis_i = dis[gals == igal][0]
        name_i = igal.replace('C', 'C ')
        name_sort = table_gals['Object Name'] == name_i

        # Galaxy information
        ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']

        path_mom0 = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom0/{}_mom0.fits'.format(igal_cube)
        path_mom1 = '../../MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(igal_cube)
        path_savefig_21_morph = '../../MUSEQuBES+CUBS/plots/{}_{}_morph.png'.format(igal_cube, '21cm')

        #
        mom0 = fits.open(path_mom0)[0].data[0]
        mom1 = fits.open(path_mom1)[0].data[0]
        hdr_Serra = fits.open(path_mom0)[0].header

        #
        w = WCS(hdr_Serra, naxis=2)
        center_gal = SkyCoord(ra_gal[0], dec_gal[0], unit='deg', frame='icrs')
        c_gal = w.world_to_pixel(center_gal)
        x_gal, y_gal = np.meshgrid(np.arange(mom1.shape[0]), np.arange(mom1.shape[1]))
        x_gal, y_gal = x_gal.flatten(), y_gal.flatten()
        pixcoord = PixCoord(x=x_gal, y=y_gal)

        # 15 arcsec in 3C57 corresponding kpc
        scale = np.pi * dis_i * 1 / 3600 / 180 * 1e3  # how many kpc per arcsec
        height = np.round(110 / scale / hdr_Serra['CDELT2'] / 3600, 0)  # 5 pix in MUSE = 1 arcsec = 7 kpc for 3C57
        width = np.round(110 / scale / hdr_Serra['CDELT2'] / 3600, 0)
        rectangle = RectanglePixelRegion(center=PixCoord(x=c_gal[0], y=c_gal[1]), width=width, height=height)
        mask_flatten = rectangle.contains(pixcoord)
        mask = mask_flatten.reshape(mom1.shape)

        y_indices, x_indices = np.where(mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        mom0 = mom0[y_min:y_max + 1, x_min:x_max + 1]
        mom1 = mom1[y_min:y_max + 1, x_min:x_max + 1]
        new_center_x = (x_max - x_min + 1) // 2
        new_center_y = (y_max - y_min + 1) // 2
        new_center = (new_center_x, new_center_y)
        # mask = np.where(mom0 == 0, mom0, 1)

        # Beam size
        kernel = Gaussian2DKernel(x_stddev=3, y_stddev=3)
        kernel.normalize()
        psf = kernel.array

        threshold = detect_threshold(mom0, 0.8)
        npixels = 20  # minimum number of connected pixels
        segmap = detect_sources(mom0, threshold, npixels)

        # Only select the largest in size
        areas = segmap.areas
        labels = segmap.labels
        segmap.data = np.where(segmap.data == labels[np.argmax(areas)], segmap.data, 0)
        segmap.data = np.where(segmap.data == 0, segmap.data, 1)

        source_morphs = source_morphology(mom0, segmap.data, gain=1e5, psf=psf, x_qso=new_center[0], y_qso=new_center[1],
                                          annulus_width=2.5, skybox_size=32, petro_extent_cas=1.5)
        morph = source_morphs[0]

        A_ZQL = CalculateAsymmetry(image=morph._segmap_shape_asym, mask=morph._mask_stamp, center=morph._asymmetry_center, type='shape')
        A_ZQL_2 = CalculateAsymmetry(image=morph._cutout_stamp_maskzeroed_no_bg, mask=morph._mask_stamp, center=morph._asymmetry_center,
                                     sky_asymmetry=morph._sky_asymmetry, type='standard')
        seg_OII_cutout = np.where(morph._cutout_stamp_maskzeroed_no_bg == 0, morph._cutout_stamp_maskzeroed_no_bg, 1)
        A_ZQL_3 = CalculateAsymmetry(image=seg_OII_cutout, mask=morph._mask_stamp, center=morph._asymmetry_center, type='shape')

        # plt.figure()
        # plt.imshow(seg_OII_cutout, origin='lower', cmap='gray')
        # plt.plot(morph._asymmetry_center[0], morph._asymmetry_center[1], 'ro', markersize=5)
        # plt.show()
        # raise Exception('segmap')

        # Print the asymmetry values
        print('A_ZQL_shape_ori', A_ZQL)
        print('A_ZQL_standard', A_ZQL_2)
        print('A_ZQL_shape', A_ZQL_3)
        print('A =', morph.asymmetry)
        print('A_rms =', morph.rms_asymmetry2)
        print('A_outer =', morph.outer_asymmetry)
        print('A_shape=', morph.shape_asymmetry)

        # Save the asymmetry values
        path_21cm_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_21cm_asymmetry.txt'

        if os.path.exists(path_21cm_asymmetry):
            t = Table.read(path_21cm_asymmetry, format='ascii.fixed_width')
        else:
            t = Table(names=('cubename', 'A', 'A_rms', 'A_outer', 'A_shape', 'A_ZQL', 'A_shape_ZQL'),
                      dtype=('S15', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

        if igal_cube in t['cubename']:
            index = np.where(t['cubename'] == igal_cube)[0][0]
            t['A'][index] = morph.asymmetry
            t['A_rms'][index] = morph.rms_asymmetry2
            t['A_outer'][index] = morph.outer_asymmetry
            t['A_shape'][index] = morph.shape_asymmetry
            t['A_ZQL'][index] = A_ZQL_2
            t['A_shape_ZQL'][index] = A_ZQL_3
        else:
            t.add_row((igal_cube, morph.asymmetry, morph.rms_asymmetry2, morph.outer_asymmetry, morph.shape_asymmetry,
                       A_ZQL_2, A_ZQL_3))
        t.write(path_21cm_asymmetry, format='ascii.fixed_width', overwrite=True)

        fig = make_figure(morph)
        plt.savefig(path_savefig_21_morph, dpi=300, bbox_inches='tight')

def AnalyzeLyaMorphology(cubename=None, savefig=False):
    # Load QSO information
    path_xyZ_Lya = '../../MUSEQuBES+CUBS/SB_Lya/QSOxyzlist.txt'
    xyZ_Lya = Table.read(path_xyZ_Lya, format='ascii')
    x, y = xyZ_Lya['X'][xyZ_Lya['QSO'] == cubename][0], xyZ_Lya['Y'][xyZ_Lya['QSO'] == cubename][0]
    cube_Lya_name, z_qso = xyZ_Lya['cube_Lya_name'][xyZ_Lya['QSO'] == cubename][0], \
                           xyZ_Lya['z'][xyZ_Lya['QSO'] == cubename][0]
    c2 = np.array([x, y])

    path_SB_Lya = '../../MUSEQuBES+CUBS/SB_Lya/Maps_unsmoothed_gsm0/{}.fits'.format(cubename)
    # path_SB_Lya_smoothed = '../../MUSEQuBES+CUBS/SB_Lya/Maps_smoothed_gsm1/{}.fits'.format(cubename)
    path_seg_Lya = '../../MUSEQuBES+CUBS/SB_Lya/Maps_smoothed_gsm1/{}_mask.fits'.format(cubename)
    path_cube_Lya = '../../MUSEQuBES+CUBS/SB_Lya/cubes/{}.fits'.format(cube_Lya_name)
    path_seg_Lya_3D = '../../MUSEQuBES+CUBS/SB_Lya/cubes/{}.Objects_Id1.fits'.format(cube_Lya_name)
    path_savefig_Lya_morph = '../../MUSEQuBES+CUBS/plots/{}_{}_morph.png'.format(cubename, 'Lya')

    # Analyze asymetry and kinematics
    SB_Lya = fits.open(path_SB_Lya)[0].data * 1e-1 # convert the unit to 1e-17 erg/s/cm^2/arcsec^2
    # SB_Lya_smoothed = fits.open(path_SB_Lya_smoothed)[0].data
    seg_Lya = fits.open(path_seg_Lya)[0].data
    cube_Lya = fits.open(path_cube_Lya)[0].data
    cube_Lya_seg = fits.open(path_seg_Lya_3D)[0].data

    # Match the physical resolution
    size_max = 10.0  # kpc
    smooth_std = size_max / (cosmo.angular_diameter_distance(z_qso).value * 1e3) * 206265 / 0.2 / 2.3548  # to pixel
    smooth_std = np.round(smooth_std, 1)  # round to 2 decimal places
    kernel = Gaussian2DKernel(x_stddev=smooth_std, y_stddev=smooth_std)
    kernel_1 = Kernel(kernel.array[np.newaxis, :, :])
    # cube_Lya = convolve(cube_Lya, kernel_1)
    SB_Lya_smoothed = np.nansum(np.where(cube_Lya_seg, cube_Lya, np.nan), axis=0) * 1.25 / 0.2 / 0.2 * 1e-3

    # Mask the centroid
    x, y = np.meshgrid(np.arange(SB_Lya.shape[1]), np.arange(SB_Lya.shape[0]))  # need to reverse for asymmetrical array
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)
    circle = CirclePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), radius=2.5)
    center_mask_flatten = ~circle.contains(pixcoord)
    center_mask = center_mask_flatten.reshape(SB_Lya.shape)
    SB_Lya = np.where(center_mask, SB_Lya, np.nan)
    SB_Lya_smoothed = np.where(center_mask, SB_Lya_smoothed, np.nan)
    seg_Lya = np.where(seg_Lya == 0, seg_Lya, 1)
    seg_Lya = np.where(center_mask, seg_Lya, 0)

    # PSF and gain do not matter for asymmetry
    # kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)
    kernel.normalize()
    psf = kernel.array

    source_morphs = source_morphology(SB_Lya, seg_Lya, mask=np.isnan(SB_Lya), gain=1e5, psf=psf,
                                      x_qso=c2[0], y_qso=c2[1], annulus_width=2.5, skybox_size=32, petro_extent_cas=1.5)
    morph = source_morphs[0]

    A_ZQL = CalculateAsymmetry(image=morph._segmap_shape_asym, mask=morph._mask_stamp,
                               center=morph._asymmetry_center, type='shape')
    A_ZQL_2 = CalculateAsymmetry(image=morph._cutout_stamp_maskzeroed_no_bg, mask=morph._mask_stamp,
                                 center=morph._asymmetry_center, sky_asymmetry=morph._sky_asymmetry, type='standard')
    seg_Lya_cutout = np.where(morph._cutout_stamp_maskzeroed_no_bg == 0, morph._cutout_stamp_maskzeroed_no_bg, 1)
    A_ZQL_3 = CalculateAsymmetry(image=seg_Lya_cutout, mask=morph._mask_stamp, center=morph._asymmetry_center,
                                 type='shape')

    # plt.figure()
    # plt.imshow(seg_Lya_cutout, origin='lower', cmap='gray')
    # plt.plot(c2[0], c2[1], 'r+', markersize=10)
    # plt.plot(morph._asymmetry_center[0], morph._asymmetry_center[1], 'b+', markersize=10)
    # plt.show()
    # raise Exception('segmap')

    # Print the asymmetry values
    # print('A_ZQL_shape_ori', A_ZQL)
    # print('A_ZQL_standard', A_ZQL_2)
    print('A_ZQL_shape', A_ZQL_3)
    # print('A =', morph.asymmetry)
    # print('A_rms =', morph.rms_asymmetry2)
    # print('A_outer =', morph.outer_asymmetry)
    # print('A_shape=', morph.shape_asymmetry)

    # Gini map
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(SB_Lya_smoothed, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=1)
    ax[1].imshow(morph._cutout_stamp_maskzeroed_no_bg, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=1)
    plt.show()
    SB_Lya = np.where(seg_Lya, SB_Lya, np.nan)
    # raise ValueError('testing')
    Gini = ComputeGini(morph._cutout_stamp_maskzeroed_no_bg, seg_Lya_cutout)
    Gini_smoothed = ComputeGini(SB_Lya, seg_Lya)
    print('Lya Gini index is', Gini)
    print('lya Gini index smoothed is', Gini_smoothed)
    raise ValueError('testing')

    # Save the asymmetry values
    path_Lya_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_Lya_asymmetry.txt'

    if os.path.exists(path_Lya_asymmetry):
        t = Table.read(path_Lya_asymmetry, format='ascii.fixed_width')
    else:
        t = Table(names=('cubename', 'A', 'A_rms', 'A_outer', 'A_shape', 'A_ZQL', 'A_shape_ZQL',
                         'Gini', 'Gini_smoothed'),
                  dtype=('S15', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    if cubename in t['cubename']:
        index = np.where(t['cubename'] == cubename)[0][0]
        t['A'][index] = np.round(morph.asymmetry, 2)
        t['A_rms'][index] = np.round(morph.rms_asymmetry2, 2)
        t['A_outer'][index] = np.round(morph.outer_asymmetry, 2)
        t['A_shape'][index] = np.round(morph.shape_asymmetry, 2)
        t['A_ZQL'][index] = np.round(A_ZQL_2, 2)
        t['A_shape_ZQL'][index] = np.round(A_ZQL_3, 2)
        t['Gini'][index] = np.round(Gini, 2)
        t['Gini_smoothed'][index] = np.round(Gini_smoothed, 2)
    else:
        t.add_row((cubename, np.round(morph.asymmetry, 2), np.round(morph.rms_asymmetry2, 2),
                   np.round(morph.outer_asymmetry, 2), np.round(morph.shape_asymmetry, 2),
                   np.round(A_ZQL_2, 2), np.round(A_ZQL_3, 2), np.round(Gini, 2), np.round(Gini_smoothed, 2)))
    t.write(path_Lya_asymmetry, format='ascii.fixed_width', overwrite=True)

    if savefig:
        fig = make_figure(morph)
        fig.savefig(path_savefig_Lya_morph, dpi=300, bbox_inches='tight')


# CUBS+MUSE host nebula itself
# AnalyzeMorphology(cubename='HE0435-5304', nums_seg_OII=[1], nums_seg_OIII=[1])
# # AnalyzeMorphology(cubename='HE0153-4520')
# AnalyzeMorphology(cubename='HE0226-4110', nums_seg_OII=[2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#                   nums_seg_OIII=[1, 5, 6, 8, 9, 10, 11, 16, 19])
# AnalyzeMorphology(cubename='PKS0405-123', nums_seg_OII=[5, 7, 10, 11, 13, 16, 17, 20], nums_seg_OIII=[15])
# AnalyzeMorphology(cubename='HE0238-1904', nums_seg_OII=[1, 6, 12, 13, 17, 19], select_seg_OII=True,
#                   nums_seg_OIII=[1, 2, 4, 9, 13, 15, 17, 20], select_seg_OIII=True)
# AnalyzeMorphology(cubename='3C57', nums_seg_OII=[2], nums_seg_OIII=[])
# AnalyzeMorphology(cubename='PKS0552-640', nums_seg_OII=[2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#                   nums_seg_OIII=[5, 6, 7, 8, 12, 15, 16, 17, 18, 20])
# AnalyzeMorphology(cubename='J0110-1648', nums_seg_OII=[1], nums_seg_OIII=[2])
# AnalyzeMorphology(cubename='J0454-6116', nums_seg_OII=[2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 17, 18], nums_seg_OIII=[2, 7, 9, 10, 18, 19])
# AnalyzeMorphology(cubename='J2135-5316', nums_seg_OII=[2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19],
#                   nums_seg_OIII=[4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# AnalyzeMorphology(cubename='J0119-2010', nums_seg_OII=[3, 4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20],
#                   nums_seg_OIII=[7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20])
# AnalyzeMorphology(cubename='HE0246-4101', nums_seg_OII=[1], select_seg_OII=True)
# AnalyzeMorphology(cubename='J0028-3305', nums_seg_OII=[2], select_seg_OII=True)
# AnalyzeMorphology(cubename='HE0419-5657', nums_seg_OII=[2, 4, 5], select_seg_OII=True)
# AnalyzeMorphology(cubename='PB6291', nums_seg_OII=[2, 6, 7], select_seg_OII=True)
# AnalyzeMorphology(cubename='Q0107-0235', nums_seg_OII=[1, 4, 5, 6], select_seg_OII=True)
# AnalyzeMorphology(cubename='PKS2242-498', nums_seg_OII=[1, 2], select_seg_OII=True)
# AnalyzeMorphology(cubename='PKS0355-483', nums_seg_OII=[2, 3, 4, 8, 9, 10, 11], select_seg_OII=True)
# AnalyzeMorphology(cubename='HE0112-4145')
# AnalyzeMorphology(cubename='HE0439-5254')
# # AnalyzeMorphology(cubename='HE2305-5315')
# AnalyzeMorphology(cubename='HE1003+0149')
# AnalyzeMorphology(cubename='HE0331-4112', nums_seg_OII=[6], select_seg_OII=True)
# AnalyzeMorphology(cubename='TEX0206-048', nums_seg_OII=[1, 8, 12, 13, 15, 20, 23, 26, 27, 28, 34, 57, 60, 79, 81,
#                                                         101, 107, 108, 114, 118, 317, 547, 552],
#                                                         select_seg_OII=True)
# AnalyzeMorphology(cubename='Q1354+048', nums_seg_OII=[1, 2])
# AnalyzeMorphology(cubename='J0154-0712', nums_seg_OII=[5])
# AnalyzeMorphology(cubename='LBQS1435-0134', nums_seg_OII=[1, 3, 7], select_seg_OII=True)
# AnalyzeMorphology(cubename='PG1522+101', nums_seg_OII=[2, 3, 8, 11], select_seg_OII=True)
# # AnalyzeMorphology(cubename='HE2336-5540')
# AnalyzeMorphology(cubename='PKS0232-04', nums_seg_OII=[2, 4, 5, 7])

# CUBS+MUSE host nebula plus galaxies
# AnalyzeMorphology(cubename='HE0435-5304', addon='_plus_galaxies')
# # AnalyzeMorphology(cubename='HE0153-4520', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0226-4110', nums_seg_OII=[12, 14, 15, 16, 17, 20],
#                   nums_seg_OIII=[5, 11, 16, 19], addon='_plus_galaxies', savefig=True)
# AnalyzeMorphology(cubename='PKS0405-123', nums_seg_OII=[5], nums_seg_OIII=[15], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0238-1904', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='3C57', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='PKS0552-640', nums_seg_OII=[2, 6, 7, 9, 10, 14, 18],
#                   nums_seg_OIII=[7, 9, 12, 19, 20], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='J0110-1648', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='J0454-6116', nums_seg_OII=[2, 6, 8, 13, 17, 18],
#                   nums_seg_OIII=[2, 9, 10, 18], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='J2135-5316', nums_seg_OII=[3, 4, 10, 12, 13, 14, 16, 17, 18, 19],
#                   nums_seg_OIII=[4, 12, 13, 14, 15, 17, 19, 20], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='J0119-2010', nums_seg_OII=[4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20],
#                   nums_seg_OIII=[7, 9, 11, 12, 14, 16, 17, 18], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0246-4101', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='J0028-3305', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0419-5657', nums_seg_OII=[1], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='PB6291', nums_seg_OII=[3, 5], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='Q0107-0235', nums_seg_OII=[7], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='PKS2242-498', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='PKS0355-483', nums_seg_OII=[6, 14], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0112-4145', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0439-5254', addon='_plus_galaxies')
# # AnalyzeMorphology(cubename='HE2305-5315', nums_seg_OII=[5, 6, 7, 8], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE1003+0149', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE0331-4112', nums_seg_OII=[1, 2], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='TEX0206-048', nums_seg_OII=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 22, 23,
#                                                         26, 27, 28, 34, 57, 60, 79, 81, 101, 107, 108, 114, 118, 317,
#                                                         547, 552],
#                                                         select_seg_OII=True, addon='_plus_galaxies')
# AnalyzeMorphology(cubename='Q1354+048', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='J0154-0712', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='LBQS1435-0134', addon='_plus_galaxies')
# AnalyzeMorphology(cubename='PG1522+101', nums_seg_OII=[6, 12], addon='_plus_galaxies')
# AnalyzeMorphology(cubename='HE2336-5540', ddon='_plus_galaxies')
# AnalyzeMorphology(cubename='PKS0232-04', nums_seg_OII=[4, 5, 7], addon='_plus_galaxies')

# HI 21 cm
# gal_list = np.array(['NGC2594', 'NGC2685', 'NGC2764', 'NGC3619', 'NGC3626', 'NGC3838', 'NGC3941',
#                      'NGC3945', 'NGC4203', 'NGC4262', 'NGC5173', 'NGC5582', 'NGC5631', 'NGC6798',
#                      'UGC06176', 'UGC09519'])
# dis_list = np.array([35.1, 13.05, 37.40, 31.967, 17.755, 23.5, 11.816,
#                      23.400, 18.836, 19.741, 38.000, 33.791, 23.933, 37.5,
#                      40.1, 27.6])  # in Mpc
# gal_list = ['NGC0680', 'NGC1023', 'NGC2594', 'NGC2685', 'NGC2764', 'NGC2768',
#             'NGC2824', 'NGC2859', 'NGC3032', 'NGC3073', 'NGC3182', 'NGC3193',
#             'NGC3384', 'NGC3414', 'NGC3457', 'NGC3489', 'NGC3499', 'NGC3522',
#             'NGC3608', 'NGC3619', 'NGC3626', 'NGC3838', 'NGC3941', 'NGC3945',
#             'NGC3998', 'NGC4026', 'NGC4036', 'NGC4111', 'NGC4150', 'NGC4203',
#             'NGC4262', 'NGC4278', 'NGC4406', 'NGC4521', 'NGC4694', 'NGC4710',
#             'NGC5103', 'NGC5173', 'NGC5198', 'NGC5422', 'NGC5557', 'NGC5582',
#             'NGC5631', 'NGC5866', 'NGC6798', 'NGC7280', 'NGC7332', 'NGC7465',
#             'PGC028887', 'UGC03960', 'UGC05408', 'UGC06176', 'UGC09519',]
#
#
#
# 'NGC2685', 'NGC2764', 'NGC3619', 'NGC3626', 'NGC3838', 'NGC3941',
#                      'NGC3945', 'NGC4203', 'NGC4262', 'NGC5173', 'NGC5582', 'NGC5631', 'NGC6798',
#                      'UGC06176', 'UGC09519'])
# dis_list = np.array([35.1, 13.05, 37.40, 31.967, 17.755, 23.5, 11.816,
#                      23.400, 18.836, 19.741, 38.000, 33.791, 23.933, 37.5,
#                      40.1, 27.6])  # in Mpc



#
# Analyze21cmMorphology(gal_list, dis_list,)
# Analyze21cmMorphology(['NGC3838'], ['23.5'])

# Lyalpha
AnalyzeLyaMorphology(cubename="J124957-015928")
AnalyzeLyaMorphology(cubename="J133254+005250")
AnalyzeLyaMorphology(cubename="J205344-354652")
AnalyzeLyaMorphology(cubename="J221527-161133")
AnalyzeLyaMorphology(cubename="J230301-093930")
AnalyzeLyaMorphology(cubename="J012403+004432")
AnalyzeLyaMorphology(cubename="J013724-422417")
AnalyzeLyaMorphology(cubename="J015741-010629")
AnalyzeLyaMorphology(cubename="J020944+051713")
AnalyzeLyaMorphology(cubename="J024401-013403")
AnalyzeLyaMorphology(cubename="J033900-013318")
AnalyzeLyaMorphology(cubename="J111008+024458")
AnalyzeLyaMorphology(cubename="J111113-080401")
AnalyzeLyaMorphology(cubename="J123055-113909")


