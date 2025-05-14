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
from astropy.table import Table
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
    # plt.imshow(np.abs(image_180), origin='lower', cmap='gray')
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
        dis_i = dis_list[gal_list == igal][0]
        ang_i = ang_list[gal_list == igal][0]

        #
        name_i = igal.replace('C', 'C ')
        name_sort = table_gals['Object Name'] == name_i

        # Galaxy information
        ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
        v_sys_gal = table_gals[name_sort]['cz (Velocity)']

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
        height = np.round(100 / scale / hdr_Serra['CDELT2'] / 3600, 0)  # 5 pix in MUSE = 1 arcsec = 7 kpc for 3C57
        width = np.round(100 / scale / hdr_Serra['CDELT2'] / 3600, 0)
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
        mask = np.where(mom0 ==0, mom0, 1)

        # plt.figure()
        # plt.imshow(mask, origin='lower', cmap='gray')
        # plt.plot(new_center[0], new_center[1], 'ro', markersize=5)
        # plt.show()
        # print(mask.shape)
        # raise Exception('segmap')

        # Beam size
        kernel = Gaussian2DKernel(x_stddev=3, y_stddev=3)
        kernel.normalize()
        psf = kernel.array

        source_morphs = source_morphology(mom0, mask, gain=1e5, psf=psf, x_qso=new_center[0], y_qso=new_center[1],
                                          annulus_width=2.5, skybox_size=32, petro_extent_cas=1.5)
        morph = source_morphs[0]
        print(morph)

        A_ZQL = CalculateAsymmetry(image=morph._segmap_shape_asym, mask=morph._mask_stamp, center=new_center, type='shape')
        A_ZQL_2 = CalculateAsymmetry(image=morph._cutout_stamp_maskzeroed_no_bg, mask=morph._mask_stamp, center=new_center,
                                     sky_asymmetry=morph._sky_asymmetry, type='standard')
        seg_OII_cutout = np.where(morph._cutout_stamp_maskzeroed_no_bg == 0, morph._cutout_stamp_maskzeroed_no_bg, 1)
        A_ZQL_3 = CalculateAsymmetry(image=seg_OII_cutout, mask=morph._mask_stamp, center=new_center, type='shape')


        # Print the asymmetry values
        print('A_ZQL_given seg', A_ZQL)
        print('A_ZQL_standard', A_ZQL_2)
        print('A_ZQL_my own seg', A_ZQL_3)
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
        t.add_row((igal_cube, morph.asymmetry, morph.rms_asymmetry2, morph.outer_asymmetry, morph.shape_asymmetry,
                   A_ZQL_2, A_ZQL_3))
        t.write(path_21cm_asymmetry, format='ascii.fixed_width', overwrite=True)

        fig = make_figure(morph)
        plt.savefig(path_savefig_21_morph, dpi=300, bbox_inches='tight')



gal_list = np.array(['NGC2594', 'NGC2685', 'NGC2764', 'NGC3619', 'NGC3626', 'NGC3838', 'NGC3941',
                     'NGC3945', 'NGC4203', 'NGC4262', 'NGC5173', 'NGC5582', 'NGC5631', 'NGC6798',
                     'UGC06176', 'UGC09519'])
dis_list = np.array([35.1, 13.05, 37.40, 31.967, 17.755, 23.5, 11.816,
                     23.400, 18.836, 19.741, 38.000, 33.791, 23.933, 37.5,
                     40.1, 27.6])  # in Mpc
ang_list = np.array([45, 180-53, 180 + 65, 180, -90, 180 + 50, -65,
                     180 + 80, -60, -60, 0, 180-60, 90, 53,
                     -55, -55])

Analyze21cmMorphology(gal_list, dis_list,)