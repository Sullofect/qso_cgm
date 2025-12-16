import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
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
from astropy.stats import mad_std
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
np.random.seed(1)

def EstimateKinematics(cubename=None, nums_seg_OII=[], nums_seg_OIII=[], select_seg_OII=False, select_seg_OIII=False, addon=''):
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
    path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
    path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)

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


    # Estimate from v50, s80
    v50 = fits.open(path_v50_plot)[1].data
    s80 = fits.open(path_s80_plot)[1].data

    v50_flatten, s80_flatten = v50.ravel(), s80.ravel()
    v50_flatten = v50_flatten[~np.isnan(v50_flatten)]
    s80_flatten = s80_flatten[~np.isnan(s80_flatten)]
    v50_5, v50_95 = np.percentile(v50_flatten, [5, 95])
    s80_mean = np.mean(s80_flatten)
    print(cubename, v50_5, v50_95, s80_mean)
    # print(cubename, mad_std(v50_flatten), mad_std(s80_flatten))

    # plt.figure()
    # plt.hist(v50_flatten, bins=100, color='blue', alpha=0.5, label='V50')
    # plt.show()

    # Calculate from OII flux
    # cube_OII = Cube(path_cube_OII)
    # cube_OII_ori = Cube(path_cube_OII_ori)
    # cube_OII_ori = cube_OII_ori.select_lambda(np.min(cube_OII.wave.coord()), np.max(cube_OII.wave.coord()))
    # flux_OII = cube_OII_ori.data * 1e-3
    # assert len(cube_OII.wave.coord()) == len(cube_OII_ori.wave.coord())

    # # Analyze asymmetry and kinematics
    # seg_OII_3D, seg_OII = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OII)[1].data
    # if select_seg_OII:
    #     nums_seg_OII = np.setdiff1d(np.arange(1, np.max(seg_OII) + 1), nums_seg_OII)
    # seg_OII = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, 0)
    # seg_OII = np.where(seg_OII == 0 , seg_OII, 1)
    # flux_OII = np.where(seg_OII[np.newaxis, :, :], flux_OII, np.nan)
    # flux_OII = np.where(seg_OII_3D, flux_OII, np.nan)

    # plt.figure()
    # plt.plot(cube_OII.wave.coord(), np.nansum(flux_OII, axis=(1, 2)), '-')
    # plt.plot(flux_OII[:, 75, 80], '-')
    # plt.imshow(np.nansum(flux_OII, axis=0), origin='lower', cmap='viridis', vmin=0, vmax=1.5e-3)
    # plt.show()


    # Save the asymmetry values
    path_kinematics = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_V50S80_summary{}.txt'.format(addon)
    if os.path.exists(path_kinematics):
        t = Table.read(path_kinematics, format='ascii.fixed_width')
    else:
        t = Table(names=('cubename', 'V_5', 'V_95', 'S_80'),
                  dtype=('S15', 'f8', 'f8', 'f8'))
    if cubename in t['cubename']:
        index = np.where(t['cubename'] == cubename)[0][0]
        t['V_5'][index] = np.round(v50_5, 0)
        t['V_95'][index] = np.round(v50_95, 0)
        t['S_80'][index] = np.round(s80_mean, 0)
    else:
        t.add_row((cubename, np.round(v50_5, 0), np.round(v50_95, 0), np.round(s80_mean, 0)))
    t.write(path_kinematics, format='ascii.fixed_width', overwrite=True)

    # OIII
    # if os.path.exists(path_SB_OIII_kin):
    #     # Cubes
    #     cube_OIII = Cube(path_cube_OIII)
    #     cube_OIII_ori = Cube(path_cube_OIII_ori)
    #     cube_OIII_ori = cube_OIII_ori.select_lambda(np.min(cube_OIII.wave.coord()), np.max(cube_OIII.wave.coord()))
    #     flux_OIII = convolve(cube_OIII_ori.data * 1e-3, kernel_1)
    #     assert len(cube_OIII.wave.coord()) == len(cube_OIII_ori.wave.coord())
    #
    #     # Analyze asymmetry and kinematics
    #     seg_OIII_3D, seg_OIII = fits.open(path_3Dseg_OIII)[0].data, fits.open(path_3Dseg_OIII)[1].data
    #     SB_OIII_smoothed = np.nansum(np.where(seg_OIII_3D != 0, flux_OIII, np.nan), axis=0)
    #     seg_OIII_3D = np.where(seg_OIII[np.newaxis, :, :] == 1, seg_OIII_3D, 0)
    #     idx = np.argmax(np.sum(seg_OIII_3D, axis=(1, 2)), axis=0)
    #     SB_OIII_smoothed = np.where(SB_OIII_smoothed != 0, SB_OIII_smoothed, 3 * flux_OIII[idx, :, :])
    #     SB_OIII_smoothed *= 1.25 / 0.2 / 0.2
    #
    #     # Analyze asymetry and kinematics
    #     SB_OIII = fits.open(path_SB_OIII_kin)[1].data
    #     SB_OIII = np.where(center_mask, SB_OIII, np.nan)
    #     seg_OIII = np.where(center_mask, seg_OIII, 0)
    #     if select_seg_OIII:
    #         nums_seg_OIII = np.setdiff1d(np.arange(1, np.max(seg_OIII) + 1), nums_seg_OIII)
    #     seg_OIII_mask = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, -1)
    #     seg_OIII = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, 0)
    #     seg_OIII = np.where(seg_OIII == 0, seg_OIII, 1)

        # bkgrd_OIII = np.where(seg_OIII_mask == 0, SB_OIII, np.nan)
        # bkgrd_OIII_random = np.random.choice(bkgrd_OIII.flatten()[~np.isnan(bkgrd_OIII.flatten())],
        #                                     bkgrd_OIII.shape, replace=True).reshape(bkgrd_OIII.shape)
        # SB_OIII = np.where(seg_OIII_mask != -1, SB_OIII, bkgrd_OIII_random)





# CUBS+MUSE host nebula itself
EstimateKinematics(cubename='HE0435-5304', nums_seg_OII=[1], nums_seg_OIII=[1])
# EstimateKinematics(cubename='HE0153-4520')
EstimateKinematics(cubename='HE0226-4110', nums_seg_OII=[2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  nums_seg_OIII=[1, 5, 6, 8, 9, 10, 11, 16, 19])
EstimateKinematics(cubename='PKS0405-123', nums_seg_OII=[5, 7, 10, 11, 13, 16, 17, 20], nums_seg_OIII=[15])
EstimateKinematics(cubename='HE0238-1904', nums_seg_OII=[1, 6, 12, 13, 17, 19], select_seg_OII=True,
                  nums_seg_OIII=[1, 2, 4, 9, 13, 15, 17, 20], select_seg_OIII=True)
EstimateKinematics(cubename='3C57', nums_seg_OII=[2], nums_seg_OIII=[])
EstimateKinematics(cubename='PKS0552-640', nums_seg_OII=[2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                  nums_seg_OIII=[5, 6, 7, 8, 12, 15, 16, 17, 18, 20])
EstimateKinematics(cubename='J0110-1648', nums_seg_OII=[1], nums_seg_OIII=[2])
EstimateKinematics(cubename='J0454-6116', nums_seg_OII=[2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 17, 18], nums_seg_OIII=[2, 7, 9, 10, 18, 19])
EstimateKinematics(cubename='J2135-5316', nums_seg_OII=[2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19],
                  nums_seg_OIII=[4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20])
EstimateKinematics(cubename='J0119-2010', nums_seg_OII=[3, 4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20],
                  nums_seg_OIII=[7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20])
EstimateKinematics(cubename='HE0246-4101', nums_seg_OII=[1], select_seg_OII=True)
EstimateKinematics(cubename='J0028-3305', nums_seg_OII=[2], select_seg_OII=True)
EstimateKinematics(cubename='HE0419-5657', nums_seg_OII=[2, 4, 5], select_seg_OII=True)
EstimateKinematics(cubename='PB6291', nums_seg_OII=[2, 6, 7], select_seg_OII=True)
EstimateKinematics(cubename='Q0107-0235', nums_seg_OII=[1, 4, 5, 6], select_seg_OII=True)
EstimateKinematics(cubename='PKS2242-498', nums_seg_OII=[1, 2], select_seg_OII=True)
EstimateKinematics(cubename='PKS0355-483', nums_seg_OII=[2, 3, 4, 8, 9, 10, 11], select_seg_OII=True)
EstimateKinematics(cubename='HE0112-4145')
EstimateKinematics(cubename='HE0439-5254')
# EstimateKinematics(cubename='HE2305-5315')
EstimateKinematics(cubename='HE1003+0149')
EstimateKinematics(cubename='HE0331-4112', nums_seg_OII=[6], select_seg_OII=True)
EstimateKinematics(cubename='TEX0206-048', nums_seg_OII=[1, 8, 12, 13, 15, 20, 23, 26, 27, 28, 34, 57, 60, 79, 81,
                                                        101, 107, 108, 114, 118, 317, 547, 552],
                                                        select_seg_OII=True)
EstimateKinematics(cubename='Q1354+048', nums_seg_OII=[1, 2])
EstimateKinematics(cubename='J0154-0712', nums_seg_OII=[5])
EstimateKinematics(cubename='LBQS1435-0134', nums_seg_OII=[1, 3, 7], select_seg_OII=True)
EstimateKinematics(cubename='PG1522+101', nums_seg_OII=[2, 3, 8, 11], select_seg_OII=True)
# EstimateKinematics(cubename='HE2336-5540')
EstimateKinematics(cubename='PKS0232-04', nums_seg_OII=[2, 4, 5, 7])

# CUBS+MUSE host nebula plus galaxies
# EstimateKinematics(cubename='HE0435-5304', addon='_plus_galaxies')
# # EstimateKinematics(cubename='HE0153-4520', addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0226-4110', nums_seg_OII=[12, 14, 15, 16, 17, 20],
#                   nums_seg_OIII=[5, 11, 16, 19], addon='_plus_galaxies', savefig=True)
# EstimateKinematics(cubename='PKS0405-123', nums_seg_OII=[5], nums_seg_OIII=[15], addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0238-1904', addon='_plus_galaxies')
# EstimateKinematics(cubename='3C57', addon='_plus_galaxies')
# EstimateKinematics(cubename='PKS0552-640', nums_seg_OII=[2, 6, 7, 9, 10, 14, 18],
#                   nums_seg_OIII=[7, 9, 12, 19, 20], addon='_plus_galaxies')
# EstimateKinematics(cubename='J0110-1648', addon='_plus_galaxies')
# EstimateKinematics(cubename='J0454-6116', nums_seg_OII=[2, 6, 8, 13, 17, 18],
#                   nums_seg_OIII=[2, 9, 10, 18], addon='_plus_galaxies')
# EstimateKinematics(cubename='J2135-5316', nums_seg_OII=[3, 4, 10, 12, 13, 14, 16, 17, 18, 19],
#                   nums_seg_OIII=[4, 12, 13, 14, 15, 17, 19, 20], addon='_plus_galaxies')
# EstimateKinematics(cubename='J0119-2010', nums_seg_OII=[4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20],
#                   nums_seg_OIII=[7, 9, 11, 12, 14, 16, 17, 18], addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0246-4101', addon='_plus_galaxies')
# EstimateKinematics(cubename='J0028-3305', addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0419-5657', nums_seg_OII=[1], addon='_plus_galaxies')
# EstimateKinematics(cubename='PB6291', nums_seg_OII=[3, 5], addon='_plus_galaxies')
# EstimateKinematics(cubename='Q0107-0235', nums_seg_OII=[7], addon='_plus_galaxies')
# EstimateKinematics(cubename='PKS2242-498', addon='_plus_galaxies')
# EstimateKinematics(cubename='PKS0355-483', nums_seg_OII=[6, 14], addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0112-4145', addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0439-5254', addon='_plus_galaxies')
# # EstimateKinematics(cubename='HE2305-5315', nums_seg_OII=[5, 6, 7, 8], addon='_plus_galaxies')
# EstimateKinematics(cubename='HE1003+0149', addon='_plus_galaxies')
# EstimateKinematics(cubename='HE0331-4112', nums_seg_OII=[1, 2], addon='_plus_galaxies')
# EstimateKinematics(cubename='TEX0206-048', nums_seg_OII=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 22, 23,
#                                                         26, 27, 28, 34, 57, 60, 79, 81, 101, 107, 108, 114, 118, 317,
#                                                         547, 552],
#                                                         select_seg_OII=True, addon='_plus_galaxies')
# EstimateKinematics(cubename='Q1354+048', addon='_plus_galaxies')
# EstimateKinematics(cubename='J0154-0712', addon='_plus_galaxies')
# EstimateKinematics(cubename='LBQS1435-0134', addon='_plus_galaxies')
# EstimateKinematics(cubename='PG1522+101', nums_seg_OII=[6, 12], addon='_plus_galaxies')
# EstimateKinematics(cubename='HE2336-5540', ddon='_plus_galaxies')
# EstimateKinematics(cubename='PKS0232-04', nums_seg_OII=[4, 5, 7], addon='_plus_galaxies')





