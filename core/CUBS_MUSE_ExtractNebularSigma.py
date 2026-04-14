import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from photutils.profiles import RadialProfile
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# QSO information
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')

c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239

def Bin(x, y, bins=None):
    n, edges = np.histogram(x, bins=bins)
    len_bins = len(bins) - 1
    y_mean, y_max, y_min, y_std = np.zeros(len_bins), np.zeros(len_bins), np.zeros(len_bins), np.zeros(len_bins)
    x_mean = (edges[:-1] + edges[1:]) / 2
    for i in range(len_bins):
        if n[i] == 0:
            y_mean[i], y_max[i], y_min[i], y_std[i] = np.nan, np.nan, np.nan, np.nan
        else:
            mask = (x > edges[i]) * (x <= edges[i + 1])
            if len(y[mask]) == 0:
                y_mean[i], y_max[i], y_min[i], y_std[i] = np.nan, np.nan, np.nan, np.nan
            else:
                y_mean[i], y_max[i], y_min[i], y_std[i] = np.nanmean(y[mask]), np.nanmax(y[mask]), \
                                                          np.nanmin(y[mask]), np.nanstd(y[mask])
    return x_mean, y_mean, y_max, y_min, y_std

def CalculateVelocityDifference(cubename=None, zapped=False, NLR='', UseDataSeg=(1.5, 'gauss', None, None)):
    # Fit parameter
    fit_param = {"OII": 1, "OII_2nd": 0, 'ResolveOII': True, 'r_max': 1.5,
                 'OII_center': wave_OII3728_vac, "OIII": 1, "OIII_2nd": 0}

    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    if (1 + z_qso) * wave_OIII5008_vac >= 9350:
        print('OIII is not covered')
        fit_param['OIII'] = 0
    else:
        print('OIII coverage is covered')

    # Define lines
    if fit_param['OII'] >= 1 and fit_param['OIII'] == 0:
        line = 'OII'
    elif fit_param['OII'] == 0 and fit_param['OIII'] >= 1:
        line = 'OIII'
    else:
        line = 'OII+OIII'

    # if zapped
    if cubename == 'TEX0206-048':
        zapped = True

    if zapped:
        str_zap = '_zapped'
    else:
        str_zap = ''

    path_fit = '../../MUSEQuBES+CUBS/fit_kin/{}{}_fit_{}{}_{}_{}_{}_{}_{}_{}.fits'.\
        format(cubename, str_zap, line, NLR, fit_param['ResolveOII'], int(fit_param['OII_center']), *UseDataSeg)

    hdul_fit = fits.open(path_fit)
    pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, \
    da_OII, b_OII, db_OII, a_OIII, da_OIII, b_OIII, db_OIII, chisqr, redchi = hdul_fit[:23]
    pri, fs, v, z, dz, sigma, dsigma, flux_OII_fit, dflux_OII_fit, flux_OIII_fit, dflux_OIII_fit, r, dr, a_OII, \
    da_OII, b_OII, db_OII, a_OIII, da_OIII, b_OIII, db_OIII, \
    chisqr, redchi = pri.data, fs.data, v.data, z.data, dz.data, sigma.data, dsigma.data, \
                     flux_OII_fit.data, dflux_OII_fit.data, flux_OIII_fit.data, dflux_OIII_fit.data, r.data, \
                     dr.data, a_OII.data, da_OII.data, b_OII.data, db_OII.data, a_OIII.data, da_OIII.data, \
                     b_OIII.data, db_OIII.data, chisqr.data, redchi.data


    vdiff = abs(v[0, :, :] - v[1, :, :])



    # Select these spaxels in sigma and print the mean vdiff, min vdiff, max vdiff, sigma_1 and sigma_2
    print(f'{cubename}')
    print('mean vdiff: {}, min vdiff:{}, max vdiff{}'.format(np.nanmean(vdiff), np.nanmin(vdiff), np.nanmax(vdiff)))

    # Select spaxels with vdiff is not nan
    mask = ~np.isnan(vdiff)
    print('sigma_1: {}, sigma_2: {}'.format(np.nanmean(sigma[0, :, :][mask]), np.nanmean(sigma[1, :, :][mask])))

    # min max
    print('min sigma_1: {}, min sigma_2: {}'.format(np.nanmin(sigma[0, :, :][mask]), np.nanmin(sigma[1, :, :][mask])))
    print('max sigma_1: {}, max sigma_2: {}'.format(np.nanmax(sigma[0, :, :][mask]), np.nanmax(sigma[1, :, :][mask])))


    plt.figure()
    plt.hist(vdiff[mask].flatten(), bins='auto')
    # plt.imshow(sigma[2, :, :], origin='lower', cmap='viridis', vmin=-200, vmax=0)
    plt.show()

def ExtractRadialProfile(allFields=None):
    r_stack, sigma_stack = [], []
    r_L_stack, r_S_stack, r_A_stack = [], [], []
    sigma_L_stack, sigma_S_stack, sigma_A_stack = [], [], []
    r_RL_stack, r_RQ_stack, r_RJ_stack = [], [], []
    sigma_RL_stack, sigma_RQ_stack, sigma_RJ_stack = [], [], []

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.00)
    for i, cubename in enumerate(allFields[:, 0]):
        RL = False
        jet = False
        data_qso_i = data_qso[data_qso['name'] == cubename]
        ra_qso, dec_qso, z_qso, logRL = data_qso_i['ra_GAIA'][0], data_qso_i['dec_GAIA'][0], data_qso_i['redshift'][0], \
                                        data_qso_i['logRL'][0]
        print(f'QSO {cubename}, with redshift is {z_qso}')

        if cubename == 'PKS0405-123' or cubename == '3C57' or cubename == 'J0110-1648' or cubename == 'Q0107-0235' \
                or cubename == 'PKS2242-498' or cubename == 'PKS0232-04':
            print('This is a radio-loud quasar and has a jet')
            jet = True

        if logRL >= 1:
            print('This is a radio-loud quasar')
            RL = True

        #
        path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
        path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)

        # Load the data and header
        hdul_v50 = fits.open(path_v50_plot)
        hdr_v50, v50 = hdul_v50[1].header, hdul_v50[1].data
        w = WCS(hdr_v50, naxis=2)
        hdul_s80 = fits.open(path_s80_plot)
        s80 = hdul_s80[1].data

        # Mask out emission associated with galaxies
        # Load data
        nums_seg_OII = allFields[i, 3]
        select_seg_OII = allFields[i, 4]
        nums_seg_OIII = allFields[i, 5]
        select_seg_OIII = allFields[i, 6]

        UseSeg = (1.5, 'gauss', 1.5, 'gauss')
        line_OII, line_OIII = 'OII', 'OIII'

        # OII SBs
        if cubename == 'TEX0206-048':
            str_zap = '_zapped'
        else:
            str_zap = ''

        # Load the segmentation map
        path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line_OII, *UseSeg)
        path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        if cubename == 'PKS0552-640':
            path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
                format(cubename, str_zap, line_OIII, *UseSeg)
        elif cubename == 'HE0226-4110':
            path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
                format(cubename, str_zap, line_OII, *UseSeg)

        seg_OII = fits.open(path_3Dseg_OII)[1].data
        if select_seg_OII:
            nums_seg_OII = np.setdiff1d(np.arange(1, np.max(seg_OII) + 1), nums_seg_OII)
        seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)

        # Fix v50 and s80 according to [O II] seg
        v50 = np.where(seg_OII_mask != -1, v50, np.nan)
        s80 = np.where(seg_OII_mask != -1, s80, np.nan)

        # Fix v50 and s80 according to [O III] seg
        if os.path.exists(path_3Dseg_OIII):
            seg_OIII_3D, seg_OIII = fits.open(path_3Dseg_OIII)[0].data, fits.open(path_3Dseg_OIII)[1].data
            if select_seg_OIII:
                nums_seg_OIII = np.setdiff1d(np.arange(1, np.max(seg_OIII) + 1), nums_seg_OIII)
            seg_OIII_mask = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, -1)

            # Fix v50 and s80 accordingly to [O II]
            v50 = np.where(seg_OIII_mask != -1, v50, np.nan)
            s80 = np.where(seg_OIII_mask != -1, s80, np.nan)

        # Draw the radial profile
        edge_radii = np.arange(5, 115, 10)
        d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
        edge_radii_pixel = edge_radii * 206265 / (0.2 * d_A_kpc) # Convert from kpc to pixel using the WCS information
        xycen = SkyCoord(ra_qso, dec_qso, unit='deg').to_pixel(w)
        rp = RadialProfile(s80, xycen, edge_radii_pixel, error=None, mask=None)

        # Convert from pixel to kpc for plotting
        d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
        r_kpc = rp.radius * 0.2 * d_A_kpc / 206265 # Convert from pixel to kpc using the WCS information
        profile = rp.profile

        # Stack all of them
        r_stack.extend(r_kpc)
        sigma_stack.extend(profile)

        # Stack by morphogical-kinematic type
        if i <= 10:  # L
            r_L_stack.extend(r_kpc)
            sigma_L_stack.extend(profile)
        elif 11 <= i < 22:
            r_S_stack.extend(r_kpc)
            sigma_S_stack.extend(profile)
        else:
            r_A_stack.extend(r_kpc)
            sigma_A_stack.extend(profile)

        # Stack by radio property
        if RL:
            r_RL_stack.extend(r_kpc)
            sigma_RL_stack.extend(profile)
            if jet:
                r_RJ_stack.extend(r_kpc)
                sigma_RJ_stack.extend(profile)
        else:
            r_RQ_stack.extend(r_kpc)
            sigma_RQ_stack.extend(profile)

    # Compute the mean and median
    r_stack, sigma_stack = np.asarray(r_stack), np.asarray(sigma_stack)
    r_L_stack, r_S_stack, r_A_stack = np.asarray(r_L_stack), np.asarray(r_S_stack), np.asarray(r_A_stack)
    sigma_L_stack, sigma_S_stack, sigma_A_stack = np.asarray(sigma_L_stack), np.asarray(sigma_S_stack), \
                                                  np.asarray(sigma_A_stack)

    r_RL_stack, r_RQ_stack, r_RJ_stack = np.asarray(r_RL_stack), np.asarray(r_RQ_stack), np.asarray(r_RJ_stack)
    sigma_RL_stack, sigma_RQ_stack, sigma_RJ_stack = np.asarray(sigma_RL_stack), np.asarray(sigma_RQ_stack), \
                                                  np.asarray(sigma_RJ_stack)

    # Profile
    r_mean, sigma_mean, sigma_max, sigma_min, sigma_std = Bin(r_stack, sigma_stack, bins=edge_radii)
    r_L_mean, sigma_L_mean, sigma_L_max, sigma_L_min, sigma_L_std = Bin(r_L_stack, sigma_L_stack, bins=edge_radii)
    r_S_mean, sigma_S_mean, sigma_S_max, sigma_S_min, sigma_S_std = Bin(r_S_stack, sigma_S_stack, bins=edge_radii)
    r_A_mean, sigma_A_mean, sigma_A_max, sigma_A_min, sigma_A_std = Bin(r_A_stack, sigma_A_stack, bins=edge_radii)
    r_RL_mean, sigma_RL_mean, sigma_RL_max, sigma_RL_min, sigma_RL_std = Bin(r_RL_stack, sigma_RL_stack, bins=edge_radii)
    r_RQ_mean, sigma_RQ_mean, sigma_RQ_max, sigma_RQ_min, sigma_RQ_std = Bin(r_RQ_stack, sigma_RQ_stack, bins=edge_radii)
    r_RJ_mean, sigma_RJ_mean, sigma_RJ_max, sigma_RJ_min, sigma_RJ_std = Bin(r_RJ_stack, sigma_RJ_stack, bins=edge_radii)

    # Plot the mean profiles
    ax[0].plot(r_mean, sigma_mean, '-', color='orange', alpha=0.8, linewidth=2.0, label='All')
    ax[0].errorbar(10, 75, yerr=np.nanmean(sigma_std), fmt='none', ecolor='k', alpha=1.0, capsize=3)
    ax[0].plot(r_RL_mean, sigma_RL_mean, '-', color='darkred', alpha=0.8, linewidth=1.7, label='Radio-loud')
    ax[0].plot(r_RJ_mean, sigma_RJ_mean, '--', color='darkred', alpha=0.8, linewidth=1.7, label='Radio-loud with jet')
    ax[0].plot(r_RQ_mean, sigma_RQ_mean, '-', color='#7B2CBF', alpha=0.8, linewidth=1.7, label='Radio-quiet')

    ax[1].plot(r_mean, sigma_mean, '-', color='orange', alpha=0.8, linewidth=2.0, label='All')
    ax[1].plot(r_L_mean, sigma_L_mean, '-', color='k', alpha=0.8, linewidth=1.7, label='Irregular, large-scale')
    ax[1].plot(r_S_mean, sigma_S_mean, '-', color="#C92A2A", alpha=0.8, linewidth=1.7, label='Host-galaxy-scale')
    ax[1].plot(r_A_mean, sigma_A_mean, '-', color='#3B5BDB', alpha=0.8, linewidth=1.7,
               label='Complex morphology \n and kinematics')

    # Figure
    ax[0].set_title('Radio Classes', size=25, x=0.5, y=0.05)
    ax[1].set_title('Morphological Classes', size=25, x=0.5, y=0.05)
    ax[0].set_xlabel('Radius [kpc]', size=25)
    ax[1].set_xlabel('Radius [kpc]', size=25)
    ax[0].set_xlim(0, 120)
    ax[0].set_ylim(0, 300)
    ax[0].set_ylabel(r'$\rm{Radial \ Profile \ of} \ \sigma \rm \, [km \, s^{-1}]$', size=25)
    ax[0].legend(loc='upper right', fontsize=15)
    ax[1].legend(loc='upper right', fontsize=15)
    plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_SigmaProfile.png', bbox_inches='tight')

L = np.array([["HE0226-4110",     150,  84, [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], False,
               [1, 5, 6, 8, 9, 10, 11, 16, 19], False],
              ["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False],
              ["HE0238-1904",     113, 103, [1, 6, 12, 13, 17, 19], True, [1, 2, 4, 9, 13, 15, 17, 20], True],
              ["PKS0552-640",     124, 153, [2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], False,
               [5, 6, 7, 8, 12, 15, 16, 17, 18, 20], False],
              ["J0454-6116",      151, 102, [2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 17, 18], False,
               [2, 7, 9, 10, 18, 19], False],
              ["J0119-2010",      166,  96, [3, 4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20], False,
               [2, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20], False],
              ["HE0246-4101",     129,  74, [1], True, [], False],
              ["PKS0355-483",     142,  50, [2, 3, 4, 8, 9, 10, 11], True, [], False],
              ["HE0439-5254",     246,  47, [], False, [], False],
              ["TEX0206-048",     194, 200, [1, 8, 12, 13, 15, 20, 23, 26, 27, 28, 34, 57, 60, 79, 81,
                                             101, 107, 108, 114, 118, 317, 547, 552], True, [], False],
              ["Q1354+048",       123, 126, [1, 2], False, [], False]], dtype=object)
S = np.array([["HE0435-5304",      87,  55, [1], False, [1], False],
              ["3C57",            151,  71, [2], False, [], False],
              ["J0110-1648",       91,  29, [1], False, [2], False],
              ["HE0112-4145",     164,  38, [], False, [], False],
              ["J0154-0712",      137,  63, [5], False, [], False],
              ["LBQS1435-0134",    261,  63, [1, 3, 7], True, [], False ],
              ["PG1522+101",       132, 50, [2, 3, 8, 11], True, [], False],
              ["J0028-3305",      133,  42, [2], True, [], False],
              ["HE0419-5657",     154,  35, [2, 4, 5], True, [], False],
              ["PB6291",          116,  28, [2, 6, 7], True, [], False],
              ["HE1003+0149",     208,  53, [3], False, [], False],
              ["HE0331-4112",     196,  32, [6], True, [], False]], dtype=object)

A = np.array([["J2135-5316",      107,  83, [2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19], False,
               [4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20], False],
              ["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
              ["PKS2242-498",     147,  71, [1, 2], True, [], False],
              ["PKS0232-04",      178, 116, [2, 4, 5, 7], False, [], False]], dtype=object)
allType = np.vstack((L, S, A))

# CalculateVelocityDifference(cubename='3C57')
# CalculateVelocityDifference(cubename='J2135-5316')
# CalculateVelocityDifference(cubename='J0119-2010')

# ExtractRadialProfile(cubename_list=['3C57'], label='test')
# ExtractRadialProfile(cubename_list=['HE0226-4110', 'PKS0405-123', 'HE0238-1904', 'PKS0552-640', 'J0454-6116',
#                                     'J0119-2010', 'HE0246-4101', 'PKS0355-483', 'HE0439-5254', 'TEX0206-048',
#                                     'Q1354+048'],
#                      label='irregular, large-scale')
# ExtractRadialProfile(cubename_list=['HE0435-5304', '3C57', 'J0110-1648', 'HE0112-4145', 'J0154-0712',
#                                     'LBQS1435-0134', 'J0028-3305', 'HE0419-5657', 'PB6291', 'HE1003+0149',
#                                     'HE0331-4112'],
#                      label='host-galaxy-scale')
# ExtractRadialProfile(cubename_list=['J2135-5316', 'Q0107-0235', 'PKS2242-498', 'PG1522+101', 'PKS0232-04'],
#                      label='complex morphology')



ExtractRadialProfile(allFields=allType)
# ExtractRadialProfile(cubename_list=['HE0435-5304', '3C57', 'J0110-1648', 'HE0112-4145', 'J0154-0712',
#                                     'LBQS1435-0134', 'J0028-3305', 'HE0419-5657', 'PB6291', 'HE1003+0149',
#                                     'HE0331-4112'],
#                      label='host-galaxy-scale')
# ExtractRadialProfile(cubename_list=['J2135-5316', 'Q0107-0235', 'PKS2242-498', 'PG1522+101', 'PKS0232-04'],
#                      label='complex morphology')