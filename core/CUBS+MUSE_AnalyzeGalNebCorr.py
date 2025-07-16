import os
import aplpy
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)



# Generate quasar nebulae summary figure
morphology = np.array(["R", "N", "I", "R+I", "R+O+I", "O+R", "I", "R", "I", "O+I", "O+I",
                       "U+I", "U", "U+I", "U", "U", "R", "I", "R", "I", "N", "U", "U",
                       "F", "I", "R", "R", "R", "N", "R"])

area = np.array([1170,   35, 2740, 8180, 4820, 1930, 5180,  530, 4350, 2250,
                 3090, 2280,  500,  620,  410, 2130, 2260, 1310,  890,  970,
                 100, 1660,  520,10800, 2860, 1340, 2010, 1220,   30, 3670], dtype=float)

size = np.array([55,    7,   84,  129,  103,   71,  153,   29,  102,   83,
                 96,   74,   42,   35,   28,    90,   71,   50,   38,   47,
                 15,   53,   32,  200,  126,   63,   63,   50,    7,  116], dtype=float)

sigma_80 = np.array([87,   np.nan, 150, 106, 113, 151, 124,  91, 151, 107,
                     166,   129, 133, 154, 116, 136, 147, 142, 164, 246,
                     np.nan, 208, 196, 194, 123, 137, 261, 132, np.nan, 178], dtype=float)

# Quasar nebulae summary figure
plt.figure()
for i in range(len(morphology)):
    if len(morphology[i]) > 1:
        morphology[i] = morphology[i][-1]
    if morphology[i] == 'R':
        marker = '*'
    elif morphology[i] == 'I':
        marker = 'o'
    elif morphology[i] == 'F':
        marker = '+'
    elif morphology[i] == 'U':
        marker = 'D'
    else:
        marker = 's'
    plt.plot(sigma_80[i], size[i], marker=marker, color='red', markersize=10)
plt.xlabel(r'$\sigma_{80}$', size=25)
plt.ylabel(r'Size', size=25)
plt.legend()
plt.show()
raise ValueError('testing')


# Nebulae info
morph = ["I", "R+I", "R+O+I", "I", "I", "O+I", "O+I", "U+I", "U+I", "I", "I", "I", "I"]
N_gal = [10, 37, 34, 11, 18, 2, 7, 5, 3, 22, 23, 7, 18]
N_enc = [2, 12, 4, 3, 2, 0, 0, 2, 1, 0, 3, 1, 2]
size = [84, 129, 103, 153, 102, 83, 96, 74, 35, 90, 50, 47, 126]
area = [2740, 8180, 4820, 5180, 4350, 2250, 3090, 2280, 620, 2130, 1310, 970, 2860]




def ComputeCorr(cubename=None, HSTcentroid=False):
    # QSO information
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # V50, S80
    path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
    path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)
    v50 = fits.open(path_v50_plot)[1].data
    s80 = fits.open(path_s80_plot)[1].data
    hdr_v50 = fits.open(path_v50_plot)[1].header
    w = WCS(hdr_v50, naxis=2)
    center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
    c2 = w.world_to_pixel(center_qso)

    # Load galaxy information
    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        v_gal = data_gal['v']
        if HSTcentroid:
            ra_gal, dec_gal, type = data_gal['ra_HST'], data_gal['dec_HST'], data_gal['type']
        else:
            ra_gal, dec_gal, type = data_gal['ra_cor'], data_gal['dec_cor'], data_gal['type']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal, ra_hst, dec_hst = [], [], [], [], []

    c_gal = w.world_to_pixel(SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs'))
    x, y = np.meshgrid(np.arange(v50.shape[1]), np.arange(v50.shape[0]))
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)

    # Test 1
    # plt.figure()
    # val_array = []
    # for i in range(len(v_gal)):
    #     dis_i = np.sqrt((c_gal[0][i] - x) ** 2 + (c_gal[1][i] - y) ** 2) * 0.2 * 50 / 8
    #     weight = 1 / np.exp(-dis_i ** 2 / 2 / (130 ** 2))
    #     normalization = np.nansum(weight)
    #     val = np.nansum(weight * ((v_gal[i] - v50.ravel()) / s80.ravel()) ** 2) / normalization
    #     val_array.append(val)
    #     print('Galaxy {}: {:.2f}'.format(i, val))
    #     plt.text(c_gal[0][i], c_gal[1][i], str(int(val)), fontsize=10, color='black', ha='center', va='center')
    # plt.plot(c_gal[0], c_gal[1], 'o', markersize=10, color='red', label='Galaxies')
    # plt.plot(c2[0], c2[1], 'o', markersize=10, color='blue', label='QSO')
    # # plt.xlim(0, v50.shape[1])
    # # plt.ylim(0, v50.shape[0])
    # plt.show()

    # Test 2
    plt.figure()
    # val_array = []
    for i in range(len(v_gal)):
        dis_i = np.sqrt((c_gal[0][i] - x) ** 2 + (c_gal[1][i] - y) ** 2) * 0.2 * 50 / 8
        # weight = np.exp(-dis_i ** 2 / 2 / (130 ** 2))
        # normalization = np.nansum(weight)

        # Specify the threshold
        if c_gal[1][i] > np.shape(v50)[1] or c_gal[0][i] > np.shape(v50)[0]:
            inside_nebula = False
        else:
            inside_nebula = ~np.isnan(v50[int(c_gal[1][i]), int(c_gal[0][i])])
        nebula_factor = 1.0 if inside_nebula else 0.4
        far_sigma = np.abs(((v_gal[i] - v50.ravel()) / s80.ravel()))
        scale_length = 130
        effective_threshold = 2 * (1 - dis_i / scale_length) * nebula_factor
        within_threshold = far_sigma <= effective_threshold
        ratio = np.sum(within_threshold) / len(far_sigma)


        print('Galaxy {}: {:.2f}'.format(i, ratio))
        plt.text(c_gal[0][i], c_gal[1][i], str(np.round(ratio, 2)), fontsize=10, color='black', ha='center', va='center')
    plt.plot(c_gal[0], c_gal[1], 'o', markersize=10, color='red', label='Galaxies')
    plt.plot(c2[0], c2[1], 'o', markersize=10, color='blue', label='QSO')
    # plt.xlim(0, v50.shape[1])
    # plt.ylim(0, v50.shape[0])
    plt.show()

# ComputeCorr(cubename='PKS0405-123', HSTcentroid=True)