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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Nebulae info
morph = ["I", "R+I", "R+O+I", "I", "I", "O+I", "O+I", "U+I", "U+I", "I", "I", "I", "I"]
N_gal = [10, 37, 34, 11, 18, 2, 7, 5, 3, 22, 23, 7, 18]
N_enc = [2, 12, 4, 3, 2, 0, 0, 2, 1, 0, 3, 1, 2]
size = [84, 129, 103, 153, 102, 83, 96, 74, 35, 90, 50, 47, 126]
area = [2740, 8180, 4820, 5180, 4350, 2250, 3090, 2280, 620, 2130, 1310, 970, 2860]


def ComputeCorr(cubename=None, HSTcentroid=False, scale_length=None, ):
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
        # scale_length = 130
        effective_threshold = 2 * (1 - dis_i / scale_length) * nebula_factor
        within_threshold = far_sigma <= effective_threshold
        ratio = np.sum(within_threshold) / len(far_sigma)
        print('Galaxy {}: {:.2f}'.format(i, ratio))
        plt.text(c_gal[0][i], c_gal[1][i], str(np.round(ratio, 2)), fontsize=10, color='black', ha='center', va='center')
    plt.imshow(v50, origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
    plt.plot(c_gal[0], c_gal[1], 'o', markersize=10, color='red', label='Galaxies')
    plt.plot(c2[0], c2[1], 'o', markersize=10, color='blue', label='QSO')
    plt.xlim(0, v50.shape[1])
    plt.ylim(0, v50.shape[0])
    plt.show()

# ComputeCorr(cubename='HE0226-4110', HSTcentroid=True, scale_length=84)
# ComputeCorr(cubename='PKS0405-123', HSTcentroid=True, scale_length=130)
# ComputeCorr(cubename='HE0238-1904', HSTcentroid=True, scale_length=103)
# ComputeCorr(cubename='PKS0552-640', HSTcentroid=True, scale_length=153)
ComputeCorr(cubename='3C57', HSTcentroid=True, scale_length=71)