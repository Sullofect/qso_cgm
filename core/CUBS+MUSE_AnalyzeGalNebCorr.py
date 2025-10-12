import os
import aplpy
import numpy as np
import matplotlib as mpl
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
rc('xtick.minor', size=4, visible=True)
rc('ytick.minor', size=4, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Part 1
# Nebulae info
morph = ["I", "R+I", "R+O+I", "I", "I", "O+I", "O+I", "U+I", "U+I", "I", "I", "I"]
N_gal = [10, 37, 34, 11, 18, 2, 7, 5, 3, 23, 7, 18]
N_enc = [2, 12, 4, 3, 2, 0, 0, 2, 1, 3, 1, 2]
size = [84, 129, 103, 153, 102, 83, 96, 74, 35, 50, 47, 126]
area = [2740, 8180, 4820, 5180, 4350, 2250, 3090, 2280, 620, 1310, 970, 2860]

# plt.figure(figsize=(5, 5), dpi=300)
# plt.plot(N_enc, size, 'o', color='k', ms=8)
# plt.xlabel(r'$N_{\rm enc}$', size=25)
# plt.ylabel('Size [kpc]', size=25)
# plt.xlim()
# plt.ylim(30, 160)
# plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_Ngal_Size.png', bbox_inches='tight')

# fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharey=True)
# fig.subplots_adjust(wspace=0.0, hspace=0.0)
# ax[0].plot(N_gal, size, 'o', color='k', ms=8)
# ax[1].plot(N_enc, size, 'o', color='k', ms=8)
# ax[0].set_xlabel(r'$N_{\rm gal}$', size=25)
# ax[1].set_xlabel(r'$N_{\rm enc}$', size=25)
# ax[0].set_ylabel('Size [kpc]', size=25)
# ax[0].set_xlim()
# ax[0].set_ylim(30, 160)
# fig.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_Ngal_Size.png', bbox_inches='tight')



# Compute correlation for individual galaxies
def ComputeCorr(cubename=None, scale_length=None, vmax=300, savefig=False, nums_seg_OII=None, select_seg_OII=False,
                nums_seg_OIII=None, select_seg_OIII=False):
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

    # Load data
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    line_OII, line_OIII = 'OII', 'OIII'

    # OII SBs
    if cubename == 'TEX0206-048':
        str_zap = '_zapped'
    else:
        str_zap = ''

    # Load the segmentation map
    path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
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


    # Load galaxy information
    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    try:
        data_gal = fits.open(path_gal)[1].data
        v_gal = data_gal['v']
        try:
            ra_gal, dec_gal, type = data_gal['ra_HST'], data_gal['dec_HST'], data_gal['type']
        except KeyError:
            ra_gal, dec_gal, type = data_gal['ra_cor'], data_gal['dec_cor'], data_gal['type']
    except FileNotFoundError:
        print('No galaxies info')
        ra_gal, dec_gal, v_gal, ra_hst, dec_hst = [], [], [], [], []
    c_gal = w.world_to_pixel(SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs'))
    x, y = np.meshgrid(np.arange(v50.shape[1]), np.arange(v50.shape[0]))
    x, y = x.flatten(), y.flatten()
    pixcoord = PixCoord(x=x, y=y)

    # Compute the velocity score
    score_array = []

    if savefig:
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
        plt.figure(figsize=(5, 5), dpi=300)

    # Compute the physical scale at the redshift of the quasar
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
    arcsec = (50 / d_A_kpc) * 206265

    # Specify the threshold
    for i in range(len(v_gal)):
        dis_i = np.sqrt((c_gal[0][i] - x) ** 2 + (c_gal[1][i] - y) ** 2) * 0.2 * 50 / arcsec

        # Determine if a galaxy is inside the nebula
        if 0 <= c_gal[1][i] <= np.shape(v50)[1] and 0 <= c_gal[0][i] <= np.shape(v50)[0]:
            inside_nebula = ~np.isnan(v50[int(c_gal[1][i]), int(c_gal[0][i])])
        else:
            inside_nebula = False

        # Compute the Association
        nebula_factor = 1.0 if inside_nebula else 0.5
        far_sigma = np.abs(((v_gal[i] - v50.ravel()) / s80.ravel()))
        val = dis_i / scale_length
        val[val >= 1.0] = 0.8
        effective_threshold = 2 * (1 - val) * nebula_factor
        within_threshold = far_sigma <= effective_threshold
        ratio = np.sum(within_threshold) / len(v50[~np.isnan(v50)])


        # Compute the ratio
        if 0 <= c_gal[0][i] <= 150 and 0 <= c_gal[1][i] <= 150:
            print('Galaxy {}: {:.2f}'.format(i, ratio))
            score_array.append(ratio)
        if savefig:
            if 0 <= c_gal[0][i] <= 150 and 0 <= c_gal[1][i] <=150:
                plt.text(c_gal[0][i], c_gal[1][i], '{:.2f}'.format(ratio),
                         fontsize=15, color='black', ha='center', va='center')
    if savefig:
        plt.imshow(v50, origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
        plt.scatter(c_gal[0], c_gal[1], marker='o', s=140, c='white', edgecolor='k', facecolor='white', label='Galaxies')
        plt.scatter(c_gal[0], c_gal[1], marker='o', s=120, c='none', edgecolor=plt.cm.coolwarm(norm(v_gal)), facecolor='none', label='Galaxies')
        plt.xlim(0, v50.shape[1])
        plt.ylim(0, v50.shape[0])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_{}_corr.png'.format(cubename), bbox_inches='tight')
    return np.asarray(score_array)

L = np.array([["HE0226-4110",     150,  84, [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], False,
               [1, 5, 6, 8, 9, 10, 11, 16, 19], False],
              ["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False],
              ["HE0238-1904",     113, 103, [1, 6, 12, 13, 17, 19], True, [1, 2, 4, 9, 13, 15, 17, 20], True],
              ["PKS0552-640",     124, 153, [2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], False,
               [5, 6, 7, 8, 12, 15, 16, 17, 18, 20], False],
              ["J0454-6116",      151, 102, [2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 17, 18], False,
               [2, 7, 9, 10, 18, 19], False],
              ["J0119-2010",      166,  96, [3, 4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20], False,
               [7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20], False],
              ["HE0246-4101",     129,  74, [1], True, [], False],
              ["PKS0355-483",     142,  50, [2, 3, 4, 8, 9, 10, 11], True, [], False],
              ["HE0439-5254",     246,  47, [], False, [], False],
              ["TEX0206-048",     194, 200, [1, 8, 12, 13, 15, 20, 23, 26, 27, 28, 34, 57, 60, 79, 81,
                                             101, 107, 108, 114, 118, 317, 547, 552], True, [], False],
              ["Q1354+048",       123, 126, [1, 2], False, [], False]], dtype=object)
S_BR = np.array([["HE0435-5304",      87,  55, [1], False, [1], False],
                 ["3C57",            151,  71, [2], False, [], False],
                 ["J0110-1648",       91,  29, [1], False, [2], False],
                 ["HE0112-4145",     164,  38, [], False, [], False],
                 ["J0154-0712",      137,  63, [5], False, [], False],
                 ["LBQS1435-0134",    261,  63, [1, 3, 7], True, [], False ]], dtype=object)
S = np.array([["J0028-3305",      133,  42, [2], True, [], False],
              ["HE0419-5657",     154,  35, [2, 4, 5], True, [], False],
              ["PB6291",          116,  28, [2, 6, 7], True, [], False],
              ["HE1003+0149",     208,  53, [], False, [], False],
              ["HE0331-4112",     196,  32, [6], True, [], False]], dtype=object)
A = np.array([["J2135-5316",      107,  83, [2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19], False,
               [4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20], False],
              ["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
              ["PKS2242-498",     147,  71, [1, 2], True, [], False],
              ["PG1522+101",      132,  50, [2, 3, 8, 11], True, [], False],
              ["PKS0232-04",      178, 116, [2, 4, 5, 7], False, [], False]], dtype=object)


def SummarizeCorr(L=None, S_BR=None, S=None, A=None):
    # Compute association for L Type
    scale_length_array_L, scale_length_array_S, scale_length_array_A = np.array([]), np.array([]), np.array([])
    largeThan05_array_L, largeThan05_array_S, largeThan05_array_A = np.array([]), np.array([]), np.array([])
    score_L = np.array([])
    for i in range(len(L)):
        score_array = ComputeCorr(cubename=L[i][0], scale_length=L[i][2],
                                  nums_seg_OII=L[i][3], select_seg_OII=L[i][4],
                                  nums_seg_OIII=L[i][5], select_seg_OIII=L[i][6],
                                  savefig=True)
        scale_length_array_L = np.hstack((scale_length_array_L, L[i][2]))
        largeThan05_array_L = np.hstack((largeThan05_array_L, len(score_array[score_array > 0.2])))
        score_L = np.hstack((score_L, score_array))

    # Compute association for S Type
    score_S_BR = np.array([])
    for i in range(len(S_BR)):
        score_array = ComputeCorr(cubename=S_BR[i][0], scale_length=S_BR[i][2],
                                  nums_seg_OII=S_BR[i][3], select_seg_OII=S_BR[i][4],
                                  nums_seg_OIII=S_BR[i][5], select_seg_OIII=S_BR[i][6],
                                  savefig=True)
        scale_length_array_S = np.hstack((scale_length_array_S, S_BR[i][2]))
        largeThan05_array_S = np.hstack((largeThan05_array_S, len(score_array[score_array > 0.2])))
        score_S_BR = np.hstack((score_S_BR, score_array))

    score_S = np.array([])
    for i in range(len(S)):
        score_array = ComputeCorr(cubename=S[i][0], scale_length=S[i][2],
                                  nums_seg_OII=S[i][3], select_seg_OII=S[i][4],
                                  nums_seg_OIII=S[i][5], select_seg_OIII=S[i][6],
                                  savefig=True)
        scale_length_array_S = np.hstack((scale_length_array_S, S[i][2]))
        largeThan05_array_S = np.hstack((largeThan05_array_S, len(score_array[score_array > 0.2])))
        score_S = np.hstack((score_S, score_array))
    score_S = np.hstack((score_S_BR, score_S))

    score_A = np.array([])
    for i in range(len(A)):
        score_array = ComputeCorr(cubename=A[i][0], scale_length=A[i][2],
                                  nums_seg_OII=A[i][3], select_seg_OII=A[i][4],
                                  nums_seg_OIII=A[i][5], select_seg_OIII=A[i][6],
                                  savefig=True)
        scale_length_array_A = np.hstack((scale_length_array_A, A[i][2]))
        largeThan05_array_A = np.hstack((largeThan05_array_A, len(score_array[score_array > 0.2])))
        score_A = np.hstack((score_A, score_array))

    print('average L', sum(score_L > 0.5) / len(L))
    print('average S', sum(score_S > 0.5) / (len(S_BR) + len(S)))

    score_L = score_L[score_L >= 0.2]
    score_S = score_S[score_S >= 0.2]
    score_A = score_A[score_A >= 0.2]

    # Histogram
    bins = np.linspace(0, 1, 11)
    mid = (bins[1:] + bins[:-1]) / 2
    mid = np.append(mid, mid[-1] + mid[-1] - mid[-2])
    counts_L, _ = np.histogram(score_L, bins=bins)
    counts_L = np.append(counts_L, counts_L[-1])
    counts_S, _ = np.histogram(score_S, bins=bins)
    counts_S = np.append(counts_S, counts_S[-1])
    counts_A, _ = np.histogram(score_A, bins=bins)
    counts_A = np.append(counts_A, counts_A[-1])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300, constrained_layout=True)
    ax.step(mid, counts_L, where="mid", alpha=0.8, color="k", linestyle="-", linewidth=2, label=r'Irregular, large-scale')
    ax.step(mid, counts_S, where="mid", alpha=0.8, color="red", linestyle="--", linewidth=2, label=r'Host-galaxy-scale')
    ax.step(mid, counts_A, where="mid", alpha=0.8, color="blue", linestyle=":", linewidth=2, label=r'Complex Morphology')
    ax.set_xlim(0.1, 1.0)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, np.ceil(ymax))
    ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1, 2))
    # ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_xlabel(r'KAF', size=25)
    ax.set_ylabel(r'$N$', size=25)
    ax.legend(loc='upper right', fontsize=20)
    plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_CorrScore_LType.png', bbox_inches='tight')

    # Scatter plot
    plt.figure(figsize=(5, 5), dpi=300, constrained_layout=True)
    plt.scatter(scale_length_array_L, largeThan05_array_L, marker="o", alpha=0.8, s=50, color='k')
    plt.scatter(scale_length_array_S, largeThan05_array_S, marker="s", alpha=0.8, s=50, color='red')
    plt.scatter(scale_length_array_A, largeThan05_array_A, marker="^", alpha=0.8, s=50, color='blue')
    plt.xlabel(r'$\rm Size \, [kpc]$', size=25)
    plt.ylabel(r'$N_{>\,0.2}$', size=25)
    plt.xlim(20, 225)
    plt.ylim(-0.5, 9)
    plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_CorrScore_ScaleLength.png', bbox_inches='tight')





# Test
# ComputeCorr(cubename='HE0226-4110', HSTcentroid=True, scale_length=84)
# ComputeCorr(cubename='PKS0405-123', HSTcentroid=True, scale_length=130)
# ComputeCorr(cubename='HE0238-1904', HSTcentroid=True, scale_length=103)
# ComputeCorr(cubename='PKS0552-640', HSTcentroid=True, scale_length=153)
# ComputeCorr(cubename='3C57', HSTcentroid=True, scale_length=71)
# ComputeCorr(cubename='Q0107-0235', HSTcentroid=True, scale_length=90)
# ComputeCorr(cubename='TEX0206-048', HSTcentroid=True, scale_length=200)
# ComputeCorr(cubename='PB6291', scale_length=28, savefig=True, nums_seg_OII=[2, 6, 7], select_seg_OII=True)

SummarizeCorr(L=L, S_BR=S_BR, S=S, A=A)