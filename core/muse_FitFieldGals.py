import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from regions import Regions
from scipy.stats import norm
from astropy import units as u
from scipy.optimize import minimize
from muse_compare_z import compare_z
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12


# Minimize likelihood
def gauss(x, mu, sigma):
    return np.exp(- (x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)

def loglikelihood(x_0, x):
    mu1, sigma1= x_0[0], x_0[1]
    return -1 * np.sum(np.log(gauss(x, mu1, sigma1)))

def loglikelihood_2(x_0, x):
    mu1, sigma1, mu2, sigma2, p1 = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
    return -1 * np.sum(np.log(p1 * gauss(x, mu1, sigma1) + (1 - p1) * gauss(x, mu2, sigma2)))


def FitGals(cubename=None, nums_gauss=1):
    path_savefig = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_galaxy_velocity.png'.format(cubename)

    #
    filename = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/{}_gal_info.fits'.format(cubename)
    data = fits.getdata(filename, 1, ignore_missing_end=True)

    # Load galxies infomation
    row_gal, ID_gal, z_gal, v_gal = data['row'], data['ID'], data['z'] , data['v']
    name_gal, ql_gal, ra_gal, dec_gal = data['name'], data['ql'], data['ra'], data['dec']
    v_gal_qso = np.hstack((np.array([0]), v_gal))  # add the quasar
    v_gal_qso_1500 = v_gal_qso[(v_gal_qso >= -1500) * (v_gal_qso <= 1500)]

    # QSO info
    # path_qso = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/gal_info/quasars.dat'
    # data_qso = ascii.read(path_qso, format='fixed_width')
    # data_qso = data_qso[data_qso['name'] == cubename]
    # ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

    # Normalization
    bin_width = 400
    bins = np.arange(-2000, 2000 + bin_width, bin_width)
    normalization_all = len(v_gal_qso_1500) * bin_width

    # Initialize outlier rejection
    v_gal_qso_i = v_gal_qso_1500
    for i in range(10):
        nums_i = len(v_gal_qso_i)
        mean, std = np.mean(v_gal_qso_i), np.std(v_gal_qso_i)
        v_gal_qso_i = v_gal_qso_i[(v_gal_qso_i < 3 * std + mean) * (v_gal_qso_i > - 3 * std + mean)]
        nums_f = len(v_gal_qso_i)
        if nums_f == nums_i:
            print("Break at {}".format(i))
            print(v_gal_qso_i)
            break

    #
    if nums_gauss == 1:
        guesses = [0, 200]
        bnds = ((-2000, 2000), (0, 1000))
        result = minimize(loglikelihood, guesses, (v_gal_qso_i), bounds=bnds, method="Powell")
    elif nums_gauss == 2:
        guesses = [0, 200, 500, 200, 0.3]
        bnds = ((-100, 0), (0, 500), (200, 1000), (0, 1000), (0, 1))
        result = minimize(loglikelihood_2, guesses, (v_gal_qso_i), bounds=bnds, method="Powell")
    print(result.x)

    # Plot
    rv = np.linspace(-3000, 3000, 1000)
    plt.figure(figsize=(8, 5), dpi=300)
    # plt.vlines(0, 0, 15, linestyles='--', color='k', label=r"$\mathrm{QSO's \; redshift}$")
    plt.hist(v_gal_qso, bins=bins, color='k', histtype='step', label=r'$v_{\rm all}$')
    plt.hist(v_gal_qso_1500, bins=bins, color='C1', alpha=0.6, histtype='stepfilled', label=r'$v_{\rm candidate}$')
    if nums_gauss == 1:
        plt.plot(rv, normalization_all * norm.pdf(rv, result.x[0], result.x[1]), '-', c='r', lw=1, alpha=1,
                 label=r'$\mu_{1} = \, $' + str("{0:.0f}".format(result.x[0])) + r'$\mathrm{\, km \, s^{-1}}$'
                       + '\n' + r'$\sigma_{1} = \, $' + str("{0:.0f}".format(result.x[1]))
                       + r'$\mathrm{\, km \, s^{-1}}$')
    elif nums_gauss == 2:
        plt.plot(rv, result.x[4] * normalization_all * norm.pdf(rv, result.x[0], result.x[1]), '--', c='b', lw=1,
                 alpha=1,
                 label=r'$P_{1} = \,$' + str("{0:.2f}".format(result.x[4])) +
                       '\n' + r'$\mu_{1} = \, $' + str("{0:.0f}".format(result.x[0]))
                       + r'$\mathrm{\, km \, s^{-1}}$' + '\n' + r'$\sigma_{1} = \, $'
                       + str("{0:.0f}".format(result.x[1])) + r'$\mathrm{\, km \, s^{-1}}$')
        plt.plot(rv, (1 - result.x[4]) * normalization_all * norm.pdf(rv, result.x[2], result.x[3]), '--', c='red',
                 lw=1,
                 alpha=1, label=r'$P_{2} = \,$' + str("{0:.2f}".format(1 - result.x[4])) +
                                '\n' + r'$\mu_{2} = \, $' + str("{0:.0f}".format(result.x[2]))
                                + r'$\mathrm{\, km \, s^{-1}}$' + '\n' + r'$\sigma_{2} = \, $'
                                + str("{0:.0f}".format(result.x[3])) + r'$\mathrm{\, km \, s^{-1}}$')
        plt.plot(rv, result.x[4] * normalization_all * norm.pdf(rv, result.x[0], result.x[1]) +
                 (1 - result.x[4]) * normalization_all * norm.pdf(rv, result.x[2], result.x[3]), '-k',
                 lw=1, alpha=1, label=r'$P_{1}N(\mu_{1}\mathrm{,} \, \sigma_{1}^2) + $'
                                      + '\n' + r'$P_{2}N(\mu_{2}\mathrm{,} \, \sigma_{2}^2)$', zorder=-100)
    plt.xlim(-3000, 3000)
    plt.ylim(0, )
    # plt.yticks([2, 4, 6, 8, 10, 12])
    plt.minorticks_on()
    plt.xlabel(r'$\Delta v [\mathrm{km \; s^{-1}}]$', size=20)
    plt.ylabel(r'$\mathrm{Numbers}$', size=20)
    plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                    labelsize=20)
    plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
    plt.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
    plt.savefig(path_savefig, bbox_inches='tight')

FitGals(cubename='PKS0552-640', nums_gauss=1)