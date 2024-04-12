import os
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
from matplotlib.ticker import FormatStrFormatter
from muse_LoadCloudy import format_cloudy
from muse_LoadCloudy import format_cloudy_nogrid
from muse_LoadCloudy import format_cloudy_nogrid_BB
from muse_LoadCloudy import format_cloudy_nogrid_AGN
from muse_LoadLineRatio import load_lineratio
from muse_gas_spectra_DenMCMC import PlotGasSpectra
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('axes', **{'labelsize':15})


# Define likelihood function
def log_prob(x, bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364,
             f_HeII4687, f_OIII5008, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel,
             logflux_Hgam, logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, dlogflux_NeV3346,
             dlogflux_OII, dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364,
             dlogflux_HeII4687, dlogflux_OIII5008):
    if mode == 'power_law' or mode == 'BB':
        logden, alpha, logz = x[0], x[1], x[2]
        if logden < bnds[0, 0] or logden > bnds[0, 1]:
            return -np.inf
        elif alpha < bnds[1, 0] or alpha > bnds[1, 1]:
            return -np.inf
        elif logz < bnds[2, 0] or logz > bnds[2, 1]:
            return -np.inf
        else:
            var_array = (logden, alpha, logz)
            pass
    elif mode == 'AGN':
        logden, logz, logT, alpha_ox, alpha_uv, alpha_x = x[0], x[1], x[2], x[3], x[4], x[5]
        if logden < bnds[0, 0] or logden > bnds[0, 1]:
            return -np.inf
        elif logz < bnds[1, 0] or logz > bnds[1, 1]:
            return -np.inf
        elif logT < bnds[2, 0] or logT > bnds[2, 1]:
            return -np.inf
        elif alpha_ox < bnds[3, 0] or alpha_ox > bnds[3, 1]:
            return -np.inf
        elif alpha_uv < bnds[4, 0] or alpha_uv > bnds[4, 1]:
            return -np.inf
        elif alpha_x < bnds[5, 0] or alpha_x > bnds[5, 1]:
            return -np.inf
        else:
            var_array = (logden, logz, logT, alpha_ox, alpha_uv, alpha_x)
            pass
    elif mode == 'AGN_nouv':
        logden, logz, logT, alpha_ox, alpha_x = x[0], x[1], x[2], x[3], x[4]
        if logden < bnds[0, 0] or logden > bnds[0, 1]:
            return -np.inf
        elif logz < bnds[1, 0] or logz > bnds[1, 1]:
            return -np.inf
        elif logT < bnds[2, 0] or logT > bnds[2, 1]:
            return -np.inf
        elif alpha_ox < bnds[3, 0] or alpha_ox > bnds[3, 1]:
            return -np.inf
        elif alpha_x < bnds[5, 0] or alpha_x > bnds[5, 1]:
            return -np.inf
        else:
            var_array = (logden, logz, logT, alpha_ox, alpha_x)
            pass

    chi2_NeV3346 = ((f_NeV3346(var_array) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
    chi2_OII = ((f_OII(var_array) - logflux_OII) / dlogflux_OII) ** 2
    chi2_NeIII3869 = ((f_NeIII3869(var_array) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
    chi2_Hdel = ((f_Hdel(var_array) - logflux_Hdel) / dlogflux_Hdel) ** 2
    chi2_Hgam = ((f_Hgam(var_array) - logflux_Hgam) / dlogflux_Hgam) ** 2
    chi2_OIII4364 = ((f_OIII4364(var_array) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
    chi2_HeII4687 = ((f_HeII4687(var_array) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
    chi2_OIII5008 = ((f_OIII5008(var_array) - logflux_OIII5008) / dlogflux_OIII5008) ** 2

    sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_NeIII3869, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
                          chi2_HeII4687, chi2_OIII5008])
    return - 0.5 * np.nansum(sum_array[line_param[:, 1]])


def log_prob_LHIS(x, bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364, f_HeII4687,
                 f_OIII5008, f_Hbeta, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel, logflux_Hgam,
                 logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, logflux_Hbeta, dlogflux_NeV3346, dlogflux_OII,
                 dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364, dlogflux_HeII4687,
                 dlogflux_OIII5008, dlogflux_Hbeta):
    # if mode == 'power_law' or mode == 'BB':
    logden, logden_HIS, ratio_a, ratio_b, alpha, logz = x[0], x[1], x[2], x[3], x[4], x[5]
    a_ab = ratio_a / (ratio_a + ratio_b)
    b_ab = ratio_b / (ratio_a + ratio_b)
    # ratio_a /= 1e4
    # ratio_b /= 1e4
    # if alpha < -0.5 or alpha > -0.1:
    #     L = -np.inf
        # return -np.inf
    if logden < bnds[0, 0] or logden > bnds[0, 1]:
        L = -np.inf
        # return -np.inf
    elif logden_HIS < bnds[0, 0] or logden_HIS > bnds[0, 1]:
        L = -np.inf
        # return -np.inf
    elif logden < logden_HIS:
        L = -np.inf
    # elif ratio_a <= 0.0 or ratio_a > 1e7:
    #     L = -np.inf
    #     # return -np.inf
    # elif ratio_b <= 0.0 or ratio_a > 1e7:
    #     L = -np.inf
        # return -np.inf
    elif a_ab < 0.1:
        L = -np.inf
    elif b_ab < 0.1:
        L = -np.inf
    elif alpha < bnds[1, 0] or alpha > bnds[1, 1]:
        L = -np.inf
        # return -np.inf
    elif logz < bnds[2, 0] or logz > bnds[2, 1]:
        L = -np.inf
        # return -np.inf
    else:
        var_array = (logden, alpha, logz)
        var_array_HIS = (logden_HIS, alpha, logz)
    #         pass
    # elif mode == 'AGN':
    #     logden, logz, logT, alpha_ox, alpha_uv, alpha_x = x[0], x[1], x[2], x[3], x[4], x[5]
    #     if logden < bnds[0, 0] or logden > bnds[0, 1]:
    #         return -np.inf
    #     elif logz < bnds[1, 0] or logz > bnds[1, 1]:
    #         return -np.inf
    #     elif logT < bnds[2, 0] or logT > bnds[2, 1]:
    #         return -np.inf
    #     elif alpha_ox < bnds[3, 0] or alpha_ox > bnds[3, 1]:
    #         return -np.inf
    #     elif alpha_uv < bnds[4, 0] or alpha_uv > bnds[4, 1]:
    #         return -np.inf
    #     elif alpha_x < bnds[5, 0] or alpha_x > bnds[5, 1]:
    #         return -np.inf
    #     else:
    #         var_array = (logden, logz, logT, alpha_ox, alpha_uv, alpha_x)
    #         pass
    # elif mode == 'AGN_nouv':
    #     logden, logz, logT, alpha_ox, alpha_x = x[0], x[1], x[2], x[3], x[4]
    #     if logden < bnds[0, 0] or logden > bnds[0, 1]:
    #         return -np.inf
    #     elif logz < bnds[1, 0] or logz > bnds[1, 1]:
    #         return -np.inf
    #     elif logT < bnds[2, 0] or logT > bnds[2, 1]:
    #         return -np.inf
    #     elif alpha_ox < bnds[3, 0] or alpha_ox > bnds[3, 1]:
    #         return -np.inf
    #     elif alpha_x < bnds[5, 0] or alpha_x > bnds[5, 1]:
    #         return -np.inf
    #     else:
    #         var_array = (logden, logz, logT, alpha_ox, alpha_x)
    #         pass

        factor = 1 / 4 / np.pi / 1.15185925e+28 ** 2 / 1e-17
        f_NeV3346_total = np.log10(ratio_a * 10 ** f_NeV3346(var_array)
                                   + ratio_b * 10 ** f_NeV3346(var_array_HIS)) + np.log10(factor)
        f_OII_total = np.log10(ratio_a * 10 ** f_OII(var_array)
                               + ratio_b * 10 ** f_OII(var_array_HIS)) + np.log10(factor)
        # f_OII_total = np.log10(10 ** f_OII(var_array)) + np.log10(factor)
        f_NeIII3869_total = np.log10(ratio_a * 10 ** f_NeIII3869(var_array)
                                     + ratio_b * 10 ** f_NeIII3869(var_array_HIS)) + np.log10(factor)
        f_Hbeta_total = np.log10(ratio_a * 10 ** f_Hbeta(var_array)
                                 + ratio_b * 10 ** f_Hbeta(var_array_HIS)) + np.log10(factor)
        f_Hdel_total = np.log10(ratio_a * 10 ** f_Hdel(var_array)
                                + ratio_b * 10 ** f_Hdel(var_array_HIS)) + np.log10(factor)
        f_Hgam_total = np.log10(ratio_a * 10 ** f_Hgam(var_array)
                                + ratio_b * 10 ** f_Hgam(var_array_HIS)) + np.log10(factor)
        # f_Hbeta_total = np.log10(10 ** f_Hbeta(var_array)) + np.log10(factor)
        # f_Hdel_total = np.log10(10 ** f_Hdel(var_array)) + np.log10(factor)
        # f_Hgam_total = np.log10(10 ** f_Hgam(var_array)) + np.log10(factor)
        f_OIII4364_total = np.log10(ratio_a * 10 ** f_OIII4364(var_array)
                                    + ratio_b * 10 ** f_OIII4364(var_array_HIS)) + np.log10(factor)
        f_HeII4687_total = np.log10(ratio_a * 10 ** f_HeII4687(var_array)
                                    + ratio_b * 10 ** f_HeII4687(var_array_HIS)) + np.log10(factor)
        f_OIII5008_total = np.log10(ratio_a * 10 ** f_OIII5008(var_array)
                                    + ratio_b * 10 ** f_OIII5008(var_array_HIS)) + np.log10(factor)
        # f_OIII5008_total = np.log10(10 ** f_OIII5008(var_array)) + np.log10(factor)

        #
        chi2_NeV3346 = ((f_NeV3346_total - f_Hbeta_total - logflux_NeV3346) / dlogflux_NeV3346) ** 2
        chi2_OII = ((f_OII_total - f_Hbeta_total - logflux_OII) / dlogflux_OII) ** 2
        chi2_NeIII3869 = ((f_NeIII3869_total - f_Hbeta_total - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
        chi2_Hdel = ((f_Hdel_total - f_Hbeta_total - logflux_Hdel) / dlogflux_Hdel) ** 2
        chi2_Hgam = ((f_Hgam_total - f_Hbeta_total - logflux_Hgam) / dlogflux_Hgam) ** 2
        chi2_OIII4364 = ((f_OIII4364_total - f_Hbeta_total - logflux_OIII4364) / dlogflux_OIII4364) ** 2
        chi2_HeII4687 = ((f_HeII4687_total - f_Hbeta_total - logflux_HeII4687) / dlogflux_HeII4687) ** 2
        chi2_OIII5008 = ((f_OIII5008_total - f_Hbeta_total - logflux_OIII5008) / dlogflux_OIII5008) ** 2

        # chi2_NeV3346 = ((f_NeV3346_total - logflux_NeV3346) / dlogflux_NeV3346) ** 2
        # # print(chi2_NeV3346)
        # chi2_OII = ((f_OII_total - logflux_OII) / dlogflux_OII) ** 2
        # chi2_NeIII3869 = ((f_NeIII3869_total - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
        # chi2_Hdel = ((f_Hdel_total - logflux_Hdel) / dlogflux_Hdel) ** 2
        # chi2_Hgam = ((f_Hgam_total - logflux_Hgam) / dlogflux_Hgam) ** 2
        # chi2_OIII4364 = ((f_OIII4364_total - logflux_OIII4364) / dlogflux_OIII4364) ** 2
        # chi2_HeII4687 = ((f_HeII4687_total - logflux_HeII4687) / dlogflux_HeII4687) ** 2
        # chi2_OIII5008 = ((f_OIII5008_total - logflux_OIII5008) / dlogflux_OIII5008) ** 2
        # chi2_Hbeta = ((f_Hbeta_total - logflux_Hbeta) / dlogflux_Hbeta) ** 2
        # print(chi2_OII)
        sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
                              chi2_HeII4687, chi2_OIII5008])
        # sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_Hdel, chi2_Hgam, chi2_Hbeta, chi2_OIII4364,
        #                       chi2_HeII4687, chi2_OIII5008])
        # sum_array = np.array([chi2_OII, chi2_OIII5008])
        L = - 0.5 * np.nansum(sum_array)
        if L == 0:
            L = -np.inf
        # print(L)
    return L


def lognormal(logn, logmu, sigma, mode='all'):
    if mode == 'logL':
        pdf = np.exp(- (np.log(10 ** logn) - np.log(10 ** logmu)) ** 2
                     / 2 / sigma ** 2) / np.sqrt(2 * np.pi) / sigma
    elif mode == 'all':
        pdf = np.exp(- (np.log(10 ** logn) - np.log(10 ** logmu[:, np.newaxis])) ** 2
                     / 2 / sigma[:, np.newaxis] ** 2) / np.sqrt(2 * np.pi) / sigma[:, np.newaxis]
    return pdf


def log_prob_lognormal(x, bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364, f_HeII4687,
                 f_OIII5008, f_Hbeta, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel, logflux_Hgam,
                 logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, logflux_Hbeta, dlogflux_NeV3346, dlogflux_OII,
                 dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364, dlogflux_HeII4687,
                 dlogflux_OIII5008, dlogflux_Hbeta):
    # if mode == 'power_law' or mode == 'BB':
    logmu, sigma, alpha, logz = x[:]

    logn = np.linspace(-2, bnds[0, 1], 1000)
    prob = lognormal(logn, logmu, sigma, mode='logL')

    if logmu < bnds[0, 0] or logmu > bnds[0, 1]:
        L = -np.inf
    elif sigma < 0 or sigma > 10:
        L = -np.inf
    elif alpha < bnds[1, 0] or alpha > bnds[1, 1]:
        L = -np.inf
        # return -np.inf
    elif logz < bnds[2, 0] or logz > bnds[2, 1]:
        L = -np.inf
        # return -np.inf
    else:
        var_array = (logn, alpha, logz)
        f_NeV3346_total = np.log10(np.sum(prob * 10 ** f_NeV3346(var_array)))
        f_OII_total = np.log10(np.sum(prob * 10 ** f_OII(var_array)))
        f_NeIII3869_total = np.log10(np.sum(prob * 10 ** f_NeIII3869(var_array)))
        f_Hbeta_total = np.log10(np.sum(prob * 10 ** f_Hbeta(var_array)))
        f_Hdel_total = np.log10(np.sum(prob * 10 ** f_Hdel(var_array)))
        f_Hgam_total = np.log10(np.sum(prob * 10 ** f_Hgam(var_array)))
        f_OIII4364_total = np.log10(np.sum(prob * 10 ** f_OIII4364(var_array)))
        f_HeII4687_total = np.log10(np.sum(prob * 10 ** f_HeII4687(var_array)))
        f_OIII5008_total = np.log10(np.sum(prob * 10 ** f_OIII5008(var_array)))

        #
        chi2_NeV3346 = ((f_NeV3346_total - f_Hbeta_total - logflux_NeV3346) / dlogflux_NeV3346) ** 2
        chi2_OII = ((f_OII_total - f_Hbeta_total - logflux_OII) / dlogflux_OII) ** 2
        chi2_NeIII3869 = ((f_NeIII3869_total - f_Hbeta_total - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
        chi2_Hdel = ((f_Hdel_total - f_Hbeta_total - logflux_Hdel) / dlogflux_Hdel) ** 2
        chi2_Hgam = ((f_Hgam_total - f_Hbeta_total - logflux_Hgam) / dlogflux_Hgam) ** 2
        chi2_OIII4364 = ((f_OIII4364_total - f_Hbeta_total - logflux_OIII4364) / dlogflux_OIII4364) ** 2
        chi2_HeII4687 = ((f_HeII4687_total - f_Hbeta_total - logflux_HeII4687) / dlogflux_HeII4687) ** 2
        chi2_OIII5008 = ((f_OIII5008_total - f_Hbeta_total - logflux_OIII5008) / dlogflux_OIII5008) ** 2

        sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_NeIII3869, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
                              chi2_HeII4687, chi2_OIII5008])
        L = - 0.5 * np.nansum(sum_array[line_param[:, 1]])
        if L == 0:
            L = -np.inf
        # print(L)
    return L


# Default values
den_default = np.linspace(-2, 2.6, 24, dtype='f2') # log # 0.25
Z_default = np.linspace(-1.5, 0.5, 11, dtype='f2')
alpha_default = np.linspace(-1.8, 0, 10, dtype='f2')


def RunCloudyMCMC(den_array=den_default, Z_array=Z_default, T_array=None, alpha_array=alpha_default,
                  alpha_ox_array=None, alpha_uv_array=None, alpha_x_array=None, region=None, trial=None, bnds=None,
                  line_param=None, deredden=True, norm='Hbeta', mode='power_law', figname_extra='', nums_chain=5000,
                  nums_disc=1000):
    # Load the actual measurement
    logflux_Hbeta, dlogflux_Hbeta, logflux_NeV3346, dlogflux_NeV3346, logflux_OII, dlogflux_OII, logr_OII, \
    dlogr_OII, logflux_NeIII3869, dlogflux_NeIII3869, logflux_Hdel, dlogflux_Hdel, logflux_Hgam, dlogflux_Hgam, \
    logflux_OIII4364, dlogflux_OIII4364, logflux_HeII4687, dlogflux_HeII4687, \
    logflux_OIII5008, dlogflux_OIII5008 = load_lineratio(region=region, deredden=deredden, norm=norm)

    # Load cloudy result
    if region == 'S2' and trial == 't1':
        den_array = np.arange(-2, 2.6, 0.1)  # log
        Z_array = np.linspace(-1.5, 0.5, 21, dtype='f2')  # log
        alpha_array = np.linspace(-1.8, -0.6, 25, dtype='f2')
        output, ind = format_cloudy(filename=[Z_array, alpha_array], path='/Users/lzq/Dropbox/Data/CGM/cloudy/trial2/')
        var_array = (den_array, alpha_array, Z_array)

    else:
        if mode == 'power_law':
            output = format_cloudy_nogrid(filename=[den_array, alpha_array, Z_array],
                                          path='/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/')
            var_array = (den_array, alpha_array, Z_array)
        elif mode == 'BB':
            output = format_cloudy_nogrid_BB(filename=[den_array, alpha_array, Z_array],
                                             path='/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/')
            var_array = (den_array, alpha_array, Z_array)
        else:
            output = format_cloudy_nogrid_AGN(filename=[den_array, Z_array, T_array, alpha_ox_array, alpha_uv_array,
                                                        alpha_x_array],
                                              path='/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/')
            var_array = (den_array, Z_array, T_array, alpha_ox_array, alpha_uv_array, alpha_x_array)
            if mode == 'AGN_nouv':
                output = output[:, :, :, :, :, 0, :]
                var_array = (den_array, Z_array, T_array, alpha_ox_array, alpha_x_array)

    if norm == 'LHIS' or norm == 'lognormal':
        output_norm = output
    elif norm == 'Hbeta':
        if region == 'S2' and trial == 't1':
            output_norm = output
        else:
            output_norm = output - output[10]
    elif norm == 'OII':
        output_norm = output - output[3]  # in log
    elif norm == 'HeII':
        output_norm = output - output[8]


    f_NeV3346 = interpolate.RegularGridInterpolator(var_array, output_norm[0],
                                                    bounds_error=False, fill_value=None)
    f_OII3727 = interpolate.RegularGridInterpolator(var_array, output_norm[1],
                                                    bounds_error=False, fill_value=None)
    f_OII3730 = interpolate.RegularGridInterpolator(var_array, output_norm[2],
                                                    bounds_error=False, fill_value=None)
    f_OII = interpolate.RegularGridInterpolator(var_array, output_norm[3],
                                                bounds_error=False, fill_value=None)
    f_NeIII3869 = interpolate.RegularGridInterpolator(var_array, output_norm[4],
                                                      bounds_error=False, fill_value=None)
    f_Hdel = interpolate.RegularGridInterpolator(var_array, output_norm[5],
                                                 bounds_error=False, fill_value=None)
    f_Hgam = interpolate.RegularGridInterpolator(var_array, output_norm[6],
                                                 bounds_error=False, fill_value=None)
    f_OIII4364 = interpolate.RegularGridInterpolator(var_array, output_norm[7],
                                                     bounds_error=False, fill_value=None)
    f_HeII4687 = interpolate.RegularGridInterpolator(var_array, output_norm[8],
                                                     bounds_error=False, fill_value=None)
    f_OIII5008 = interpolate.RegularGridInterpolator(var_array, output_norm[9],
                                                     bounds_error=False, fill_value=None)
    if norm != 'Hbeta':
        f_Hbeta = interpolate.RegularGridInterpolator(var_array, output_norm[10],
                                                      bounds_error=False, fill_value=None)
    # Run MCMC
    nwalkers = 40
    if mode == 'power_law':
        ndim = 3
        p0 = np.array([1.8, -1.5, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
        labels = [r"$\mathrm{log_{10}(n/cm^{-3})}$", r"$\mathrm{\alpha}$", r"$\mathrm{log_{10}(Z/Z_{\odot})}$"]

        if norm == 'LHIS':
            ndim = 6
            p0 = np.array([1.8, 0.8, 0.9, 0.5, -1.4, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
            labels = [r"$\mathrm{log_{10}(n_{lI}/cm^{-3})}$", r"$\mathrm{log_{10}(n_{hI}/cm^{-3})}$",
                      r"$\mathrm{\frac{a}{a + b}}$", r"$\mathrm{\frac{b}{a + b}}$", r"$\mathrm{\alpha}$",
                      r"$\mathrm{log_{10}(Z/Z_{\odot})}$"]

        elif norm == 'lognormal':
            ndim = 4
            p0 = np.array([1.5, 3, -1.4, -0.3]) + 0.01 * np.random.randn(nwalkers, ndim)
            labels = [r"$\mathrm{log_{10}(\mu/cm^{-3})}$", r"$\mathrm{\sigma}$", r"$\mathrm{\alpha}$",
                      r"$\mathrm{log_{10}(Z/Z_{\odot})}$"]

    elif mode == 'BB':
        ndim = 3
        p0 = np.array([1.8, 5, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
        labels = [r"$\mathrm{log_{10}(n/cm^{-3})}$", r"$\mathrm{log_{10}(T/K)}$", r"$\mathrm{log_{10}(Z/Z_{\odot})}$"]
    elif mode == 'AGN':
        ndim = 6
        p0 = np.array([1.6, -0.3, 5.2, -0.7, -0.5, -1.0]) + 0.1 * np.random.randn(nwalkers, ndim)
        labels = [r"$\mathrm{log_{10}(n/cm^{-3})}$", r"$\mathrm{log_{10}(Z/Z_{\odot})}$", r"$\mathrm{log_{10}(T)}$",
                  r"$\mathrm{\alpha_{ox}}$", r"$\mathrm{\alpha_{uv}}$", r"$\mathrm{\alpha_x}$"]
    elif mode == 'AGN_nouv':
        ndim = 5
        p0 = np.array([1.6, -0.3, 5.2, -0.7, -1.0]) + 0.1 * np.random.randn(nwalkers, ndim)
        labels = [r"$\mathrm{log_{10}(n/cm^{-3})}$", r"$\mathrm{log_{10}(Z/Z_{\odot})}$", r"$\mathrm{log_{10}(T)}$",
                  r"$\mathrm{\alpha_{ox}}$", r"$\mathrm{\alpha_x}$"]
    if deredden:
        filename = '/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra + '_dered.h5'
        figname_MCMC = '/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial \
                       + figname_extra + '_dered.pdf'
    else:
        filename = '/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra + '.h5'
        figname_MCMC = '/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra + '.pdf'
    backend = emcee.backends.HDFBackend(filename)

    if os.path.exists(filename):
        samples = backend.get_chain(flat=True, discard=nums_disc)
        samples_corner = np.copy(samples)
        if norm == 'LHIS':
            samples_corner[:, 2] = samples[:, 2] / (samples[:, 2] + samples[:, 3])
            samples_corner[:, 3] = samples[:, 3] / (samples[:, 2] + samples[:, 3])
    else:
        if norm == 'LHIS':
            args = (bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364, f_HeII4687,
                    f_OIII5008, f_Hbeta, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel, logflux_Hgam,
                    logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, logflux_Hbeta, dlogflux_NeV3346,
                    dlogflux_OII, dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364,
                    dlogflux_HeII4687, dlogflux_OIII5008, dlogflux_Hbeta)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_LHIS, args=args, backend=backend)
            state = sampler.run_mcmc(p0, nums_chain)
            samples = sampler.get_chain(flat=True, discard=nums_disc)
            samples_corner = np.copy(samples)
            samples_corner[:, 2] = samples[:, 2] / (samples[:, 2] + samples[:, 3])
            samples_corner[:, 3] = samples[:, 3] / (samples[:, 2] + samples[:, 3])
        elif norm == 'lognormal':
            args = (bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364, f_HeII4687,
                    f_OIII5008, f_Hbeta, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel, logflux_Hgam,
                    logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, logflux_Hbeta, dlogflux_NeV3346,
                    dlogflux_OII, dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364,
                    dlogflux_HeII4687, dlogflux_OIII5008, dlogflux_Hbeta)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_lognormal, args=args, backend=backend)
            state = sampler.run_mcmc(p0, nums_chain)
            samples = sampler.get_chain(flat=True, discard=nums_disc)
            samples_corner = np.copy(samples)
        else:
            args = (bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364,
                    f_HeII4687, f_OIII5008, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel, logflux_Hgam,
                    logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, dlogflux_NeV3346, dlogflux_OII,
                    dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364,
                    dlogflux_HeII4687, dlogflux_OIII5008)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, backend=backend)
            state = sampler.run_mcmc(p0, nums_chain)
            samples = sampler.get_chain(flat=True, discard=nums_disc)
            samples_corner = np.copy(samples)

    figure = corner.corner(samples_corner, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k',
                           title_kwargs={"fontsize": 13}, smooth=1., smooth1d=1., bins=25)

    for i, ax in enumerate(figure.get_axes()):
        if not np.isin(i, np.arange(0, ndim ** 2, ndim + 1)):
            ax.tick_params(axis='both', direction='in', top='on', bottom='on', left='on', right='on')
        ax.tick_params(axis='both', direction='in', top='on', bottom='on')
    figure.savefig(figname_MCMC, bbox_inches='tight')

    # Violin plot
    gridspec = dict(wspace=0, width_ratios=[1, 0.25, 3, 0.5, 1])
    fig, ax = plt.subplots(1, 5, figsize=(15, 5), gridspec_kw=gridspec, dpi=300)
    ax[1].set_visible(False)
    ax[3].set_visible(False)
    E_NeV3346, E_NeIII3869 = 97.11, 40.96  # in ev
    E_OIII4364, E_OIII5008, E_OII = 35.12, 35.12, 13.6
    E_HeII4687, E_Hbeta = 54.42, 13.6

    # Data
    line_param_violin = np.array([line_param[1, 1], line_param[7, 1], line_param[6, 1], line_param[0, 1]])
    data_x1 = [0, 4]
    data_x2 = [E_OII / E_Hbeta, E_OIII5008 / E_Hbeta - 0.5, E_HeII4687 / E_Hbeta + 0.3, E_NeV3346 / E_Hbeta - 2]
    data_x2_plot = np.arange(1, 5, 1)
    draw = np.random.choice(len(samples), size=4000, replace=False)
    samples_draw = samples[draw]
    #
    if line_param[5, 1]:
        data_y1 = np.array([logr_OII, logflux_OIII4364 - logflux_OIII5008], dtype=object).reshape(len(data_x1))
        data_y1err = np.array([dlogr_OII, np.sqrt(dlogflux_OIII4364 ** 2
                                                  + dlogflux_OIII5008 ** 2)], dtype=object).reshape(len(data_x1))
    else:
        data_y1 = np.array([logr_OII, np.nan], dtype=object).reshape(len(data_x1))
        data_y1err = np.array([dlogr_OII, np.nan], dtype=object).reshape(len(data_x1))

    if norm == 'LHIS':
        # Reload data
        logflux_Hbeta, dlogflux_Hbeta, logflux_NeV3346, dlogflux_NeV3346, logflux_OII, dlogflux_OII, logr_OII, \
        dlogr_OII, logflux_NeIII3869, dlogflux_NeIII3869, logflux_Hdel, dlogflux_Hdel, logflux_Hgam, dlogflux_Hgam, \
        logflux_OIII4364, dlogflux_OIII4364, logflux_HeII4687, dlogflux_HeII4687, \
        logflux_OIII5008, dlogflux_OIII5008 = load_lineratio(region=region, deredden=deredden, norm='Hbeta')

        #
        data_y2 = np.array([logflux_OII, logflux_OIII5008, logflux_HeII4687, logflux_NeV3346],
                           dtype=object).reshape(len(data_x2))
        data_y2err = np.array([dlogflux_OII, dlogflux_OIII5008, dlogflux_HeII4687, dlogflux_NeV3346],
                              dtype=object).reshape(len(data_x2))
        data_y2 = np.where(line_param_violin, data_y2, np.nan)
        data_y2err = np.where(line_param_violin, data_y2err, 0)

        #
        model_ratio_a = samples_draw[:, 2]
        model_ratio_b = samples_draw[:, 3]
        model = (samples_draw[:, 0], samples_draw[:, 4], samples_draw[:, 5])
        model_HIS = (samples_draw[:, 1], samples_draw[:, 4], samples_draw[:, 5])
        norm_Hbeta = np.log10(model_ratio_a * 10 ** f_Hbeta(model) + model_ratio_b * 10 ** f_Hbeta(model_HIS))
        model_y2 = np.array([np.log10(model_ratio_a * 10 ** f_OII(model)
                                      + model_ratio_b * 10 ** f_OII(model_HIS)) - norm_Hbeta,
                             np.log10(model_ratio_a * 10 ** f_OIII5008(model)
                                      + model_ratio_b * 10 ** f_OIII5008(model_HIS)) - norm_Hbeta,
                             np.log10(model_ratio_a * 10 ** f_HeII4687(model)
                                      + model_ratio_b * 10 ** f_HeII4687(model_HIS)) - norm_Hbeta,
                             np.log10(model_ratio_a * 10 ** f_NeV3346(model)
                                      + model_ratio_b * 10 ** f_NeV3346(model_HIS)) - norm_Hbeta])
        OII_ratio = model_ratio_a * 10 ** f_OII(model) / (model_ratio_a * 10 ** f_OII(model)
                                                          + model_ratio_b * 10 ** f_OII(model_HIS))
        OIII_ratio = model_ratio_a * 10 ** f_OIII5008(model) / (model_ratio_a * 10 ** f_OIII5008(model)
                                                          + model_ratio_b * 10 ** f_OIII5008(model_HIS))
        plt.figure(figsize=(3, 3), dpi=300)
        plt.hist(OII_ratio, histtype='step', label='[O II]')
        plt.hist(OIII_ratio, label='[O III]')
        plt.legend(loc=2)
        plt.xlabel(r'$\rm Flux\_highden / (Flux\_highden + Flux\_lowden)$')
        plt.ylabel(r'Posterior Probability')
        plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial +
                    '_OII_ratio_check.png', bbox_inches='tight')

        #
        if line_param[5, 1]:
            model_y1 = [np.log10(model_ratio_a * 10 ** f_OII3730(model)
                                 + model_ratio_b * 10 ** f_OII3730(model_HIS))
                        - np.log10(model_ratio_a * 10 ** f_OII3727(model)
                                   + model_ratio_b * 10 ** f_OII3727(model_HIS)),
                        np.log10(model_ratio_a * 10 ** f_OIII4364(model)
                                 + model_ratio_b * 10 ** f_OIII4364(model_HIS))
                        - np.log10(model_ratio_a * 10 ** f_OIII5008(model)
                        + model_ratio_b * 10 ** f_OIII5008(model_HIS))]
        else:
            model_y1 = [np.log10(model_ratio_a * 10 ** f_OII3730(model)
                                 + model_ratio_b * 10 ** f_OII3730(model_HIS))
                        - np.log10(model_ratio_a * 10 ** f_OII3727(model)
                                   + model_ratio_b * 10 ** f_OII3727(model_HIS)),
                        np.nan * np.zeros(len(f_OII3727(model)))]

    elif norm == 'lognormal':
        data_y2 = np.array([logflux_OII, logflux_OIII5008, logflux_HeII4687, logflux_NeV3346],
                           dtype=object).reshape(len(data_x2))
        data_y2err = np.array([dlogflux_OII, dlogflux_OIII5008, dlogflux_HeII4687, dlogflux_NeV3346],
                              dtype=object).reshape(len(data_x2))
        data_y2 = np.where(line_param_violin, data_y2, np.nan)
        data_y2err = np.where(line_param_violin, data_y2err, 0)

        logmu_draw, sigma_draw, alpha_draw, logz_draw = samples_draw[:, 0], samples_draw[:, 1], \
                                                        samples_draw[:, 2], samples_draw[:, 3]

        logn_draw = np.linspace(bnds[0, 0], bnds[0, 1], 1000)
        prob_draw = lognormal(logn_draw, logmu_draw, sigma_draw)

        #
        logn_X, alpha_Y = np.meshgrid(logn_draw, alpha_draw)
        logn_X, logz_Z = np.meshgrid(logn_draw, logz_draw)
        model = (logn_X, alpha_Y, logz_Z)
        norm_Hbeta = np.log10(np.sum(prob_draw * 10 ** f_Hbeta(model), axis=1))
        model_y2 = [np.log10(np.sum(prob_draw * 10 ** f_OII(model), axis=1)) - norm_Hbeta,
                             np.log10(np.sum(prob_draw * 10 ** f_OIII5008(model), axis=1)) - norm_Hbeta,
                             np.log10(np.sum(prob_draw * 10 ** f_HeII4687(model), axis=1)) - norm_Hbeta,
                             np.log10(np.sum(prob_draw * 10 ** f_NeV3346(model), axis=1)) - norm_Hbeta]
        model_y2[1] = model_y2[1][~np.isnan(model_y2[0])]
        model_y2[2] = model_y2[2][~np.isnan(model_y2[0])]
        model_y2[3] = model_y2[3][~np.isnan(model_y2[0])]
        model_y2[0] = model_y2[0][~np.isnan(model_y2[0])]
        model_y2 = np.asarray(model_y2)
        print(np.shape(model_y2[0]), np.shape(model_y2[1]), np.shape(model_y2[2]), np.shape(model_y2[3]))

        # For subplot
        if line_param[5, 1]:
            model_y1 = [np.log10(np.sum(prob_draw * 10 ** f_OII3730(model), axis=1))
                        - np.log10(np.sum(prob_draw * 10 ** f_OII3727(model), axis=1)),
                        np.log10(np.sum(prob_draw * 10 ** f_OIII4364(model), axis=1))
                        - np.log10(np.sum(prob_draw * 10 ** f_OIII5008(model), axis=1))]
        else:
            model_y1 = [np.log10(np.sum(prob_draw * 10 ** f_OII3730(model), axis=1))
                        - np.log10(np.sum(prob_draw * 10 ** f_OII3727(model), axis=1)),
                        np.nan * np.zeros(len(f_OII3727(model)))]
        model_y1[0] = model_y1[0][~np.isnan(model_y1[0])]

        # Lognormal diagnosis plot # 1
        plt.figure(figsize=(3, 3), dpi=300)
        plt.plot(logn_draw, prob_draw.T, lw=0.1)
        plt.legend(loc=2)
        plt.xlabel(r'$\rm log(\mu)$')
        plt.ylabel(r'PDF')
        plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial +
                    '_CheckLognormal#1.png', bbox_inches='tight')

        # 2
        plt.figure(figsize=(3, 3), dpi=300)
        plt.scatter(logmu_draw, sigma_draw, c=np.log10(np.sum(prob_draw * 10 ** f_OII3730(model), axis=1))
                        - np.log10(np.sum(prob_draw * 10 ** f_OII3727(model), axis=1)), s=0.1)
        plt.colorbar(label=r'$\mathrm{\frac{[O \, II3729]}{[O \, II3727]}}$')
        plt.legend(loc=2)
        plt.xlabel(r'$\rm log(\mu)$')
        plt.ylabel(r'$\rm sigma$')
        plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial +
                    '_CheckLognormal#2.png', bbox_inches='tight')

        # 3
        logn_OII_weighted = np.sum(logn_draw * prob_draw * 10 ** f_OII(model), axis=1) / \
                            np.sum(prob_draw * 10 ** f_OII(model), axis=1)
        plt.figure(figsize=(3, 3), dpi=300)
        plt.scatter(logmu_draw, sigma_draw, c=logn_OII_weighted, s=0.1)
        plt.colorbar(label=r'$\rm logn_{[O \, II]}$')
        plt.legend(loc=2)
        plt.xlabel(r'$\rm log(\mu)$')
        plt.ylabel(r'$\rm sigma$')
        plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial +
                    '_CheckLognormal#3.png', bbox_inches='tight')

    else:
        data_y2 = np.array([logflux_OII, logflux_OIII5008, logflux_HeII4687, logflux_NeV3346],
                           dtype=object).reshape(len(data_x2))
        data_y2err = np.array([dlogflux_OII, dlogflux_OIII5008, dlogflux_HeII4687, dlogflux_NeV3346],
                              dtype=object).reshape(len(data_x2))
        data_y2 = np.where(line_param_violin, data_y2, np.nan)
        data_y2err = np.where(line_param_violin, data_y2err, 0)

        #
        if mode == 'power_law' or mode == 'BB':
            model = (samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2])
        elif mode == 'AGN':
            model = (samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2],
                     samples_draw[:, 3], samples_draw[:, 4], samples_draw[:, 5])
        elif mode == 'AGN_nouv':
            model = (samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2],
                     samples_draw[:, 3], samples_draw[:, 4])
        model_y2 = np.array([f_OII(model), f_OIII5008(model), f_HeII4687(model), f_NeV3346(model)])

        if line_param[5, 1]:
            model_y1 = [f_OII3730(model) - f_OII3727(model), f_OIII4364(model) - f_OIII5008(model)]
        else:
            model_y1 = [f_OII3730(model) - f_OII3727(model), np.nan * np.zeros(len(f_OII3727(model)))]

    # Plot data
    model_y2 = np.where(line_param_violin[:, np.newaxis], model_y2, np.nan)
    model_y2 = [np.array(model_y2[0, :]), np.array(model_y2[1, :]), np.array(model_y2[2, :]),
                np.array(model_y2[3, :])]
    print(np.shape(model_y2))
    ax[0].errorbar(data_x1, data_y1, data_y1err, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
    ax[2].errorbar(data_x2_plot, data_y2, data_y2err, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
    violin_parts_0 = ax[0].violinplot(model_y1, positions=data_x1, points=500, widths=1.5, showmeans=False,
                                      showextrema=False, showmedians=False)
    violin_parts_1 = ax[2].violinplot(model_y2, positions=data_x2_plot, points=500, widths=0.5, showmeans=False,
                                      showextrema=False, showmedians=False)

    for pc in violin_parts_0['bodies']:
        pc.set_facecolor('C1')
    for pc in violin_parts_1['bodies']:
        pc.set_facecolor('C1')
    ax[0].set_xlim(-1, 6)
    ax[2].set_xlim(data_x2_plot.min() - 0.3, data_x2_plot.max() + 0.3)
    region_i = region
    if len(region_i) > 5:
        region_i = region_i[:-4]
    ax[2].annotate(region_i, xy=(0.7, 0.75), size=30, xycoords='subfigure fraction',)
    ax[2].set_title(r'$\mathrm{Ionization \, energy}$', size=20, y=1.1)
    ax[2].annotate("", xy=(0.63, 0.920), xytext=(0.4, 0.920), xycoords='subfigure fraction',
                   arrowprops=dict(arrowstyle="->"))
    ax[0].set_ylabel(r'$\mathrm{log[line \, ratio]}$', size=20)
    if norm == 'Hbeta':
        ax[2].set_xticks(data_x2_plot, [r'$\mathrm{\frac{[O \, II]}{H\beta}}$', r'$\mathrm{\frac{[O \, III]}{H\beta}}$',
                                        r'$\mathrm{\frac{He \, II}{H\beta}}$', r'$\mathrm{\frac{[Ne \, V]}{H\beta}}$'])
    elif norm == 'OII':
        ax[2].set_xticks(data_x2_plot, [r'$\mathrm{\frac{[O \, II]}{[O \, II]}}$',
                                        r'$\mathrm{\frac{[O \, III]}{[O \, II]}}$',
                                        r'$\mathrm{\frac{He \, II}{[O \, II]}}$',
                                        r'$\mathrm{\frac{[Ne \, V]}{[O \, II]}}$'])
    elif norm == 'HeII':
        ax[2].set_xticks(data_x2_plot, [r'$\mathrm{\frac{[O \, II]}{He \, II}}$',
                                        r'$\mathrm{\frac{[O \, III]}{He \, II}}$',
                                        r'$\mathrm{\frac{He \, II}{He \, II}}$',
                                        r'$\mathrm{\frac{[Ne \, V]}{He \, II}}$'])
    elif norm == 'LHIS' or norm == 'lognormal':
        ax[2].set_xticks(data_x2_plot, [r'$\mathrm{\frac{[O \, II]}{H\beta}}$', r'$\mathrm{\frac{[O \, III]}{H\beta}}$',
                                        r'$\mathrm{\frac{He \, II}{H\beta}}$', r'$\mathrm{\frac{[Ne \, V]}{H\beta}}$'])
    ax[2].annotate(r'$13.6\mathrm{eV}$', xy=(0.30, 0.87), xycoords='subfigure fraction', size=15)
    ax[2].annotate(r'$35.1\mathrm{eV}$', xy=(0.43, 0.87), xycoords='subfigure fraction', size=15)
    ax[2].annotate(r'$54.4\mathrm{eV}$', xy=(0.56, 0.87), xycoords='subfigure fraction', size=15)
    ax[2].annotate(r'$97.1\mathrm{eV}$', xy=(0.70, 0.87), xycoords='subfigure fraction', size=15)
    ax[0].set_xticks([0, 4], [r'$\mathrm{[O \, II]}$' + r'$\mathrm{\frac{\lambda 3729}{\lambda 3727}}$',
                              r'$\mathrm{[O \, III]}$' + r'$\mathrm{\frac{\lambda 4364}{\lambda 5008}}$'])
    ax[0].minorticks_on()
    ax[2].minorticks_on()
    ax[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', top='on',
                      labelsize=20, size=5)
    ax[0].tick_params(axis='both', which='minor', direction='in', bottom=False, left='on', right='on', size=3)
    ax[2].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', top='on',
                      labelsize=20, size=5)
    ax[2].tick_params(axis='both', which='minor', direction='in', bottom=False, left='on', right='on', size=3)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Hist comparison
    samples_OII = PlotGasSpectra(region, figname='spectra_gas_' + region, deredden=deredden,
                                 save_table=False, save_figure=False, return_samples=True,
                                 nums_chain=nums_chain, nums_disc=nums_disc)
    sample_min_OII, sample_max_OII = np.nanmin(samples_OII[:, 0]), np.nanmax(samples_OII[:, 0])
    sample_min_cloudy, sample_max_cloudy = np.nanmin(10 ** samples[:, 0]), np.nanmax(10 ** samples[:, 0])
    sample_min = np.minimum(sample_min_OII, sample_min_cloudy)
    sample_max = np.maximum(sample_max_OII, sample_max_cloudy)
    bins_cloudy = np.arange(1.5, np.log10(sample_max//10 * 10 + 10), 0.1)

    #
    # bins_mid = (bins_cloudy[:-1] + bins_cloudy[1:]) / 2
    nums_cloudy, _ = np.histogram(samples[:, 0], bins=bins_cloudy,
                                  range=(1.5, np.log10(sample_max//10 * 10 + 10)))
    weights = np.ones_like(samples[:, 0]) / nums_cloudy.max()
    # nums_cloudy /= nums_cloudy.max()
    # ax[4].plot(bins_mid, nums_cloudy, color='brown', fillstyle='full', alpha=0.5, label=r'$\rm Cloudy$',
    #            drawstyle='steps-mid')
    ax[4].hist(samples[:, 0], bins=bins_cloudy, range=(1.5, np.log10(sample_max//10 * 10 + 10)),
               weights=weights, color='C1', histtype='stepfilled', lw=2, alpha=0.6, label=r'$\rm Cloudy$')
    nums_OII, _ = np.histogram(np.log10(samples_OII[:, 0]), bins=bins_cloudy,
                               range=(1.5, np.log10(sample_max//10 * 10 + 10)))
    weights = np.ones_like(samples_OII[:, 0]) / nums_OII.max()
    nums, _, __ = ax[4].hist(np.log10(samples_OII[:, 0]), bins=bins_cloudy,
                             range=(1.5, np.log10(sample_max//10 * 10 + 10)),
                             weights=weights, color='red', histtype='step', lw=2,
                             alpha=1, label=r'$\rm [O \, II]$')
    ax[4].annotate("", xy=(1.52, np.max(nums) + 0.05), xytext=(1.9, np.max(nums) + 0.05), xycoords='data',
                   arrowprops=dict(arrowstyle="->", color='red'))
    ax[4].minorticks_on()
    ax[4].set_yticks([0.25, 0.5, 0.75, 1.0])
    ax[4].set_xlim(1.5, np.log10(sample_max//10 * 10 + 10))
    if region == 'S2' and trial == 't1':
        ax[4].set_xticks([1.5, 1.75])
    ax[4].set_ylim(0, np.max(nums) + 0.2)
    ax[4].set_xlabel(r'$\mathrm{log(} n \mathrm{/cm^{-3})}$', size=20)
    ax[4].set_ylabel(r'$\mathrm{Peak \, Normalized \, Posterior}$', size=20)
    ax[4].tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on',
                      labelleft='on', labelright=False, labelsize=20, size=5)
    ax[4].tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on',
                      size=3)
    ax[4].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[4].legend(loc=1)
    if deredden:
        fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra +
                    '_violin_dered.png', bbox_inches='tight')
    else:
        fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra +
                    '_violin.png', bbox_inches='tight')

    # Make a histogram for poster
    bins_cloudy = np.arange(1.5, np.log10(sample_max//10 * 10 + 10), 0.05)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    nums_cloudy, _ = np.histogram(samples[:, 0], bins=bins_cloudy,
                                  range=(1.5, np.log10(sample_max//10 * 10 + 10)))
    weights = np.ones_like(samples[:, 0]) / nums_cloudy.max()
    ax.hist(samples[:, 0], bins=bins_cloudy, range=(1.5, np.log10(sample_max//10 * 10 + 10)),
               weights=weights, color='C1', histtype='stepfilled', lw=2, alpha=0.6, label=r'$\rm Cloudy$')
    nums_OII, _ = np.histogram(np.log10(samples_OII[:, 0]), bins=bins_cloudy,
                               range=(1.5, np.log10(sample_max//10 * 10 + 10)))
    weights = np.ones_like(samples_OII[:, 0]) / nums_OII.max()
    nums, _, __ = ax.hist(np.log10(samples_OII[:, 0]), bins=bins_cloudy,
                             range=(1.5, np.log10(sample_max//10 * 10 + 10)),
                             weights=weights, color='red', histtype='step', lw=2,
                             alpha=1, label=r'$\rm [O \, II]$')
    # ax.annotate("", xy=(1.52, np.max(nums) + 0.05), xytext=(1.9, np.max(nums) + 0.05), xycoords='data',
    #                arrowprops=dict(arrowstyle="->", color='red'))
    ax.minorticks_on()
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    # ax.set_xlim(1.5, np.log10(sample_max//10 * 10 + 10))
    if region == 'S2' and trial == 't1':
        ax.set_xticks([1.5, 1.75])
    ax.set_ylim(0, np.max(nums) + 0.2)
    ax.set_xlabel(r'$\mathrm{log(} n \mathrm{/cm^{-3})}$', size=20)
    ax.set_ylabel(r'$\mathrm{Peak \, Normalized \, Posterior}$', size=20)
    ax.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on',
                      labelleft='on', labelright=False, labelsize=20, size=5)
    ax.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on',
                      size=3)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.legend(loc=1)
    if deredden:
        fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra +
                    '_violin_dered_poster.png', bbox_inches='tight')
    else:
        fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + figname_extra +
                    '_violin_poster.png', bbox_inches='tight')


# S1
# S1_bnds = np.array([[-2, 2.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S1_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S1', trial='t1', bnds=S1_bnds, line_param=S1_param, deredden=True)

# S1 low ionizaton
# S1_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', False]], dtype=bool)
# RunCloudyMCMC(region='S1', trial='t1', figname_extra='LIP', bnds=S1_bnds, line_param=S1_param, deredden=True)

# S1 middle ionization
# S1_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S1', trial='t1', figname_extra='MIP', bnds=S1_bnds, line_param=S1_param, deredden=True)

# S1 high ionizaton
# S1_param = np.array([['NeV3346', True],
#                      ['OII', False],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S1', trial='t1', figname_extra='HIP', bnds=S1_bnds, line_param=S1_param, deredden=True, norm='HeII')

# AGN model
# S1_bnds = np.array([[1.0, 2.4],
#                     [-0.5, 0.5],
#                     [5, 5.5],
#                     [-1.2, -0.2],
#                     [-1.0, 0],
#                     [-1.5, -0.5]])
# S1_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# den_array_AGN = np.linspace(1.0, 2.4, 8, dtype='f2')
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(5, 5.5, 3, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, -0.2, 6, dtype='f2')
# alpha_uv_array_AGN = np.linspace(-1.0, 0, 3, dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, -0.5, 3, dtype='f2')
# RunCloudyMCMC_AGN(den_array_AGN, z_array_AGN, T_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
#                   region='S1', trial='AGN', bnds=S1_bnds, line_param=S1_param, deredden=False)

# AGN_2 model
# S1_bnds = np.array([[1.0, 2.4],
#                     [-0.5, 0.5],
#                     [4.75, 5.75],
#                     [-1.2, 0],
#                     [-0.52, -0.48],
#                     [-1.5, 0.5]])
# S1_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# den_array_AGN = np.linspace(1.0, 2.4, 8, dtype='f2')
# Z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(4.75, 5.75, 5, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# RunCloudyMCMC(den_array=den_array_AGN, Z_array=Z_array_AGN, T_array=T_array_AGN, alpha_ox_array=alpha_ox_array_AGN,
#               alpha_uv_array=alpha_uv_array_AGN, alpha_x_array=alpha_x_array_AGN, region='S1', trial='AGN_2',
#               bnds=S1_bnds, line_param=S1_param, deredden=False, mode='AGN_nouv', nums_chain=5000, nums_disc=1000)
#


# S1 LHIS
# S1_bnds = np.array([[-2, 2.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S1_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S1', trial='t1_Emi', norm='LHIS', bnds=S1_bnds, line_param=S1_param, deredden=True,
#               nums_chain=5000, nums_disc=1000)

# S1 lognormal
# S1_bnds = np.array([[0, 3.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S1_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# den_array_S1_Emi = np.linspace(-2, 3.6, 29, dtype='f2')
# RunCloudyMCMC(den_array=den_array_S1_Emi, region='S1', trial='t1_Emi_PP', norm='lognormal', bnds=S1_bnds, line_param=S1_param,
#               deredden=True, nums_chain=5000, nums_disc=2000, figname_extra='_lognormal_4')

# S2
# S2_bnds = np.array([[-2, 2.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S2_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S2', trial='t1', bnds=S2_bnds, line_param=S2_param, deredden=True)


# S2 low ionizaton
# S2_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', False]], dtype=bool)
# RunCloudyMCMC(region='S2', trial='t1', figname_extra='LIP', bnds=S2_bnds, line_param=S2_param, deredden=True)

# S2 middle ionizaton
# S2_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S2', trial='t1', figname_extra='MIP', bnds=S2_bnds, line_param=S2_param, deredden=True)

# S2 high ionizaton
# S2_param = np.array([['NeV3346', True],
#                      ['OII', False],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S2', trial='t1', figname_extra='HIP', bnds=S2_bnds, line_param=S2_param,
#               deredden=True, norm='HeII')

# S2 BB
# T_array = np.linspace(4, 6.5, 13, dtype='f2')[1:10]
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))
# S2_bnds = np.array([[-2, 4.6],
#                     [4, 6.0],
#                     [-1.5, 0.5]])
# S2_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(den_array=Hden_ext, alpha_array=T_array, region='S2', trial='BB_t1', mode='BB',
#               bnds=S2_bnds, line_param=S2_param, deredden=True)

# # S3
# S3_bnds = np.array([[-2, 3.0],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S3_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S3', trial='t1', bnds=S3_bnds, line_param=S3_param, deredden=False)

# # S4
# S4_bnds = np.array([[-2, 3.0],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S4_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S4', trial='t1', bnds=S4_bnds, line_param=S4_param, deredden=False)

# S5
# S5_bnds = np.array([[-2, 6.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S5_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 6.6, 20, dtype='f2')))
# RunCloudyMCMC(region='S5', trial='t1', bnds=S5_bnds, line_param=S5_param, deredden=True,
#               den_array=Hden_ext)

# S5 lognormal
# S5_bnds = np.array([[1, 5.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S5_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# den_array_S5_Emi = np.linspace(-2, 5.6, 39, dtype='f2')
# RunCloudyMCMC(den_array=den_array_S5_Emi, region='S5', trial='t1_Emi', norm='lognormal', bnds=S5_bnds, line_param=S5_param,
#               deredden=True, nums_chain=1000, nums_disc=200, figname_extra='_lognormal')

# S5_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))
# RunCloudyMCMC(region='S5', trial='t2', bnds=S5_bnds, line_param=S5_param, deredden=True, den_array=Hden_ext)


# S6
S6_bnds = np.array([[-2, 3.4],
                    [-1.8, 0],
                    [-1.5, 0.5]])
S6_param = np.array([['NeV3346', True],
                     ['OII', True],
                     ['NeIII3869', False],
                     ['Hdel', True],
                     ['Hgam', True],
                     ['OIII4364', True],
                     ['HeII4687', True],
                     ['OIII5008', True]], dtype=bool)
Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 3.4, 4, dtype='f2')))
RunCloudyMCMC(den_array=Hden_ext, region='S6', trial='t1', bnds=S6_bnds, line_param=S6_param, deredden=True)
# RunCloudyMCMC(region='S6', trial='t2', bnds=S6_bnds, line_param=S6_param, deredden=True)

# S6 lognormal
# S6_bnds = np.array([[1, 6.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S6_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# den_array_S6_Emi = np.linspace(-2, 6.6, 44, dtype='f2')
# RunCloudyMCMC(den_array=den_array_S6_Emi, region='S6', trial='t1_Emi_PP', norm='lognormal', bnds=S6_bnds,
#               line_param=S6_param, deredden=True, nums_chain=5000, nums_disc=1000, figname_extra='_lognormal')

# S6 LHIS
# S6_bnds = np.array([[-2, 3.4],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S6_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 3.4, 4, dtype='f2')))
# RunCloudyMCMC(den_array=Hden_ext, region='S6', trial='t1_Emi', norm='LHIS', bnds=S6_bnds, line_param=S6_param,
#               deredden=True, nums_chain=5000, nums_disc=1000)

# S6 low ionizaton
# S6_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', False]], dtype=bool)
# RunCloudyMCMC(den_array=Hden_ext, region='S6', trial='t1', figname_extra='LIP',
#               bnds=S6_bnds, line_param=S6_param, deredden=True)

# S6 middle ionizaton
# S6_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(den_array=Hden_ext, region='S6', trial='t1', figname_extra='MIP',
#               bnds=S6_bnds, line_param=S6_param, deredden=True)

# S6 high ionizaton
# S6_param = np.array([['NeV3346', True],
#                      ['OII', False],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(den_array=Hden_ext, region='S6', trial='t1', figname_extra='HIP',
#               bnds=S6_bnds, line_param=S6_param, deredden=True, norm='HeII')

# S6 AGN
# S6_bnds = np.array([[1.0, 3.4],
#                     [-0.5, 0.5],
#                     [4.75, 5.75],
#                     [-1.2, 0],
#                     [-0.52, -0.48],
#                     [-1.5, 0.5]])
# S6_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# den_array_AGN = np.linspace(1.0, 3.4, 13, dtype='f2')
# Z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(4.75, 5.75, 5, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# RunCloudyMCMC(den_array=den_array_AGN, Z_array=Z_array_AGN, T_array=T_array_AGN, alpha_ox_array=alpha_ox_array_AGN,
#               alpha_uv_array=alpha_uv_array_AGN, alpha_x_array=alpha_x_array_AGN, region='S6', trial='AGN',
#               bnds=S6_bnds, line_param=S6_param, deredden=False, mode='AGN_nouv', nums_chain=5000, nums_disc=1000)

# S7
# S7_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S7_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))
# RunCloudyMCMC(region='S7', trial='t1', bnds=S7_bnds, line_param=S7_param, deredden=True, den_array=Hden_ext)
# RunCloudyMCMC(region='S7', trial='t2', bnds=S7_bnds, line_param=S7_param, deredden=True, den_array=Hden_ext)

# S8
# S8_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S8_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', False],
#                      ['Hgam', True],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))
# RunCloudyMCMC(den_array=Hden_ext, region='S8', trial='t1', bnds=S8_bnds, line_param=S8_param, deredden=True)
# RunCloudyMCMC(den_array=Hden_ext, region='S8', trial='t2', bnds=S8_bnds, line_param=S8_param, deredden=True)


# S9
# S9_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S9_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))
# RunCloudyMCMC(den_array=Hden_ext, region='S9', trial='t1', bnds=S9_bnds, line_param=S9_param, deredden=True)
# RunCloudyMCMC(den_array=Hden_ext, region='S9', trial='t2', bnds=S9_bnds, line_param=S9_param, deredden=True)

# S9 AGN
# S9_bnds = np.array([[1.0, 4.6],
#                     [-0.5, 0.5],
#                     [4.75, 5.75],
#                     [-1.2, 0],
#                     [-0.52, -0.48],
#                     [-1.5, 0.5]])
# S9_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# den_array_AGN = np.linspace(1.0, 4.6, 19, dtype='f2')
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(4.75, 5.75, 5, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# RunCloudyMCMC(den_array=den_array_AGN, Z_array=z_array_AGN, T_array=T_array_AGN, alpha_ox_array=alpha_ox_array_AGN,
#               alpha_uv_array=alpha_uv_array_AGN, alpha_x_array=alpha_x_array_AGN, region='S9', trial='AGN',
#               mode='AGN_nouv', bnds=S9_bnds, line_param=S9_param, deredden=True)

# S10
# S10_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# S10_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', True],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))
# RunCloudyMCMC(den_array=Hden_ext, region='S10', trial='t1', bnds=S10_bnds, line_param=S10_param, deredden=True)
# RunCloudyMCMC(den_array=Hden_ext, region='S10', trial='t2', bnds=S10_bnds, line_param=S10_param, deredden=True)


# # B1
# B1_bnds = np.array([[-2, 2.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# B1_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='B1', trial='t1', bnds=B1_bnds, line_param=B1_param, deredden=True, norm='OII')

# # B2
# B2_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# B2_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='B2', trial='t1', bnds=B2_bnds, line_param=B2_param, deredden=True, norm='OII')

# B3
# B3_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# B3_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='B3', trial='t1', bnds=B3_bnds, line_param=B3_param, deredden=True, norm='OII')

# B3_new
# B3_new_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# B3_new_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='B3_new', trial='t1_Emi', bnds=B3_new_bnds, line_param=B3_new_param, deredden=True, norm='OII')

# B4_new
# B4_new_bnds = np.array([[-2, 4.6],
#                     [-1.8, 0],
#                     [-1.5, 0.5]])
# B4_new_param = np.array([['NeV3346', False],
#                      ['OII', True],
#                      ['NeIII3869', False],
#                      ['Hdel', False],
#                      ['Hgam', False],
#                      ['OIII4364', False],
#                      ['HeII4687', False],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='B4_new', trial='t1_Emi', bnds=B4_new_bnds, line_param=B4_new_param, deredden=True, norm='OII')