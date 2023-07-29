import os
import emcee
import corner
from scipy import integrate
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


def lognormal(logn, logmu, sigma):
    return np.exp(- (np.log(10 ** logn[:]) - np.log(10 ** logmu[:, np.newaxis])) ** 2 / 2 / sigma[:, np.newaxis] ** 2) / np.sqrt(2 * np.pi) / sigma[:, np.newaxis]


def log_prob_lognormal(x, bnds, line_param, mode, f_NeV3346, f_OII, f_NeIII3869, f_Hdel, f_Hgam, f_OIII4364, f_HeII4687,
                 f_OIII5008, f_Hbeta, logflux_NeV3346, logflux_OII, logflux_NeIII3869, logflux_Hdel, logflux_Hgam,
                 logflux_OIII4364, logflux_HeII4687, logflux_OIII5008, logflux_Hbeta, dlogflux_NeV3346, dlogflux_OII,
                 dlogflux_NeIII3869, dlogflux_Hdel, dlogflux_Hgam, dlogflux_OIII4364, dlogflux_HeII4687,
                 dlogflux_OIII5008, dlogflux_Hbeta):
    # if mode == 'power_law' or mode == 'BB':
    logmu, sigma, alpha, logz = x[:]

    logn = np.linspace(bnds[0, 0], bnds[0, 1], 1000)
    prob = lognormal(logn, logmu, sigma)

    if logmu < bnds[0, 0] or logmu > bnds[0, 1]:
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

        sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
                              chi2_HeII4687, chi2_OIII5008])
        L = - 0.5 * np.nansum(sum_array)
        if L == 0:
            L = -np.inf
        # print(L)
    return L

# den_test = np.linspace(1.0, 2.4, 8, dtype='f2')
den_test = np.linspace(-5, 2.6, 100)
prob_test = lognormal(den_test, np.array([1, 2]), np.array([5, 6]))
print(prob_test)
plt.figure()
plt.plot(den_test, prob_test.T)


sum = np.trapz(prob_test, x=np.log(10 ** den_test))
print(integrate.cumtrapz(prob_test, np.log(10 ** den_test)))
print(sum)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/lognormal_test')
