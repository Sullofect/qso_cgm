import os
import emcee
import corner
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('axes', **{'labelsize':15})

# Load the actual measurement
# Load S2 line ratio
# Use OII [Ne V] [Ne III] Hdel Hgam [O III] He II [O III]
path_fit_info_sr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                                'moreline_profile_selected_region.fits')
data_fit_info_sr = fits.getdata(path_fit_info_sr, ignore_missing_end=True)
data_fit_info_sr = data_fit_info_sr[7]  ### S8

flux_Hbeta, dflux_Hbeta = data_fit_info_sr['flux_Hbeta'], data_fit_info_sr['dflux_Hbeta']
flux_OII = data_fit_info_sr['flux_OII'] / flux_Hbeta
dflux_OII = flux_OII * np.sqrt((data_fit_info_sr['dflux_OII'] / data_fit_info_sr['flux_OII']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
r_OII, dr_OII = data_fit_info_sr['r_OII'], data_fit_info_sr['dr_OII']
flux_NeV3346 = data_fit_info_sr['flux_NeV3346'] / flux_Hbeta
dflux_NeV3346 = flux_NeV3346 * np.sqrt((data_fit_info_sr['dflux_NeV3346'] / data_fit_info_sr['flux_NeV3346']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
flux_NeIII3869 = data_fit_info_sr['flux_NeIII3869'] / flux_Hbeta
dflux_NeIII3869 = flux_NeIII3869 * np.sqrt((data_fit_info_sr['dflux_NeIII3869'] / data_fit_info_sr['flux_NeIII3869']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
flux_Hdel = data_fit_info_sr['flux_Hdel'] / flux_Hbeta
dflux_Hdel = flux_Hdel * np.sqrt((data_fit_info_sr['dflux_Hdel'] / data_fit_info_sr['flux_Hdel']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
flux_Hgam = data_fit_info_sr['flux_Hgam'] / flux_Hbeta
dflux_Hgam = flux_Hgam * np.sqrt((data_fit_info_sr['dflux_Hgam'] / data_fit_info_sr['flux_Hgam']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
flux_OIII4364 = data_fit_info_sr['flux_OIII4364'] / flux_Hbeta
dflux_OIII4364 = flux_OIII4364 * np.sqrt((data_fit_info_sr['dflux_OIII4364'] / data_fit_info_sr['flux_OIII4364']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
flux_HeII4687 = data_fit_info_sr['flux_HeII4687'] / flux_Hbeta
dflux_HeII4687 = flux_HeII4687 * np.sqrt((data_fit_info_sr['dflux_HeII4687'] / data_fit_info_sr['flux_HeII4687']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)
flux_OIII5008 = data_fit_info_sr['flux_OIII5008'] / flux_Hbeta
dflux_OIII5008 = flux_OIII5008 * np.sqrt((data_fit_info_sr['dflux_OIII5008'] / data_fit_info_sr['flux_OIII5008']) ** 2
                               + (dflux_Hbeta / flux_Hbeta) ** 2)

# Take the log
logflux_Hbeta, dlogflux_Hbeta = np.log10(flux_Hbeta), dflux_Hbeta / (flux_Hbeta * np.log(10))
logflux_NeV3346, dlogflux_NeV3346 =  np.log10(flux_NeV3346), np.sqrt((dflux_NeV3346 / (flux_NeV3346 * np.log(10))) ** 2
                                                                     + 0.0 ** 2)
logflux_OII, dlogflux_OII = np.log10(flux_OII), np.sqrt((dflux_OII / (flux_OII * np.log(10))) ** 2 + 0.0 ** 2)
logr_OII, dlogr_OII = np.log10(r_OII), np.sqrt((dr_OII / (r_OII * np.log(10))) ** 2 + 0.0 ** 2)
logflux_NeIII3869, dlogflux_NeIII3869 =  np.log10(flux_NeIII3869), np.sqrt((dflux_NeIII3869 /
                                                                            (flux_NeIII3869 * np.log(10))) ** 2
                                                                           + 0.0 ** 2)
logflux_Hdel, dlogflux_Hdel =  np.log10(flux_Hdel), np.sqrt((dflux_Hdel / (flux_Hdel * np.log(10))) ** 2 + 0.0 ** 2)
logflux_Hgam, dlogflux_Hgam =  np.log10(flux_Hgam), np.sqrt((dflux_Hgam / (flux_Hgam * np.log(10))) ** 2 + 0.0 ** 2)
logflux_OIII4364, dlogflux_OIII4364 =  np.log10(flux_OIII4364), np.sqrt((dflux_OIII4364 /
                                                                         (flux_OIII4364 * np.log(10))) ** 2 + 0.0 ** 2)
# logflux_HeII4687, dlogflux_HeII4687 =  np.log10(flux_HeII4687), np.sqrt((dflux_HeII4687 /
#                                                                          (flux_HeII4687 * np.log(10))) ** 2 + 0.0 ** 2)
logflux_OIII5008, dlogflux_OIII5008 =  np.log10(flux_OIII5008), np.sqrt((dflux_OIII5008 /
                                                                         (flux_OIII5008 * np.log(10))) ** 2 + 0.0 ** 2)
# print(dlogflux_NeV3346)
# print(dlogflux_OII)
# print(dlogflux_NeIII3869)
# print(dlogflux_Hdel)
# print(dlogflux_Hgam)
# print(dlogflux_OIII4364)
# print(dlogflux_HeII4687)
# print(dlogflux_OIII5008)


# Load cloudy result
Hden = np.arange(-2, 2.6, 0.1)  # log
metal = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])  # log
alpha = np.array([-1.8, -1.75, -1.7, -1.65, -1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2,
                  -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])

# Load lineratio
def load_cloudy(filename=None, path=None):
    # Line profile
    line = np.genfromtxt(path + filename + '.lin', delimiter=None)
    NeV3346, OII3727, OII3730 = line[:, 2], line[:, 3], line[:, 4]
    NeIII3869, Hdel, Hgam = line[:, 5], line[:, 8], line[:, 9]
    OIII4364, HeII4687, OIII5008 = line[:, 10], line[:, 11], line[:, 13]
    data = np.vstack((NeV3346, OII3727, OII3730, OII3727 + OII3730, NeIII3869, Hdel, Hgam, OIII4364, HeII4687, OIII5008))
    return np.log10(data)


def format_cloudy_t2(filename=None, path=None):
    print(np.shape(filename[0]))
    print(np.shape(filename[1]))
    for i in range(len(filename[0])):
        metal_i = filename[0][i]
        for j in range(len(filename[1])):
            alpha_j = filename[1][j]
            filename_ij = 'alpha_' + str(alpha_j) + '_' + str(metal_i)
            if j == 0:
                output_j = load_cloudy(filename_ij, path=path)
                ind_j = np.array([[alpha_j, metal_i]])
            else:
                ind_jj = np.array([[alpha_j, metal_i]])
                c_i = load_cloudy(filename_ij, path=path)
                output_j = np.dstack((output_j, c_i))
                ind_j = np.dstack((ind_j, ind_jj))

        if i == 0:
            ind = ind_j[:, :, :, np.newaxis]
            output = output_j[:, :, :, np.newaxis]
        else:
            output = np.concatenate((output, output_j[:, :, :, np.newaxis]), axis=3)
            ind =  np.concatenate((ind, ind_j[:, :, :, np.newaxis]), axis=3)
    return output, ind

output, ind = format_cloudy_t2(filename=[metal, alpha], path='/Users/lzq/Dropbox/Data/CGM/cloudy/trial2/')

f_NeV3346 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[0, :, :, :],
                                                bounds_error=False, fill_value=None)
f_OII3727 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[1, :, :, :],
                                            bounds_error=False, fill_value=None)
f_OII3730 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[2, :, :, :],
                                            bounds_error=False, fill_value=None)
f_OII = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[3, :, :, :],
                                            bounds_error=False, fill_value=None)
f_NeIII3869 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[4, :, :, :],
                                                  bounds_error=False, fill_value=None)
f_Hdel = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[5, :, :, :],
                                             bounds_error=False, fill_value=None)
f_Hgam = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[6, :, :, :],
                                             bounds_error=False, fill_value=None)
f_OIII4364 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[7, :, :, :],
                                                 bounds_error=False, fill_value=None)
f_HeII4687 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[8, :, :, :],
                                                 bounds_error=False, fill_value=None)
f_OIII5008 = interpolate.RegularGridInterpolator((Hden, alpha, metal), output[9, :, :, :],
                                                 bounds_error=False, fill_value=None)

# Define the log likelihood function and run MCMC
def log_prob(x):
    logden, alpha, logz = x[0], x[1], x[2]
    # if alpha > -1.2:
    #     return -np.inf
    # else:
    return - 0.5 * (
                    #((f_NeV3346((logden, alpha, logz)) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
                    ((f_OII((logden, alpha, logz)) - logflux_OII) / dlogflux_OII) ** 2
                    # + ((f_NeIII3869((logden, alpha, logz)) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
                    # + ((f_Hdel((logden, alpha, logz)) - logflux_Hdel) / dlogflux_Hdel) ** 2
                    # + ((f_Hgam((logden, alpha, logz)) - logflux_Hgam) / dlogflux_Hgam) ** 2
                    # + ((f_OIII4364((logden, alpha, logz)) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
                    # + ((f_HeII4687((logden, alpha, logz)) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
                    + ((f_OIII5008((logden, alpha, logz)) - logflux_OIII5008) / dlogflux_OIII5008) ** 2)


ndim, nwalkers = 3, 40
p0 = np.array([1.1, -1.4, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(p0, 2000)
samples = sampler.get_chain(flat=True, discard=1000)

# chain_emcee = sampler.get_chain()
# f, ax = plt.subplots(1, 2, figsize=(15, 7))
# for j in range(2):
#     for i in range(chain_emcee.shape[1]):
#         ax[j].plot(np.arange(chain_emcee.shape[0]), chain_emcee[:, i, j], lw=1)
#     ax[j].set_xlabel("Step Number")
#     ax[j].set_xlim(0, 1000)
#     ax[j].set_ylim(0, 5)
# f.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test_mcmc_chain.png', bbox_inches='tight')

#
# ax = plt.figure(figsize=(10, 10), dpi=300)
figure = corner.corner(samples, labels=[r"$\mathrm{log_{10}(n)}$", r"$\mathrm{\alpha}$",
                                        r"$\mathrm{log_{10}(Z/Z_{\odot})}$"],
                       quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k', title_kwargs={"fontsize": 13},
                       smooth=1., smooth1d=1., bins=25)

for i, ax in enumerate(figure.get_axes()):
    if i == 2:
        ax.tick_params(axis='both', direction='in', top='on', bottom='on', left='on', right='on')
    ax.tick_params(axis='both', direction='in', top='on', bottom='on')
figure.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_test_mcmc_S8.pdf', bbox_inches='tight')

#
# NeV3346_NeIII3869_error = np.sqrt(dlogflux_NeV3346 ** 2 + dlogflux_NeIII3869 ** 2)
# OIII5008_OII_error = np.sqrt(dlogflux_OIII5008 ** 2 + dlogflux_OII ** 2)
# fig, ax = plt.subplots(3, 1, figsize=(5, 12), dpi=300, sharex=True)
# for z_i in np.array([-1.5, -1., -0.5, 0, 0.3, 0.4, 0.5]):
#     i = np.where(metal == z_i)
#     i = i[0][0]
#     ax[0].plot(Hden, output[0, :, i] - output[2, :, i], label="Z = " + str(z_i))
#     ax[0].plot(Hden, f_NeV3346((Hden, z_i)) - f_NeIII3869((Hden, z_i)), '--k', lw=0.2)
#     ax[1].plot(Hden, output[7, :, i] - output[1, :, i], label="Z = " + str(z_i))
#     ax[1].plot(Hden, f_OIII5008((Hden, z_i)) - f_OII((Hden, z_i)), '--k', lw=0.2)
#     ax[2].plot(Hden, output[6, :, i], label="Z = " + str(z_i))
#     ax[2].plot(Hden, f_HeII4687((Hden, z_i)), '--k', lw=0.2)
# ax[0].axhline(logflux_NeV3346 - logflux_NeIII3869, xmin=-2, xmax=2.6, ls='--', color='r')
# ax[1].axhline(logflux_OIII5008 - logflux_OII, xmin=-2, xmax=2.6, ls='--', color='r')
# ax[2].axhline(logflux_HeII4687 - logflux_Hbeta, xmin=-2, xmax=2.6, ls='--', color='r')
# ax[0].minorticks_on()
# ax[1].minorticks_on()
# ax[2].minorticks_on()
# ax[0].set_xlim(-2.5, 3)
# # ax[1].set_ylim(-50, 50)
# ax[2].set_xlabel(r'$\mathrm{log(n)}$', size=20)
# ax[0].set_ylabel(r'$\mathrm{log(\frac{[Ne \, V]}{[Ne \, III])}}$', size=20)
# ax[1].set_ylabel(r'$\mathrm{log(\frac{[O \, III]}{[O \, II])}}$', size=20)
# ax[2].set_ylabel(r'$\mathrm{log(\frac{[He \, II]}{H\beta)}}$', size=20)
# ax[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# ax[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# ax[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# ax[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# ax[2].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# ax[2].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# ax[0].legend()
# fig.tight_layout()
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_check_MCMC.png', bbox_inches='tight')


# Check
# fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 3]}, dpi=300)
# E_NeV3346, E_NeIII3869 = 97.11, 40.96  # in ev
# E_OIII4364, E_OIII5008, E_OII = 35.12, 35.12, 13.6
# E_HeII4687, E_Hbeta = 54.42, 13.6
# data_x = [E_OII / E_Hbeta, E_OIII4364 / E_Hbeta - 0.5, E_OIII5008 / E_Hbeta, E_NeIII3869 / E_Hbeta + 0.2,
#           E_HeII4687 / E_Hbeta, E_NeV3346 / E_Hbeta - 2]
# data_y = [logflux_OII, logflux_OIII4364, logflux_OIII5008, logflux_NeIII3869, logflux_HeII4687, logflux_NeV3346]
# data_yerr = [dlogflux_OII, dlogflux_OIII4364, dlogflux_OIII5008, dlogflux_NeIII3869,
#              dlogflux_HeII4687, dlogflux_NeV3346]
# ax[1].errorbar(data_x, data_y,  data_yerr, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
# ax[0].errorbar([0, 5], [logr_OII, logflux_OIII4364 - logflux_OIII5008],
#                [dlogr_OII, np.sqrt(dlogflux_OIII4364 ** 2 + dlogflux_OIII5008 ** 2)]
#                , fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
#
# best_fit = (1.67, -0.79, 0.16)
# bestfit_y = [f_OII(best_fit), f_OIII4364(best_fit), f_OIII5008(best_fit), f_NeIII3869(best_fit), f_HeII4687(best_fit),
#              f_NeV3346(best_fit)]
# ax[1].plot(data_x, bestfit_y, '-k', alpha=1)
# ax[0].plot([0, 5], [f_OII3730(best_fit) - f_OII3727(best_fit),
#                     f_OIII4364(best_fit) - f_OIII5008(best_fit)], '-k', alpha=1)
#
# inds = np.random.randint(len(samples), size=50)
# for i in inds:
#     sample = samples[i]
#     model = (sample[0], sample[1], sample[2])
#     model_y = [f_OII(model), f_OIII4364(model), f_OIII5008(model), f_NeIII3869(model), f_HeII4687(model),
#                f_NeV3346(model)]
#     ax[1].plot(data_x, model_y, '-C1', alpha=0.1)
#     ax[0].plot([0, 5], [f_OII3730(model) - f_OII3727(model),
#                     f_OIII4364(model) - f_OIII5008(model)], '-C1', alpha=0.1)
#
# ax[1].set_xlabel(r'$\mathrm{Ionization \, energy}$', size=20)
# ax[0].set_ylabel(r'$\mathrm{Line \, ratio}$', size=20)
# ax[1].set_xticks(data_x, [r'$\mathrm{\frac{[O \, II]}{H\beta}}$', r'$\mathrm{\frac{[O \, III]}{H\beta}}$',
#                           r'$\mathrm{}$', r'$\mathrm{\frac{[Ne \, III]}{H\beta}}$',
#                           r'$\mathrm{\frac{He \, II}{H\beta}}$', r'$\mathrm{\frac{[Ne \, V]}{H\beta}}$'], y=0.8)
# ax[0].set_xticks([2, 4], [r'$\mathrm{\frac{\lambda 3729}{\lambda 3727}}$',
#                           r'$\mathrm{\frac{\lambda 4363}{\lambda 5007}}$'], y=0.8)
# ax[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# ax[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# ax[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=15, size=5)
# ax[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_check_MCMC_S8.png', bbox_inches='tight')