import os
import emcee
import corner
import lmfit
import pyneb as pn
import numpy as np
import astropy.io.fits as fits
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

# [OII] 3726/3729" "[OIII] 4363/5007" "
O2 = pn.Atom('O', 2)
O3 = pn.Atom('O', 3)

# Load S2 line ratio
path_fit_info_sr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'moreline_profile_selected_region.fits')
data_fit_info_sr = fits.getdata(path_fit_info_sr, 0, ignore_missing_end=True)
r_OII = data_fit_info_sr[:, 11]
dr_OII = data_fit_info_sr[:, 25]
r_OIII = data_fit_info_sr[:, 8] / data_fit_info_sr[:, 13]
dr_OIII = r_OIII * np.sqrt((data_fit_info_sr[:, 22] / data_fit_info_sr[:, 8] ) ** 2
                           + (data_fit_info_sr[:, 27] / data_fit_info_sr[:, 13]) ** 2)
logr_OII = np.log10(r_OII)
logr_OIII = np.log10(r_OIII)
logdr_OII = dr_OII / (r_OII * np.log(dr_OII))
logdr_OIII = dr_OIII / (r_OIII * np.log(dr_OIII))

# [Ne V] 3346.79, [Ne III] 3869, He I 3889 and H8, NeIII3968 and Hepsilon. Hdelta, Hgamma, [O III] 4364, He II 4687
# lines_more = (1 + z) * np.array([3346.79, 3869.86, 3889.00, 3890.16, 3968.59, 3971.20, 4102.89, 4341.68,
#                                  4364.44, 4687.31])
# [O II] 3727, 3729, Hbeta, [O III] 4960 5008
# lines = (1 + z) * np.array([3727.092, 3729.8754960, 4862.721, 4960.295, 5008.239])
fig, axarr = plt.subplots(2, 1, figsize=(5, 10), dpi=300)
O2.plotEmiss(tem_min=1000, tem_max=20000, ionic_abund=1.0, den=1e3, style='-',
             legend_loc=4, temLog=False, plot_total=False, plot_only_total=False,
             legend=True, total_color='black', total_label='TOTAL', ax=axarr[0])
O3.plotEmiss(tem_min=1000, tem_max=20000, ionic_abund=1.0, den=1e3, style='-',
             legend_loc=4, temLog=False, plot_total=False, plot_only_total=False,
             legend=True, total_color='black', total_label='TOTAL', ax=axarr[1])
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test.png', bbox_inches='tight')


# [O II] vs density
fig, axarr = plt.subplots(2, 1, figsize=(5, 10), dpi=300)
tem_array_den = np.array([8000, 10000, 20000])
den_array_den = 10 ** np.linspace(-1, 6, 50)
OII3727_den = O2.getEmissivity(tem=tem_array_den, den=den_array_den, wave=3727)
OII3729_den = O2.getEmissivity(tem=tem_array_den, den=den_array_den, wave=3729)
for i in range(len(tem_array_den)):
    axarr[0].plot(np.log10(den_array_den), np.log10(OII3729_den/OII3727_den)[i, :], '-',
                  label='T=' + str(tem_array_den[i]))
axarr[0].set_xlabel(r"$\mathrm{log_{10}[Hydrogen \, density]}$", size=15)
axarr[0].set_ylabel(r'$\mathrm{log( [O \, II] \lambda 3729 / \lambda 3727)}$', size=15)
axarr[0].legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)

# [O II] vs temperature
tem_array_tem = 10 ** np.linspace(1, 5, 1000)
den_array_tem = np.array([1, 100, 1000, 10000])
OII3727_tem = O2.getEmissivity(tem=tem_array_tem, den=den_array_tem, wave=3727)
OII3729_tem = O2.getEmissivity(tem=tem_array_tem, den=den_array_tem, wave=3729)
for i in range(len(den_array_tem)):
    axarr[1].plot(np.log10(tem_array_tem), np.log10(OII3729_tem/OII3727_tem)[:, i], '-', label='n=' +
                                                                                            str(den_array_tem[i]))
axarr[1].set_xlabel(r"$\mathrm{log(Temperature \, [K])}$", size=15)
axarr[1].set_ylabel(r'$\mathrm{log( [O \, II] \lambda 3729 / \lambda 3727)}$', size=15)
axarr[1].legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test_OII.png', bbox_inches='tight')

# # ###
# [O III] vs density
fig, axarr = plt.subplots(2, 1, figsize=(5, 10), dpi=300)
tem_array_den = np.array([8000, 10000, 20000])
den_array_den = 10 ** np.linspace(-1, 6, 50)
OIII4363_den = O3.getEmissivity(tem=tem_array_den, den=den_array_den, wave=4363)
OIII5007_den = O3.getEmissivity(tem=tem_array_den, den=den_array_den, wave=5007)
for i in range(len(tem_array_den)):
    axarr[0].plot(np.log10(den_array_den), np.log10(OIII4363_den/OIII5007_den)[i, :],
                  '-', label='T=' + str(tem_array_den[i]))
axarr[0].set_xlabel(r"$\mathrm{log_{10}[Hydrogen \, density]}$", size=15)
axarr[0].set_ylabel(r'$\mathrm{log( [O \, III] \lambda 4363 / \lambda 5007)}$', size=15)
axarr[0].legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)

# [O II] vs temperature
tem_array_tem = 10 ** np.linspace(3.7, 5.7, 1000)
den_array_tem = np.array([1, 100, 1000, 10000])
OIII4363_tem = O3.getEmissivity(tem=tem_array_tem, den=den_array_tem, wave=4363)
OIII5007_tem = O3.getEmissivity(tem=tem_array_tem, den=den_array_tem, wave=5007)
for i in range(len(den_array_tem)):
    axarr[1].plot(np.log10(tem_array_tem), np.log10(OIII4363_tem/OIII5007_tem)[:, i],
                  '-', label='n=' + str(den_array_tem[i]))
axarr[1].set_xlabel(r"$\mathrm{log(Temperature \, [K])}$", size=15)
axarr[1].set_ylabel(r'$\mathrm{log( [O \, III] \lambda 4363 / \lambda 5007)}$', size=15)
axarr[1].legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test_OIII.png', bbox_inches='tight')


# MCMC and interpolation
tem_array = 10 ** np.linspace(3.5, 4.5, 200)
den_array = 10 ** np.linspace(1.0, 2.0, 200)
Tem, Den = np.meshgrid(tem_array, den_array)
OII3727_array = O2.getEmissivity(tem=tem_array, den=den_array, wave=3727)
OII3729_array = O2.getEmissivity(tem=tem_array, den=den_array, wave=3729)
OIII4363_array = O3.getEmissivity(tem=tem_array, den=den_array, wave=4363)
OIII5007_array = O3.getEmissivity(tem=tem_array, den=den_array, wave=5007)

#

# Interpolation
f_OII = interpolate.interp2d(np.log10(Tem.flatten()), np.log10(Den.flatten()),
                         np.log10(OII3729_array / OII3727_array).flatten(), kind='cubic')
f_OIII = interpolate.interp2d(np.log10(Tem.flatten()), np.log10(Den.flatten()),
                         np.log10(OIII4363_array / OIII5007_array).flatten(), kind='cubic')

# print(np.shape(Tem), np.shape(Den))
# print(np.shape(OII3727_array))
# print(f(tem_array, den_array)[50:, 50:])
# print(np.shape(f(np.log10(tem_array), np.log10(den_array))))

# Plot the result of interpolation
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(np.log10(Tem), np.log10(Den), np.log10(OII3729_array / OII3727_array))
# ax.plot_surface(np.log10(Tem), np.log10(Den), f(np.log10(tem_array), np.log10(den_array)))
# ax.plot_surface(np.log10(Tem), np.log10(Den), f(np.log10(Tem.flatten()), np.log10(Den.flatten()))[0, :].reshape((100, 100)))
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test_interp2d.png', bbox_inches='tight')
#
# true_line_ratio = -0.1
# print(- 0.5 * ((f(4.11, 3.01) - true_line_ratio) / 1e-8) ** 2)
def log_prob(x):
    logtem, logden = x[0], x[1]
    return - 0.5 * (((f_OII(logtem, logden) - logr_OII) / logdr_OII) ** 2 +
                    ((f_OIII(logtem, logden) - logr_OIII) / logdr_OIII) ** 2)
#
ndim, nwalkers = 2, 10
p0 = np.array([4.1, 1.5]) + 0.001 * np.random.randn(nwalkers, ndim)
print(p0)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(p0, 100)
samples = sampler.get_chain(flat=True)

chain_emcee = sampler.get_chain()
# Plot each walker with a separate line.  This is c1.
f, ax = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={'wspace':0.3})
for j in range(2):
    for i in range(chain_emcee.shape[1]):
        ax[j].plot(np.arange(chain_emcee.shape[0]), chain_emcee[:, i, j], lw=1)
    ax[j].set_xlabel("Step Number")
    ax[j].set_xlim(0, 100)
    ax[j].set_ylim(0, 5)
f.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test_mcmc_chain.png', bbox_inches='tight')
#
# lnprob = sampler.get_log_prob()[-1]

# ndim, nsamples = 2, 10000
# np.random.seed(42)
# samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
# print(np.shape(samples))
# figure = corner.corner(samples)
# samples = OII3729_tem / OII3727_tem
# plt.figure(figsize=(10, 10), dpi=300)
# figure = corner.corner(samples, title_fmt = '.3f',labels=[r"$T$", r"$\mathrm{Density}$"],
#                        quantiles=[0.25, 0.68, 0.95], show_titles=True, color='k')
# figure.savefig('/Users/lzq/Dropbox/Data/CGM_plots/pyneb_test_mcmc.png', bbox_inches='tight')