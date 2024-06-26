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
data_fit_info_sr = data_fit_info_sr[0]

# t = Table(flux_info, names=('flux_NeV3346', 'flux_NeIII3869', 'flux_HeI3889', 'flux_H8', f'lux_NeIII3968',
#                             'flux_Heps', 'flux_Hdel', 'flux_Hgam', 'flux_OIII4364', 'flux_HeII4687',
#                             'flux_OII', 'r_OII', 'flux_Hbeta', 'flux_OIII5008', 'dflux_NeV3346',
#                             'dflux_NeIII3869', 'dflux_HeI3889', 'dflux_H8', 'dflux_NeIII3968',
#                             'dflux_Heps', 'dflux_Hdel', 'dflux_Hgam', 'dflux_OIII4364',
#                             'dflux_HeII4687', 'dflux_OII', 'dr_OII', 'dflux_Hbeta', 'dflux_OIII5008'))

flux_Hbeta, dflux_Hbeta = data_fit_info_sr['flux_Hbeta'], data_fit_info_sr['dflux_Hbeta']

flux_OII, dflux_OII = data_fit_info_sr['flux_OII'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_OII'] / flux_Hbeta
flux_NeV3346, dflux_NeV3346 = data_fit_info_sr['flux_NeV3346'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_NeV3346'] / flux_Hbeta
flux_NeIII3869, dflux_NeIII3869 = data_fit_info_sr['flux_NeIII3869'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_NeIII3869'] / flux_Hbeta
flux_Hdel, dflux_Hdel = data_fit_info_sr['flux_Hdel'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_Hdel'] / flux_Hbeta
flux_Hgam, dflux_Hgam = data_fit_info_sr['flux_Hgam'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_Hgam'] / flux_Hbeta
flux_OIII4364, dflux_OIII4364 = data_fit_info_sr['flux_OIII4364'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_OIII4364'] / flux_Hbeta
flux_HeII4687, dflux_HeII4687 = data_fit_info_sr['flux_HeII4687'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_HeII4687'] / flux_Hbeta
flux_OIII5008, dflux_OIII5008 = data_fit_info_sr['flux_OIII5008'] / flux_Hbeta, \
                              data_fit_info_sr['dflux_OIII5008'] / flux_Hbeta

# Take the log
logflux_Hbeta, dlogflux_Hbeta = np.log10(flux_Hbeta), dflux_Hbeta / (flux_Hbeta * np.log(10))
logflux_NeV3346, dlogflux_NeV3346 =  np.log10(flux_NeV3346), np.sqrt((dflux_NeV3346 / (flux_NeV3346 * np.log(10))) ** 2
                                                                     + 0.03 ** 2)

logflux_OII, dlogflux_OII = np.log10(flux_OII), np.sqrt((dflux_OII / (flux_OII * np.log(10))) ** 2 + 0.03 ** 2)
logflux_NeIII3869, dlogflux_NeIII3869 =  np.log10(flux_NeIII3869), np.sqrt((dflux_NeIII3869 /
                                                                            (flux_NeIII3869 * np.log(10))) ** 2
                                                                           + 0.03 ** 2)
logflux_Hdel, dlogflux_Hdel =  np.log10(flux_Hdel), np.sqrt((dflux_Hdel / (flux_Hdel * np.log(10))) ** 2 + 0.03 ** 2)
logflux_Hgam, dlogflux_Hgam =  np.log10(flux_Hgam), np.sqrt((dflux_Hgam / (flux_Hgam * np.log(10))) ** 2 + 0.03 ** 2)
logflux_OIII4364, dlogflux_OIII4364 =  np.log10(flux_OIII4364), np.sqrt((dflux_OIII4364 /
                                                                         (flux_OIII4364 * np.log(10))) ** 2 + 0.03 ** 2)
logflux_HeII4687, dlogflux_HeII4687 =  np.log10(flux_HeII4687), np.sqrt((dflux_HeII4687 /
                                                                         (flux_HeII4687 * np.log(10))) ** 2 + 0.03 ** 2)
logflux_OIII5008, dlogflux_OIII5008 =  np.log10(flux_OIII5008), np.sqrt((dflux_OIII5008 /
                                                                         (flux_OIII5008 * np.log(10))) ** 2 + 0.03 ** 2)
print(dlogflux_NeV3346)
print(dlogflux_OII)
print(dlogflux_NeIII3869)
print(dlogflux_Hdel)
print(dlogflux_Hgam)
print(dlogflux_OIII4364)
print(dlogflux_HeII4687)
print(dlogflux_OIII5008)


# Load cloudy result
Hden = np.arange(-2, 2.6, 0.1)  # log
print(len(Hden))
metal = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])  # log
print(len(metal))

# Load lineratio

def load_cloudy(filename=None, path=None):
    # Line profile
    line = np.genfromtxt(path + filename + '.lin', delimiter=None)
    NeV3346, OII3727, OII3730 = line[:, 2], line[:, 3], line[:, 4]
    NeIII3869, Hdel, Hgam = line[:, 5], line[:, 8], line[:, 9]
    OIII4364, HeII4687, OIII5008 = line[:, 10], line[:, 11], line[:, 13]
    data = np.vstack((NeV3346, OII3727 + OII3730, NeIII3869, Hdel, Hgam, OIII4364, HeII4687, OIII5008))
    return np.log10(data)


def format_cloudy(filename=None, path=None):
    for i in range(len(filename)):
        filename_i = 'alpha_1.4_' + str(filename[i])
        if i == 0:
            output = load_cloudy(filename_i, path=path)
        else:
            c_i = load_cloudy(filename_i, path=path)
            output = np.dstack((output, c_i))
    return output

output = format_cloudy(filename=metal, path='/Users/lzq/Dropbox/Data/CGM/cloudy/trial1/')

f_NeV3346 = interpolate.RegularGridInterpolator((Hden, metal), output[0, :, :], bounds_error=False, fill_value=None)
f_OII = interpolate.RegularGridInterpolator((Hden, metal), output[1, :, :], bounds_error=False, fill_value=None)
f_NeIII3869 = interpolate.RegularGridInterpolator((Hden, metal), output[2, :, :], bounds_error=False, fill_value=None)
f_Hdel = interpolate.RegularGridInterpolator((Hden, metal), output[3, :, :], bounds_error=False, fill_value=None)
f_Hgam = interpolate.RegularGridInterpolator((Hden, metal), output[4, :, :], bounds_error=False, fill_value=None)
f_OIII4364 = interpolate.RegularGridInterpolator((Hden, metal), output[5, :, :], bounds_error=False, fill_value=None)
f_HeII4687 = interpolate.RegularGridInterpolator((Hden, metal), output[6, :, :], bounds_error=False, fill_value=None)
f_OIII5008 = interpolate.RegularGridInterpolator((Hden, metal), output[7, :, :], bounds_error=False, fill_value=None)

# Define the log likelihood function and run MCMC
def log_prob(x):
    logden, logz = x[0], x[1]
    # if logden < 1:
    #     return -np.inf
    # else:
    return - 0.5 * (((f_NeV3346((logden, logz)) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
                    + ((f_OII((logden, logz)) - logflux_OII) / dlogflux_OII) ** 2
                    + ((f_NeIII3869((logden, logz)) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
                    + ((f_Hdel((logden, logz)) - logflux_Hdel) / dlogflux_Hdel) ** 2
                    + ((f_Hgam((logden, logz)) - logflux_Hgam) / dlogflux_Hgam) ** 2
                    + ((f_OIII4364((logden, logz)) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
                    + ((f_HeII4687((logden, logz)) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
                    + ((f_OIII5008((logden, logz)) - logflux_OIII5008) / dlogflux_OIII5008) ** 2)


ndim, nwalkers = 2, 40
p0 = np.array([1.1, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(p0, 10000)
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
figure = corner.corner(samples, labels=[r"$\mathrm{log_{10}(n)}$", r"$\mathrm{log_{10}(Z/Z_{\odot})}$"],
                       quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k', title_kwargs={"fontsize": 13},
                       smooth=1., smooth1d=1., bins=25)

for i, ax in enumerate(figure.get_axes()):
    if i == 2:
        ax.tick_params(axis='both', direction='in', top='on', bottom='on', left='on', right='on')
    ax.tick_params(axis='both', direction='in', top='on', bottom='on')
figure.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_test_mcmc_all.pdf', bbox_inches='tight')


NeV3346_NeIII3869_error = np.sqrt(dlogflux_NeV3346 ** 2 + dlogflux_NeIII3869 ** 2)
OIII5008_OII_error = np.sqrt(dlogflux_OIII5008 ** 2 + dlogflux_OII ** 2)
fig, ax = plt.subplots(3, 1, figsize=(5, 12), dpi=300, sharex=True)
for z_i in np.array([-1.5, -1., -0.5, 0, 0.3, 0.4, 0.5]):
    i = np.where(metal == z_i)
    i = i[0][0]
    ax[0].plot(Hden, output[0, :, i] - output[2, :, i], label="Z = " + str(z_i))
    ax[0].plot(Hden, f_NeV3346((Hden, z_i)) - f_NeIII3869((Hden, z_i)), '--k', lw=0.2)
    ax[1].plot(Hden, output[7, :, i] - output[1, :, i], label="Z = " + str(z_i))
    ax[1].plot(Hden, f_OIII5008((Hden, z_i)) - f_OII((Hden, z_i)), '--k', lw=0.2)
    ax[2].plot(Hden, output[6, :, i], label="Z = " + str(z_i))
    ax[2].plot(Hden, f_HeII4687((Hden, z_i)), '--k', lw=0.2)
ax[0].axhline(logflux_NeV3346 - logflux_NeIII3869, xmin=-2, xmax=2.6, ls='--', color='r')
ax[1].axhline(logflux_OIII5008 - logflux_OII, xmin=-2, xmax=2.6, ls='--', color='r')
ax[2].axhline(logflux_HeII4687 - logflux_Hbeta, xmin=-2, xmax=2.6, ls='--', color='r')
ax[0].minorticks_on()
ax[1].minorticks_on()
ax[2].minorticks_on()
ax[0].set_xlim(-2.5, 3)
# ax[1].set_ylim(-50, 50)
ax[2].set_xlabel(r'$\mathrm{log(n)}$', size=20)
ax[0].set_ylabel(r'$\mathrm{log(\frac{[Ne \, V]}{[Ne \, III])}}$', size=20)
ax[1].set_ylabel(r'$\mathrm{log(\frac{[O \, III]}{[O \, II])}}$', size=20)
ax[2].set_ylabel(r'$\mathrm{log(\frac{[He \, II]}{H\beta)}}$', size=20)
ax[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax[2].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax[2].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax[0].legend()
fig.tight_layout()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_check_MCMC.png', bbox_inches='tight')


# Check
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300, sharex=True)
E_NeV3346, E_NeIII3869 = 97.11, 40.96  # in ev
E_OIII4364, E_OIII5008, E_OII = 35.12, 35.12, 13.6
E_HeII4687, E_Hbeta = 54.42, 13.6
data_x = [E_OII / E_Hbeta, E_OIII4364 / E_Hbeta - 0.5, E_OIII5008 / E_Hbeta, E_NeIII3869 / E_Hbeta + 0.2,
          E_HeII4687 / E_Hbeta, E_NeV3346 / E_Hbeta - 2]
data_y = [logflux_OII - logflux_Hbeta, logflux_OIII4364 - logflux_Hbeta, logflux_OIII5008 - logflux_Hbeta,
          logflux_NeIII3869 - logflux_Hbeta, logflux_HeII4687 - logflux_Hbeta, logflux_NeV3346 - logflux_Hbeta]
data_yerr = [np.sqrt(dlogflux_OII ** 2 + dlogflux_Hbeta ** 2),
             np.sqrt(dlogflux_OIII4364 ** 2 + dlogflux_Hbeta ** 2),
             np.sqrt(dlogflux_OIII5008 ** 2 + dlogflux_Hbeta ** 2),
             np.sqrt(dlogflux_NeIII3869 ** 2 + dlogflux_Hbeta ** 2),
             np.sqrt(dlogflux_HeII4687 ** 2 + dlogflux_Hbeta ** 2),
             np.sqrt(dlogflux_NeV3346 ** 2 + dlogflux_Hbeta ** 2)]
ax.errorbar(data_x, data_y,  data_yerr, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
best_fit = (1.49, -0.19)
bestfit_y = [f_OII(best_fit), f_OIII4364(best_fit), f_OIII5008(best_fit), f_NeIII3869(best_fit), f_HeII4687(best_fit),
             f_NeV3346(best_fit)]
ax.plot(data_x, bestfit_y, '-k', alpha=1)

inds = np.random.randint(len(samples), size=50)
for i in inds:
    sample = samples[i]
    model = (sample[0], sample[1])
    model_y = [f_OII(model), f_OIII4364(model), f_OIII5008(model), f_NeIII3869(model), f_HeII4687(model),
               f_NeV3346(model)]
    ax.plot(data_x, model_y, '-C1', alpha=0.1)
ax.set_xlabel(r'$\mathrm{Ionization \, energy}$', size=20)
ax.set_ylabel(r'$\mathrm{Line \, ratio}$', size=20)
ax.set_xticks(data_x, [r'$\mathrm{\frac{[O \, II]}{H\beta}}$', r'$\mathrm{\frac{[O \, III]}{H\beta}}$',
                       r'$\mathrm{}$', r'$\mathrm{\frac{[Ne \, III]}{H\beta}}$',
                       r'$\mathrm{\frac{He \, II}{H\beta}}$', r'$\mathrm{\frac{[Ne \, V]}{H\beta}}$'],
              y=0.8)
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_check_MCMC_2.png', bbox_inches='tight')