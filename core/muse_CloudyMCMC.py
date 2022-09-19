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
logflux_NeV3346, logdflux_NeV3346 =  np.log10(flux_NeV3346), dflux_NeV3346 / (flux_NeV3346 * np.log(10))
logflux_OII, logdflux_OII = np.log10(flux_OII), dflux_OII / (flux_OII * np.log(10))
logflux_NeIII3869, logdflux_NeIII3869 =  np.log10(flux_NeIII3869), dflux_NeIII3869 / (flux_NeIII3869 * np.log(10))
logflux_Hdel, logdflux_Hdel =  np.log10(flux_Hdel), dflux_Hdel / (flux_Hdel * np.log(10))
logflux_Hgam, logdflux_Hgam =  np.log10(flux_Hgam), dflux_Hgam / (flux_Hgam * np.log(10))
logflux_OIII4364, logdflux_OIII4364 =  np.log10(flux_OIII4364), dflux_OIII4364 / (flux_OIII4364 * np.log(10))
logflux_HeII4687, logdflux_HeII4687 =  np.log10(flux_HeII4687), dflux_HeII4687 / (flux_HeII4687 * np.log(10))
logflux_OIII5008, logdflux_OIII5008 =  np.log10(flux_OIII5008), dflux_OIII5008 / (flux_OIII5008 * np.log(10))



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
    return - 0.5 * (((f_NeV3346((logden, logz)) - logflux_NeV3346) / logdflux_NeV3346) ** 2
                    + ((f_OII((logden, logz)) - logflux_OII) / logdflux_OII) ** 2
                    + ((f_NeIII3869((logden, logz)) - logflux_NeIII3869) / logdflux_NeIII3869) ** 2
                    + ((f_Hdel((logden, logz)) - logflux_Hdel) / logdflux_Hdel) ** 2
                    + ((f_Hgam((logden, logz)) - logflux_Hgam) / logdflux_Hgam) ** 2
                    + ((f_OIII4364((logden, logz)) - logflux_OIII4364) / logdflux_OIII4364) ** 2
                    + ((f_HeII4687((logden, logz)) - logflux_HeII4687) / logdflux_HeII4687) ** 2
                    + ((f_OIII5008((logden, logz)) - logflux_OIII5008) / logdflux_OIII5008) ** 2)


ndim, nwalkers = 2, 40
p0 = np.array([1.1, -0.1]) + 0.01 * np.random.randn(nwalkers, ndim)
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