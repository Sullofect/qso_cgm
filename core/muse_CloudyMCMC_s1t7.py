import emcee
import corner
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rc
from muse_LoadCloudy import format_cloudy_nogrid
from muse_LoadLineRatio import load_lineratio
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('axes', **{'labelsize':15})

# Load the actual measurement
logflux_Hbeta, dlogflux_Hbeta, logflux_NeV3346, dlogflux_NeV3346, logflux_OII, dlogflux_OII, logr_OII, \
dlogr_OII, logflux_NeIII3869, dlogflux_NeIII3869, logflux_Hdel, dlogflux_Hdel, logflux_Hgam, dlogflux_Hgam, \
logflux_OIII4364, dlogflux_OIII4364, logflux_HeII4687, dlogflux_HeII4687, \
logflux_OIII5008, dlogflux_OIII5008 = load_lineratio(region='S1')

# Load cloudy result
Hden = np.linspace(-2, 2.6, 24, dtype='f2') # log # 0.25
metal = np.linspace(-1.5, 0.5, 11, dtype='f2')
alpha = np.linspace(-1.8, 0, 10, dtype='f2')
output = format_cloudy_nogrid(filename=[Hden, alpha, metal], path='/Users/lzq/Dropbox/Data/CGM/cloudy/trial7/')

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
    # if alpha < -1.2:
    #     return -np.inf
    # if alpha > -0.5:
    #     return -np.inf
    # if logz > 0.:
    #     return -np.inf
    # if logden > 2.6:
    #     return -np.inf
    # elif logden < 0:
    #     return -np.inf
    # else:
    return - 0.5 * (((f_NeV3346((logden, alpha, logz)) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
                    + ((f_OII((logden, alpha, logz)) - logflux_OII) / dlogflux_OII) ** 2
                    + ((f_NeIII3869((logden, alpha, logz)) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
                    + ((f_Hdel((logden, alpha, logz)) - logflux_Hdel) / dlogflux_Hdel) ** 2
                    + ((f_Hgam((logden, alpha, logz)) - logflux_Hgam) / dlogflux_Hgam) ** 2
                    + ((f_OIII4364((logden, alpha, logz)) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
                    + ((f_HeII4687((logden, alpha, logz)) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
                    + ((f_OIII5008((logden, alpha, logz)) - logflux_OIII5008) / dlogflux_OIII5008) ** 2)


ndim, nwalkers = 3, 40
p0 = np.array([1.8, -1.5, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(p0, 3000)
samples = sampler.get_chain(flat=True, discard=100)

figure = corner.corner(samples, labels=[r"$\mathrm{log_{10}(n)}$", r"$\mathrm{\alpha}$",
                                        r"$\mathrm{log_{10}(Z/Z_{\odot})}$"],
                       quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k', title_kwargs={"fontsize": 13},
                       smooth=1., smooth1d=1., bins=25)

for i, ax in enumerate(figure.get_axes()):
    if not np.isin(i, np.arange(0, ndim ** 2, ndim + 1)):
        ax.tick_params(axis='both', direction='in', top='on', bottom='on', left='on', right='on')
    ax.tick_params(axis='both', direction='in', top='on', bottom='on')
figure.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_test_mcmc_S1_t7.pdf', bbox_inches='tight')