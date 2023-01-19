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


def RunCloudyMCMC(Hden=np.linspace(-2, 2.6, 24, dtype='f2'),
                  metal=np.linspace(-1.5, 0.5, 11, dtype='f2'),
                  alpha=np.linspace(-1.8, 0, 10, dtype='f2'),
                  region=None, trial=None, bnds=None, line_param=None, deredden=True):
    # Load the actual measurement
    logflux_Hbeta, dlogflux_Hbeta, logflux_NeV3346, dlogflux_NeV3346, logflux_OII, dlogflux_OII, logr_OII, \
    dlogr_OII, logflux_NeIII3869, dlogflux_NeIII3869, logflux_Hdel, dlogflux_Hdel, logflux_Hgam, dlogflux_Hgam, \
    logflux_OIII4364, dlogflux_OIII4364, logflux_HeII4687, dlogflux_HeII4687, \
    logflux_OIII5008, dlogflux_OIII5008 = load_lineratio(region=region, deredden=deredden)

    # Load cloudy result
    # Hden = np.linspace(-2, 2.6, 24, dtype='f2') # log # 0.25
    # metal = np.linspace(-1.5, 0.5, 11, dtype='f2')
    # alpha = np.linspace(-1.8, 0, 10, dtype='f2')
    output = format_cloudy_nogrid(filename=[Hden, alpha, metal],
                                  path='/Users/lzq/Dropbox/Data/CGM/cloudy/'
                                       + region + '_' + trial + '/')

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
    def log_prob(x, bnds, line_param):
        logden, alpha, logz = x[0], x[1], x[2]
        if logden < bnds[0, 0] or logden > bnds[0, 1]:
            return -np.inf
        elif alpha < bnds[1, 0] or alpha > bnds[1, 1]:
            return -np.inf
        elif logz < bnds[2, 0] or logz > bnds[2, 1]:
            return -np.inf
        else:
            chi2_NeV3346 = ((f_NeV3346((logden, alpha, logz)) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
            chi2_OII = ((f_OII((logden, alpha, logz)) - logflux_OII) / dlogflux_OII) ** 2
            chi2_NeIII3869 = ((f_NeIII3869((logden, alpha, logz)) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
            chi2_Hdel = ((f_Hdel((logden, alpha, logz)) - logflux_Hdel) / dlogflux_Hdel) ** 2
            chi2_Hgam = ((f_Hgam((logden, alpha, logz)) - logflux_Hgam) / dlogflux_Hgam) ** 2
            chi2_OIII4364 = ((f_OIII4364((logden, alpha, logz)) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
            chi2_HeII4687 = ((f_HeII4687((logden, alpha, logz)) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
            chi2_OIII5008 = ((f_OIII5008((logden, alpha, logz)) - logflux_OIII5008) / dlogflux_OIII5008) ** 2

            sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_NeIII3869, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
                                  chi2_HeII4687, chi2_OIII5008])
            return - 0.5 * np.nansum(sum_array[line_param[:, 1]])

        # if np.all(np.invert(np.array(bnds, dtype=bool))):
        #     chi2_NeV3346 = ((f_NeV3346((logden, alpha, logz)) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
        #     chi2_OII = ((f_OII((logden, alpha, logz)) - logflux_OII) / dlogflux_OII) ** 2
        #     chi2_NeIII3869 = ((f_NeIII3869((logden, alpha, logz)) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
        #     chi2_Hdel = ((f_Hdel((logden, alpha, logz)) - logflux_Hdel) / dlogflux_Hdel) ** 2
        #     chi2_Hgam = ((f_Hgam((logden, alpha, logz)) - logflux_Hgam) / dlogflux_Hgam) ** 2
        #     chi2_OIII4364 = ((f_OIII4364((logden, alpha, logz)) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
        #     chi2_HeII4687 = ((f_HeII4687((logden, alpha, logz)) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
        #     chi2_OIII5008 = ((f_OIII5008((logden, alpha, logz)) - logflux_OIII5008) / dlogflux_OIII5008) ** 2
        #
        #     sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_NeIII3869, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
        #                           chi2_HeII4687, chi2_OIII5008])
        #     return - 0.5 * np.nansum(sum_array[line_param[:, 1]])
        # else:
        #     mask_1 = bnds[:, 0] == None
        #     bnds[:, 0][mask_1] = -np.inf
        #     mask_2 = bnds[:, 1] == None
        #     bnds[:, 1][mask_2] = np.inf
        #     if logden < bnds[0, 0] or logden > bnds[0, 1]:
        #         return -np.inf
        #     elif alpha < bnds[1, 0] or alpha > bnds[1, 1]:
        #         return -np.inf
        #     elif logz < bnds[2, 0] or logz > bnds[2, 1]:
        #         return -np.inf
        #     else:
        #         chi2_NeV3346 = ((f_NeV3346((logden, alpha, logz)) - logflux_NeV3346) / dlogflux_NeV3346) ** 2
        #         chi2_OII = ((f_OII((logden, alpha, logz)) - logflux_OII) / dlogflux_OII) ** 2
        #         chi2_NeIII3869 = ((f_NeIII3869((logden, alpha, logz)) - logflux_NeIII3869) / dlogflux_NeIII3869) ** 2
        #         chi2_Hdel = ((f_Hdel((logden, alpha, logz)) - logflux_Hdel) / dlogflux_Hdel) ** 2
        #         chi2_Hgam = ((f_Hgam((logden, alpha, logz)) - logflux_Hgam) / dlogflux_Hgam) ** 2
        #         chi2_OIII4364 = ((f_OIII4364((logden, alpha, logz)) - logflux_OIII4364) / dlogflux_OIII4364) ** 2
        #         chi2_HeII4687 = ((f_HeII4687((logden, alpha, logz)) - logflux_HeII4687) / dlogflux_HeII4687) ** 2
        #         chi2_OIII5008 = ((f_OIII5008((logden, alpha, logz)) - logflux_OIII5008) / dlogflux_OIII5008) ** 2
        #
        #         sum_array = np.array([chi2_NeV3346, chi2_OII, chi2_NeIII3869, chi2_Hdel, chi2_Hgam, chi2_OIII4364,
        #                               chi2_HeII4687, chi2_OIII5008])
        #         return - 0.5 * np.nansum(sum_array[line_param[:, 1]])


    ndim, nwalkers = 3, 40
    p0 = np.array([1.8, -1.5, -0.3]) + 0.1 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(bnds, line_param))
    state = sampler.run_mcmc(p0, 5000)
    samples = sampler.get_chain(flat=True, discard=1000)

    figure = corner.corner(samples, labels=[r"$\mathrm{log_{10}(n)}$", r"$\mathrm{\alpha}$",
                                            r"$\mathrm{log_{10}(Z/Z_{\odot})}$"],
                           quantiles=[0.16, 0.5, 0.84], show_titles=True, color='k', title_kwargs={"fontsize": 13},
                           smooth=1., smooth1d=1., bins=25)

    for i, ax in enumerate(figure.get_axes()):
        if not np.isin(i, np.arange(0, ndim ** 2, ndim + 1)):
            ax.tick_params(axis='both', direction='in', top='on', bottom='on', left='on', right='on')
        ax.tick_params(axis='both', direction='in', top='on', bottom='on')
    if deredden:
        figure.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial +
                       '_dered.pdf', bbox_inches='tight')
    else:
        figure.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial +
                       '.pdf', bbox_inches='tight')

    # Violin plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 3]}, dpi=300)

    E_NeV3346, E_NeIII3869 = 97.11, 40.96  # in ev
    E_OIII4364, E_OIII5008, E_OII = 35.12, 35.12, 13.6
    E_HeII4687, E_Hbeta = 54.42, 13.6

    line_param_violin = np.array([line_param[1, 1], line_param[5, 1], line_param[7, 1],
                                  line_param[2, 1], line_param[6, 1], line_param[0, 1]])
    data_x1 = [0, 5]
    data_x2 = [E_OII / E_Hbeta, E_OIII4364 / E_Hbeta - 0.5, E_OIII5008 / E_Hbeta - 0.5, E_NeIII3869 / E_Hbeta + 0.2,
              E_HeII4687 / E_Hbeta, E_NeV3346 / E_Hbeta - 2]
    data_y2 = np.array([logflux_OII, logflux_OIII4364, logflux_OIII5008,
                        logflux_NeIII3869, logflux_HeII4687, logflux_NeV3346], dtype=object).reshape(len(data_x2))
    data_y2err = np.array([dlogflux_OII, dlogflux_OIII4364, dlogflux_OIII5008,
                           dlogflux_NeIII3869, dlogflux_HeII4687, dlogflux_NeV3346], dtype=object).reshape(len(data_x2))

    # Mask
    data_y2 = np.where(line_param_violin, data_y2, np.nan)
    data_y2err = np.where(line_param_violin, data_y2err, 0)

    draw = np.random.choice(len(samples), size=500, replace=False)
    samples_draw = samples[draw]
    model = (samples_draw[:, 0], samples_draw[:, 1], samples_draw[:, 2])

    model_y2 = np.array([f_OII(model), f_OIII4364(model), f_OIII5008(model), f_NeIII3869(model),
                         f_HeII4687(model), f_NeV3346(model)])

    if line_param[5, 1]:
        data_y1 = np.array([logr_OII, logflux_OIII4364 - logflux_OIII5008], dtype=object).reshape(len(data_x1))
        data_y1err = np.array([dlogr_OII, np.sqrt(dlogflux_OIII4364 ** 2
                                                  + dlogflux_OIII5008 ** 2)], dtype=object).reshape(len(data_x1))
        model_y1 = [f_OII3730(model) - f_OII3727(model), f_OIII4364(model) - f_OIII5008(model)]
    else:
        data_y1 = np.array([logr_OII, np.nan], dtype=object).reshape(len(data_x1))
        data_y1err = np.array([dlogr_OII, np.nan], dtype=object).reshape(len(data_x1))
        model_y1 = [f_OII3730(model) - f_OII3727(model), np.nan * np.zeros(len(f_OII3727(model)))]

    model_y2 = np.where(line_param_violin[:, np.newaxis], model_y2, np.nan)
    model_y2 = [np.array(model_y2[0, :]), np.array(model_y2[1, :]), np.array(model_y2[2, :]),
                np.array(model_y2[3, :]), np.array(model_y2[4, :]), np.array(model_y2[5, :])]
    ax[0].errorbar(data_x1, data_y1, data_y1err, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
    ax[1].errorbar(data_x2, data_y2, data_y2err, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
    violin_parts_0 = ax[0].violinplot(model_y1, positions=data_x1, points=500, widths=0.5, showmeans=False,
                                      showextrema=False, showmedians=False)
    violin_parts_1 = ax[1].violinplot(model_y2, positions=data_x2, points=500, widths=0.5, showmeans=False,
                                      showextrema=False, showmedians=False)
    for pc in violin_parts_0['bodies']:
        pc.set_facecolor('C1')
    for pc in violin_parts_1['bodies']:
        pc.set_facecolor('C1')
    ax[1].set_title(r'$\mathrm{Ionization \, energy}$', size=20, y=1.04)
    ax[1].annotate("", xy=(0.7, 0.90), xytext=(0.47, 0.90), xycoords='figure fraction',
                   arrowprops=dict(arrowstyle="->"))
    ax[0].set_ylabel(r'$\mathrm{log[line \, ratio]}$', size=20)
    ax[1].set_xticks(data_x2, [r'$\mathrm{\frac{[O \, II]}{H\beta}}$', r'$\mathrm{}$',
                              r'$\mathrm{\frac{[O \, III]}{H\beta}}$', r'$\mathrm{\frac{[Ne \, III]}{H\beta}}$',
                              r'$\mathrm{\frac{He \, II}{H\beta}}$', r'$\mathrm{\frac{[Ne \, V]}{H\beta}}$'])
    ax[1].annotate(r'$\mathrm{\lambda 4363}$', xy=(0.40, 0.2), xycoords='figure fraction', size=15)
    ax[1].annotate(r'$\mathrm{\lambda 5007}$', xy=(0.40, 0.8), xycoords='figure fraction', size=15)
    ax[0].set_xticks([1, 4], [r'$\mathrm{[O \, II]}$' + '\n' + r'$\mathrm{\frac{\lambda 3729}{\lambda 3727}}$',
                              r'$\mathrm{[O \, III]}$' + '\n' + r'$\mathrm{\frac{\lambda 4363}{\lambda 5007}}$'], y=0.9)
    ax[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=15,
                      size=5)
    ax[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
    ax[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20,
                      size=5)
    ax[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
    if deredden:
        fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + '_violin_dered.png',
                    bbox_inches='tight')
    else:
        fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_MCMC/' + region + '_' + trial + '_violin.png',
                    bbox_inches='tight')


# # S3
# S3_bnds = np.array([[-2, 3.0],
#                     [-2, 0],
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
#                     [-2, 0],
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
S5_bnds = np.array([[-2, 6.6],
                    [-2, 0],
                    [-1.5, 0.5]])
S5_param = np.array([['NeV3346', False],
                     ['OII', True],
                     ['NeIII3869', True],
                     ['Hdel', False],
                     ['Hgam', False],
                     ['OIII4364', False],
                     ['HeII4687', False],
                     ['OIII5008', True]], dtype=bool)
Hden_ext = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 6.6, 20, dtype='f2')))
RunCloudyMCMC(region='S5', trial='t1', bnds=S5_bnds, line_param=S5_param, deredden=False,
              Hden=Hden_ext)
# RunCloudyMCMC(region='S5', trial='t2', bnds=S5_bnds, line_param=S5_param, deredden=False)

# S6
# S6_bnds = np.array([[-2, 3.4],
#                     [-2, 0],
#                     [-1.5, 0.5]])
# S6_param = np.array([['NeV3346', True],
#                      ['OII', True],
#                      ['NeIII3869', True],
#                      ['Hdel', True],
#                      ['Hgam', True],
#                      ['OIII4364', True],
#                      ['HeII4687', True],
#                      ['OIII5008', True]], dtype=bool)
# RunCloudyMCMC(region='S6', trial='t1', bnds=S6_bnds, line_param=S6_param, deredden=False,
#               Hden=Hden_ext)
# RunCloudyMCMC(region='S6', trial='t2', bnds=S6_bnds, line_param=S6_param, deredden=False)

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
# RunCloudyMCMC(region='S7', trial='t1', bnds=S7_bnds, line_param=S7_param, deredden=False, Hden=Hden_ext)

# S8
# S8_bnds = np.array([[-2, 4.6],
#                     [-1.2, 0],
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
# RunCloudyMCMC(region='S8', trial='t1', bnds=S8_bnds, line_param=S8_param, deredden=False, Hden=Hden_ext)


# S9
# S9_bnds = np.array([[-2, 4.6],
#                     [-1.2, 0],
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
# RunCloudyMCMC(region='S9', trial='t1', bnds=S9_bnds, line_param=S9_param, deredden=False, Hden=Hden_ext)


# # S10
# S10_bnds = np.array([[-2, 2.6],
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
# RunCloudyMCMC(region='S10', trial='t1', bnds=S10_bnds, line_param=S10_param, deredden=False)
#
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
# RunCloudyMCMC(region='B1', trial='t1', bnds=B1_bnds, line_param=B1_param, deredden=False)

# # B2
# B2_bnds = np.array([[-2, 2.6],
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
# RunCloudyMCMC(region='B2', trial='t1', bnds=B2_bnds, line_param=B2_param, deredden=False)
#
# # B3
# B3_bnds = np.array([[-2, 2.6],
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
# RunCloudyMCMC(region='B3', trial='t1', bnds=B3_bnds, line_param=B3_param, deredden=False)