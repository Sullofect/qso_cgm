import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate, interpolate
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('lines', lw=2)
rc('xtick', top=True, direction='in', labelsize=12)
rc('ytick', right=True, direction='in', labelsize=12)
rc('xtick.major', size=5)
rc('ytick.major', size=5)
rc('xtick.minor', size=3, visible=True)
rc('ytick.minor', size=3, visible=True)

fig, ax = plt.subplots(4, 1, figsize=(4, 16), dpi=300, sharex=True)
ax = ax.ravel()
fig.subplots_adjust(hspace=0., wspace=0.25)
for i, i_val in enumerate([1.3]):
    # color = 'C' + str(i)
    color='k'
    label = r'$\rm log(n_H)=$ ' + str(i_val)
    path_S5_con = '../../Data/CGM/cloudy/ComputeLogU/S5_distance_{}.con'.format(i_val)
    S5_con = np.loadtxt(path_S5_con, usecols=[0, 1, 2, 3, 4])

    path_S5_hyd = '../../Data/CGM/cloudy/ComputeLogU/S5_distance_{}.hyd'.format(i_val)
    S5_hyd = np.loadtxt(path_S5_hyd)
    T_e, Hden, HI_H = S5_hyd[:, 1], S5_hyd[:, 2], S5_hyd[:, 4]

    path_S5_emi = '../../Data/CGM/cloudy/ComputeLogU/S5_distance_{}.emi'.format(i_val)
    LineEmis = np.loadtxt(path_S5_emi)
    depth, NeV = LineEmis[:, 0], LineEmis[:, 1]
    OII = LineEmis[:, 2] + LineEmis[:, 3]
    Hbeta, OIII = LineEmis[:, 11], LineEmis[:, 12]

    path_S5_oxy = '../../Data/CGM/cloudy/ComputeLogU/S5_distance_{}.oxy_2'.format(i_val)
    S5_oxy = np.loadtxt(path_S5_oxy)
    abun = S5_oxy

    N_HI = integrate.cumulative_trapezoid(y=Hden * HI_H, x=depth, initial=0)
    tau_912 = N_HI * 6.30e-18  # X.Prochaska 2017
    depth /= 3.086e+21  # change unit

    ax[0].plot(depth, OIII / OIII.max(), '-b', lw=2)
    ax[0].plot(depth, OII / OII.max(), '--r', lw=2)
    ax[0].plot(depth, Hbeta / Hbeta.max(), '-k', lw=2)
    # ax[0].plot(depth, NeV / NeV.max(), ':', lw=2, c=color)
    # ax[0, 0].plot(depth, OIII / Hbeta, '-k', lw=2)

    if i == 0:
        ax[0].plot([], [], '-b', label='[O III]')
        ax[0].plot([], [], '--r', label='[O II]')
        ax[0].plot([], [], '-k', label='Hbeta')
        # ax[0, 0].plot([], [], ':k', label='NeV')

    ax[0].set_ylabel(r'Cumulative flux', size=15)
    ax[0].set_xlim(0.1, 5)
    # ax[0, 0].set_ylim(1, 100)
    # ax[0].set_xscale('log')
    # ax[0, 0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(depth, T_e, '-', c=color, label=label)
    ax[1].set_ylabel(r'Electron  temperature [K]', size=15)
    ax[1].set_yscale('log')


    f = interpolate.interp1d(tau_912, depth)
    depth_tau_1 = f(1)
    f_OIII_Hbeta = interpolate.interp1d(depth, OIII / Hbeta)
    print('depth is {}'.format(f(1)))
    print('ratio at tau = 1 is ', f_OIII_Hbeta(depth_tau_1))
    # ax[2].axvline(x=depth_tau_1, ymin=tau_912.min(), ymax=tau_912.max())
    # ax[2].axhline(y=1, xmin=0, xmax=100)
    ax[2].plot(depth, tau_912, '-', c=color, label=label)
    ax[2].set_xlabel(r'Depth into the cloud [kpc]', size=15)
    ax[2].set_ylabel(r'Optical depth at 912 $\rm \AA (\tau_{912})$', size=15)
    ax[2].set_yscale('log')

    # ax[3].plot(depth, abun, '-', c=color, label=label)
    ax[3].plot(depth, abun[:, 2], '-', c='k', label=r'$\rm O^+$', zorder=100)
    ax[3].plot(depth, abun[:, 3], '--', c='b', label=r'$\rm O^{2+}$')
    ax[3].plot(depth, abun[:, 8] + abun[:, 9], '-', c='red', label=r'$\rm O^{7+} + O^{8+}$')
    ax[3].set_xlabel(r'Depth into the cloud [kpc]', size=15)
    ax[3].set_ylabel(r'Ion fraction', size=15)
    ax[3].legend()
    # fig.tight_layout()
    fig.savefig('../../Data/CGM_plots/CheckLineratio.png', bbox_inches='tight')



# fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
# ax.plot(S5_con[::5, 0], S5_con[::5, 1], '-', lw=0.2,  label=r'incident')
# ax.plot(S5_con[::5, 0], S5_con[::5, 2], '-b', lw=0.2,  label=r'transmitted')
# # ax.plot(S5_con[::5, 0], S5_con[::5, 3], '-r', lw=0.2,  label=r'incident')
# ax.set_xlim(1e-3, 1e6)
# ax.set_ylim(1e45, 5e49)
# ax.set_xlabel(r'$h\nu (\mathrm{Ryd})$', size=15)
# ax.set_ylabel(r'$\nu f_{\nu}(\mathrm{arbitrary \, unit})$', size=15)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=10, size=5)
# ax.tick_params(axis='x', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/CheckTranmission.png', bbox_inches='tight')

