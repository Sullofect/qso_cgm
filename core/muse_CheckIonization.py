import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate
from muse_LoadCloudy import load_cloudy_nogrid
from astropy.table import Table
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

h = 6.6261e-27
ev = 1.60218e-12
k = 1.3807e-16
hv_low, hv_high = 0.37, 73.5
nuL_nu = 46.54
nu_912 = 13.6 / 4.1357e-15

# formula
def f(nu, alpha):
    norm = 10 ** nuL_nu / nu_912 / (nu_912 ** alpha)
    return norm * nu ** alpha


def f_AGN(nu, alpha_ox, alpha_uv, alpha_x, T_BB):
    kT_BB = k * T_BB
    kT_IR = 0.01 * 13.6 * ev
    hnu = h * nu

    # Norm 1
    nu_2500 = 3e10 / 2500 / 1e-8
    A_2500 = nu_2500 ** alpha_uv * np.exp(- h * nu_2500 / kT_BB) * np.exp(-kT_IR / (h * nu_2500))
    norm = A_2500 * 403.3 ** alpha_ox

    #
    nu_2kev = 2e3 * ev / h
    C = norm / nu_2kev ** alpha_x
    # C = norm / (nu_2kev ** alpha_uv * np.exp(- h * nu_2kev / kT_BB) * np.exp(-kT_IR / (h * nu_2kev))
    #             + nu_2kev ** alpha_x)

    # Norm 2
    norm = 10 ** nuL_nu / nu_912 / \
           (nu_912 ** alpha_uv * np.exp(-13.6 * ev / kT_BB) * np.exp(-kT_IR / (13.6 * ev)))
    # + C * nu_912 ** alpha_x
    term1 = norm * nu ** alpha_uv * np.exp(- hnu / kT_BB) * np.exp(- kT_IR / hnu)
    term2 = norm * C * nu ** alpha_x
    term2 = np.where((nu < 100e3 * ev / h) * (nu > 136 * ev / h), term2, 0)
    total = term1 + term2
    return total



for i, i_val in enumerate([-1, 0, 1]):
    path_S5_con = '/Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU/S5_distance_{}.con'.format(i_val)
    S5_con = np.loadtxt(path_S5_con, usecols=[0, 1, 2, 3, 4])

    path_S5_hyd = '/Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU/S5_distance_{}.hyd'.format(i_val)
    S5_hyd = np.loadtxt(path_S5_hyd)

    path_S5_emi = '/Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU/S5_distance_{}.emi'.format(i_val)
    LineEmis = np.loadtxt(path_S5_emi)
    distance, NeV = LineEmis[:, 0], LineEmis[:, 1]
    OII = LineEmis[:, 2] + LineEmis[:, 3]
    Hbeta, OIII = LineEmis[:, 11], LineEmis[:, 12]
# Sum_Hbeta = integrate.simpson(LineEmis[:, 1], LineEmis[:, 0])
# Sum_OII = integrate.simpson(LineEmis[:, 4], LineEmis[:, 0])
# Sum_OIII = integrate.simpson(LineEmis[:, 2], LineEmis[:, 0])
# print(LineEmis[:, 2][:50])
# print('EMISS=', Sum_OII / Sum_Hbeta)
# print('EMISS=', Sum_OIII / Sum_Hbeta)


fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
ax.plot(S5_con[::5, 0], S5_con[::5, 1], '-', lw=0.2,  label=r'incident')
ax.plot(S5_con[::5, 0], S5_con[::5, 2], '-b', lw=0.2,  label=r'transmitted')
# ax.plot(S5_con[::5, 0], S5_con[::5, 3], '-r', lw=0.2,  label=r'incident')
ax.set_xlim(1e-3, 1e6)
ax.set_ylim(1e45, 5e49)
ax.set_xlabel(r'$h\nu (\mathrm{Ryd})$', size=15)
ax.set_ylabel(r'$\nu f_{\nu}(\mathrm{arbitrary \, unit})$', size=15)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=10, size=5)
ax.tick_params(axis='x', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/CheckTranmission.png', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
# ax.plot(distance / 3.086e+21, np.log10(OIII / Hbeta), '-', lw=0.2,  label='[O III] / Hbeta')
ax.plot(distance / 3.086e+21, OIII / OIII.max(), '-', lw=0.2,  label='[O III]')
ax.plot(distance / 3.086e+21, OII / OII.max(), '-b', lw=0.2,  label='[O II]')
ax.plot(distance / 3.086e+21, Hbeta / Hbeta.max(), '-r', lw=0.2,  label='Hbeta')
ax.plot(distance / 3.086e+21, NeV / NeV.max(), '--k', lw=0.2,  label='NeV')
# ax.plot(S5_con[::5, 0], S5_con[::5, 3], '-r', lw=0.2,  label=r'incident')
ax.set_xlim(1, 20)
# ax.set_ylim(-27.5, -20)
ax.set_xlabel(r'distance [kpc]', size=15)
ax.set_ylabel(r'log(line ratio)', size=15)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=10, size=5)
ax.tick_params(axis='x', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.set_xscale('log')
# ax.set_yscale('log')
ax.legend()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/CheckLineratio.png', bbox_inches='tight')
