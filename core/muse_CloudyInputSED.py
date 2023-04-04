import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
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


#* np.exp(- nu / nu_high) * np.exp(- nu_low / nu)
data = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy', 'CheckContinuum', 'continuum.txt')
CheckContinuum = np.loadtxt(data, usecols=[0, 1, 2])
data_AGN = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy',
                        'CheckContinuum', 'continuum_AGN.txt')
CheckContinuum_AGN = np.loadtxt(data_AGN, usecols=[0, 1, 2])


hv = np.linspace(hv_low, hv_high, 1000)
hv_AGN = np.logspace(-5, 100, 1000)
nu = hv * 13.6 / 4.1357e-15
nu_AGN = hv_AGN * 13.6 / 4.1357e-15
fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
ax.plot(hv, nu * f(nu, -0.8), lw=0.2, label=r'$\alpha = -0.8$')
ax.plot(hv, nu * f(nu, -1), lw=0.2, label=r'$\alpha = -1$')
ax.plot(hv, nu * f(nu, -1.2), lw=0.2, label=r'$\alpha = -1.2$')
ax.plot(hv_AGN, nu_AGN * f_AGN(nu_AGN, -1.2, -0.5, -1, 10 ** 5.25), lw=0.2, label=r'AGN')
ax.plot(CheckContinuum[::5, 0], CheckContinuum[::5, 1], '-', lw=0.2,  label=r'Cloudy $\alpha = -1.4$')
ax.plot(CheckContinuum_AGN[::5, 0], CheckContinuum_AGN[::5, 1], '-', lw=0.1, label=r'Cloudy AGN')
ax.set_xlim(1e-3, 1e6)
ax.set_ylim(1e45, 5e47)
ax.set_xlabel(r'$h\nu (\mathrm{Ryd})$', size=15)
ax.set_ylabel(r'$\nu f_{\nu}(\mathrm{arbitrary \, unit})$', size=15)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=10, size=5)
ax.tick_params(axis='x', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_powerlaw.png', bbox_inches='tight')