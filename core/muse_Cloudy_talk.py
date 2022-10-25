import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

hv_low, hv_high = 0.37, 73.5
def f(v, alpha):
    return v ** alpha
#* np.exp(- nu / nu_high) * np.exp(- nu_low / nu)
hv = np.linspace(hv_low, hv_high, 100)
nu = hv * 13.6 / 4.1357e-15
fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
ax.plot(hv, 1e2 * nu * f(nu, -0.8), label=r'$\alpha = -0.8$')
ax.plot(hv, 1e4 * nu * f(nu, -1), label=r'$\alpha = -1$')
ax.plot(hv, 1e6 * nu * f(nu, -1.2), label=r'$\alpha = -1.2$')
ax.set_xlim(1e-3, 1e6)
ax.set_ylim(1, 1e8)
ax.set_xlabel(r'$h\nu (\mathrm{Ryd})$', size=15)
ax.set_ylabel(r'$\nu f_{\nu}(\mathrm{arbitrary \, unit})$', size=15)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=10, size=5)
ax.tick_params(axis='x', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_powerlaw.png', bbox_inches='tight')