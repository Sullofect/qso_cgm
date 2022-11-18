import pyneb as pn
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from muse_LoadLineRatio import load_lineratio

# [OII] 3726/3729" "[OIII] 4363/5007" "
O2 = pn.Atom('O', 2)
O3 = pn.Atom('O', 3)


logflux_Hbeta, dlogflux_Hbeta, logflux_NeV3346, dlogflux_NeV3346, logflux_OII, dlogflux_OII, logr_OII, \
dlogr_OII, logflux_NeIII3869, dlogflux_NeIII3869, logflux_Hdel, dlogflux_Hdel, logflux_Hgam, dlogflux_Hgam, \
logflux_OIII4364, dlogflux_OIII4364, logflux_HeII4687, dlogflux_HeII4687, \
logflux_OIII5008, dlogflux_OIII5008 = load_lineratio(region='all')



den_array = 10 ** np.linspace(-1, 6, 100)
OII3727 = O2.getEmissivity(tem=20000, den=den_array, wave=3727)
OII3729 = O2.getEmissivity(tem=20000, den=den_array, wave=3729)
r_OII_pyneb = OII3729 / OII3727

f_r_OII = interpolate.interp1d(np.log10(r_OII_pyneb), np.log10(den_array), bounds_error=False)

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
ax.plot(np.log10(den_array), np.log10(r_OII_pyneb), '-k')
# ax.plot(f_r_OII(np.log10(r_OII_pyneb)), np.log10(r_OII_pyneb), '.r', ms=5)
ax.plot(f_r_OII(logr_OII), logr_OII, '.r', ms=5)
ax.plot(f_r_OII(logr_OII - 3 * dlogr_OII), logr_OII - 3 * dlogr_OII, '.b', ms=5)

print(logr_OII)
print(dlogr_OII)
print(f_r_OII(logr_OII))
print(f_r_OII(logr_OII - 3 * dlogr_OII))
ax.set_xlabel(r"$\mathrm{log_{10}[Hydrogen \, density]}$", size=15)
ax.set_ylabel(r'$\mathrm{log( [O \, II] \lambda 3729 / \lambda 3727)}$', size=15)
ax.legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/RegionPyneb_OII.png', bbox_inches='tight')