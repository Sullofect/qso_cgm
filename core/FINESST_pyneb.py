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

#
# [O III] vs temperature
fig, axarr = plt.subplots(2, 1, figsize=(5, 10), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.)
tem_array = 10 ** np.linspace(3.7, 4.5, 1000)
den_array = np.array([0.1, 1, 10, 100])
OIII4363 = O3.getEmissivity(tem=tem_array, den=den_array, wave=4363)
OIII5007 = O3.getEmissivity(tem=tem_array, den=den_array, wave=5007)
for i in range(len(den_array)):
    axarr[0].plot(np.log10(tem_array), np.log10(OIII4363/OIII5007)[:, i],
                  '-', label=r'$\rm{log}(n_{e}/\rm{cm^{-3}})=$ ' + str(np.log10(den_array[i])))
axarr[0].set_xlabel(r"$\mathrm{log(Temperature / K)}$", size=25)
axarr[0].set_ylabel(r'$\mathrm{log( [O \, III] \lambda 4363 / \lambda 5007)}$', size=25)

# T vs metal
for i in range(len(den_array)):
    O2_abund = O2.getIonAbundance(int_ratio=500, tem=tem_array,
                                  den=np.ones_like(tem_array) * den_array[i], to_eval='L(3726)+L(3729)',
                                  Hbeta=100.0)
    O3_abund = O3.getIonAbundance(int_ratio=500, tem=tem_array,
                                  den=np.ones_like(tem_array) * den_array[i], to_eval='L(5007)+L(4959)',
                                  Hbeta=100.0)
    abun = 12 + np.log10(O3_abund + O2_abund)
    axarr[1].plot(np.log10(tem_array), abun, '-')
axarr[1].set_xlabel(r"$\mathrm{log(Temperature / K)}$", size=25)
axarr[1].set_ylabel(r'$\rm 12 + log(O/H)$', size=25)
axarr[0].minorticks_on()
axarr[1].minorticks_on()
axarr[0].legend(prop={'size': 15}, framealpha=0, loc=4, fontsize=15)
axarr[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
axarr[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
axarr[1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
axarr[1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Proposal/FINESST/pyneb_test_OIII.png', bbox_inches='tight')


# [O III] vs temperature
fig, axarr = plt.subplots(1, 1, figsize=(5, 5), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.)
tem_array = 10 ** np.linspace(3.7, 4.5, 1000)
den_array = np.array([0.1, 1, 10, 100])
OIII4363 = O3.getEmissivity(tem=tem_array, den=den_array, wave=4363)
OIII5007 = O3.getEmissivity(tem=tem_array, den=den_array, wave=5007)
for i in range(len(den_array)):
    axarr.plot(np.log10(tem_array), np.log10(OIII4363/OIII5007)[:, i],
                  '-', label=r'$\rm{log}(n_{e}/\rm{cm^{-3}})=$ ' + str(np.log10(den_array[i])))
axarr.set_xlabel(r"$\mathrm{log(Temperature / K)}$", size=25)
axarr.set_ylabel(r'$\mathrm{log( [O \, III] \lambda 4363 / \lambda 5007)}$', size=25)
axarr.minorticks_on()
axarr.legend(prop={'size': 15}, framealpha=0, loc=4, fontsize=15)
axarr.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
axarr.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Proposal/FINESST/pyneb_OIII.png', bbox_inches='tight')
