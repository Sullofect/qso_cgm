import os
import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import integrate
from scipy.stats import norm
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import minimize
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

SM = fits.getdata(path_savetab + 'gal_list_StellarMass.fits')
row = SM['row']
M_8_50 = SM['M_8_50']
M_9_50 = SM['M_9_50']
M_10_50 = SM['M_10_50']

resi_8_10 = M_8_50 - M_10_50
resi_9_10 = M_9_50 - M_10_50

resi_8_10 = resi_8_10[~np.isnan(resi_8_10)]
resi_9_10 = resi_9_10[~np.isnan(resi_9_10)]


plt.figure(figsize=(8, 5), dpi=300)
plt.hist(resi_8_10, bins=np.linspace(-0.5, 0.5, 21), facecolor='red', histtype='stepfilled', alpha=0.5, label=r'T8-T10')
plt.hist(resi_9_10, bins=np.linspace(-0.5, 0.5, 21), facecolor='blue', histtype='stepfilled', alpha=0.5, label=r'T9-T10')
# plt.xlim(-2000, 2000)
# plt.ylim(0, 12)
plt.minorticks_on()
plt.xlabel(r'$\Delta M_{ste}$', size=20)
plt.ylabel(r'$\mathrm{Numbers}$', size=20)
plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                labelsize=20)
plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
plt.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
plt.savefig(path_savefig + 'SM_residue', bbox_inches='tight')