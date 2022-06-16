import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import bagpipes as pipes
from muse_compare_z import compare_z
import muse_kc
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/lzq/Dropbox/pyobs/')
from kcorrect import apptoabs
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)



Westra2010_r_5Gyr_SSP = Table.read('/Users/lzq/Dropbox/pyobs/data/kcorrect/kcorrections_r_5Gyr_Westra2010.dat', format='ascii')

Westra2010_r_5Myr_SSP = Table.read('/Users/lzq/Dropbox/pyobs/data/kcorrect/kcorrections_r_5Myr_Westra2010.dat', format='ascii')

print(Westra2010_r_5Gyr_SSP)
print(Westra2010_r_5Myr_SSP)


redshifts = Table()
redshifts['redshift'] = np.arange(0.0, 1.1, 0.01)
redshifts['kcorrection_5Gyr'] = -99.9
redshifts['kcorrection_5Myr'] = -99.9

for redshift in redshifts:
   redshift['kcorrection_5Gyr'] = muse_kc.KC(model='ssp_5Gyr_Z02', filter_o='SDSS_r.dat',
                                                   filter_e='SDSS_r.dat', z=redshift['redshift'])
   redshift['kcorrection_5Myr'] = muse_kc.KC(model='ssp_5Myr_Z02', filter_o='SDSS_r.dat',
                                                   filter_e='SDSS_r.dat', z=redshift['redshift'])

print(redshifts)
# k_array = np.zeros(len( Westra2010_r_5Gyr_SSP['redshift']))
# for i, z_i in enumerate(Westra2010_r_5Gyr_SSP['redshift']):
#    k_array[i] = muse_kc.KC(model='ssp_5Gyr_Z02', filter_o='SDSS_r.dat',
#                                               filter_e='SDSS_r.dat', z=z_i)
#
# print(k_array / Westra2010_r_5Gyr_SSP['kcorrection_r'])
plt.figure(figsize=(8, 8), dpi=300)

plt.plot(Westra2010_r_5Gyr_SSP['redshift'], Westra2010_r_5Gyr_SSP['kcorrection_r'],
         '.', color='k', mec='black', ms=15,
         label=r'$\rm Westra\ 2010\ 5\ Gyr\ SSP$')
plt.plot(redshifts['redshift'], redshifts['kcorrection_5Gyr'],
          color='black', linestyle='-', label=r'$\rm pyobs\ kcorrect\ 5\ Gyr\ SSP$')

plt.plot(Westra2010_r_5Myr_SSP['redshift'], Westra2010_r_5Myr_SSP['kcorrection_r'],
         '.', color='red', mec='black', ms=15,
         label=r'$\rm Westra\ 2010\ 5\ Myr\ SSP$')
plt.plot(redshifts['redshift'], redshifts['kcorrection_5Myr'],
         color='black', linestyle='-', label=r'$\rm pyobs\ kcorrect\ 5\ Myr\ SSP$')

plt.xlabel(r'$\rm redshift$')
plt.ylabel(r'$\rm k\ correction\ r$')
plt.tick_params(axis='both', which='major', direction='in', top='on', size=5, labelsize=20)
plt.tick_params(axis='both', which='minor', direction='in', top='on', size=3)
plt.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
plt.savefig('/Users/lzq/Dropbox/data/cgm_plots/kcorrection_check.png')