import os
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import stats
from astropy.io import ascii
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.wcs import WCS
import time as tm
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

c_kms = 2.998e5
wave_Halpha_vac = 6564.61
wave_Hbeta_vac = 4862.721
wave_OIII4960_vac = 4960.30
wave_OIII5008_vac = 5008.239
wave_NII6549_vac = 6549.86
wave_NII6585_vac = 6585.27
wave_SII6718_vac = 6718.29
wave_SII6732_vac = 6732.67

line_list = np.array([wave_Halpha_vac, wave_Hbeta_vac, wave_NII6549_vac, wave_NII6585_vac,
                      wave_OIII4960_vac, wave_OIII5008_vac, wave_SII6718_vac, wave_SII6732_vac])
cubename = 'Q0107-0235'
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
print('wavelength in um', line_list * (1 + z_qso) / 1e4)

# Unit is in 10^-17
OII_flux = 1.0
lines_flux = OII_flux * np.asarray([1.0, 0.3, 0.16, 0.5, 0.8, 2.33, 0.5, 0.5])
print('OII flux', OII_flux)
print('line fluxes are', lines_flux)


#
data = np.loadtxt('../../Proposal/HST+JWST/mktrans_zm_10_10.dat.txt')

plt.figure(figsize=(10, 10), dpi=300)
plt.plot(data[::2, 0], data[::2, 1], '-k', lw=1)
plt.vlines(line_list * (1 + z_qso) / 1e4, ymin=0, ymax=1, color='r', linestyle='--', lw=1)
plt.savefig('../../Proposal/HST+JWST/transmission.png')

# write a line do vertical lines

