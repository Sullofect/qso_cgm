#!/usr/bin/env python
import os
import webbpsf
import time as tm
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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# JWST PSF
# os.env('export WEBBPSF_PATH="/Users/lzq/Dropbox/Zhuoqi Liu/Proposal/data/webbpsf-data:$WEBBPSF_PATH"')
# ns = webbpsf.NIRSpec()
# plt.figure(figsize=(8, 12))
# ns = webbpsf.NIRSpec()
# ns.image_mask='MSA all open'
# ns.display()
# plt.savefig('../../Proposal/HST+JWST/example_nirspec_msa_optics.png')
# raise ValueError('Stop here')

# HE1003+0149 m_z = 16.49, m_i = 16.56

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
OII_flux = 1.1
lines_flux = OII_flux * np.asarray([1.0, 0.3, 0.16, 0.5, 0.8, 2.33, 0.5, 0.5])
print('OII flux', OII_flux)
print('line fluxes are', lines_flux)


#
data = np.loadtxt('../../Proposal/HST+JWST/mktrans_zm_10_10.dat.txt')
data_sky = np.loadtxt('../../Proposal/HST+JWST/OH_JHK_band.dat')

plt.figure(figsize=(10, 10), dpi=300)
plt.plot(data[::40, 0], data[::40, 1], '-k', lw=1)
plt.plot(data_sky[:, 0], data_sky[:, 1] / np.max(data_sky[:, 1]), '-b', lw=1)
plt.vlines(line_list * (1 + z_qso) / 1e4, ymin=0, ymax=1.05, color='r', linestyle='--', lw=1)
plt.xlim(0.8, 2.0)
plt.ylim(0, 1.05)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Transmission')
plt.savefig('../../Proposal/HST+JWST/transmission_{}.png'.format(cubename))
