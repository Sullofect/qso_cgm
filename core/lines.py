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
cubename = 'PKS0405-123'
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
print('wavelength in um', line_list * (1 + z_qso) / 1e4)

