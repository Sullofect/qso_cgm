import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

# path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits_SUBTRACTED.fits')
path_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
path_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_Hbeta_line_offset.fits')
path_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_4960_line_offset.fits')
path_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OIII_5008_line_offset.fits')

cube_OII = Cube(path_OII)
cube_Hbeta = Cube(path_Hbeta)
cube_OIII4960 = Cube(path_OIII4960)
cube_OIII5008 = Cube(path_OIII5008)
wave_OII = pyasl.airtovac2(cube_OII.wave.coord())
wave_Hbeta = pyasl.airtovac2(cube_Hbeta.wave.coord())
wave_OIII4960 = pyasl.airtovac2(cube_OIII4960.wave.coord())
wave_OIII5008 = pyasl.airtovac2(cube_OIII5008.wave.coord())
wave_stack = np.hstack((wave_OII, wave_Hbeta, wave_OIII4960, wave_OIII5008))

flux_OII, flux_Hbeta = cube_OII.data * 1e-3, cube_Hbeta.data * 1e-3
flux_OIII4960, flux_OIII5008 = cube_OIII4960.data * 1e-3, cube_OIII5008.data * 1e-3
flux_OII_err, flux_Hbeta_err = np.sqrt(cube_OII.var) * 1e-3, np.sqrt(cube_Hbeta.var) * 1e-3
flux_OIII4960_err = np.sqrt(cube_OIII4960.var) * 1e-3
flux_OIII5008_err = np.sqrt(cube_OIII5008.var) * 1e-3

# flux_all = np.vstack((flux_OII, flux_Hbeta, flux_OIII4960, flux_OIII5008))
# flux_err_all = np.vstack((flux_OII_err, flux_Hbeta_err, flux_OIII4960_err, flux_OIII5008_err))

# Direct integration
line_OII = integrate.simps(flux_OII, axis=0)
line_Hbeta =  integrate.simps(flux_Hbeta, axis=0)
line_OIII4960 = integrate.simps(flux_OIII4960, axis=0)
line_OIII5008 = integrate.simps(flux_OIII5008, axis=0)

print(np.shape(line_OIII4960))
plt.imshow(line_OIII5008 / line_OII, vmin=0, vmax=10, origin='lower', cmap='hot_r')
plt.show()
# fitted result
# line_fit_OII =

# Taking aperture info
# spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
# spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
# spe_OIII_4960_i = cube_OIII_4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
# spe_OIII_5008_i = cube_OIII_5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
#
#
# flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
# flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
# flux_OIII_4960_i, flux_OIII_4960_err_i = spe_OIII_4960_i.data * 1e-3, np.sqrt(spe_OIII_4960_i.var) * 1e-3
# flux_OIII_5008_i, flux_OIII_5008_err_i = spe_OIII_5008_i.data * 1e-3, np.sqrt(spe_OIII_5008_i.var) * 1e-3
# flux_stack = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII_4960_i, flux_OIII_5008_i))
# flux_err_stack = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII_4960_err_i, flux_OIII_5008_err_i))




