import os
import glob
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

# Constants
c_kms = 2.998e5
wave_Halpha_vac = 6564.614

def Gaussian(wave_vac, z, sigma_kms, flux, wave_line_vac):
    wave_obs = wave_line_vac * (1 + z)
    sigma_A = (sigma_kms / c_kms * wave_obs)

    peak = flux / np.sqrt(2 * sigma_A ** 2 * np.pi)
    gaussian = peak * np.exp(-(wave_vac - wave_obs) ** 2 / 2 / sigma_A ** 2)

    return gaussian

table_gals = ascii.read('../../MUSEQuBES+CUBS/MASSIVE/gals.csv')
object, z_gals = table_gals['Object Name'], table_gals['Redshift (z)']
print(object)
gals = glob.glob('../../MUSEQuBES+CUBS/MASSIVE/inspec/*.dat')
# print(gals)
# raise ValueError('Stop here')
plt.figure()
for i, gal in enumerate(gals):
    i_sort = np.where()
    data_massive = ascii.read(gal)
    wave, flux, flux_err, weight = data_massive['col1'], data_massive['col2'], data_massive['col3'], data_massive['col4']
    wave /= (1 + z_i)
    flux /= np.median(flux)
    plt.plot(wave, flux)
# plt.xlim(6500, 6600)
plt.ylim(0, 1.5)
plt.show()