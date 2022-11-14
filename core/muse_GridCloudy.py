import os
import emcee
import corner
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe


# load the region
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

# Calculate the distance to a specific region
z = 0.6282144177077355
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.angular_diameter_distance(z=z)
ratio = (1 * u.radian).to(u.arcsec).value
arcsec_15 = (15 * d_l / ratio).to(u.kpc).value
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_s2, dec_s2 =  40.1364401, -18.8655766


c_qso = SkyCoord(ra_qso_muse, dec_qso_muse, frame='icrs', unit='deg')
c_s2 = SkyCoord(ra_array, dec_array, frame='icrs', unit='deg')
ang_sep = c_s2.separation(c_qso).to(u.arcsec).value
distance = np.log10((ang_sep * d_l / ratio).to(u.cm).value)
print(distance, text_array)
# = 23.049 = 23.05 for S2
# = 22.753 = 22.75 for S8

#### Define the grid
### Trial 1:
# Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])

for i in range(len(z)):
    lines = np.array(['Table power law spectral index -1.4, low=0.37, high=73.5 ',
                      'nuL(nu) = 46.54 at 1.0 Ryd',
                      'hden 4 vary',
                      'grid -2 2.5 0.1',
                      'save grid "alpha_1.4_' + str(z[i]) + '.grd"',
                      'metals ' + str(z[i]) + ' log',
                      'radius 23.05',
                      'iterative to convergence',
                      'save averages, file="alpha_1.4_' + str(z[i]) +  '.avr" last no clobber',
                      'temperature, hydrogen 1 over volume',
                      'end of averages',
                      'save line list "alpha_1.4_' + str(z[i]) + '.lin" from "linelist.dat" last'])
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial1/alpha_1.4_' + str(z[i]) + '.in', lines, fmt="%s")

### Trial 2:
# Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
alpha_array = np.array([-1.8, -1.75, -1.7, -1.65, -1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2])
z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
for i in range(len(z)):
    for j in range(len(alpha_array)):
        lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
                          'nuL(nu) = 46.54 at 1.0 Ryd',
                          'hden 4 vary',
                          'grid -2 2.5 0.1',
                          'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
                          'metals ' + str(z[i]) + ' log',
                          'radius 23.05',
                          'iterative to convergence',
                          'save averages, file="alpha_' + str(alpha_array[j])
                          + '_' + str(z[i]) + '.avr" last no clobber',
                          'temperature, hydrogen 1 over volume',
                          'end of averages',
                          'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
                          + '.lin" from "linelist.dat" last'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial2/alpha_' + str(alpha_array[j]) + '_'
                   + str(z[i]) + '.in', lines, fmt="%s")


### Trial 2 Part 2:
# Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
for i in range(len(z)):
    for j in range(len(alpha_array)):
        lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
                          'nuL(nu) = 46.54 at 1.0 Ryd',
                          'hden 4 vary',
                          'grid -2 2.5 0.1',
                          'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
                          'metals ' + str(z[i]) + ' log',
                          'radius 23.05',
                          'iterative to convergence',
                          'save averages, file="alpha_' + str(alpha_array[j])
                          + '_' + str(z[i]) + '.avr" last no clobber',
                          'temperature, hydrogen 1 over volume',
                          'end of averages',
                          'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
                          + '.lin" from "linelist.dat" last'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial2_p2/alpha_' + str(alpha_array[j]) + '_'
                   + str(z[i]) + '.in', lines, fmt="%s")

### Trial 3
# Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
for i in range(len(z)):
    for j in range(len(alpha_array)):
        lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
                          'nuL(nu) = 46.54 at 1.0 Ryd',
                          'hden 4 vary',
                          'grid -2 2.5 0.1',
                          'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
                          'metals ' + str(z[i]) + ' log',
                          'radius 22.75',
                          'iterative to convergence',
                          'save averages, file="alpha_' + str(alpha_array[j])
                          + '_' + str(z[i]) + '.avr" last no clobber',
                          'temperature, hydrogen 1 over volume',
                          'end of averages',
                          'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
                          + '.lin" from "linelist.dat" last'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial3/alpha_' + str(alpha_array[j]) + '_'
                   + str(z[i]) + '.in', lines, fmt="%s")


### Trial 4
# Luminosity, alpha=?, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
for i in range(len(z)):
    for j in range(len(alpha_array)):
        lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
                          'nuL(nu) = 45.54 at 1.0 Ryd',
                          'hden 4 vary',
                          'grid -2 2.5 0.1',
                          'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
                          'metals ' + str(z[i]) + ' log',
                          'radius 22.75',
                          'iterative to convergence',
                          'save averages, file="alpha_' + str(alpha_array[j])
                          + '_' + str(z[i]) + '.avr" last no clobber',
                          'temperature, hydrogen 1 over volume',
                          'end of averages',
                          'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
                          + '.lin" from "linelist.dat" last'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial4/alpha_' + str(alpha_array[j]) + '_'
                   + str(z[i]) + '.in', lines, fmt="%s")

### Trial 5
# Luminosity, alpha=?, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
alpha_array = np.linspace(-1.2, 0, 13, dtype='f2')
# z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
#               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
z = np.linspace(-1.5, 0.5, 11, dtype='f2')
for i in range(len(z)):
    for j in range(len(alpha_array)):
        lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
                          'nuL(nu) = 45.54 at 1.0 Ryd',
                          'hden 4 vary',
                          'grid -2 2.5 0.1',
                          'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
                          'metals ' + str(z[i]) + ' log',
                          'radius 22.75',
                          'iterative to convergence',
                          'save averages, file="alpha_' + str(alpha_array[j])
                          + '_' + str(z[i]) + '.avr" last no clobber',
                          'temperature, hydrogen 1 over volume',
                          'end of averages',
                          'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
                          + '.lin" from "linelist.dat" last'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial5/alpha_' + str(alpha_array[j]) + '_'
                   + str(z[i]) + '.in', lines, fmt="%s")