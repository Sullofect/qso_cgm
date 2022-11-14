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


## Trial test stopping condition
alpha_array = -0.4
z = -1
T = np.array(['4e3', '5e3', '6e3', '7e3', '8e3', '9e3', '1e4'])
den = np.linspace(1, 3, 5)
for i in range(len(T)):
        lines = np.array(['Table power law spectral index ' + str(alpha_array) + ', low=0.37, high=73.5 ',
                          'nuL(nu) = 46.54 at 1.0 Ryd',
                          'hden 1 vary',
                          'grid 1 3 0.5',
                          'save grid "T_' + str(T[i]) + '.grd"',
                          'metals ' + str(z) + ' log',
                          'radius 22.75',
                          'iterative to convergence',
                          'save averages, file="T_' + str(T[i]) + '.avr" last no clobber',
                          'temperature, hydrogen 1 over volume',
                          'end of averages',
                          'save line list "T_' + str(T[i])  + '.lin" from "linelist.dat" last',
                          'stop temperature ' + str(T[i]) + ' K'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trialStopCond/T_' + str(T[i]) + '.in', lines, fmt="%s")

# Load lineratio
def load_cloudy(filename=None, path=None):
    # Line profile
    line = np.genfromtxt(path + filename + '.lin', delimiter=None)
    NeV3346, OII3727, OII3730 = line[:, 2], line[:, 3], line[:, 4]
    NeIII3869, Hdel, Hgam = line[:, 5], line[:, 8], line[:, 9]
    OIII4364, HeII4687, OIII5008 = line[:, 10], line[:, 11], line[:, 13]
    data = np.vstack((OII3727 + OII3730, OIII4364, OIII5008, NeIII3869, HeII4687, NeV3346))
    return np.log10(data)


def format_cloudy(filename=None, path=None):
    for i in range(len(filename[0])):
        metal_i = filename[0][i]
        for j in range(len(filename[1])):
            T_j = filename[1][j]
            filename_ij = 'T_' + str(T_j)
            if j == 0:
                output_j = load_cloudy(filename_ij, path=path)
                ind_j = np.array([[T_j, metal_i]])
            else:
                ind_jj = np.array([[T_j, metal_i]])
                c_i = load_cloudy(filename_ij, path=path)
                output_j = np.dstack((output_j, c_i))
                ind_j = np.dstack((ind_j, ind_jj))
        if i == 0:
            ind = ind_j[:, :, :, np.newaxis]
            output = output_j[:, :, :, np.newaxis]
        else:
            output = np.concatenate((output, output_j[:, :, :, np.newaxis]), axis=3)
            ind =  np.concatenate((ind, ind_j[:, :, :, np.newaxis]), axis=3)
    return output, ind

metal = np.array([z])

# Load Cloudy output
output, ind = format_cloudy(filename=[metal, T], path='/Users/lzq/Dropbox/Data/CGM/cloudy/trialStopCond/')

print(np.shape(output))


# Make diagnostic plot
fig, ax = plt.subplots(2, 3, figsize=(12, 9), sharex=True, dpi=300)
ax = ax.ravel()

T_array = np.linspace(4, 10, 7, dtype='f2') * 1e3
elements = [r'$\mathrm{\frac{[O \, II]}{H\beta}}$', r'$\mathrm{\frac{[O \, III 4363]}{H\beta}}$',
            r'$\mathrm{\frac{[O \, III 5007]}{H\beta}}$', r'$\mathrm{\frac{[Ne \, III]}{H\beta}}$',
            r'$\mathrm{\frac{He \, II}{H\beta}}$', r'$\mathrm{\frac{[Ne \, V]}{H\beta}}$']
for i in range(6):
    for j in range(len(den)):
        ax[i].plot(T_array, output[i, j, :, 0], label='n=' + str(den[j]))
        ax[i].set_title(elements[i], y=0.1)
        ax[0].legend()
fig.tight_layout()
fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Cloudy_CheckStopCond.png', bbox_inches='tight')
