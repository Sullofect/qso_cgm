import os
import muse_kc
import pyneb as pn
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from regions import PixCoord
from scipy.stats import norm
from astropy import units as u
from astropy.table import Table
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
from muse_RenameGal import ReturnGalLabel
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
c_kms = 2.998e5

# 3C57 M_BH M_ste
M_BH = 8.9
print((M_BH - 8.95) / 1.40 + 11)  # Different IMF
print((M_BH - 9 - np.log10(0.49)) / 1.16 + 11)
print((M_BH  - 0.4 - 9 - 0.3 - np.log10(0.49)) / 1.16 + 11)  # 0.3 dex error on M_BH vs M_ste relation

# Dynamical mass 3C57
G = 6.67e-8
R_3C57, v_3C57 = 5 / 7 * 50 * 3.086e21, 250 * 1e5
M_3C57 = R_3C57 * v_3C57 ** 2 / G / 2e33
print(np.format_float_scientific(M_3C57))

# 3C57 gaia magnitude
m_gaia = 16.196356
m_AB = m_gaia - 25.6884 + 25.7934
print('m_G in AB is', m_AB)

# 3C57 halo mass
# fit a Gaussian
# Minimize likelihood
cubename = '3C57'
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
t = fits.open(filename)[1].data
row, ID, z, v, name, ql, ra, dec = t['row'], t['ID'], t['z'], t['v'], t['name'], t['ql'], t['ra'], t['dec']

# Add the quasar and one more galaxy
z_G2 = 0.67240
v_G2 = (z_G2 - z_qso) / (1 + z_qso) * c_kms
v_gal = np.append(v, v_G2)
v_gal = np.append(v_gal, 0)
print(v_gal)

def gauss(x, mu, sigma):
    return np.exp(- (x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)

def loglikelihood(x_0, x):
    mu1, sigma1 = x_0[0], x_0[1]
    return -1 * np.sum(np.log(gauss(x, mu1, sigma1)))

# Normalization
bins_final = np.arange(-3000, 3200, 200)
nums, v_edge = np.histogram(v_gal, bins=bins_final)
normalization_all = np.sum(nums) * 200

#
guesses = [-80, 300]
bnds = ((-500, 500), (0, 1000))
result = minimize(loglikelihood, guesses, (v_gal), bounds=bnds, method="Powell")
print(result.x)

# Plot
rv = np.linspace(-2000, 2000, 1000)
plt.figure(figsize=(8, 5))
plt.hist(v_gal, bins=bins_final, color='k', histtype='step', label=r'$v_{\rm all}$')
plt.plot(rv, normalization_all * norm.pdf(rv, result.x[0], result.x[1]), '--', c='b', lw=1, alpha=1,
         label=r'$\mu_{1} = \, $' + str("{0:.0f}".format(result.x[0]))
               + r'$\mathrm{\, km \, s^{-1}}$' + '\n' + r'$\sigma_{1} = \, $'
               + str("{0:.0f}".format(result.x[1])) + r'$\mathrm{\, km \, s^{-1}}$')
plt.xlim(-2000, 2000)
plt.ylim(0, 13)
plt.yticks([2, 4, 6, 8, 10, 12])
plt.minorticks_on()
plt.xlabel(r'$\Delta v [\mathrm{km \; s^{-1}}]$', size=20)
plt.ylabel(r'$\mathrm{Numbers}$', size=20)
plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                labelsize=20)
plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
plt.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
plt.show()


# Munari et al. 2013
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h_z = cosmo.H(z_qso).value / 100
sigma = result.x[1] * len(v_gal) / (len(v_gal) - 1)
D, B, beta = 0.25, -0.0016, 1
sigma_cor = sigma * (1 + (D / (len(v_gal) - 1) ** beta + B))
print(sigma_cor)
M_200 = (sigma_cor / 1100) ** 3 * 1e15 / h_z
print(np.log10(M_200))
M_200_test = (370 / 1100) ** 3 * 1e15 / h_z
print('Halo mass is', np.format_float_scientific(M_200))
print('Halo mass test is', np.format_float_scientific(M_200_test))

#
sys_Li = [7, 3, 3, 9, 8, 7, 6, 8, 23, 4, 19, 3, 9, 5, 13, 7]
sys_loud = [3, 9, 23, 13, 7]
mass_Li = [13.2, 13.6, 13.9, 13.7, 12.1, 13.3, 14.1, 13.3, 13.1, 14.6, 13.5]
mass_loud = [13.6, 14.1, 14.6, 13.5]

print('N group', np.mean(sys_Li), np.std(sys_Li))
print('N group loud', np.mean(sys_loud), np.std(sys_loud))
print('halo mass', np.mean(mass_Li), np.std(mass_Li))
print('halo mass loud', np.mean(mass_loud), np.std(mass_loud))

