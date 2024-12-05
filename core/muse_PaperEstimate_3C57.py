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
# plt.show()


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


# OIII / Hbeta ratio
# os.system('muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_Hbeta -t 3.0 -s 1.5 -k gauss '
#           '-s_spe 1.5 -k_spe gauss -ssf False -l 0.3 -sl 8110 8160 -n 5')
# os.system('muse_MakeNBImageWith3DSeg.py -m 3C57_ESO-DEEP_subtracted_Hbeta -t 1.0 -s 1.5 -k gauss '
#           '-ssf False -l 0.3 -sl 8110 8160 -n 5')
UseSeg = (1.5, 'gauss', 1.5, 'gauss')
path_SB_Hbeta = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_Hbeta_SB_3DSeg_1.5_gauss_1.5_gauss.fits'
path_SB_OIII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_OIII_SB_3DSeg_1.5_gauss_1.5_gauss.fits'
path_3Dseg_Hbeta = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format('Hbeta', *UseSeg)
path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/3C57_ESO-DEEP_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format('OIII', *UseSeg)
hdul_Hbeta, hdul_OIII = fits.open(path_SB_Hbeta), fits.open(path_SB_OIII)
SB_Hbeta, SB_OIII = hdul_Hbeta[1].data, hdul_OIII[1].data
seg_3D_Hbeta_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_Hbeta)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_Hbeta, mask_seg_OIII = np.sum(seg_3D_Hbeta_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
mask_seg = mask_seg_Hbeta + mask_seg_OIII
OIII_Hbeta = np.log10(SB_OIII / SB_Hbeta)
OIII_Hbeta = np.where(mask_seg > 0, OIII_Hbeta, np.nan)
print(np.nanmedian(OIII_Hbeta), np.nanstd(OIII_Hbeta))
plt.figure()
plt.imshow(OIII_Hbeta, origin='lower')
plt.show()



# path_fit = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
# hdul = fits.open(path_fit)
# fs, hdr = hdul[1].data, hdul[2].header
# v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
# sigma, dsigma = hdul[5].data, hdul[6].data
# flux_OII_fit, dflux_OII_fit = hdul[7].data, hdul[8].data
# flux_OIII_fit, dflux_OIII_fit = hdul[9].data, hdul[10].data
# r, dr = hdul[11].data, hdul[12].data
# a_OII, da_OII = hdul[13].data, hdul[14].data
# a_OIII, da_OIII = hdul[17].data, hdul[18].data
# b_OII, db_OII = hdul[15].data, hdul[16].data
# b_OIII, db_OIII = hdul[19].data, hdul[20].data
