import os
import aplpy
import statmorph
import numpy as np
import skimage.measure
import skimage.transform
import skimage.feature
import skimage.segmentation
from statmorph_ZQL import source_morphology
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from photutils.segmentation import detect_threshold, detect_sources
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
# from statmorph.utils.image_diagnostics import make_figure
from image_diagnostics import make_figure
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                EllipticalAperture, EllipticalAnnulus)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.minor', size=4, visible=True)
rc('ytick.minor', size=4, visible=True)
rc('xtick', direction='in', labelsize=20, top='on')
rc('ytick', direction='in', labelsize=20, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Save the asymmetry values
path_OII_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OII_asymmetry.txt'
t_OII = Table.read(path_OII_asymmetry, format='ascii.fixed_width')

path_OIII_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OIII_asymmetry.txt'
t_OIII = Table.read(path_OIII_asymmetry, format='ascii.fixed_width')

path_OII_asymmetry_plus = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OII_asymmetry_plus_galaxies.txt'
t_OII_plus = Table.read(path_OII_asymmetry_plus, format='ascii.fixed_width')

path_OIII_asymmetry_plus = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OIII_asymmetry_plus_galaxies.txt'
t_OIII_plus = Table.read(path_OIII_asymmetry_plus, format='ascii.fixed_width')

path_21cm_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_21cm_asymmetry.txt'
t_21cm = Table.read(path_21cm_asymmetry, format='ascii.fixed_width')

path_Lya_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_Lya_asymmetry.txt'
t_Lya = Table.read(path_Lya_asymmetry, format='ascii.fixed_width')


# Conduct K-S test
A_OII, A_OII_shape = t_OII['A_ZQL'], t_OII['A_shape_ZQL']
A_OIII, A_OIII_shape = t_OIII['A_ZQL'], t_OIII['A_shape_ZQL']
A_21cm, A_21cm_shape = t_21cm['A_ZQL'], t_21cm['A_shape_ZQL']
A_Lya, A_Lya_shape = t_Lya['A_ZQL'], t_Lya['A_shape_ZQL']

#
from scipy.stats import ks_2samp
stat, pval = ks_2samp(A_OII_shape, A_21cm_shape)
print('shape OII and 21', stat, pval)

stat, pval = ks_2samp(A_OII_shape, A_Lya_shape)
print('shape OII and Lyalpha', stat, pval)

stat, pval = ks_2samp(A_OII, A_21cm)
print('OII and 21', stat, pval)

stat, pval = ks_2samp(A_OII, A_Lya)
print('OII and Lyalpha', stat, pval)


# Asymmetry Histogram
bins = np.linspace(0, 2, 11)
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
fig.subplots_adjust(wspace=0.15)
ax.hist(A_OII_shape, bins=bins, color='brown', alpha=0.5, histtype='stepfilled', lw=1.2,
           label=r'$\rm [O\,II]$', zorder=100, weights=np.ones_like(A_OII_shape) / len(A_OII_shape))
mid = (bins[1:] + bins[:-1]) / 2
mid = np.append(mid, mid[-1] + mid[-1] - mid[-2])
counts1, _ = np.histogram(A_21cm_shape, bins=bins, weights=np.ones_like(A_21cm_shape) / len(A_21cm_shape))
counts1 = np.append(counts1, counts1[-1])
counts2, _ = np.histogram(A_Lya_shape, bins=bins, weights=np.ones_like(A_Lya_shape) / len(A_Lya_shape))
counts2 = np.append(counts2, counts2[-1])
plt.step(mid, counts1, where="mid", alpha=0.8, color="blue", linestyle="--", linewidth=2, label=r'$\rm H\,I \, 21 \,cm$')
plt.step(mid, counts2, where="mid", alpha=0.8, color="red", linestyle="--", linewidth=2, label=r'$\rm Ly \alpha$')
ax.set_xlim(0, 2)
ax.set_xlabel('Asymmetry', size=25)
ax.set_ylabel(r'Fraction', size=25)
ax.legend(loc='upper right', fontsize=15)
for x in bins:
    ax.axvline(x, color='0.85', lw=0.8, zorder=0)
plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_asymmetry_distribution.png', bbox_inches='tight')


# Asymmetry and Gini index histograms
# bins = np.linspace(0, 2, 11)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
# fig.subplots_adjust(wspace=0.15)
# # ax[0].hist(t_OII['A_shape_ZQL'], bins=bins, color='brown', alpha=0.5, label=r'$\rm [O\,II]$')
# # ax[0].hist(t_OIII['A_shape_ZQL'], bins=bins, color='blue', alpha=0.5, label=r'$\rm [O\,III]$')
# ax[0].hist(t_OII_plus['A_shape_ZQL'], bins=bins, color='brown', alpha=0.5, histtype='stepfilled', lw=2,
#            label=r'$\rm [O\,II]$ plus')
# # ax[0].hist(t_OIII_plus['A_shape_ZQL'], bins=bins, color='blue', alpha=1.0, histtype='step', lw=2,
# #            label=r'$\rm [O\,III]$ plus')
# ax[0].hist(t_21cm['A_shape_ZQL'], bins=bins, color='red', alpha=1.0, label=r'$\rm HI \, 21\,cm$', histtype='step', lw=2)
# ax[0].hist(t_Lya['A_shape_ZQL'], bins=bins, color='k', alpha=1.0, label=r'$\rm Ly \alpha$', histtype='step', lw=2)
#
# bins = np.linspace(0, 1, 6)
# # ax[1].hist(t_OII['Gini_smoothed'], bins=bins, color='brown', alpha=0.5, label=r'$\rm [O\,II]$')
# # ax[1].hist(t_OIII['Gini_smoothed'], bins=bins, color='blue', alpha=0.5, label=r'$\rm [O\,III]$')
# ax[1].hist(t_OII_plus['Gini_smoothed'], bins=bins, color='brown', alpha=0.5, histtype='stepfilled', lw=2, label=r'$\rm [O\,II]$ plus')
# # ax[1].hist(t_OIII_plus['Gini_smoothed'], bins=bins, color='blue', alpha=1.0, histtype='step', lw=2, label=r'$\rm [O\,III]$ plus')
# ax[1].hist(t_21cm['Gini_smoothed'], bins=bins, color='red', alpha=1.0, label=r'$\rm HI \, 21\,cm$', histtype='step', lw=2)
# ax[1].hist(t_Lya['Gini_smoothed'], bins=bins, color='k', alpha=1.0, label=r'$\rm Ly \alpha$', histtype='step', lw=2)
# ax[0].set_xlim(0, 2)
# ax[1].set_xlim(0, 1)
# ax[0].set_xlabel('Asymmetry', size=20)
# ax[1].set_xlabel('Gini Index', size=20)
# ax[0].set_ylabel('N', size=20)
# ax[0].legend(loc='upper right', fontsize=15)
# plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_asymmetry_distribution.png', bbox_inches='tight')


