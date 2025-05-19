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
rc('xtick.minor', size=3, visible=False)
rc('ytick.minor', size=3, visible=False)
rc('xtick', direction='in', labelsize=15)
rc('ytick', direction='in', labelsize=15)
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Save the asymmetry values
path_OII_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OII_asymmetry.txt'
t_OII = Table.read(path_OII_asymmetry, format='ascii.fixed_width')

path_OIII_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_OIII_asymmetry.txt'
t_OIII = Table.read(path_OIII_asymmetry, format='ascii.fixed_width')

path_21cm_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_21cm_asymmetry.txt'
t_21cm = Table.read(path_21cm_asymmetry, format='ascii.fixed_width')

path_Lya_asymmetry = '../../MUSEQuBES+CUBS/asymmetry/CUBS+MUSE_Lya_asymmetry.txt'
t_Lya = Table.read(path_Lya_asymmetry, format='ascii.fixed_width')

bins = np.linspace(0, 2, 11)
plt.figure(figsize=(5, 5), dpi=300)
plt.hist(t_OII['A_shape_ZQL'], bins=bins, color='brown', alpha=0.5, label=r'$\rm [O\,II]$')
plt.hist(t_OIII['A_shape_ZQL'], bins=bins, color='blue', alpha=0.5, label=r'$\rm [O\,III]$')
plt.hist(t_21cm['A_shape_ZQL'], bins=bins, color='red', alpha=1.0, label=r'$\rm HI \, 21\,cm$', histtype='step', lw=2)
plt.hist(t_Lya['A_shape_ZQL'], bins=bins, color='k', alpha=1.0, label=r'$\rm Ly \alpha$', histtype='step', lw=2)
plt.xlim(0, 2)
plt.xlabel('Asymmetry', size=20)
plt.ylabel('N', size=20)
plt.legend(loc='upper right', fontsize=15)
plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_asymmetry_distribution.png', bbox_inches='tight')


