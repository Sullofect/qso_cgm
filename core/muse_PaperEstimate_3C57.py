import os
import muse_kc
import pyneb as pn
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from muse_RenameGal import ReturnGalLabel
from regions import RectangleSkyRegion, RectanglePixelRegion
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


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