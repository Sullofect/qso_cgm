import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from PyAstronomy import pyasl


c_left = SkyCoord(37.0744327 * u.deg, -40.9534193 * u.deg, frame='icrs')
c_mid = SkyCoord(37.0657292 * u.deg, -40.9534194 * u.deg, frame='icrs')
c_right = SkyCoord(37.0570988 * u.deg, -40.9534194 * u.deg, frame='icrs')



sep_lm = c_left.separation(c_mid)
sep_rm = c_right.separation(c_mid)

print('separation left mid', sep_lm.arcsecond)
print('separation right mid', sep_rm.arcsecond)