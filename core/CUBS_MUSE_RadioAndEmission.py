import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from photutils.profiles import RadialProfile
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Also - for the radio thing - can you, centered on the quasar, and with zero degrees pointing towards one of the lobes,
# do a plot of the emission intensity as a function of angle? I.e., is there a peak of emission at 0 and 180 degrees,
# so that even though the situation is complicated, that nonetheless there is an excess of emission in the radio jet directions?


L = np.array([["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False]], dtype=object)
S = np.array([["3C57",            151,  71, [2], False, [], False],
              ["J0110-1648",       91,  29, [1], False, [2], False]], dtype=object)
A = np.array([["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
              ["PKS2242-498",     147,  71, [1, 2], True, [], False],
              ["PKS0232-04",      178, 116, [2, 4, 5, 7], False, [], False]], dtype=object)
allType = np.vstack((L, S, A))




def