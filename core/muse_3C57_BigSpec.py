import os
# import aplpy
# import lmfit
import numpy as np
import matplotlib as mpl
import gala.potential as gp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import stats
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import PixCoord
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian2DKernel
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
import biconical_outflow_model_3d as bicone
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from gala.units import galactic, solarsystem, dimensionless
from photutils.isophote import EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.isophote import Ellipse
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Generate random 2D map data and 1D velocity data
map_data = np.random.rand(10, 10)
velocity_data = np.random.rand(10)

# Create a figure with subplots
fig, axs = plt.subplots(nrows=map_data.shape[0], ncols=map_data.shape[1], figsize=(20, 20))

# Iterate over each spaxel and plot the corresponding 1D spectrum
for i in range(map_data.shape[0]):
    for j in range(map_data.shape[1]):
        ax = axs[i, j]
        spectrum = np.random.rand(100)  # Example 1D spectrum data
        velocity = velocity_data[j]  # Example velocity data for color coding

        ax.plot(spectrum, color='black')
        ax.set_title(f'Spaxel ({i}, {j})')
        ax.set_xlim(0, len(spectrum))
        ax.set_ylim(0, 1)  # Adjust y-axis limits as needed
        ax.set_xticks([])  # Disable x-axis ticks
        ax.set_yticks([])  # Disable y-axis ticks

        # Color the full panel according to its velocity
        ax.set_facecolor(plt.cm.viridis(velocity))

plt.tight_layout()
plt.show()