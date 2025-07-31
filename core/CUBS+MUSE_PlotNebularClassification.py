import os
import aplpy
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# Generate quasar nebulae summary figure
morphology = np.array(["R", "N", "I", "R+I", "R+O+I", "R+O", "I", "R", "I", "O+I", "O+I",
                       "U+I", "U", "U+I", "U", "U", "R", "I", "R", "I", "N", "U", "U",
                       "F", "I", "R", "R", "R", "N", "R"])

area = np.array([1170,   35, 2740, 8180, 4820, 1930, 5180,  530, 4350, 2250,
                 3090, 2280,  500,  620,  410, 2130, 2260, 1310,  890,  970,
                 100, 1660,  520,10800, 2860, 1340, 2010, 1220,   30, 3670], dtype=float)

size = np.array([55,    7,   84,  129,  103,   71,  153,   29,  102,   83,
                 96,   74,   42,   35,   28,    90,   71,   50,   38,   47,
                 15,   53,   32,  200,  126,   63,   63,   50,    7,  116], dtype=float)

sigma_80 = np.array([87,   np.nan, 150, 106, 113, 151, 124,  91, 151, 107,
                     166,   129, 133, 154, 116, 136, 147, 142, 164, 246,
                     np.nan, 208, 196, 194, 123, 137, 261, 132, np.nan, 178], dtype=float)

# Create upper half-circle — do NOT close across the flat side
a, b = 0.8, 1.0
theta_upper = np.linspace(0, np.pi, 50)
verts_upper = np.column_stack([a * np.cos(theta_upper), b * np.sin(theta_upper)])
codes_upper = [Path.MOVETO] + [Path.LINETO] * (len(verts_upper) - 1)
upper_half = Path(verts_upper, codes_upper)

# Create lower half-circle — same idea
theta_lower = np.linspace(np.pi, 2 * np.pi, 50)
verts_lower = np.column_stack([a * np.cos(theta_lower), b * np.sin(theta_lower)])
codes_lower = [Path.MOVETO] + [Path.LINETO] * (len(verts_lower) - 1)
lower_half = Path(verts_lower, codes_lower)

# Jellyfish
# verts = [
#     # Dome (bell) - upper semi ellipse using Bezier curves
#     (0.0, 0.5),  # start at top center
#     (0.3, 0.5),  # control point 1
#     (0.5, 0.0),  # control point 2
#     (0.5, -0.3), # end point
#
#     (0.5, -0.3), # move down right side
#     (0.5, -0.5), # control point 3
#     (0.0, -0.5), # control point 4
#     (0.0, -0.3), # end bottom middle
#
#     (0.0, -0.3), # move down left side
#     (0.0, -0.5), # control point 5
#     (-0.5, -0.5),# control point 6
#     (-0.5, -0.3),# end left bottom
#
#     (-0.5, -0.3),# left side up
#     (-0.5, 0.0), # control point 7
#     (-0.3, 0.5), # control point 8
#     (0.0, 0.5),  # back to top center
#
#     # Tentacles as curves (just one example tentacle here)
#     (0.0, -0.3),
#     (0.1, -0.6),
#     (0.2, -0.8),
#     (0.1, -1.0),
#
#     (0.0, -0.3),
#     (-0.1, -0.6),
#     (-0.2, -0.8),
#     (-0.1, -1.0),
# ]
#
# codes = [
#     Path.MOVETO,
#     Path.CURVE4,
#     Path.CURVE4,
#     Path.CURVE4,
#
#     Path.LINETO,
#     Path.CURVE4,
#     Path.CURVE4,
#     Path.CURVE4,
#
#     Path.LINETO,
#     Path.CURVE4,
#     Path.CURVE4,
#     Path.CURVE4,
#
#     Path.LINETO,
#     Path.CURVE4,
#     Path.CURVE4,
#     Path.CURVE4,
#
#     Path.MOVETO,
#     Path.CURVE4,
#     Path.CURVE4,
#     Path.CURVE4,
#
#     Path.MOVETO,
#     Path.CURVE4,
#     Path.CURVE4,
#     Path.CURVE4,
# ]
# jellyfish_path = Path(verts, codes)

# Quasar nebulae summary figure
plt.figure(figsize=(5, 5), dpi=300)
for i in range(len(morphology)):
    for j in range(len(morphology[i])):
        morpho = morphology[i][j]
        if morpho == 'R':
            plt.scatter(sigma_80[i], size[i], marker=upper_half, color='none', facecolor='blue', edgecolor='blue',
                        s=60, alpha=0.5)
            plt.scatter(sigma_80[i], size[i], marker=lower_half, color='none', facecolor='red', edgecolor='red',
                        s=60, alpha=0.5)
        elif morpho == 'U':
            plt.scatter(sigma_80[i], size[i], marker='d', color='none', facecolor='none', edgecolor='black', s=40, alpha=0.7)
        elif morpho == 'I':
            plt.scatter(sigma_80[i], size[i], marker='s', color='none', facecolor='none', edgecolor='brown', s=100, alpha=0.8)
        elif morpho == 'F':
            plt.scatter(sigma_80[i], size[i], marker='+', color='none', facecolor='purple', edgecolor='black', s=70, alpha=0.8)
        elif morpho == 'O':
            plt.scatter(sigma_80[i], size[i], marker='2', color='red', facecolor='red', edgecolor='black', s=50, alpha=0.5)


class SplitCircleLegend:
    def __init__(self, upper_path, lower_path):
        self.upper_path = upper_path
        self.lower_path = lower_path
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import PathPatch
        center = handlebox.xdescent + handlebox.width / 2, handlebox.ydescent + handlebox.height / 2
        scale = min(handlebox.width, handlebox.height) / 2
        trans = (mtransforms.Affine2D()
                 .scale(scale)
                 .translate(*center))
        # Draw lower half (red)
        patch1 = PathPatch(self.lower_path, transform=trans + handlebox.get_transform(),
                           facecolor='red', edgecolor='red', alpha=0.5)
        # Draw upper half (blue)
        patch2 = PathPatch(self.upper_path, transform=trans + handlebox.get_transform(),
                           facecolor='blue', edgecolor='blue', alpha=0.5)
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return patch1

# Dummy handle (content doesn't matter)
split_marker = object()
handles = [split_marker,
           Line2D([], [], marker='d', color='none', markerfacecolor='none', markeredgecolor='black', alpha=0.7),
           Line2D([], [], marker='s', color='none', markerfacecolor='none', markeredgecolor='brown', alpha=0.8),
           Line2D([], [], marker='+', color='none', markerfacecolor='purple', markeredgecolor='purple', alpha=0.8),
           Line2D([], [], marker='2', color='none', markerfacecolor='red', markeredgecolor='red', alpha=0.5)]
plt.legend(handles=handles, labels=['R', 'U', ' I', 'F', 'O'],
           handler_map={split_marker: SplitCircleLegend(upper_half, lower_half)}, loc='upper right', fontsize=15)

plt.xlabel(r'$\sigma_{80} \, \rm [km\,s^{-1}]$', size=25)
plt.ylabel(r'$\rm Size \, [kpc]$', size=25)
plt.xlim(60, 280)
plt.ylim(20, 220)
plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_Label_Distribution.png', bbox_inches='tight')
