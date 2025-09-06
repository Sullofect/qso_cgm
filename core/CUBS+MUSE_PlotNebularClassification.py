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

L = np.array([["HE0226-4110",     150,  84],
              ["PKS0405-123",     106, 129],
              ["HE0238-1904",     113, 103],
              ["PKS0552-640",     124, 153],
              ["J0454-6116",      151, 102],
              ["J0119-2010",      166,  96],
              ["HE0246-4101",     129,  74],
              ["PKS0355-483",     142,  50],
              ["HE0439-5254",     246,  47],
              ["TXS0206-048",     194, 200],
              ["Q1354+048",       123, 126]], dtype=object)

S_BR = np.array([["HE0435-5304",      87,  55],
                 ["3C57",            151,  71],
                 ["J0110-1648",       91,  29],
                 ["HE0112-4145",     164,  38],
                 ["J0154-0712",      137,  63],
                 ["Q1435-0134",      261,  63]], dtype=object)

S = np.array([["J0028-3305",      133,  42],
              ["HE0419-5657",     154,  35],
              ["Q0107-025",       116,  28],
              ["HE1003+0149",     208,  53],
              ["HE0331-4112",     196,  32]], dtype=object)

A = np.array([["J2135-5316",      107,  83],
              ["Q0107-0235",      136,  90],
              ["PKS2242-498",     147,  71],
              ["PG1522+101",      132,  50],
              ["PKS0232-04",      178, 116]], dtype=object)



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

# Quasar nebulae summary figure
plt.figure(figsize=(5, 5), dpi=300)

# for i in range(len(morphology)):
#     for j in range(len(morphology[i])):
#         morpho = morphology[i][j]
#         if morpho == 'R':
#             plt.scatter(sigma_80[i], size[i], marker=upper_half, color='none', facecolor='blue', edgecolor='blue',
#                         s=60, alpha=0.5)
#             plt.scatter(sigma_80[i], size[i], marker=lower_half, color='none', facecolor='red', edgecolor='red',
#                         s=60, alpha=0.5)
#         elif morpho == 'U':
#             plt.scatter(sigma_80[i], size[i], marker='d', color='none', facecolor='none', edgecolor='black', s=40, alpha=0.7)
#         elif morpho == 'I':
#             plt.scatter(sigma_80[i], size[i], marker='s', color='none', facecolor='none', edgecolor='brown', s=100, alpha=0.8)
#         elif morpho == 'F':
#             plt.scatter(sigma_80[i], size[i], marker='+', color='none', facecolor='purple', edgecolor='black', s=70, alpha=0.8)
#         elif morpho == 'O':
#             plt.scatter(sigma_80[i], size[i], marker='2', color='red', facecolor='red', edgecolor='black', s=50, alpha=0.5)
#

for i in range(len(L)):
    # BR
    if L[i][0] == 'PKS0405-123' or L[i][0] == 'HE0238-1904':
        plt.scatter(L[i][1], L[i][2], marker=upper_half, color='none', facecolor='blue', edgecolor='blue',
                    s=50, alpha=0.5)
        plt.scatter(L[i][1], L[i][2], marker=lower_half, color='none', facecolor='red', edgecolor='red',
                    s=50, alpha=0.5)
    # Outflow
    if L[i][0] == 'HE0238-1904' or L[i][0] == 'J0119-2010':
        plt.scatter(L[i][1], L[i][2], marker='x', color='red', facecolor='red', edgecolor='black', s=50, alpha=0.8)
    plt.scatter(L[i][1], L[i][2], marker='s', color='none', facecolor='none', edgecolor='black', s=100, alpha=0.8)
for i in range(len(S_BR)):
    # Outflow
    if S_BR[i][0] == '3C57':
        plt.scatter(S_BR[i][1], S_BR[i][2], marker='x', color='red', facecolor='red', edgecolor='black', s=50, alpha=0.8)
    plt.scatter(S_BR[i][1], S_BR[i][2], marker=upper_half, color='none', facecolor='blue', edgecolor='blue',
                s=30, alpha=0.5)
    plt.scatter(S_BR[i][1], S_BR[i][2], marker=lower_half, color='none', facecolor='red', edgecolor='red',
                s=30, alpha=0.5)
    plt.scatter(S_BR[i][1], S_BR[i][2], marker='d', color='none', facecolor='none', edgecolor='black', s=70, alpha=0.7)
for i in range(len(S)):
    plt.scatter(S[i][1], S[i][2], marker='d', color='none', facecolor='none', edgecolor='black', s=70, alpha=0.7)
for i in range(len(A)):
    # Outflows
    if A[i][0] == 'J2135-5316':
        plt.scatter(A[i][1], A[i][2], marker='x', color='red', facecolor='red', edgecolor='black', s=50, alpha=0.8)
    plt.scatter(A[i][1], A[i][2], marker='p', color='none', facecolor='none', edgecolor='black', s=100, alpha=0.7)



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
ax = plt.gca()
split_marker = object()
handles = [Line2D([], [], marker='s', color='none', markerfacecolor='none', markeredgecolor='black', alpha=0.7),
           Line2D([], [], marker='d', color='none', markerfacecolor='none', markeredgecolor='black', alpha=0.8),
           Line2D([], [], marker='p', color='none', markerfacecolor='none', markeredgecolor='black', alpha=0.8)]
legend1 = plt.legend(handles=handles, labels=['Large, Irregular', 'Host-Galaxy-Scale', 'Complex Morphology \n '
                                                                                       'and Kinematics',
                                              'Blueshifted-Redshifted', 'Outflows'],
           handler_map={split_marker: SplitCircleLegend(upper_half, lower_half)}, loc='upper left', fontsize=10)
ax.add_artist(legend1)
legend2 = plt.legend(handles=[split_marker,
                              Line2D([], [], marker='x', color='none', markerfacecolor='red', markeredgecolor='red',
                                     alpha=0.5)],
                     labels=['Blueshifted-\nRedshifted', 'Outflows'],
                     handler_map={split_marker: SplitCircleLegend(upper_half, lower_half)}, loc='upper right',
                     fontsize=10)
ax.add_artist(legend2)
plt.xlabel(r'$\sigma \, \rm [km\,s^{-1}]$', size=25)
plt.ylabel(r'$\rm Size \, [kpc]$', size=25)
plt.xlim(60, 280)
plt.ylim(20, 220)
plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_Label_Distribution.png', bbox_inches='tight')
