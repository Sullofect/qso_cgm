import os
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord
from photutils.centroids import centroid_quadratic
from photutils.profiles import RadialProfile
from astropy.wcs import WCS
from mpdaf.obj import Image, Cube, WaveCoord, iter_spe, iter_ima
from astropy import units as u
from photutils.background import Background2D, MedianBackground
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
path_data = '/Users/lzq/Dropbox/Data/CGM/'

def arc2kpc(x):
    return x * 100 / 15

def kpc2arc(x):
    return x * 15 / 100

# QSO info
z_qso = 0.6282144177077355
OII_air_2 = 3728.815
OIII_air = 5006.843
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
c_qso = SkyCoord(ra=ra_qso_muse*u.degree, dec=dec_qso_muse*u.degree, frame='fk5')


path_cube_OII = path_data + 'cube_narrow/CUBE_OII_line_offset_zapped.fits'
path_cube_OIII = path_data + 'cube_narrow/CUBE_OIII_5008_line_offset_zapped.fits'
f = fits.open(path_data + 'image_plot/image_OOHbeta_fitline_revised.fits')
w = WCS(f[0].header)
x, y = w.world_to_pixel(c_qso)

#
cube_OII = Cube(path_cube_OII)
cube_OIII = Cube(path_cube_OIII)

#
for i in [2]:
    # Split by velocity
    dv_i, dv_f = -600, 600
    wave_i_OII = OII_air_2 * (1 + z_qso) * (dv_i / 3e5 + 1)
    wave_f_OII = OII_air_2 * (1 + z_qso) * (dv_f / 3e5 + 1)

    wave_i_OIII = OIII_air * (1 + z_qso) * (dv_i / 3e5 + 1)
    wave_f_OIII = OIII_air * (1 + z_qso) * (dv_f / 3e5 + 1)

    sub_cube_OII = cube_OII.select_lambda(wave_i_OII, wave_f_OII)
    sub_cube_OII = sub_cube_OII.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2

    sub_cube_OIII = cube_OIII.select_lambda(wave_i_OIII, wave_f_OIII)
    sub_cube_OIII = sub_cube_OIII.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2


edge_radii = np.logspace(0, 2.5, 26) / 0.2 * 15 / 100
rp_OII = RadialProfile(sub_cube_OII.data * 1e17, [x, y], edge_radii,
                       error=np.sqrt(sub_cube_OII.var) * 1e17, mask=None)
rp_OIII = RadialProfile(sub_cube_OIII.data * 1e17, [x, y], edge_radii,
                        error=np.sqrt(sub_cube_OIII.var) * 1e17, mask=None)
radius, profile_OII, dprofile_OII = rp_OII.radius, rp_OII.profile, rp_OII.profile_error
radius, profile_OIII, dprofile_OIII = rp_OIII.radius, rp_OIII.profile, rp_OIII.profile_error

#
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# ax.plot(radius, profile_OII, '.', color='C0')
# ax.plot(radius, profile_OIII, '.', color='C1')
# ax.fill_between(radius, profile_OII - dprofile_OII, profile_OII + dprofile_OII, color='C0',
#                 alpha=0.2, label='[O II]')
# ax.fill_between(radius, profile_OIII - dprofile_OIII, profile_OIII + dprofile_OIII, color='C1',
#                 alpha=0.2, label='[O III]')
ax.errorbar(radius, profile_OII, dprofile_OII, fmt='.k', capsize=8, elinewidth=0.7, mfc='C0',
            ms=15, markeredgewidth=0.7, label=r'$\rm [O II]$')
ax.errorbar(radius, profile_OIII, dprofile_OIII, fmt='.k', capsize=8, elinewidth=0.7, mfc='C1',
            ms=15, markeredgewidth=0.7, label=r'$\rm [O III]$')
ax.set_xlim(6, 100)
# plt.ylim(1, )
ax.set_xscale('log')
ax.set_yscale('log')
ax.minorticks_on()
ax.legend(loc=1, prop={'size': 15})
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=7)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.set_xlabel(r"$\rm Radial \; distance [pkpc]$", size=20)
ax.set_ylabel(r'$\mathrm{SB \; [10^{-17} \; erg \; cm^{-2} \; s^{-1} \; arcsec^{-2}]}$', size=20)

# Second axis
secax = ax.secondary_xaxis('top', functions=(kpc2arc, arc2kpc))
secax.minorticks_on()
secax.set_xscale('log')
secax.set_xlabel(r"$\rm Radial \; distance [arcsec]$", size=20)
secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/RadialProfile_test.png', bbox_inches='tight')


# Check if center is correct
# plt.figure()
# plt.imshow(sub_cube.data * 1e17, origin='lower')
# plt.plot(x, y, '*', ms=10)
# plt.show()

# Calculate the centroid
from photutils.segmentation import detect_sources
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel, Box2DKernel
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
data_OII = sub_cube_OII.data
bkg_estimator = MedianBackground()
bkg = Background2D(data_OII, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
# data_OII -= bkg.background  # subtract the background
threshold = 1.2 * bkg.background_rms

# Convol
kernel = Box2DKernel(3)
convolved_data = convolve(data_OII, kernel)

# Seg
segment_map = detect_sources(convolved_data, threshold, npixels=10)
idx = segment_map.labels[np.argmax(segment_map.areas)]
segment_map_data = np.where(segment_map.data == idx, segment_map.data, 0)
data_OII_seg = np.where(segment_map.data == idx, data_OII, np.nan)
data_OII_convolved_seg = np.where(segment_map.data == idx, convolved_data, np.nan)
print(segment_map)

# Normalize
norm = ImageNormalize(stretch=SqrtStretch())

from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic)
x_OII, y_OII = centroid_com(data_OII_seg)
x_OII_convolved, y_OII_convolved = centroid_com(data_OII_convolved_seg)

# Fig
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15.0))
ax1.imshow(data_OII_convolved_seg, origin='lower', cmap='Reds', norm=norm)
ax1.plot(x_OII, y_OII, '*', ms=10)
ax1.plot(x_OII_convolved, y_OII_convolved, '.k', ms=10)
ax1.set_title('Background-subtracted Data')
ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
ax2.set_title('Segmentation Image')
ax3.imshow(segment_map_data, origin='lower', interpolation='nearest')
ax3.plot(x_OII, y_OII, '*', ms=10)
# plt.show()
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Segementation_test.png', bbox_inches='tight')

