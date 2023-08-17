import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from mpdaf.obj import Image, Cube, WaveCoord, iter_spe, iter_ima
from astropy import units as u
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel, Box2DKernel
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
path_data = '/Users/lzq/Dropbox/Data/CGM/'

# QSO info
z_qso = 0.6282144177077355
OII_air_2, OIII_air = 3728.815, 5006.843
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
c_qso = SkyCoord(ra=ra_qso_muse*u.degree, dec=dec_qso_muse*u.degree, frame='fk5')


# Cubes
path_cube_OII = path_data + 'cube_narrow/CUBE_OII_line_offset_zapped.fits'
path_cube_OIII5008 = path_data + 'cube_narrow/CUBE_OIII_5008_line_offset_zapped.fits'
f = fits.open(path_data + 'image_plot/image_OOHbeta_fitline_revised.fits')
w = WCS(f[0].header)
x, y = w.world_to_pixel(c_qso)
cube_OII = Cube(path_cube_OII)
cube_OIII5008 = Cube(path_cube_OIII5008)

#
size = np.shape(cube_OII.data)
wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
wave_OIII5008_vac = pyasl.airtovac2(cube_OIII5008.wave.coord())
flux_OIII5008 = cube_OIII5008.data * 1e-3
flux_OIII5008_err = np.sqrt(cube_OIII5008.var) * 1e-3

def keep_longest_true_ori(a):
    # Convert to array
    a = np.asarray(a)

    # Attach sentients on either sides w.r.t True
    b = np.r_[False, a, False]

    # Get indices of group shifts
    s = np.flatnonzero(b[:-1] != b[1:])
    print(b)
    print(b[:-1] != b[1:])
    print(s)

    # Get group lengths and hence the max index group
    m = (s[1::2] - s[::2]).argmax(axis=0)
    print(s[1:2])
    print(s[::2])
    print(m)

    # Initialize array and assign only the largest True island as True.
    out = np.zeros_like(a)
    out[s[2*m]:s[2*m+1]] = 1
    return out

def keep_longest_true(a):
    # Convert to array
    a = np.asarray(a)

    # Attach sentients on either sides w.r.t True
    # b = np.r_[False, a, False]
    b = np.zeros((np.shape(a)[0] + 2, np.shape(a)[1], np.shape(a)[2]))
    b[1:np.shape(a)[0]+1, :, :] = a
    b[0, :, :] = False
    b[-1, :, :] = False

    # Get indices of group shifts
    # s = np.flatnonzero(b[:-1] != b[1:])
    # print(b[:-1, :, :] != b[1:, :, :])
    # s = np.nonzero(b[:-1, :, :] != b[1:, :, :])
    # s = np.take_along_axis(s, np.expand_dims(s, axis=-1), axis=-1)
    s_0 = np.arange(np.shape(a)[0] + 1)
    s_1 = np.repeat(s_0[:, np.newaxis], np.shape(a)[1], axis=1)
    s_2 = np.repeat(s_1[:, :, np.newaxis], np.shape(a)[2], axis=2)
    s = np.where(b[:-1, :, :] != b[1:, :, :], s_2, False)
    print(s[:, 0, 0])

    # Get group lengths and hence the max index group
    print(np.shape(s))
    m = np.nanargmax(s[1::2, :, :] - s[::2, :, :], axis=0)
    # m = np.where(m == False, m, np.nan)
    # print(m[20, 20])

    # Initialize array and assign only the largest True island as True.
    out = np.zeros_like(a)
    m2 = np.take_along_axis(s, np.expand_dims(2 * m, axis=0), axis=1)
    m21 = np.take_along_axis(s, np.expand_dims(2 * m + 1, axis=0), axis=1)
    print(m2[:, 0, 0])
    # out = np.where( , , 1)
    out =
    out[s[2*m]:s[2*m+1]] = 1
    return out

# Vectorize everything
flux_where = np.where(flux_OIII5008 > 1.1 * np.median(flux_OIII5008_err), flux_OIII5008, False)
flux_where = np.asarray(flux_where, dtype=bool)
flux_where = keep_longest_true(flux_where[:3, :, :])
# x = keep_longest_true_ori(flux_where[:21, 92, 92])
# print(flux_where[:, 90, 90])
flux_sum = np.nansum(flux_OIII5008[flux_where], axis=0)
# print(np.shape(flux_sum))
#
#
# plt.figure(figsize=(3, 3))
# plt.imshow(flux_sum, origin='lower')
# plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/MakeNBImage_MC_test.png')

# fig, ax = plt.subplots(3, 3, figsize=(20, 20))
# # axarr = ax.ravel()
# for i, i_val in enumerate([90, 91, 92]):
#     for j, j_val in enumerate([90, 91, 92]):
#         flux_OIII5008_ij, flux_OIII5008_err_ij = flux_OIII5008[:, i_val, j_val], flux_OIII5008_err[:, i_val, j_val]
#         flux_max, flux_min = np.nanmax(flux_OIII5008_ij), np.nanmin(flux_OIII5008_ij)
#
#         flux_where = np.where(flux_OIII5008_ij > 0.5 * np.median(flux_OIII5008_err_ij), flux_OIII5008_ij, False)
#         flux_where = np.asarray(flux_where, dtype=bool)
#         flux_where = keep_longest_true(flux_where)
#         # flux_sum = np.nansum()
#         #
#         ax[i, j].plot(wave_OIII5008_vac, flux_OIII5008_ij, '-k')
#         ax[i, j].plot(wave_OIII5008_vac, flux_OIII5008_err_ij, '-C0')
#         ax[i, j].fill_between(wave_OIII5008_vac[flux_where], y1=np.zeros_like(wave_OIII5008_vac[flux_where]), y2=flux_OIII5008_ij[flux_where])
#
#
# plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/MakeNBImage_MC.png')



