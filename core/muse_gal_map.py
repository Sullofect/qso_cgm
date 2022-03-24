import os
import aplpy
import matplotlib
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from mpdaf.obj import WCS as mpdaf_WCS
from matplotlib import rc
from matplotlib import cm
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.colors import ListedColormap
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from muse_gal_check_astrometry import offset
from astropy.table import Table
path_savetable = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

path_s = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects_sean.fits')
data_s = fits.getdata(path_s, 1, ignore_missing_end=True)

path_w = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects.fits')
data_w = fits.getdata(path_w, 1, ignore_missing_end=True)

ra_w, dec_w = data_w['ra'], data_w['dec']

row_s = data_s['row']
row_w = data_w['row']

ID_s = data_s['id']
ID_w = data_w['id']

name_s = data_s['name']
name_w = data_w['name']

ql_s = data_s['quality']
ql_w = data_w['quality']

cl_s = data_s['class']
cl_w = data_w['class']

cl_s_num = np.zeros_like(cl_s)
cl_w_num = np.zeros_like(cl_w)
classes = ['galaxy', 'star', 'quasar', 'hizgal']

for i in range(4):
    cl_s_num = np.where(cl_s != classes[i], cl_s_num, i)
    cl_w_num = np.where(cl_w != classes[i], cl_w_num, i)

cl_s_num = cl_s_num.astype(float)
cl_w_num = cl_w_num.astype(float)

z_s = data_s['redshift']
z_w = data_w['redshift']
z_qso = 0.6282144177077355

v_w = 3e5 * (z_w - z_qso) / (1 + z_qso)
v_s = 3e5 * (z_s - z_qso) / (1 + z_qso)

ct_s = data_s['comment']
ct_w = data_w['comment']

select_gal = np.where(cl_w == 'galaxy')
row_gal = row_w[select_gal]
ID_gal = ID_w[select_gal]
z_gal = z_w[select_gal]
name_gal = name_w[select_gal]
ql_gal = ql_w[select_gal]
ra_gal, dec_gal = ra_w[select_gal], dec_w[select_gal]

select_qua = np.where((ql_gal == 1) | (ql_gal == 2))
row_qua = row_gal[select_qua]
ID_qua = ID_gal[select_qua]
z_qua = z_gal[select_qua]
v_qua = 3e5 * (z_qua - z_qso) / (1 + z_qso)
name_qua = name_gal[select_qua]
ql_qua = ql_gal[select_qua]
ra_qua, dec_qua = ra_gal[select_qua], dec_gal[select_qua]

bins = np.arange(-2000, 2200, 200)
select_z = np.where((v_qua > bins[0]) * (v_qua < bins[-1]))
row_final = row_qua[select_z]
ID_final = ID_qua[select_z]
z_final = z_qua[select_z]
v_final = v_qua[select_z]
name_final = name_qua[select_z]
ql_final = ql_qua[select_z]
ra_final, dec_final = ra_qua[select_z], dec_qua[select_z]

# Muse image
# path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits')
#
# cube = Cube(path)
# hdul = fits.open(path)  # open a FITS file
# hdr = hdul[1].header
# wcs = mpdaf_WCS(hdr)
#
# # Calculate the white image
# image_white = cube.sum(axis=0)
# p, q = image_white.peak()['p'], image_white.peak()['q']
# p_q = wcs.sky2pix(np.vstack((dec_final, ra_final)).T, nearest=True)
# p_gal, q_gal = p_q.T[0], p_q.T[1]
#
Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)
#
# plt.figure(figsize=(8, 5), dpi=300)
# plt.imshow(image_white.data, origin='lower', cmap=newcmp, norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar()
# # cbar.set_label(r'$\mathrm{Arcsinh}$')
# plt.contour(image_white.data, levels=[1e5, 1e6, 1e7, 1e8], colors=newcolors_red[200::30, :], linewidths=0.5, alpha=0.5,
#             norm=matplotlib.colors.LogNorm())
# plt.plot(q_gal, p_gal, 'o', color='brown', ms=7, alpha=0.4, markerfacecolor='None', markeredgecolor='red',
#          markeredgewidth=0.5)
# for i in range(len(row_final)):
#     plt.annotate(str(row_final[i]), (q_gal[i], p_gal[i]), fontsize=5)
#
# plt.axis('off')
# plt.xlim(200, 250)
# plt.ylim(200, 250)

# Getting photometry zero point
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'HE0238-1904_sex.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)

path_image = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'check.fits')
data_image = fits.getdata(path_image, 1, ignore_missing_end=True)

ra_pho = data_pho['ALPHAWIN_J2000']
dec_pho = data_pho['DELTAWIN_J2000']

# w_pho = WCS(fits.open(path_image)[1].header)
# catalog = pixel_to_skycoord(data_pho['X_IMAGE'], data_pho['Y_IMAGE'], w_pho)
catalog = SkyCoord(ra_pho, dec_pho, unit="deg")
c = SkyCoord(ra_final, dec_final, unit="deg")
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# Hubble image
f_hb = fits.open(path_hb)
w_hb = WCS(f_hb[1].header)
x, y = skycoord_to_pixel(c, w_hb)

ra_offset, dec_offset = offset()

# check consistency
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_image, figure=fig, north=True)
gc.show_colorscale(vmin=0, vmax=3, cmap=newcmp)
gc.set_xaxis_coord_type('scalar')
gc.set_yaxis_coord_type('scalar')
gc.recenter(40.1359, -18.8643, width=0.02, height=0.02)
# gc.show_circles(ra_final, dec_final, 0.0002)
gc.show_circles(ra_pho[idx] - ra_offset, dec_pho[idx] - dec_offset, 0.0002)
# plt.show()

#-------Create reg file for DS9-----------
# galaxy_list = np.array([])
# for i in range(len(ra_final)):
#     galaxy_list = np.hstack((galaxy_list, np.array(['fk5; circle(' + str(ra_final[i]) + ', '
#                                                     + str(dec_final[i]) + ', 1") ' + ' # text = "' + str(row_final[i])
#                                                    + '"'])))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg', galaxy_list, fmt='%s')
# -------------------------------------------

#----Gettting zero point--------------------
# from acstools import acszpt
# date = '2017-06-19'
# detector = 'WFC'
#
# q = acszpt.Query(date=date, detector=detector)
# zpt_table = q.fetch()
#
# print(zpt_table)
#-------------------------------------------


#----Compare with Legacy survey--------------------
from astropy.io import ascii
data = Table()
data["RA"] = ra_pho[idx] - ra_offset
data["DEC"] = dec_pho[idx] - dec_offset
data['ID'] = row_final
#data["NAME"] =  np.core.defchararray.add(np.char.mod('%d', row_final), np.repeat(np.array(['o']), len(row_final)))
#data["COLOR"] = np.repeat(np.array(['black']), len(row_final))
#data["RADIUS"] = np.repeat(np.array(['1']), len(row_final))
ascii.write(data, path_savetable + 'galaxys_list_xmatch.csv', format='csv', overwrite=True)
#-------------------------------------------

# fig = plt.figure(figsize=(8, 8), dpi=300)
# plot_extents = 0, 4500, 0, 4500
# transform = Affine2D().rotate_deg(angle * 180 / np.pi)

# helper = floating_axes.GridHelperCurveLinear(transform, plot_extents, grid_locator1=MaxNLocator(nbins=5),
#                                              grid_locator2=MaxNLocator(nbins=5))
# axarr = fig.add_subplot(111, axes_class=floating_axes.FloatingAxes, grid_helper=helper)
# aux_ax = axarr.get_aux_axes(transform)

# cax = aux_ax.imshow(data_image, origin='lower', vmin=0, vmax=3, cmap=newcmp, aspect='equal')
# # aux_ax.arrow(x_r1, y_r1, x_r2 - x_r1, y_r2 - y_r1, head_width=50, head_length=50, linewidth=2, color='k', length_includes_head=True)
# aux_ax.plot(x, y, 'o', color='brown', ms=5, alpha=0.4, markerfacecolor='None', markeredgecolor='red', markeredgewidth=0.5)
# aux_ax.plot(x_image, y_image, 'o', color='brown', ms=3, alpha=0.4, markerfacecolor='None', markeredgecolor='k', markeredgewidth=0.5)
# for i in range(len(row_final)):
#    aux_ax.annotate(str(row_final[i]), (x_image[i], y_image[i]), fontsize=7)
# # cbar = fig.colorbar(cax, ax=aux_ax)
# # cbar.set_label(r'$\mathrm{Arcsinh}$')
# axarr.axis["bottom"].set_visible(False)
# axarr.axis["top"].set_visible(False)
# axarr.axis["left"].set_visible(False)
# axarr.axis["right"].set_visible(False)
# # aux_ax.xaxis.set_view_interval(0, 5000, ignore=True)
# # aux_ax.yaxis.set_view_interval(-5000, -3000, ignore=True)
# aux_ax.set_xlim(0, 1000)
# aux_ax.set_ylim(-5000, -3000)

#
# plt.figure(figsize=(10, 5), dpi=300)
# plt.imshow(10 ** data_hb, origin='lower', vmin=0, vmax=3, cmap=newcmp)
# cbar = plt.colorbar()
# # cbar.set_label(r'$\mathrm{Arcsinh}$')
# plt.plot(x, y, 'o', color='brown', ms=7, alpha=0.4, markerfacecolor='None', markeredgecolor='red',
# markeredgewidth=0.5)
# plt.plot(x_image, y_image, 'o', color='brown', ms=5, alpha=0.4, markerfacecolor='None', markeredgecolor='k',
# markeredgewidth=0.5)
# for i in range(len(row_final)):
#     plt.annotate(str(row_final[i]), (x[i], y[i]), fontsize=7)
# plt.xlim(2000, 4000)
# plt.ylim(2500, 4000)



