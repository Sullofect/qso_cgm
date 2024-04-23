import os
import aplpy
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
from palettable.scientific.sequential import Acton_6
from palettable.cubehelix import red_16
from palettable.cmocean.sequential import Dense_20_r
from scipy.ndimage import rotate
from astropy.table import Table
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


def APLpyStyle(gc, type=None, cubename=None, ra_qso=None, dec_qso=None, z_qso=None, name_gal='NGC 3945', dis_gal=None):
    scale_phy_3C57 = 30 * 50 / 7
    scale = np.pi * dis_gal * 1 / 3600 / 180 * 1e3
    width_gal = np.round(scale_phy_3C57 / scale, 0)
    if np.isnan(dis_gal):
        gc.recenter(ra_qso, dec_qso, width=1500 / 3600, height=1500 / 3600)
    else:
        gc.recenter(ra_qso, dec_qso, width=width_gal / 3600, height=width_gal / 3600)
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey', edgecolors='k',
                    linewidths=0.5, s=600, zorder=100)
    gc.set_system_latex(True)

    # Colorbar
    gc.add_colorbar()
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.0)
    gc.colorbar.set_font(size=20)
    gc.colorbar.set_axis_label_font(size=20)
    if type == 'NarrowBand':
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')
        gc.add_scalebar(length=7 * u.arcsecond)
        gc.scalebar.set_corner('top left')
        gc.scalebar.set_label(r"$6'' \approx 50 \mathrm{\; kpc}$")
        gc.scalebar.set_font_size(35)
        gc.add_label(0.98, 0.94, cubename, size=35, relative=True, horizontalalignment='right')
        gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'FieldImage':
        gc.colorbar.hide()
    elif type == 'GasMap':
        gc.colorbar.set_ticks([-300, -200, -100, 0, 100, 200, 300])
        gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
        # gc.colorbar.hide()
        gc.add_label(0.98, 0.94, name_gal, size=35, relative=True, horizontalalignment='right')
        # gc.add_label(0.98, 0.87, r'$z={}$'.format(z_qso), size=35, relative=True, horizontalalignment='right')
    elif type == 'GasMap_sigma':
        # gc.colorbar.set_ticks([25, 50, 75, 100, 125, 150, 175])
        gc.colorbar.set_axis_label_text(r'$\sigma \mathrm{\; [km \, s^{-1}]}$')
    else:
        gc.colorbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        gc.colorbar.set_axis_label_text(r'$\rm log([O \, III]/[O \, II])$')

    # Scale bar
    # gc.add_scalebar(length=3 * u.arcsecond)
    gc.add_scalebar(length=50 / scale * u.arcsecond)
    # gc.scalebar.set_corner('top left')
    # gc.scalebar.set_label(r"$450'' \approx \,$" + '{:.0f}'.format(450 * scale) + r"$\mathrm{\; pkpc}$")
    # gc.scalebar.set_label(r"$3'' \approx 20 \mathrm{\; pkpc}$")
    gc.scalebar.set_label('{:.0f}'.format(50 / scale) +  r"$'' \approx 50 \mathrm{\; pkpc}$")
    # gc.scalebar.set_font_size(30)

    # Hide
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc.ticks.set_length(30)

    # Label
    # xw, yw = gc.pixel2world(146, 140)  # original figure
    xw, yw = gc.pixel2world(140, 140)
    # gc.show_arrows(xw, yw, -0.000035 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000035 * yw, color='k')
    # xw, yw = 40.1333130960119, -18.864847747328896
    # gc.show_arrows(xw, yw, -0.000020 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.000020 * yw, color='k')
    # gc.add_label(0.9778, 0.81, r'N', size=20, relative=True)
    # gc.add_label(0.88, 0.70, r'E', size=20, relative=True)

# load
gal_name = 'NGC5582'
path_table_gals = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/table_gals.fits'
table_gals = fits.open(path_table_gals)[1].data
gal_name_ = gal_name.replace('C', 'C ')
name_sort = table_gals['Object Name'] == gal_name_
ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
v_sys_gal = table_gals[name_sort]['cz (Velocity)']

# NGC3945
path_ETG = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(gal_name)
path_ETG_new = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1_new.fits'.format(gal_name)
path_ETG_mom2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom2.fits'.format(gal_name)
path_ETG_cube = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/allcubes/{}_cube.fits'.format(gal_name)
path_figure_mom1 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_mom1.png'.format(gal_name)
path_figure_mom2 = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_mom2.png'.format(gal_name)
path_figure_spec = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_spec.png'.format(gal_name)

# Load the kinematic map
hdul_ETG = fits.open(path_ETG)
hdr_ETG = hdul_ETG[0].header
hdr_ETG['NAXIS'] = 2
hdr_ETG.remove('NAXIS3')
hdr_ETG.remove('CTYPE3')
hdr_ETG.remove('CDELT3')
hdr_ETG.remove('CRPIX3')
hdr_ETG.remove('CRVAL3')
v_ETG = hdul_ETG[0].data[0, :, :] - v_sys_gal
hdul_ETG_new = fits.ImageHDU(v_ETG, header=hdr_ETG)
hdul_ETG_new.writeto(path_ETG_new, overwrite=True)

# Load the cube
hdul_ETG_cube = fits.open(path_ETG_cube)
hdr_ETG_cube = hdul_ETG_cube[0].header
flux = hdul_ETG_cube[0].data
flux = np.where(~np.isnan(v_ETG)[np.newaxis, :, :], flux, np.nan)


v_array = np.arange(hdr_ETG_cube['CRVAL3'], hdr_ETG_cube['CRVAL3'] + flux.shape[0] * hdr_ETG_cube['CDELT3'],
                    hdr_ETG_cube['CDELT3']) / 1e3  - v_sys_gal  # Convert from m/s to km/s, # and then shift wrt v_gal
# print(v_array)
mask_v1 = (v_array < 0) * (v_array > -350)
mask_v2 = (v_array < 250) * (v_array > 0)
mask_v3 = (v_array < 100) * (v_array > -150)

v_1 = v_array[mask_v1]
v_2 = v_array[mask_v2]
v_3 = v_array[mask_v3]
flux_1 = flux[mask_v1, :, :]
flux_2 = flux[mask_v2, :, :]
flux_3 = flux[mask_v3, :, :]
# print(np.shape(flux_1), np.shape(v_2))

# Moments
flux_cumsum_1 = np.cumsum(flux_1, axis=0)
flux_cumsum_1 /= flux_cumsum_1.max(axis=0)
v_1_array = np.zeros_like(flux_1)
v_1_array[:] = v_1[:, np.newaxis, np.newaxis]

v_10_1 = np.take_along_axis(v_1_array, np.argmin(np.abs(flux_cumsum_1 - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
v_50_1 = np.take_along_axis(v_1_array, np.argmin(np.abs(flux_cumsum_1 - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
v_90_1 = np.take_along_axis(v_1_array, np.argmin(np.abs(flux_cumsum_1 - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
sigma_1 = v_90_1 - v_10_1
sigma_1 /= 2.563  # W_80 = 2.563sigma

#
flux_cumsum_2 = np.cumsum(flux_2, axis=0)
flux_cumsum_2 /= flux_cumsum_2.max(axis=0)
v_2_array = np.zeros_like(flux_2)
v_2_array[:] = v_2[:, np.newaxis, np.newaxis]

v_10_2 = np.take_along_axis(v_2_array, np.argmin(np.abs(flux_cumsum_2 - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
v_50_2 = np.take_along_axis(v_2_array, np.argmin(np.abs(flux_cumsum_2 - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
v_90_2 = np.take_along_axis(v_2_array, np.argmin(np.abs(flux_cumsum_2 - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
sigma_2 = v_90_2 - v_10_2
sigma_2 /= 2.563  # W_80 = 2.563sigma

flux_cumsum_3 = np.cumsum(flux_3, axis=0)
flux_cumsum_3 /= flux_cumsum_3.max(axis=0)
v_3_array = np.zeros_like(flux_3)
v_3_array[:] = v_3[:, np.newaxis, np.newaxis]

v_10_3 = np.take_along_axis(v_3_array, np.argmin(np.abs(flux_cumsum_3 - 0.10), axis=0)[np.newaxis, :, :], axis=0)[0]
v_50_3 = np.take_along_axis(v_3_array, np.argmin(np.abs(flux_cumsum_3 - 0.50), axis=0)[np.newaxis, :, :], axis=0)[0]
v_90_3 = np.take_along_axis(v_3_array, np.argmin(np.abs(flux_cumsum_3 - 0.90), axis=0)[np.newaxis, :, :], axis=0)[0]
sigma_3 = v_90_3 - v_10_3
sigma_3 /= 2.563  # W_80 = 2.563sigma

#
sigma = np.where(v_ETG <= 0, sigma_1, sigma_2)
sigma = np.where((v_ETG > 50) | (v_ETG < -50), sigma, sigma_3)
sigma = np.where(~np.isnan(v_ETG), sigma, np.nan)

# Guess the velocity field and compare with the original one
v_guess = np.where(v_ETG <= 0, v_50_1, v_50_2)
v_guess = np.where((v_ETG > 50) | (v_ETG < -50), v_guess, v_50_3)
v_guess = np.where(~np.isnan(v_ETG), v_guess, np.nan)
v_rot = np.where(~np.isnan(v_guess), v_guess, -1)
v_rot = rotate(v_rot, 45)
v_rot = np.where(v_rot != -1, v_rot, np.nan)

path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_fit.fits'.format(gal_name)
hdul_fit = fits.open(path_fit)
v_fit, sigma_fit = hdul_fit[1].data, hdul_fit[3].data


# Save the moment 1 and moment 2 maps
sigma_fit = np.where(~np.isnan(v_ETG), sigma_fit, np.nan)
hdul_ETG_sigma = fits.ImageHDU(sigma_fit, header=hdr_ETG)
hdul_ETG_sigma.writeto(path_ETG_mom2, overwrite=True)

#
# fig = plt.figure(figsize=(10, 5), dpi=300)
# plt.plot(v_array, flux[:, 183, 174], 'k')
# plt.xlabel('Velocity [km/s]')
# plt.ylabel('Flux [arbitrary unit]')
# fig.savefig(path_figure_spec, bbox_inches='tight')


# Plot the kinematic map
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_ETG_new, figure=fig, hdu=1)
gc.show_colorscale(vmin=-350, vmax=350, cmap='coolwarm')
APLpyStyle(gc, type='GasMap', ra_qso=ra_gal, dec_qso=dec_gal, name_gal=gal_name, dis_gal=33.791)
fig.savefig(path_figure_mom1, bbox_inches='tight')

# Plot the sigma map
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure(path_ETG_mom2, figure=fig, hdu=1)
gc.show_colorscale(vmin=0, vmax=300, cmap=Dense_20_r.mpl_colormap)
APLpyStyle(gc, type='GasMap_sigma', ra_qso=ra_gal, dec_qso=dec_gal, name_gal=gal_name, dis_gal=33.791)
fig.savefig(path_figure_mom2, bbox_inches='tight')

# testing
# plt.figure(figsize=(5, 5))
# plt.imshow(v_fit, origin='lower', cmap='coolwarm', vmin=-350, vmax=350)
# plt.colorbar()
# plt.show()

raise ValueError('Stop here')
#
# plt.figure(figsize=(8, 8))
# plt.imshow(data[0, :, :], origin='lower', cmap='RdBu_r', vmin=-200, vmax=200)
# plt.show()

gal_list = ['NGC2594', 'NGC2685', 'NGC2764', 'NGC3619', 'NGC3626', 'NGC3838', 'NGC3941',
            'NGC3945', 'NGC4203', 'NGC4262', 'NGC5173', 'NGC5582', 'NGC5631', 'NGC6798',
            'UGC06176', 'UGC09519']
dis_list = [35.1, 13.05, 37.40, 31.967, 17.755, 23.5, 11.816,
            23.400, 18.836, 19.741, 38.000, 33.791, 23.933, 37.5,
            40.1, 27.6]  # in Mpc

def PlotEachGal(gals, dis):
    # path_table_gals = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/table_gals.fits'
    # table_gals = fits.open(path_table_gals)[1].data
    for ind, i in enumerate(gals):
        # fix missed name
        if i == 'NGC2594':
            i_cube = 'NGC2592'
        elif i == 'NGC3619':
            i_cube = 'NGC3613'
        else:
            i_cube = i

        #
        name_i = i.replace('C', 'C ')
        name_sort = table_gals['Object Name'] == name_i


        # Galaxy information
        ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
        v_sys_gal = table_gals[name_sort]['cz (Velocity)']
        print(i, v_sys_gal)

        path_ETG_i = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(i_cube)
        path_ETG_new_i = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1_new.fits'.format(i_cube)
        path_figure_mom1_i = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/plots/{}_mom1_noframe.png'.format(i)
        hdul_ETG = fits.open(path_ETG_i)
        hdr_ETG = hdul_ETG[0].header
        hdr_ETG['NAXIS'] = 2
        hdr_ETG.remove('NAXIS3')
        hdr_ETG.remove('CTYPE3')
        hdr_ETG.remove('CDELT3')
        hdr_ETG.remove('CRPIX3')
        hdr_ETG.remove('CRVAL3')
        v_ETG = hdul_ETG[0].data[0, :, :] - v_sys_gal
        hdul_ETG_new = fits.ImageHDU(v_ETG, header=hdr_ETG)
        hdul_ETG_new.writeto(path_ETG_new_i, overwrite=True)

        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_ETG_new_i, figure=fig, hdu=1)
        gc.show_colorscale(vmin=-350, vmax=350, cmap='coolwarm')
        gc.frame.set_color('white')
        APLpyStyle(gc, type='GasMap', ra_qso=ra_gal, dec_qso=dec_gal, name_gal=name_i, dis_gal=dis[ind])
        fig.savefig(path_figure_mom1_i, bbox_inches='tight')

# PlotEachGal(gal_list, dis_list)


# Place sudo slit
def PlaceSudoSlitOnEachGal(gals, dis):
    for ind, i in enumerate(gals):
        # fix missed name
        if i == 'NGC2594':
            i_cube = 'NGC2592'
        elif i == 'NGC3619':
            i_cube = 'NGC3613'
        else:
            i_cube = i

        # Galaxy information
        name_i = i.replace('C', 'C ')
        name_sort = table_gals['Object Name'] == name_i
        ra_gal, dec_gal = table_gals[name_sort]['RA'], table_gals[name_sort]['Dec']
        v_sys_gal = table_gals[name_sort]['cz (Velocity)']
        print(i, v_sys_gal)

        #
        path_Serra = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_mom1/{}_mom1.fits'.format(i_cube)
        hdul_Serra = fits.open(path_Serra)
        hdr_Serra = hdul_Serra[0].header
        v_Serra = hdul_Serra[0].data[0, :, :] - v_sys_gal

        #
        path_fit = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/Serra2012_Atlas3D_Paper13/all_fit/{}_fit.fits'.format(i)
        hdul_fit = fits.open(path_fit)
        v_fit, sigma_fit = hdul_fit[1].data, hdul_fit[3].data

        #
        w = WCS(hdr_Serra, naxis=2)
        center_gal = SkyCoord(ra_gal[0], dec_gal[0], unit='deg', frame='icrs')
        c_gal = w.world_to_pixel(center_gal)
        x_gal, y_gal = np.meshgrid(np.arange(v_ETG.shape[0]), np.arange(v_ETG.shape[1]))
        x_gal, y_gal = x_gal.flatten(), y_gal.flatten()
        pixcoord_gal = PixCoord(x=x_gal, y=y_gal)

        # mask a slit
        rectangle_gal = RectanglePixelRegion(center=PixCoord(x=c_gal[0], y=c_gal[1]), width=60, height=5,
                                             angle=Angle(-60, 'deg'))
        mask_gal = rectangle_gal.contains(pixcoord_gal)
        dis_gal = np.sqrt((x_gal - c_gal[0]) ** 2 + (y_gal - c_gal[1]) ** 2) * 2.77777798786E-03 * 3600 * 50 / 305
        dis_mask_gal = dis_gal[mask_gal]

        # mask each side
        red_gal = ((x_gal[mask_gal] - c_gal[0]) < 0) * ((y_gal[mask_gal] - c_gal[1]) > 0)
        blue_gal = ~red_gal
        dis_red_gal = dis_mask_gal[red_gal]
        dis_blue_gal = dis_mask_gal[blue_gal] * -1