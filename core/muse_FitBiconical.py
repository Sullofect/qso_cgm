import os
import aplpy
import lmfit
import numpy as np
import matplotlib as mpl
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
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# QSO information
cubename = '3C57'
str_zap = ''
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

# Measure kinematics
path_fit_N1 = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N1.fits'
path_fit_N2 = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
path_fit_N3 = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None_N3.fits'

# N1
hdul_N1 = fits.open(path_fit_N1)
fs_N1, hdr_N1 = hdul_N1[1].data, hdul_N1[2].header
v_N1, z_N1, dz_N1 = hdul_N1[2].data, hdul_N1[3].data, hdul_N1[4].data
sigma_N1, dsigma_N1 = hdul_N1[5].data, hdul_N1[6].data
flux_OIII_N1 = hdul_N1[9].data
chisqr_N1, redchi_N1 = hdul_N1[15].data, hdul_N1[16].data

# Hedaer information
hdr = hdr_N1
path_sub_white_gaia = '../../MUSEQuBES+CUBS/fit_kin/{}{}_WCS_subcube.fits'.format(cubename, str_zap)
hdr_sub_gaia = fits.open(path_sub_white_gaia)[1].header
hdr['CRVAL1'] = hdr_sub_gaia['CRVAL1']
hdr['CRVAL2'] = hdr_sub_gaia['CRVAL2']
hdr['CRPIX1'] = hdr_sub_gaia['CRPIX1']
hdr['CRPIX2'] = hdr_sub_gaia['CRPIX2']
hdr['CD1_1'] = hdr_sub_gaia['CD1_1']
hdr['CD2_1'] = hdr_sub_gaia['CD2_1']
hdr['CD1_2'] = hdr_sub_gaia['CD1_2']
hdr['CD2_2'] = hdr_sub_gaia['CD2_2']
w = WCS(hdr, naxis=2)
center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
c2 = w.world_to_pixel(center_qso)

# Path to the data
UseDetectionSeg = (1.5, 'gauss', 1.5, 'gauss')
UseSmoothedCubes = True
line_OII, line_OIII = 'OII', 'OIII'
path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OIII, *UseDetectionSeg)
path_SB_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_SB_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OIII, *UseDetectionSeg)
path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
    format(cubename, str_zap, line_OII)
path_cube_smoothed_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                         '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
    format(cubename, str_zap, line_OIII)
path_cube_smoothed_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                          '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDetectionSeg)
# Load data and smoothing
if UseSmoothedCubes:
    cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
else:
    cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(
    cube_OIII.wave.coord())
flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori

class DrawBiconeModel:
    def __init__(self, A=0.00, tau=5.00, D=1.0, fn=1.0e3, theta_in_deg=10.0, theta_out_deg=50.0, theta_B1_deg=10,
                 theta_B2_deg=60, theta_B3_deg=0, theta_D1_deg=0, theta_D2_deg=0, theta_D3_deg=0, vmax=1000.0,
                 vtype='increasing', sampling=100, azim=45, elev=45, map_interpolation='none', obs_res=20, nbins=60):
        # Model Parameters
        self.A = A  # dust extinction level (0.0 - 1.0)
        self.tau = tau  # shape of flux profile
        self.D = D  # length of bicone (arbitrary units)
        self.fn = fn  # initial flux value at center

        # Bicone Parameters
        self.theta_in_deg = theta_in_deg # inner opening angle of bicone
        self.theta_out_deg = theta_out_deg # outer opening angle of bicone

        # # Bicone inclination and PA
        self.theta_B1_deg = theta_B1_deg  # rotation along x
        self.theta_B2_deg = theta_B2_deg  # rotation along y
        self.theta_B3_deg = theta_B3_deg  # rotation along z

        # Dust plane inclination and PA
        self.theta_D1_deg = theta_D1_deg  # rotation along x
        self.theta_D2_deg = theta_D2_deg  # rotation along y
        self.theta_D3_deg = theta_D3_deg  # rotation along z

        # Velocity Parameters
        self.vmax = vmax  # km/s
        self.vtype = vtype  # 'increasing','decreasing', or 'constant'
        self.sampling = sampling  # point sampling

        # 3d Plot orientation
        self.azim = azim
        self.elev = elev

        # 2d Map options
        self.map_interpolation = map_interpolation
        self.obs_res = obs_res  # resolution of SDSS for emission line model
        self.nbins = nbins   # number of bins for emission line histogram

    def GenerateBicone(self):
        # Bicone coordinate, flux, and velocity grids
        xbgrid, ybgrid, zbgrid, fgrid, vgrid = bicone.generate_bicone(self.theta_in_deg, self.theta_out_deg,
                                                                      self.theta_B1_deg, self.theta_B2_deg,
                                                                      self.theta_B3_deg, self.theta_D1_deg,
                                                                      self.theta_D2_deg, self.theta_D3_deg,
                                                                      D=self.D, tau=self.tau, fn=self.fn, A=self.A,
                                                                      vmax=self.vmax, vtype=self.vtype,
                                                                      sampling=self.sampling, plot=True,
                                                                      orientation=(self.azim, self.elev),
                                                                      save_fig=True)
        self.xbgrid = xbgrid
        self.ybgrid = ybgrid
        self.zbgrid = zbgrid
        self.fgrid = fgrid
        self.vgrid = vgrid

    def Make2Dmap(self):
        # 2d map
        fmap, vmap, dmap, v_int, d_int = bicone.map_2d(self.xbgrid, self.ybgrid, self.zbgrid, self.fgrid, self.vgrid,
                                                       D=self.D, sampling=self.sampling, interpolation=self.map_interpolation,
                                                       plot=True, save_fig=True)
        self.fmap = fmap
        self.vmap = vmap
        self.dmap = dmap
        self.v_int = v_int
        self.d_int = d_int

    def emission_pixel(fgrid, vgrid, vmax, coord_MUSE=None, nbins=25, sampling=100, z=z_qso):
        global c2
        # Having an odd sampling ensures that there is a value at (0,0,0)
        if int(sampling) % 2 == 0:
            sampling = int(sampling) + 1

        X_sample, Y_sample = np.meshgrid(np.arange(sampling), np.arange(sampling))
        X_sample, Y_sample = X_sample.ravel(), Y_sample.ravel()
        pixcoord_sample = PixCoord(x=X_sample, y=Y_sample)

        #
        center_x, center_y = (sampling - 1) / 2, (sampling - 1) / 2
        pixel_scale = 40 / (sampling - 1)
        coord_sample = int(coord_MUSE[0] - c2[0]) / pixel_scale + center_x, \
                       int(coord_MUSE[1] - c2[1]) / pixel_scale + center_y
        rect_sample = RectanglePixelRegion(center=PixCoord(x=coord_sample[0], y=coord_sample[1]),
                                           width=1 / pixel_scale, height=1 / pixel_scale)
        mask_sample = rect_sample.contains(pixcoord_sample)

        # fig, ax = plt.subplots(1, 1)
        # # plt.imshow(v_N1[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
        # ax.imshow(np.flip(vmap, 1), cmap='coolwarm', vmin=-300, vmax=300, origin='lower')
        # patch = rect_sample.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
        # ax.plot(c2[0], c2[1], '*', markersize=15)
        # fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_cone_test.png', bbox_inches='tight')

        # Reshape grids into cubes
        mask_sample = mask_sample.reshape(sampling, sampling)
        fgrid = fgrid.reshape(sampling, sampling, sampling)
        vgrid = vgrid.reshape(sampling, sampling, sampling)
        bins = np.linspace(-vmax, vmax, nbins)

        # Convolve with seeing
        # kernel = Gaussian2DKernel(x_stddev=0.5 / pixel_scale, y_stddev=0.5 / pixel_scale,
        #                           x_size=9, y_size=9)
        # kernel_1 = Kernel(kernel.array[np.newaxis, :, :])
        # fgrid = convolve(fgrid, kernel_1)

        # find specific pixel
        v_xy = np.where(mask_sample[np.newaxis, :, :], np.flip(np.swapaxes(vgrid, 1, 2), 2), np.nan)
        f_xy = np.where(mask_sample[np.newaxis, :, :], np.flip(np.swapaxes(fgrid, 1, 2), 2), np.nan)

        # Remove NaNs
        v_xy = v_xy.ravel()
        f_xy = f_xy.ravel()
        v_xy = v_xy[f_xy > 0]
        f_xy = f_xy[f_xy > 0]
        # print(v_xy, f_xy)

        v_hist, v_edges = np.histogram(v_xy, bins=bins, weights=f_xy)
        v_mid = (v_edges[1:] + v_edges[:-1]) / 2

        # convert x in km/s to angstroms
        c = 299792.  # speed of light in km/s
        cw = 5008.239 * (1 + z)  # central wavelength; [OIII]5007 (SDSS)
        lambda_mid = cw + (v_mid * cw) / c
        return lambda_mid, v_hist / v_hist.max()

    # fmap, vmap, dmap, v_int, d_int = bicone.map_2d(xbgrid, ybgrid, zbgrid, fgrid, vgrid,
    #                                                D=D, sampling=sampling, interpolation=map_interpolation,
    #                                                plot=True, save_fig=True)
    # x, emline = bicone.emission_model(fgrid, vgrid, vmax=vmax, obs_res=obs_res, nbins=nbins, sampling=sampling,
    #                                   plot=True, save_fig=True)
    # return fgrid, vgrid, vmax

# fgrid, vgrid, vmax = DrawBiconeModel()


# remap the vmap to the observed grid
coord_MUSE = (60, 80)

# Plot the cone
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 8))
# # ax.imshow(v_N1[0, :, :], origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
# ax.imshow(np.flip(vmap, 1), cmap='coolwarm', extent=[c2[0] - 20, c2[0] + 20, c2[1] - 20, c2[1] + 20],
#           vmin=-300, vmax=300, origin='lower')
# ax.plot(c2[0], c2[1], '*', markersize=15)
# ax.set_xlim(0, 150)
# ax.set_ylim(0, 150)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_sudo_cone.png', bbox_inches='tight')




# Define the pixel coordinates
lambda_mid, v_hist = emission_pixel(fgrid, vgrid, coord_MUSE=coord_MUSE, vmax=vmax, nbins=nbins, sampling=sampling)

# plt.close('all')
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
ax.plot(wave_OIII_vac, flux_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max(),
        '-k', drawstyle='steps-mid')
ax.plot(wave_OIII_vac, flux_err_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max()
        , '-C0', drawstyle='steps-mid')
ax.plot(lambda_mid, v_hist, '-r', drawstyle='steps-mid')
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size = 20)
# ax.set_xlim(wave_OIII_vac.min(), wave_OIII_vac.max())
fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_cone_flux.png', bbox_inches='tight')
# plt.show()

# VVD
# v1, v2 = v[0, :, :], v[1, :, :]
# sigma1, sigma2 = sigma[0, :, :], sigma[1, :, :]
# v1 = np.where(v2 != 0, v1, 0)
# sigma1 = np.where(sigma2 != 0, sigma1, 0)
#
# #
# fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
# # ax.plot(v[0, :, :].flatten(), sigma[0, :, :].flatten(), 'k.')
# ax.plot(v1.flatten(), sigma1.flatten(), 'k.', ms=3)
# ax.plot(v2.flatten(), sigma2.flatten(), 'r.', ms=3)
# ax.set_xlim(-500, 500)
# ax.set_ylim(0, 500)
# ax.set_xlabel('V (km/s)')
# ax.set_ylabel(r'$\sigma$ (km/s)')
# fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_VVD_profile.png', bbox_inches='tight')

