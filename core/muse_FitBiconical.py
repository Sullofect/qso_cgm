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
c_kms = 2.998e5
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
wave_Hbeta_vac = 4862.721
wave_OIII5008_vac = 5008.239

# QSO information
cubename = '3C57'
str_zap = ''
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')
data_qso = data_qso[data_qso['name'] == cubename]
ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

# Measure kinematics
path_fit = '../../MUSEQuBES+CUBS/fit_kin/3C57_fit_OII+OIII_True_3728_1.5_gauss_None_None.fits'
hdul = fits.open(path_fit)
fs, hdr = hdul[1].data, hdul[2].header
# v, z, dz = hdul[2].data, hdul[3].data, hdul[4].data
# sigma, dsigma = hdul[5].data, hdul[6].data
# flux_OII_fit, dflux_OII_fit = hdul[7].data, hdul[8].data
# flux_OIII_fit, dflux_OIII_fit = hdul[9].data, hdul[10].data
# r, dr = hdul[11].data, hdul[12].data
# a_OII, da_OII = hdul[13].data, hdul[14].data
# a_OIII, da_OIII = hdul[17].data, hdul[18].data
# b_OII, db_OII = hdul[15].data, hdul[16].data
# b_OIII, db_OIII = hdul[19].data, hdul[20].data

# Hedaer information
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
UseDataSeg = (1.5, 'gauss', None, None)
UseDetectionSeg = (1.5, 'gauss', 1.5, 'gauss')
UseSmoothedCubes = True
line_OII, line_OIII = 'OII', 'OIII'
path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OII, *UseDetectionSeg)
path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
    format(cubename, str_zap, line_OIII, *UseDetectionSeg)
path_cube_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
    format(cubename, str_zap, line_OII)
path_cube_smoothed_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                         '{}_{}_{}.fits'.format(cubename, str_zap, line_OII, *UseDataSeg)
path_cube_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}.fits'. \
    format(cubename, str_zap, line_OIII)
path_cube_smoothed_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_{}_' \
                          '{}_{}_{}.fits'.format(cubename, str_zap, line_OIII, *UseDataSeg)
path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/3C57_V50_plot.fits'
path_w80_plot = '../../MUSEQuBES+CUBS/fit_kin/3C57_W80_plot.fits'
hdul_v50_plot = fits.open(path_v50_plot)
hdul_w80_plot = fits.open(path_w80_plot)
v50, w80 = hdul_v50_plot[1].data, hdul_w80_plot[1].data

# Load data and smoothing
if UseSmoothedCubes:
    cube_OII, cube_OIII = Cube(path_cube_smoothed_OII), Cube(path_cube_smoothed_OIII)
else:
    cube_OII, cube_OIII = Cube(path_cube_OII), Cube(path_cube_OIII)
wave_OII_vac, wave_OIII_vac = pyasl.airtovac2(cube_OII.wave.coord()), pyasl.airtovac2(cube_OIII.wave.coord())
flux_OII, flux_err_OII = cube_OII.data * 1e-3, np.sqrt(cube_OII.var) * 1e-3
flux_OIII, flux_err_OIII = cube_OIII.data * 1e-3, np.sqrt(cube_OIII.var) * 1e-3
seg_3D_OII_ori, seg_3D_OIII_ori = fits.open(path_3Dseg_OII)[0].data, fits.open(path_3Dseg_OIII)[0].data
mask_seg_OII, mask_seg_OIII = np.sum(seg_3D_OII_ori, axis=0), np.sum(seg_3D_OIII_ori, axis=0)
flux_seg_OII, flux_seg_OIII = flux_OII * seg_3D_OII_ori, flux_OIII * seg_3D_OIII_ori
size = np.shape(flux_OIII)[1:]

# Create a bin according to the MUSE data
dlambda = wave_OIII_vac[1] - wave_OIII_vac[0]
wave_OIII_ext = np.zeros(len(wave_OIII_vac) + 20)
wave_OIII_ext[10:-10] = wave_OIII_vac
wave_OIII_ext[:10] = wave_OIII_vac[0] - np.flip(np.arange(1, 11)) * dlambda
wave_OIII_ext[-10:] = wave_OIII_vac[-1] + dlambda * np.arange(1, 11)

#
wave_bin = np.zeros(len(wave_OIII_ext) + 1)
wave_bin[:-1] = wave_OIII_ext - dlambda / 2
wave_bin[-1] = wave_OIII_ext[-1] + dlambda
wave_OIII_obs = wave_OIII5008_vac * (1 + z_qso)
bins = (wave_bin - wave_OIII_obs) / wave_OIII_obs * c_kms
mask = np.zeros(size)
mask[55:100, 54:100] = 1

class DrawBiconeModel:
    def __init__(self, A=0.0, tau=5.00, D=1.0, fn=1.0e3, theta_in_deg=0.0, theta_out_deg=40.0, theta_B1_deg=90,
                 theta_B2_deg=60, theta_B3_deg=0, theta_D1_deg=0, theta_D2_deg=0, theta_D3_deg=0, vmax=300.0,
                 vtype='constant', sampling=100, azim=45, elev=45, map_interpolation='none', obs_res=100, nbins=60,
                 bins=bins):
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
        if int(sampling) % 2 == 0:  # Having an odd sampling ensures that there is a value at (0,0,0)
            self.sampling = int(sampling) + 1

        # 3d Plot orientation
        self.azim = azim
        self.elev = elev

        # 2d Map options
        self.map_interpolation = map_interpolation
        self.obs_res = obs_res  # resolution of SDSS for emission line model
        self.nbins = nbins   # number of bins for emission line histogram
        self.bins = bins
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

    def EmissionModel(self):
        # Emission model
        emmap, emline = bicone.emission_model(self.fgrid, self.vgrid, vmax=self.vmax, obs_res=self.obs_res,
                                              nbins=self.nbins, sampling=self.sampling,
                                              plot=True, save_fig=True)
        self.emmap = emmap
        self.emline = emline

    def emission_pixel(self, coord_MUSE=None):
        X_sample, Y_sample = np.meshgrid(np.arange(self.sampling), np.arange(self.sampling))
        X_sample, Y_sample = X_sample.ravel(), Y_sample.ravel()
        pixcoord_sample = PixCoord(x=X_sample, y=Y_sample)

        #
        center_x, center_y = (self.sampling - 1) / 2, (self.sampling - 1) / 2
        pixel_scale = 40 / (self.sampling - 1)
        coord_sample = int(coord_MUSE[0] - c2[0]) / pixel_scale + center_x, \
                       int(coord_MUSE[1] - c2[1]) / pixel_scale + center_y
        rect_sample = RectanglePixelRegion(center=PixCoord(x=coord_sample[0], y=coord_sample[1]),
                                           width=1 / pixel_scale, height=1 / pixel_scale)
        mask_sample = rect_sample.contains(pixcoord_sample)

        # Reshape grids into cubes
        mask_sample = mask_sample.reshape(self.sampling, self.sampling)
        fgrid = self.fgrid.reshape(self.sampling, self.sampling, self.sampling)
        vgrid = self.vgrid.reshape(self.sampling, self.sampling, self.sampling)

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

        v_hist, v_edges = np.histogram(v_xy, bins=self.bins, weights=f_xy)

        return wave_OIII_ext, v_hist / v_hist.max()

    def emission_cube(self):
        center_x, center_y = (self.sampling - 1) / 2, (self.sampling - 1) / 2
        pixel_scale = 40 / (self.sampling - 1)
        fgrid = self.fgrid.reshape(self.sampling, self.sampling, self.sampling)
        vgrid = self.vgrid.reshape(self.sampling, self.sampling, self.sampling)
        vgrid = np.flip(np.swapaxes(vgrid, 1, 2), 2)
        fgrid = np.flip(np.swapaxes(fgrid, 1, 2), 2)
        flux_model = np.zeros((len(wave_OIII_ext), *size))

        for i in range(size[0]):
            for j in range(size[1]):
                # if ~np.isnan(v50[j, i]):
                if mask[j, i]:
                    i_1, i_2 = int((i - c2[0]) / pixel_scale + center_x), int((i + 1 - c2[0]) / pixel_scale + center_x)
                    j_1, j_2 = int((j - c2[1]) / pixel_scale + center_y), int((j + 1 - c2[1]) / pixel_scale + center_y)
                    print(i_1, i_2)
                    print(j_1, j_2)
                    v_xy_ij = vgrid[:, j_1:j_2, i_1:i_2]
                    f_xy_ij = fgrid[:, j_1:j_2, i_1:i_2]

                    # Remove NaNs
                    v_xy = v_xy_ij[f_xy_ij > 0]
                    f_xy = f_xy_ij[f_xy_ij > 0]
                    if (len(v_xy) > 0) and (len(f_xy) > 0):
                        v_hist, v_edges = np.histogram(v_xy, bins=self.bins, weights=f_xy)
                        flux_model[:, j, i] = v_hist / v_hist.max()

        return flux_model



# remap the vmap to the observed grid
coord_MUSE = (67, 80)

# Define the pixel coordinates
func = DrawBiconeModel()
func.GenerateBicone()
func.Make2Dmap()
# func.EmissionModel()
lambda_mid, v_hist = func.emission_pixel(coord_MUSE=coord_MUSE)
flux_model_cube = func.emission_cube()

# Save the results
path_test = '../../MUSEQuBES+CUBS/fit_kin/Biconical_cube_test.fits'
hdul_test = fits.ImageHDU(flux_model_cube, header=None)
hdul_test.writeto(path_test, overwrite=True)

# plt.close('all')
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
ax.plot(wave_OIII_vac, flux_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max(),
        '-k', drawstyle='steps-mid')
ax.plot(wave_OIII_vac, flux_err_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max()
        , '-C0', drawstyle='steps-mid')
ax.plot(lambda_mid, v_hist, '-r', drawstyle='steps-mid')
ax.plot(lambda_mid, flux_model_cube[:, coord_MUSE[1], coord_MUSE[0]], '-b', drawstyle='steps-mid')
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size = 20)
# ax.set_xlim(wave_OIII_vac.min(), wave_OIII_vac.max())
fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_cone_flux.png', bbox_inches='tight')
# plt.show()

