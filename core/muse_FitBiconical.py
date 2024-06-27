import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import biconical_outflow_model_3d as bicone
from astropy.io import ascii
from matplotlib import rc
from scipy.integrate import simps
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import PixCoord, RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian2DKernel
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

class DrawBiconeModel:
    def __init__(self, A=0.0, tau=5.00, D=1.0, fn=1.0e3, theta_in_deg=0.0, theta_out_deg=40.0, theta_B1_deg=90,
                 theta_B2_deg=60, theta_B3_deg=0, theta_D1_deg=0, theta_D2_deg=0, theta_D3_deg=0, vmax=300.0,
                 vtype='constant', sampling=100, azim=45, elev=45, map_interpolation='none', obs_res=100, nbins=60,
                 bins=None):
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

        self.GenerateBicone()

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

    def emission_cube(self, pix_qso=None, size_ext=None, size=None, mask=None):
        center_x, center_y = (self.sampling - 1) / 2, (self.sampling - 1) / 2
        pixel_scale = 40 / (self.sampling - 1)  # rescale two pixels coordinate 40 pixel in MUSE = 100 pixel in model

        #
        fgrid = self.fgrid.reshape(self.sampling, self.sampling, self.sampling)
        vgrid = self.vgrid.reshape(self.sampling, self.sampling, self.sampling)
        vgrid = np.flip(np.swapaxes(vgrid, 1, 2), 2)
        fgrid = np.flip(np.swapaxes(fgrid, 1, 2), 2)
        flux_model = np.zeros((size_ext, *size))
        # vmap_MUSE = np.zeros(size) * np.nan


        for i in range(size[0]):
            for j in range(size[1]):
                if mask[j, i]:
                    i_1, i_2 = int((i - pix_qso[0]) / pixel_scale + center_x), \
                               int((i + 1 - pix_qso[0]) / pixel_scale + center_x)
                    j_1, j_2 = int((j - pix_qso[1]) / pixel_scale + center_y), \
                               int((j + 1 - pix_qso[1]) / pixel_scale + center_y)
                    v_xy_ij = vgrid[:, j_1:j_2, i_1:i_2]
                    f_xy_ij = fgrid[:, j_1:j_2, i_1:i_2]


                    # Remove NaNs
                    v_xy = v_xy_ij[f_xy_ij > 0]
                    f_xy = f_xy_ij[f_xy_ij > 0]
                    if (len(v_xy) > 0) and (len(f_xy) > 0):
                        # vmap_MUSE[j, i] = simps(v_xy.flatten() * f_xy.flatten()) / simps(f_xy.flatten())


                        v_hist, v_edges = np.histogram(v_xy, bins=self.bins, weights=f_xy)
                        flux_model[:, j, i] = v_hist / v_hist.max()

        # return flux_model, vmap_MUSE
        return flux_model

# # remap the vmap to the observed grid
# coord_MUSE = (67, 80)
#
# # Define the pixel coordinates
# func = DrawBiconeModel()
# func.Make2Dmap()
# # func.EmissionModel()
# lambda_mid, v_hist = func.emission_pixel(coord_MUSE=coord_MUSE)
# flux_model_cube = func.emission_cube()
#
# # Save the results
# path_test = '../../MUSEQuBES+CUBS/fit_kin/Biconical_cube_test.fits'
# hdul_test = fits.ImageHDU(flux_model_cube, header=None)
# hdul_test.writeto(path_test, overwrite=True)
#
# # chi2, chi2_all = likelihood(flux_OIII, flux_err_OIII, flux_model_cube)
# # print(chi2_all)
#
# # plt.close('all')
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 5))
# ax.plot(wave_OIII_vac, flux_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max(),
#         '-k', drawstyle='steps-mid')
# ax.plot(wave_OIII_vac, flux_err_OIII[:, coord_MUSE[1], coord_MUSE[0]] / flux_OIII[:, coord_MUSE[1], coord_MUSE[0]].max()
#         , '-C0', drawstyle='steps-mid')
# ax.plot(lambda_mid, v_hist, '-r', drawstyle='steps-mid')
# ax.plot(lambda_mid, flux_model_cube[:, coord_MUSE[1], coord_MUSE[0]], '-b', drawstyle='steps-mid')
# ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
# ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size = 20)
# # ax.set_xlim(wave_OIII_vac.min(), wave_OIII_vac.max())
# fig.savefig('../../MUSEQuBES+CUBS/fit_kin/3C57_cone_flux.png', bbox_inches='tight')
# # plt.show()
#
