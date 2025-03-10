import os
import aplpy
import numpy as np
import matplotlib as mpl
import gala.potential as gp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy import units as u
from astropy import stats
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.convolution import convolve, Kernel, Gaussian2DKernel
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
import biconical_outflow_model_3d as bicone
from mpdaf.obj import Cube, WaveCoord, Image
from PyAstronomy import pyasl
from muse_kin_ETP import PlaceSudoSlitOnEachGal
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

class PlotV50W80Profile:
    def __init__(self, cubename_list=None, width_list=None, height_list=None, angle_list=None, marker_list=None):
        self.cubename_list = cubename_list
        self.width_list = width_list
        self.height_list = height_list
        self.angle_list = angle_list
        self.marker_list = marker_list


    def Plot(self):
        # Giant plot
        # fig, ax = plt.subplots(2, 1, figsize=(15, 7), dpi=300, sharex=True)
        # fig.subplots_adjust(hspace=0.0)
        #
        # for i in range(len(self.cubename_list)):
        #     color_i = 'C{}'.format(i)
        #     print(color_i)
        #     self.PlaceSudoSlit(cubename=self.cubename_list[i], angle=self.angle_list[i], width=self.width_list[i],
        #                        height=self.height_list[i])
        #     ax[0].scatter(self.d5080_red, self.v50_red_mean, s=20, marker=self.marker_list[0], edgecolors='k', linewidths=0.5, color=color_i,
        #                   label=self.cubename_list[i])
        #     ax[0].scatter(self.d5080_blue, self.v50_blue_mean, s=20, marker=self.marker_list[0], edgecolors='k', linewidths=0.5, color=color_i)
        #     ax[1].scatter(self.d5080_red, self.w80_red_mean, s=20, marker=self.marker_list[0], edgecolors='k', linewidths=0.5, color=color_i)
        #     ax[1].scatter(self.d5080_blue, self.w80_blue_mean, s=20, marker=self.marker_list[0], edgecolors='k', linewidths=0.5, color=color_i)
        #
        # ax[0].axhline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
        # ax[0].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
        # ax[1].axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
        # ax[0].set_xlim(-40, 40)
        # ax[0].set_ylim(-450, 450)
        # # ax[1].set_ylim(0, 200)
        # ax[0].set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
        # ax[1].set_xlabel(r'$\rm Distance \, [pkpc]$', size=25)
        # ax[1].set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25, labelpad=20)
        # ax[0].legend(loc='best', fontsize=15)
        # fig.savefig('../../MUSEQuBES+CUBS/fit_kin/CUBS+MUSE_velocity_profile.png', bbox_inches='tight')

        # Create main figure
        fig = plt.figure(figsize=(15, 10), dpi=300)
        outer_grid = gridspec.GridSpec(2, 3, figure=fig, wspace=0.01, hspace=0.01)

        # Flatten the 2×3 grid into a 1D list
        for i, outer in enumerate(outer_grid):
            # Define a nested 1×2 grid inside each main subplot
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, hspace=0.0)

            # Create two vertically stacked subplots inside this grid
            ax_top = fig.add_subplot(inner_grid[0])
            ax_bottom = fig.add_subplot(inner_grid[1], sharex=ax_top)

            self.PlaceSudoSlit(cubename=self.cubename_list[i], angle=self.angle_list[i], width=self.width_list[i],
                               height=self.height_list[i])

            ax_top.scatter(self.d5080_red, self.v50_red_mean, s=20, marker=self.marker_list[0],
                           edgecolors='k', linewidths=0.5, color='red')
            ax_top.scatter(self.d5080_blue, self.v50_blue_mean, s=20, marker=self.marker_list[0],
                           edgecolors='k', linewidths=0.5, color='blue')
            ax_bottom.scatter(self.d5080_red, self.w80_red_mean, s=20, marker=self.marker_list[0],
                              edgecolors='k', linewidths=0.5, color='red')
            ax_bottom.scatter(self.d5080_blue, self.w80_blue_mean, s=20, marker=self.marker_list[0],
                              edgecolors='k', linewidths=0.5, color='blue')

            ax_top.axhline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
            ax_top.axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
            ax_bottom.axvline(0, linestyle='--', color='k', linewidth=1, zorder=-100)
            ax_top.set_xlim(-40, 40)
            ax_top.set_ylim(-450, 450)
            ax_bottom.set_ylim(0, 200)

            # ax_top.set_xticklabels([])

            # Set labels and ticks
            if i == 0:
                ax_top.set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
                ax_bottom.set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25, labelpad=20)
            elif i == 3:
                ax_bottom.set_xlabel(r'$\rm Distance \, [pkpc]$', size=25)
                ax_top.set_ylabel(r'$\rm V_{50} \rm \, [km \, s^{-1}]$', size=25)
                ax_bottom.set_ylabel(r'$\rm W_{80} \rm \, [km \, s^{-1}]$', size=25, labelpad=20)
            elif i == 4 or i == 5:
                ax_bottom.set_xlabel(r'$\rm Distance \, [pkpc]$', size=25)
                ax_top.set_yticklabels([])
                ax_bottom.set_yticklabels([])
            else:
                ax_top.set_yticklabels([])
                ax_bottom.set_xticklabels([])
                ax_bottom.set_yticklabels([])
            # ax_top.legend(loc='best', fontsize=15)
            ax_top.set_title(self.cubename_list[i], y=0.75, x=0.75, size=20)

        fig.savefig('../../MUSEQuBES+CUBS/fit_kin/CUBS+MUSE_velocity_profile_all.png', bbox_inches='tight')

    def PlaceSudoSlit(self, cubename=None, angle=None, width=None, height=None):
        # QSO information
        path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
        data_qso = ascii.read(path_qso, format='fixed_width')
        data_qso = data_qso[data_qso['name'] == cubename]
        ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

        # Calculate physical distance
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3

        # V50, S80
        path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
        path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)
        v50 = fits.open(path_v50_plot)[1].data
        s80 = fits.open(path_s80_plot)[1].data
        hdr_v50 = fits.open(path_v50_plot)[1].header
        w = WCS(hdr_v50, naxis=2)
        center_qso = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
        c2 = w.world_to_pixel(center_qso)

        #
        x, y = np.meshgrid(np.arange(v50.shape[0]), np.arange(v50.shape[1]))
        x, y = x.flatten(), y.flatten()
        pixcoord = PixCoord(x=x, y=y)

        # mask a slit
        rectangle = RectanglePixelRegion(center=PixCoord(x=c2[0], y=c2[1]), width=width,
                                         height=height, angle=Angle(angle, 'deg'))
        mask = rectangle.contains(pixcoord)
        x_c = x[mask] - c2[0]
        y_c = y[mask] - c2[1]
        x_slit = np.cos(angle) * x_c + np.sin(angle) * y_c
        y_slit = - np.sin(angle) * x_c + np.cos(angle) * y_c
        dis = np.sqrt(x_slit ** 2 + y_slit ** 2) * 0.2 * d_A_kpc / 206265

        # Mask each side
        red = x_slit > 0
        blue = ~red
        dis_red = dis[red] * -1
        dis_blue = dis[blue]

        # Slit position
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # plt.imshow(v50, origin='lower', cmap='coolwarm', vmin=-300, vmax=300)
        # rectangle.plot(ax=ax, facecolor='none', edgecolor='red', lw=2, label='Rectangle')
        # plt.show()

        # Extract
        v50_flatten, s80_flatten = v50.flatten(), s80.flatten()
        v50_blue, v50_red = v50_flatten[mask][blue], v50_flatten[mask][red]
        s80_blue, s80_red = s80_flatten[mask][blue], s80_flatten[mask][red]
        self.d5080_blue, self.v50_blue_mean, _, _, self.w80_blue_mean, _, _ = self.Bin(dis_blue, v50_blue, s80_blue, bins=20)
        self.d5080_red, self.v50_red_mean, _, _, self.w80_red_mean, _, _ = self.Bin(dis_red, v50_red, s80_red, bins=20)

    def Bin(self, x, y, z, bins=20):
        n, edges = np.histogram(x, bins=bins)
        y_mean, y_max, y_min = np.zeros(bins), np.zeros(bins), np.zeros(bins)
        z_mean, z_max, z_min = np.zeros(bins), np.zeros(bins), np.zeros(bins)
        x_mean = (edges[:-1] + edges[1:]) / 2
        for i in range(bins):
            if n[i] == 0:
                y_mean[i], y_max[i], y_min[i] = np.nan, np.nan, np.nan
                z_mean[i], z_max[i], z_min[i] = np.nan, np.nan, np.nan
            else:
                mask = (x > edges[i]) * (x <= edges[i + 1])
                if  len(y[mask]) == 0 or len(z[mask]) == 0:
                    y_mean[i], y_max[i], y_min[i] = np.nan, np.nan, np.nan
                    z_mean[i], z_max[i], z_min[i] = np.nan, np.nan, np.nan
                else:
                    y_mean[i], y_max[i], y_min[i] = np.nanmean(y[mask]), np.nanmax(y[mask]), np.nanmin(y[mask])
                    z_mean[i], z_max[i], z_min[i] = np.nanmean(z[mask]), np.nanmax(z[mask]), np.nanmin(z[mask])
        return x_mean, y_mean, y_max, y_min, z_mean, z_max, z_min


cubename_list = ['HE0435-5304', 'PKS0405-123', 'HE0238-1904', '3C57', 'PKS2242-498', 'HE0112-4145']
width_list = [50, 50, 30, 50, 50, 30]
height_list = [5, 5, 5, 5, 5, 5]
angle_list = [90, -90, 90, -30, 180, 240]
marker_list = ['o', 's', 'D', '^', 'v', '<']


func = PlotV50W80Profile(cubename_list=cubename_list, width_list=width_list, height_list=height_list,
                         angle_list=angle_list, marker_list=marker_list)
func.Plot()