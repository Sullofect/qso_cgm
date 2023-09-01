import sys
import aplpy
import argparse
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.io import fits
from PyAstronomy import pyasl
from astropy.stats import sigma_clip
from mpdaf.obj import WCS, Image, Cube
from astropy.convolution import convolve
from astropy.convolution import Kernel, Gaussian2DKernel, Box2DKernel
from photutils.segmentation import detect_sources
warnings.filterwarnings("ignore")
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10


# Set up the parser
parser = argparse.ArgumentParser(description='Make Narrow band Surface brightness map with 3D segmentation')
parser.add_argument('-m', metavar='cubename', help='MUSE cube name (without .fits)', required=True, type=str)
parser.add_argument('-t', metavar='S_N_thr', help='S/N threshold', required=True, default=1.0, type=float)
parser.add_argument('-s', metavar='smooth_val', help='width of the 2D Box smoothing filter kernel', required=True, type=float)
parser.add_argument('-npixels', metavar='npixels', help='The minimum number of connected pixels', default=100, type=int)
parser.add_argument('-connectivity', metavar='connectivity', help='The type of pixel connectivity used in determining '
                                                                  'how pixels are grouped into a detected source',
                    default=8, type=float)
parser.add_argument('-n', metavar='max_num_nebulae', help='Maximum allowed number of nebulae', default=10, type=int)
parser.add_argument('-ns', metavar='num_bkg_slice', help='Number of integration of the background',
                    default=3, type=int)
parser.add_argument('-rv', metavar='RescaleVariance', help='Whether rescale variance', default=True, type=bool)
parser.add_argument('-ab', metavar='AddBackground', help='Whether add background', default=True, type=bool)
parser.add_argument('-csm', metavar='CheckSegmentation', help='Whether add background', default=False, type=bool)
parser.add_argument('-cs', metavar='CheckSpectra',  help='The pixel position of checked spectra', default=None, type=list)
parser.add_argument('-pi', metavar='PlotNBImage',  help='Whether plot the SB map', default=True, type=bool)
args = parser.parse_args() # parse the arguments


def MakeNBImage_MC(cubename='CUBE_OIII_5008_line_offset', S_N_thr=1, smooth_val=3, npixels=100, connectivity=8,
                   max_num_nebulae=10, num_bkg_slice=3, RescaleVariance=True, AddBackground=False,
                   CheckSegmentation=False, CheckSpectra=None, PlotNBImage=True):
    # Cubes
    cubename = '{}'.format(cubename)
    path_cube = cubename + '.fits'
    filename_SB = cubename + '_SB.fits'
    filename_3Dseg = cubename + '_3DSeg.fits'
    cube = Cube(path_cube)
    header = fits.getheader(path_cube, ext=1)
    size = np.shape(cube.data)
    wave_vac = pyasl.airtovac2(cube.wave.coord())
    flux = cube.data * 1e-3
    flux_err = np.sqrt(cube.var) * 1e-3
    seg_3D = np.zeros(size)

    if RescaleVariance:
        flux_wl = np.nanmax(flux, axis=0)
        select_bkg = ~sigma_clip(flux_wl, sigma_lower=3, sigma_upper=3, cenfunc='median', masked=True).mask
        flux_mask = np.where(select_bkg[np.newaxis, :, :], flux, np.nan)
        flux_err_mask = np.where(select_bkg[np.newaxis, :, :], flux_err, np.nan)
        bkg_seg = np.where(select_bkg[np.newaxis, :, :], np.ones_like(flux_err[0, :, :]), np.nan)

        flux_var = np.nanvar(flux_mask, axis=(1, 2))
        flux_var_mean = np.nanmean(flux_err_mask ** 2, axis=(1, 2))
        value_rescale = flux_var / flux_var_mean
        print('Variance rescaling factor has mean value of {} and std of {}'.format(np.mean(value_rescale),
                                                                                    np.std(value_rescale)))
        flux_err = flux_err * np.sqrt(value_rescale)[:, np.newaxis, np.newaxis]

    # Copy object
    flux_ori = np.copy(flux)
    flux_err_ori = np.copy(flux_err)

    # Smoothing
    if smooth_val is not None:
        kernel = Box2DKernel(smooth_val)
        # kernel = Gaussian2DKernel(x_stddev=5.0, x_size=3, y_size=3)
        kernel = Kernel(kernel.array[np.newaxis, :, :])
        flux = convolve(flux, kernel)
        # flux_err = np.sqrt(convolve(flux_err ** 2, kernel))  ## perhaps wrong?
    flux_smooth_ori = np.copy(flux)

    # Iterate over nebulae
    for k in range(max_num_nebulae):
        area_array = np.zeros(size[0]) * np.nan
        for i in range(size[0]):
            flux_i, flux_err_i = flux[i, :, :], flux_err[i, :, :]
            S_N_i = flux_i / flux_err_i
            seg_i = detect_sources(S_N_i, S_N_thr, npixels=npixels, connectivity=connectivity)
            try:
                area_array[i] = np.nanmax(seg_i.areas)
            except AttributeError:
                pass
        try:
            idx_max = np.nanargmax(area_array)
        except ValueError:
            print('No enough number of nebulae are detected')
            break
        S_N_max = flux[idx_max, :, :] / flux_err[idx_max, :, :]
        seg_max = detect_sources(S_N_max, S_N_thr, npixels=npixels, connectivity=connectivity)
        label_max = seg_max.labels[np.nanargmax(seg_max.areas)]
        mask = np.where(seg_max.data == label_max, np.ones_like(flux[idx_max, :, :]), 0)

        # Initialize
        if k == 0:
            data_final = np.zeros_like(mask)
            nebulae_seg = np.full_like(mask, np.nan)
            idx = idx_max

        # Over two direction
        wave_grid_s = np.ones_like(S_N_max) * wave_vac[idx_max]
        wave_grid_b = np.ones_like(S_N_max) * wave_vac[idx_max]
        conti_s, conti_b = np.ones_like(S_N_max), np.ones_like(S_N_max)
        idx_s, idx_b = np.arange(0, idx_max), np.arange(idx_max, size[0])
        for i in np.flip(idx_s):
            flux_i, flux_err_i = flux[i, :, :], flux_err[i, :, :]
            S_N_i = flux_i / flux_err_i
            mask_i = np.where(conti_s == 1, mask, 0)
            mask_i = np.where(S_N_i >= S_N_thr, mask_i, 0)
            flux_cut = np.where(mask_i != 0, flux_i, 0)
            conti_s = np.where(mask_i != 0, conti_s, 0)
            wave_grid_s = np.where(mask_i == 0, wave_grid_s, wave_vac[i])
            seg_3D[i, :, :] = np.where(mask_i == 0, seg_3D[i, :, :], 1)
            if np.all(conti_s.flatten() == 0):
                break
            else:
                data_final += flux_cut

        for i in idx_b:
            flux_i, flux_err_i = flux[i, :, :], flux_err[i, :, :]
            S_N_i = flux_i / flux_err_i
            mask_i = np.where(conti_b == 1, mask, 0)
            mask_i = np.where(S_N_i >= S_N_thr, mask_i, 0)
            flux_cut = np.where(mask_i != 0, flux_i, 0)
            conti_b = np.where(mask_i != 0, conti_b, 0)
            wave_grid_b = np.where(mask_i == 0, wave_grid_b, wave_vac[i])
            seg_3D[i, :, :] = np.where(mask_i == 0, seg_3D[i, :, :], 1)
            if np.all(conti_b.flatten() == 0):
                break
            else:
                data_final += flux_cut

        # Mask out current nebulae
        flux = np.where(seg_max.data[np.newaxis, :, :] != label_max, flux[:, :, :], 0)
        nebulae_seg = np.where(seg_max.data[np.newaxis, :, :] != label_max, nebulae_seg, k + 1)

        if k == 0:
            if CheckSpectra is not None:
                fig, ax = plt.subplots(5, 5, figsize=(20, 20))
                for ax_i in range(5):
                    for ax_j in range(5):
                        i_j, j_j = ax_i + CheckSpectra[0], ax_j + CheckSpectra[1]
                        ax[ax_i, ax_j].plot(wave_vac, flux_ori[:, i_j, j_j], '-k')
                        ax[ax_i, ax_j].plot(wave_vac, flux_smooth_ori[:, i_j, j_j], '-b')
                        ax[ax_i, ax_j].plot(wave_vac, flux_err_ori[:, i_j, j_j], '-C0')
                        ax[ax_i, ax_j].plot(wave_vac, flux_err[:, i_j, j_j], '-C2')
                        ax[ax_i, ax_j].fill_between([wave_grid_s[i_j, j_j], wave_grid_b[i_j, j_j]], y1=np.zeros(2),
                                                    y2=np.ones(2) * np.nanmax(flux_ori[:, i_j, j_j]), color='C1',
                                                    alpha=0.2)
                        # ax[ax_i, ax_j].set_ylim(top=0.01)
                figurename = cubename + '_CheckSpectra.pdf'
                plt.savefig(figurename)

    if CheckSegmentation:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
        ax.imshow(nebulae_seg[0, :, :], origin='lower', cmap=plt.get_cmap('tab20c'))
        ax.imshow(bkg_seg[0, :, :], origin='lower', cmap=plt.get_cmap('binary_r'))
        figurename = cubename + '_CheckSegmentation.pdf'
        plt.savefig(figurename)

    if AddBackground:
        if smooth_val is not None:
            flux_bkg = flux_smooth_ori[idx, :, :]
        else:
            flux_bkg = flux_ori[idx, :, :]
        data_final = np.where(data_final != 0, data_final, num_bkg_slice * flux_bkg)

    # Save data
    fits.writeto(filename_3Dseg, seg_3D, overwrite=True)
    fits.setval(filename_3Dseg, 'CTYPE1', value=header['CTYPE1'])
    fits.setval(filename_3Dseg, 'CTYPE2', value=header['CTYPE2'])
    fits.setval(filename_3Dseg, 'EQUINOX', value=header['EQUINOX'])
    fits.setval(filename_3Dseg, 'CD1_1', value=header['CD1_1'])
    fits.setval(filename_3Dseg, 'CD2_1', value=header['CD2_1'])
    fits.setval(filename_3Dseg, 'CD1_2', value=header['CD1_2'])
    fits.setval(filename_3Dseg, 'CD2_2', value=header['CD2_2'])
    fits.setval(filename_3Dseg, 'CRPIX1', value=header['CRPIX1'])
    fits.setval(filename_3Dseg, 'CRPIX2', value=header['CRPIX2'])
    fits.setval(filename_3Dseg, 'CRVAL1', value=header['CRVAL1'])
    fits.setval(filename_3Dseg, 'CRVAL2', value=header['CRVAL2'])
    fits.setval(filename_3Dseg, 'LONPOLE', value=header['LONPOLE'])
    fits.setval(filename_3Dseg, 'LATPOLE', value=header['LATPOLE'])

    fits.writeto(filename_SB, data_final * 1.25 / 0.2 / 0.2, overwrite=True)
    fits.setval(filename_SB, 'CTYPE1', value=header['CTYPE1'])
    fits.setval(filename_SB, 'CTYPE2', value=header['CTYPE2'])
    fits.setval(filename_SB, 'EQUINOX', value=header['EQUINOX'])
    fits.setval(filename_SB, 'CD1_1', value=header['CD1_1'])
    fits.setval(filename_SB, 'CD2_1', value=header['CD2_1'])
    fits.setval(filename_SB, 'CD1_2', value=header['CD1_2'])
    fits.setval(filename_SB, 'CD2_2', value=header['CD2_2'])
    fits.setval(filename_SB, 'CRPIX1', value=header['CRPIX1'])
    fits.setval(filename_SB, 'CRPIX2', value=header['CRPIX2'])
    fits.setval(filename_SB, 'CRVAL1', value=header['CRVAL1'])
    fits.setval(filename_SB, 'CRVAL2', value=header['CRVAL2'])
    fits.setval(filename_SB, 'LONPOLE', value=header['LONPOLE'])
    fits.setval(filename_SB, 'LATPOLE', value=header['LATPOLE'])

    if PlotNBImage:
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(filename_SB, figure=fig)
        gc.show_colorscale(vmin=0, vmid=0.2, vmax=15, cmap=plt.get_cmap('Reds'), stretch='arcsinh')
        gc.set_system_latex(True)

        # Colorbar
        gc.add_colorbar()
        gc.colorbar.set_ticks([1, 5, 10])
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_pad(0.0)
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_axis_label_font(size=20)
        gc.colorbar.set_location('bottom')
        gc.colorbar.set_font(size=20)
        gc.colorbar.set_axis_label_text(r'$\mathrm{Surface \; Brightness \; [10^{-17} \; erg \; cm^{-2} \; '
                                        r's^{-1} \; arcsec^{-2}]}$')

        # Hide
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.axis_labels.hide()
        gc.ticks.set_length(30)
        figurename = cubename + '_SB.pdf'
        plt.savefig(figurename, bbox_inches='tight')


MakeNBImage_MC(cubename=args.m, S_N_thr=args.t, smooth_val=args.s, npixels=args.npixels,
               connectivity=args.connectivity, max_num_nebulae=args.n, num_bkg_slice=args.ns, RescaleVariance=args.rv,
               AddBackground=args.ab, CheckSegmentation=args.csm, CheckSpectra=args.cs, PlotNBImage=args.pi)