#!/usr/bin/env python
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
from astropy.convolution import Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
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
parser.add_argument('-m', metavar='cubename', help='MUSE cube name (without .fits), required', required=True, type=str)
parser.add_argument('-t', metavar='S_N_thr', help='S/N threshold, required', required=True, type=float)
parser.add_argument('-s', metavar='std_2D', help='width/2 or sigma of the 2D smoothing filter kernel, default is None',
                    required=False, type=float, default=None)
parser.add_argument('-l', metavar='level', help='level of contour, default is 0.2',
                    type=float, default=0.2)
parser.add_argument('-k', metavar='kernel_2D', help='2D smoothing kernel ("box" or "gauss"), default is None',
                    required=False, type=str, default=None)
parser.add_argument('-s_spe', metavar='std_spectra', help='width/2 or sigma of the 1D smoothing filter kernel, '
                                                          'default is None',
                    required=False, type=float, default=None)
parser.add_argument('-k_spe', metavar='kernel_1D', help='1D smoothing kernel ("box" or "gauss"), default is None',
                    required=False, type=str, default=None)
parser.add_argument('-npixels', metavar='npixels', help='The minimum number of connected pixels '
                                                        'for nebula determination, default is 10', default=10, type=int)
parser.add_argument('-connectivity', metavar='connectivity', help='The type of pixel connectivity used in determining '
                                                                  'how pixels are grouped into a detected source, '
                                                                  'default is 8', default=8, type=float)
parser.add_argument('-n', metavar='max_num_nebulae', help='Maximum allowed number of nebulae, default is 20',
                    default=20, type=int)
parser.add_argument('-rv', metavar='RescaleVariance', help='Whether rescale variance, default is True', default='True',
                    type=str)
parser.add_argument('-ab', metavar='AddBackground', help='Whether add background fluctuation, default is True',
                    default='True', type=str)
parser.add_argument('-ns', metavar='num_bkg_slice', help='Number of integration of the background, default is 3',
                    default=3, type=int)
parser.add_argument('-csm', metavar='CheckSegmentationMap', help='Whether check segmentation map, default is False',
                    default='False', type=str)
parser.add_argument('-cs', metavar='CheckSpectra', help='Plot 25 spectra around a certain spaxel position '
                                                        'for example: "80 90" means extracting 25 spectra '
                                                        'around x=80, y=90; default is None',
                    default=None, nargs='+', type=int)
parser.add_argument('-pi', metavar='PlotNBImage', help='Whether plot the SB map, default is True', default='True',
                    type=str)
parser.add_argument('-sl', metavar='SelectLambda', help='Wavelength interval for subcube extraction '
                                                        'for example: "3727 3929"; default is None',
                    default=None, nargs='+', type=float)
parser.add_argument('-ssf', metavar='SumSmoothedFlux', help='Whether sum smoothed or unsmoothed flux', default='True',
                    type=str)
toBool = {'True': True, 'False': False}
args = parser.parse_args()  # parse the arguments


def MakeNBImage_MC(cubename=None, S_N_thr=None, smooth_2D=None, kernel_2D=None, smooth_1D=None, kernel_1D=None,
                   npixels=None, connectivity=None, max_num_nebulae=None, num_bkg_slice=None, RescaleVariance=None,
                   AddBackground=None, CheckSegmentation=None, CheckSpectra=None, PlotNBImage=None, SelectLambda=None,
                   SumSmoothedFlux=None, level=None):
    # Cubes
    cubename = '{}'.format(cubename)
    path_cube = cubename + '.fits'
    header = fits.getheader(path_cube, ext=1)
    filename_SB = cubename + '_SB_3DSeg_{}_{}_{}_{}.fits'.format(smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    filename_3Dseg = cubename + '_3DSeg_{}_{}_{}_{}.fits'.format(smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    filename_smoothed = '{}_{}_{}_{}_{}.fits'.format(cubename, smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    figurename = cubename + '_SB_3DSeg_{}_{}_{}_{}.pdf'.format(smooth_2D, kernel_2D, smooth_1D, kernel_1D)
    cube = Cube(path_cube, ext=(1,2))
    if SelectLambda is not None:
        cube = cube.select_lambda(SelectLambda[0], SelectLambda[1])

    # Define center of rotation (in pixel or world coordinates)
    # center = cube.wcs.get_center(unit='deg')
    # # angle = 45
    # # rotated_cube = cube.rotate(angle, center=center[:2])
    # cube = cube.subcube(tuple(center[:2]), size=30)


    size = np.shape(cube.data)
    wave_vac = pyasl.airtovac2(cube.wave.coord())
    flux = cube.data * 1e-3
    flux_err = np.sqrt(cube.var) * 1e-3
    seg_3D = np.zeros(size)
    seg_label = np.zeros(size[1:])

    # Copy object
    flux_ori = np.copy(flux)
    flux_err_ori = np.copy(flux_err)

    # Smoothing
    if smooth_2D is not None:
        if kernel_2D == 'gauss':
            kernel = Gaussian2DKernel(x_stddev=smooth_2D, y_stddev=smooth_2D)
        elif kernel_2D == 'box':
            kernel = Box2DKernel(smooth_2D * 2)
        else:
            raise AttributeError('kernel name is invalid; must be "gauss" or "box"')
        kernel_1 = Kernel(kernel.array[np.newaxis, :, :])
        flux = convolve(flux, kernel_1)

    if smooth_1D is not None:
        if kernel_1D == 'gauss':
            kernel = Gaussian1DKernel(smooth_1D)
        elif kernel_1D == 'box':
            kernel = Box1DKernel(smooth_1D * 2)
        else:
            raise AttributeError('kernel name is invalid; must be "gauss" or "box"')
        kernel_1 = Kernel(kernel.array[:, np.newaxis, np.newaxis])
        flux = convolve(flux, kernel_1)
    flux_smooth_ori = np.copy(flux)

    if RescaleVariance:
        flux_wl = np.nanmax(flux, axis=0)
        select_bkg = ~sigma_clip(flux_wl, sigma_lower=3, sigma_upper=3, cenfunc='median', masked=True).mask
        flux_mask = np.where(select_bkg[np.newaxis, :, :], flux, np.nan)
        flux_err_mask = np.where(select_bkg[np.newaxis, :, :], flux_err, np.nan)
        bkg_seg = np.where(select_bkg[np.newaxis, :, :], np.ones_like(flux_err[0, :, :]), np.nan)

        flux_var = np.nanvar(flux_mask, axis=(1, 2))
        flux_var_mean = np.nanmean(flux_err_mask ** 2, axis=(1, 2))
        value_rescale = flux_var / flux_var_mean
        print('Variance rescaling factor has mean value of {} and std of {}'.format(np.nanmean(value_rescale),
                                                                                    np.nanstd(value_rescale)))
        flux_err = flux_err * np.sqrt(value_rescale)[:, np.newaxis, np.newaxis]
    cube_smoothed = cube.clone(data_init=np.empty, var_init=np.empty)
    cube_smoothed.data = flux * 1e3
    cube_smoothed.var = (flux_err * 1e3) ** 2
    cube_smoothed.write(filename_smoothed)

    # Iterate over nebulae
    for k in range(max_num_nebulae):
        area_array = np.zeros(size[0]) * np.nan
        label_array = np.copy(area_array)
        for i in range(size[0]):
            flux_i, flux_err_i = flux[i, :, :], flux_err[i, :, :]
            S_N_i = flux_i / flux_err_i
            seg_i = detect_sources(S_N_i, S_N_thr, npixels=npixels, connectivity=connectivity)
            try:
                sort_area = np.flip(np.argsort(seg_i.areas))
                area_sort = seg_i.areas[sort_area]
                label_sort = seg_i.labels[sort_area]
                mask_isnan = np.isnan(area_sort)
                area_sort = area_sort[~mask_isnan]
                label_sort = label_sort[~mask_isnan]
                num = np.minimum(len(seg_i.areas), 5)
                area_ii, label_ii = np.zeros(num), np.zeros(num)
                for ii in range(num):
                    weight = np.where(seg_i.data == label_sort[ii], S_N_i, np.nan)
                    area_ii[ii] = np.nansum(weight) / area_sort[ii]
                    label_ii[ii] = label_sort[ii]
                area_array[i] = np.nanmax(area_ii)
                label_array[i] = label_ii[np.nanargmax(area_ii)]
            except AttributeError:
                pass
        try:
            idx_max = np.nanargmax(area_array)
            label_max = label_array[idx_max]
        except ValueError:
            print('No enough number of nebulae are detected')
            break

        S_N_max = flux[idx_max, :, :] / flux_err[idx_max, :, :]
        seg_max = detect_sources(S_N_max, S_N_thr, npixels=npixels, connectivity=connectivity)
        mask = np.where(seg_max.data == label_max, np.ones_like(flux[idx_max, :, :]), 0)
        seg_label = np.where(mask == 0, seg_label, k + 1)

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
            if SumSmoothedFlux:
                flux_cut = np.where(mask_i != 0, flux_i, 0)
            else:
                flux_cut = np.where(mask_i != 0, flux_ori[i, :, :], 0)
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
            if SumSmoothedFlux:
                flux_cut = np.where(mask_i != 0, flux_i, 0)
            else:
                flux_cut = np.where(mask_i != 0, flux_ori[i, :, :], 0)
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
                        i_j, j_j = ax_i + CheckSpectra[1], ax_j + CheckSpectra[0]
                        ax[ax_i, ax_j].plot(wave_vac, flux_ori[:, i_j, j_j], '-k')
                        ax[ax_i, ax_j].plot(wave_vac, flux_smooth_ori[:, i_j, j_j], '-b')
                        # ax[ax_i, ax_j].plot(wave_vac, flux_smooth_ori[:, i_j, j_j]
                        #                     - np.median(flux_smooth_ori, axis=(1, 2)), '-r')
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
        if smooth_2D is not None or smooth_1D is not None:
            if SumSmoothedFlux:
                flux_bkg = flux_smooth_ori[idx, :, :]
            else:
                flux_bkg = flux_ori[idx, :, :]
        else:
            flux_bkg = flux_ori[idx, :, :]
        data_final = np.where(data_final != 0, data_final, num_bkg_slice * flux_bkg)

    # Save data
    header['WCSAXES'] = 2
    header.remove('NAXIS3')
    header.remove('CTYPE3')
    header.remove('CUNIT3')
    header.remove('CRPIX3')
    header.remove('CRVAL3')
    try:
        header.remove('CRDER3')
    except KeyError:
        pass
    header.remove('BUNIT')
    header.remove('CD3_3')
    header.remove('CD1_3')
    header.remove('CD2_3')
    header.remove('CD3_1')
    header.remove('CD3_2')
    hdul_seg_3D = fits.PrimaryHDU(seg_3D, header=header)
    hdul_seg_label = fits.ImageHDU(seg_label, header=header)
    hdul_seg = fits.HDUList([hdul_seg_3D, hdul_seg_label])
    hdul_seg.writeto(filename_3Dseg, overwrite=True)

    hdul_SB = fits.ImageHDU(data_final * 1.25 / 0.2 / 0.2, header=header)
    hdul_SB.writeto(filename_SB, overwrite=True)

    if PlotNBImage:
        ra_center, dec_center = header['CRVAL1'], header['CRVAL2']
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(filename_SB, figure=fig, hdu=1, north=True)
        # gc.show_colorscale(vmin=0, vmid=0.2, vmax=15, cmap=plt.get_cmap('gist_heat_r'), stretch='arcsinh')
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        # gc.recenter(ra_center, dec_center, width=30 / 3600, height=30 / 3600)
        gc.show_contour(filename_SB, hdu=1, levels=[level], color='k', linewidths=2, smooth=5, kernel='box')
        gc.set_system_latex(True)

        # Colorbar
        gc.add_colorbar()
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
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
        plt.savefig(figurename, bbox_inches='tight')

        # Plot Original
        fig = plt.figure(figsize=(8, 8), dpi=300)
        hdul_SB_ori = fits.open(cubename + '_SB.fits')
        try:
            hdul_SB_ori[1].header.remove('CRDER3')
        except KeyError:
            pass
        hdul_SB_ori[1].data /= 1e-17
        hdul_SB_ori.writeto(cubename + '_SB_ori.fits', overwrite=True)
        gc = aplpy.FITSFigure(cubename + '_SB_ori.fits', figure=fig, hdu=1, north=True)
        # gc.show_colorscale(vmin=0, vmid=0.2, vmax=15, cmap=plt.get_cmap('gist_heat_r'), stretch='arcsinh')
        gc.show_colorscale(vmin=-0.05, vmax=5, cmap=plt.get_cmap('gist_heat_r'), stretch='linear')
        # gc.recenter(ra_center, dec_center, width=30 / 3600, height=30 / 3600)
        gc.show_contour(cubename + '_SB_ori.fits', hdu=1, levels=[level], color='k', linewidths=2, smooth=5, kernel='box')
        gc.set_system_latex(True)

        # Colorbar
        gc.add_colorbar()
        gc.colorbar.set_ticks([0, 1, 2, 3, 4, 5])
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
        plt.savefig(cubename + '_SB.pdf', bbox_inches='tight')


MakeNBImage_MC(cubename=args.m, S_N_thr=args.t, smooth_2D=args.s, kernel_2D=args.k, smooth_1D=args.s_spe,
               kernel_1D=args.k_spe, npixels=args.npixels, connectivity=args.connectivity, max_num_nebulae=args.n,
               num_bkg_slice=args.ns, RescaleVariance=toBool[args.rv], AddBackground=toBool[args.ab],
               CheckSegmentation=toBool[args.csm], CheckSpectra=args.cs, PlotNBImage=toBool[args.pi],
               SumSmoothedFlux=toBool[args.ssf], SelectLambda=args.sl, level=args.l)