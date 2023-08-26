import aplpy
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord
# from astropy.wcs import WCS
from astropy.convolution import convolve
from mpdaf.obj import WCS, Image, Cube, WaveCoord, iter_spe, iter_ima
from muse_MakeGasImages import APLpyStyle
from astropy import units as u
from astropy.convolution import convolve
from astropy.convolution import Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic)
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
path_data = '/Users/lzq/Dropbox/Data/CGM/'

def keep_longest_true_ori(a):
    # Convert to array
    a = np.asarray(a)

    # Attach sentients on either sides w.r.t True
    b = np.r_[False, a, False]

    # Get indices of group shifts
    s = np.flatnonzero(b[:-1] != b[1:])

    # Get group lengths and hence the max index group
    try:
        m = (s[1::2] - s[::2]).argmax(axis=0)

        # Initialize array and assign only the largest True island as True.
        out = np.zeros_like(a)
        out[s[2 * m]:s[2 * m + 1]] = 1
    except ValueError:
        # m = 0
        out = np.zeros_like(a)
    # out = np.zeros_like(a)
    # out[s[2*m]:s[2*m+1]] = 1
    return out

def keep_longest_true(a):
    # Convert to array
    a = np.asarray(a)

    # Attach sentients on either sides w.r.t True
    b = np.zeros((np.shape(a)[0] + 2, np.shape(a)[1], np.shape(a)[2]))
    b[1:np.shape(a)[0]+1, :, :] = a
    b[0, :, :] = False
    b[-1, :, :] = False

    # Get indices of group shifts
    s_0 = np.arange(np.shape(b)[0] - 1)
    s_1 = np.repeat(s_0[:, np.newaxis], np.shape(b)[1], axis=1)
    s_2 = np.repeat(s_1[:, :, np.newaxis], np.shape(b)[2], axis=2)
    s = np.where(b[:-1, :, :] != b[1:, :, :], s_2, np.nan)
    # print(s[:, 0, 0])

    # Get group lengths and hence the max index group
    s_mid = np.tril((s[:, np.newaxis, :, :] - s[:, :, :]).T, k=-1).T
    # s_mid =
    print(np.shape(s_mid))
    s_mid = np.tr
    # s_mid = s_mid[:, ~np.all(np.isnan(s_mid), axis=0), :, :]
    # s_mid = s_mid[~np.all(np.isnan(s_mid), axis=1), :, :, :]
    print(np.shape(s_mid))
    # print(s[:, 0, 0])
    # print(s[1::2, 0, 0])
    # print(s[::2, 0, 0])

    s_mid = np.where(s_mid != 0, s_mid, np.nan)
    s_mid = np.where(s_mid != np.nan, s_mid, -100)
    print(np.shape(s_mid))
    print(np.shape(s_mid)[0] * np.shape(s_mid)[1])



    s_mid = np.reshape(s_mid, (np.shape(s_mid)[0] * np.shape(s_mid)[1], np.shape(s_mid)[2], np.shape(s_mid)[3]))
    print(s_mid[:, 0, 0])
    m = np.nanargmax(s_mid, axis=0)
    print(m[0, 0])

    # Initialize array and assign only the largest True island as True.
    s2 = np.take_along_axis(s, np.expand_dims(2 * m + 0, axis=0), axis=0)
    s21 = np.take_along_axis(s, np.expand_dims(2 * m + 1, axis=0), axis=0)
    # print(2 * m)
    # print(s2)
    # print(s21)

    out_idx = np.arange(np.shape(a)[0])
    out_idx_1 = np.repeat(out_idx[:, np.newaxis], np.shape(a)[1], axis=1)
    out_idx_2 = np.repeat(out_idx_1[:, :, np.newaxis], np.shape(a)[2], axis=2)
    out = np.zeros_like(a)
    out = np.where((out_idx_2 < s2) | (out_idx_2 > s21), out, 1)
    return out

# def MakeNBImage_LI():
    # print(wave_vac[np.nanargmax(area_array)])
    # if save_figure_test:
    #     fig, ax = plt.subplots(5, 5, figsize=(20, 20))

    # for i, i_val in enumerate(range(0, size[1])):
    #     for j, j_val in enumerate(range(0, size[2])):
    #         flux_ij, flux_err_ij = flux[:, i_val, j_val], flux_err[:, i_val, j_val]
    #         flux_max, flux_min = np.nanmax(flux_ij), np.nanmin(flux_ij)
    #         flux_where = np.where(flux_ij > 1.2 * np.median(flux_err_ij), flux_ij, False)
    #         flux_where = np.asarray(flux_where, dtype=bool)
    #         flux_where = keep_longest_true_ori(flux_where)
    #         flux_sum = np.nansum(flux_ij[flux_where])
    #         flux_sum_array[i, j] = flux_sum
    #
    #         if save_figure_test:
    #             if (i >= 90) * (i < 95):
    #                 if (j >= 90) * (j < 95):
    #                     ax[i-90, j-90].plot(wave_vac, flux_ij, '-k')
    #                     ax[i-90, j-90].plot(wave_vac, flux_err_ij, '-C0')
    #                     ax[i-90, j-90].fill_between(wave_vac[flux_where],
    #                                                 y1=np.zeros_like(wave_vac[flux_where]),
    #                                                 y2=flux_ij[flux_where])

    #
    # if save_figure_test:
    #     plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/MakeNBImage_MC.png')


    # flux_data = flux_sum_array * 1.25 / 0.2 / 0.2
    # if smooth:
    #     kernel = Box2DKernel(smooth_val)
    #     # kernel = Gaussian2DKernel(x_stddev=5.0, x_size=3, y_size=3)
    #     flux_data = convolve(flux_data, kernel)


    # Save data
    # ima = Image(data=flux_data, wcs=wcs)
    # ima.write('/Users/lzq/Dropbox/Data/CGM/NBImage_MC/' + band + '_test.fits')

# Vectorize everything

# flux_where = np.where(flux_OIII5008 > 1.1 * np.nanmedian(flux_OIII5008_err, axis=0), flux_OIII5008, False)
# flux_where = np.asarray(flux_where, dtype=bool)
#

# x = keep_longest_true_ori(flux_where[:, 0, 0])

# print(flux_where[:, 0, 0])
# flux_where = keep_longest_true(flux_where[:, :, :])
# x =
# print(flux_where[:, 90, 90])
# flux_2besum = np.where(flux_where is True, flux_OIII5008[:, :, :], np.nan)
# flux_sum = np.nansum(flux_2besum, axis=0)
# print(flux_sum)
# print(np.shape(flux_sum))
#
#

def MakeNBImage_MC(cubename='CUBE_OIII_5008_line_offset.fits', S_N_thr=1, npixels=100, connectivity=8, smooth=True,
                   smooth_val=3, nums=1, AddBackground=False, CheckSegmentation=False, CheckSpectra=None):
    # Cubes
    path_cube = path_data + 'cube_narrow/' + cubename
    cube = Cube(path_cube)
    wcs = WCS(fits.open(path_data + 'image_MakeMovie/OIII_-300_-100_contour_revised.fits')[0].header)
    size = np.shape(cube.data)
    wave_vac = pyasl.airtovac2(cube.wave.coord())
    flux = cube.data * 1e-3
    flux_err = np.sqrt(cube.var) * 1e-3
    flux_ori = np.copy(flux)
    flux_err_ori = np.copy(flux_err)

    # Smoothing
    if smooth:
        kernel = Box2DKernel(smooth_val)
        # kernel = Gaussian2DKernel(x_stddev=5.0, x_size=3, y_size=3)
        kernel = Kernel(kernel.array[np.newaxis, :, :])
        flux = convolve(flux, kernel)
        # flux_err = np.sqrt(convolve(flux_err ** 2, kernel))  ## perhaps wrong?
    flux_smooth_ori = np.copy(flux)

    # Iterate over nebulae
    for k in range(nums):
        area_array = np.zeros(size[0]) * np.nan
        for i in range(size[0]):
            flux_i, flux_err_i = flux[i, :, :], flux_err[i, :, :]
            S_N_i = flux_i / flux_err_i
            seg_i = detect_sources(S_N_i, S_N_thr, npixels=npixels, connectivity=connectivity)
            try:
                area_array[i] = np.nanmax(seg_i.areas)
            except AttributeError:
                pass

        idx_max = np.nanargmax(area_array)
        S_N_max = flux[idx_max, :, :] / flux_err[idx_max, :, :]
        seg_max = detect_sources(S_N_max, S_N_thr, npixels=npixels, connectivity=connectivity)
        label_max = seg_max.labels[np.nanargmax(seg_max.areas)]
        data_mask = np.where(seg_max.data == label_max, flux[idx_max, :, :], 0)
        mask = np.where(seg_max.data == label_max, np.ones_like(flux[idx_max, :, :]), 0)
        if k == 0:
            data_final = np.copy(data_mask)
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
            if np.all(conti_b.flatten() == 0):
                break
            else:
                data_final += flux_cut

        # Mask out current nebulae
        flux = np.where(seg_max.data[np.newaxis, :, :] != label_max, flux[:, :, :], 0)

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
                plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + cubename[5:-5] + '_CheckSpectra' + str(k) + '.png')

    if CheckSegmentation:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(np.where(flux[0, :, :] == 0, flux[0, :, :], np.nan), origin='lower')
        plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + cubename[5:-5] + '_CheckSegmentation.png')

    if AddBackground:
        if smooth:
            flux_bkg = flux_smooth_ori[idx, :, :]
        else:
            flux_bkg = flux_ori[idx, :, :]
        data_final = np.where(data_final != 0, data_final, 3 * flux_bkg)

    # Save data
    ima = Image(data=data_final * 1.25 / 0.2 / 0.2, wcs=wcs)
    ima.write('/Users/lzq/Dropbox/Data/CGM/NBImage_MC/' + cubename[5:-5] + '_NBImage_MC.fits')


#
MakeNBImage_MC(cubename='CUBE_OII_line_offset.fits', S_N_thr=0.7, smooth=True, smooth_val=3, nums=10, npixels=10,
               CheckSegmentation=True, AddBackground=True, CheckSpectra=[90, 90])

# Plot the data
fig = plt.figure(figsize=(8, 8), dpi=300)
gc = aplpy.FITSFigure('/Users/lzq/Dropbox/Data/CGM/NBImage_MC/OII_line_offset_NBImage_MC.fits', figure=fig)
# gc.show_colorscale(vmin=0, vmax=10, cmap=plt.get_cmap('Blues'))
gc.show_colorscale(vmin=0, vmid=0.2, vmax=15, cmap=plt.get_cmap('Blues'), stretch='arcsinh')
# gc.show_contour('/Users/lzq/Dropbox/Data/CGM/NBImage_MC/OII_test.fits', levels=[0.3, 2], colors='k', linewidths=0.8)
APLpyStyle(gc, type='NarrowBand')
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/MakeNBImage_MC_OII_test.png', bbox_inches='tight')
