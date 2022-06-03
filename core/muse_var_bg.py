import os
import numpy as np
import astropy.io.fits as fits
from astropy.stats import mad_std
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
from mpdaf.sdetect import findSkyMask
# Calculate the white image
path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits')
cube = Cube(path)
cube = cube.subcube((-18.8643, 40.1359), 40)
wave = cube.wave.coord()
print(len(wave))
# image_white = cube.sum(axis=0)
# image_white.write('/Users/lzq/Dropbox/Data/CGM/image_white.fits')

# image_white.plot(vmin=1950, vmax=7000, colorbar='v')
#
# ksel = np.where(image_white.data < 2000)
# image_white.mask_selection(ksel)
# image_white.plot(vmin=1950, vmax=7000, colorbar='v')
# plt.show()


# Use segmentation map
path_seg_wl = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'bg_estimate',
                           'check_seg_wl.fits')
path_seg_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'bg_estimate',
                            'check_seg_OII.fits')
path_seg_OIII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'bg_estimate',
                             'check_seg_OIII.fits')
data_seg_wl = fits.getdata(path_seg_wl, 1)
data_seg_OII = fits.getdata(path_seg_OII, 1)
data_seg_OIII = fits.getdata(path_seg_OIII, 1)

# mask = findSkyMask(data_seg_wl)

ksel_wl = np.where(data_seg_wl > 0)
ksel_OII = np.where(data_seg_wl > 0)
ksel_OIII = np.where(data_seg_wl > 0)

std_array = np.zeros(np.shape(cube)[0])
mad_std_array = np.zeros(np.shape(cube)[0])
for i in range(np.shape(cube)[0] - 1):
    image = cube[i, :, :]
    image.mask_selection(ksel_wl)
    image.mask_selection(ksel_OII)
    image.mask_selection(ksel_OIII)
    flux = image.data * 1e-3
    flux_err = np.sqrt(image.var) * 1e-3

    # print(np.nanvar(flux))
    # print(i)
    std_array[i] = np.nanstd(flux)
    mad_std_array[i] = mad_std(flux, ignore_nan=True)

bg_var_info = np.array([wave[:-1], std_array, mad_std_array])
fits.writeto('/Users/lzq/Dropbox/Data/CGM/bg_std_info.fits', bg_var_info, overwrite=True)