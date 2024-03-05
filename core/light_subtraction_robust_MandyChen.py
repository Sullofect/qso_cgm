#!/usr/bin/env python

# import necessary classes
import os
import sys
import argparse
import numpy as np

# set up the parser
parser = argparse.ArgumentParser(description='Non-negative matrix factorization of point source spectrum.')
parser.add_argument('-f', metavar='filename', help='MUSE cube filename.', required=True, type=str)
parser.add_argument('-x', metavar='x', help='Quasar initial x position (pixels).', required=True, type=int)
parser.add_argument('-y', metavar='y', help='Quasar initial y position (pixels).', required=True, type=int)
parser.add_argument('-r0', metavar='ro',
                    help='Pixels within this radius (pixels) will be excluded from the QSO spectrum', required=False,
                    type=float)
parser.add_argument('-r1', metavar='r1',
                    help='Pixels outside this radius (pixels) will be excluded from the QSO spectrum', required=True,
                    type=float)
parser.add_argument('-z', metavar='redshift', help='Redshift for the galaxy component.', required=True, type=float)
parser.add_argument('-r', metavar='radius', help='Subtraction radius (pixels).', required=True, type=float)
parser.add_argument('-w0', metavar='wave_low', help='Minimum wavelength used.', required=False, type=float,
                    default=4800)
parser.add_argument('-w1', metavar='wave_high', help='Maximum wavelength used.', required=False, type=float,
                    default=9300)
parser.add_argument('-m', metavar='mask', help='Should masks be used?', required=False, type=int, default=1)
parser.add_argument('-limitslope', metavar='limitslope', help='Ignore quasar model if slope < 0', required=False,
                    type=bool, default=True)
parser.add_argument('-limitflux', metavar='limitflux', help='Ignore quasar model if <5% of median flux or < some value',
                    required=False, type=float, default=0.0)
parser.add_argument('-ao', metavar='ao', help='AO mode?', required=False, type=bool, default=False)
parser.add_argument('-saveall', metavar='saveall', help='Save the galaxy and quasar model arrays?', required=False,
                    type=bool, default=False)

args = parser.parse_args()

# if the parser is setup, then import necessary classes again
from astropy.table import Table
from scipy.interpolate import interp1d
from mpdaf.obj import Cube, iter_spe
from pydl.goddard.astro import airtovac, vactoair
# from NonnegMFPy import nmf
from astropy.io import fits
import warnings
import numpy.ma as ma
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

# read in the cube
print('Reading in the cube {}...'.format(args.f))
cube = Cube(args.f)
cube = cube.select_lambda(args.w0, args.w1)

# Create a white light image **note for now this is just a single wavelength slice to speed up.
print('Creating white-light image')
# image_white = cube.median(axis=0)
image_white = cube[200, :, :]
print(image_white)

# Find the QSO center
# print('Fitting gaussian to quasar position in the white-light image...')
# gaussian = image_white.gauss_fit(center=(args.y, args.x), circular=True, fwhm=3, weight=True, unit_center=None)

# extract the subcube near the quasar
# yx = (gaussian.center[0], gaussian.center[1])
yx = (args.y, args.x)
print(yx)
cube_qso = cube.subcube(yx, args.r1 + 1, unit_center=None, unit_size=None)

# Get an updated centroid for the quasar based on the smaller cube which is likely more outlier resistant.
image_white = cube_qso[200, :, :]
print(image_white.shape)
# gaussian_subcube = image_white.gauss_fit(center=np.array(image_white.shape) / 2, circular=True, fwhm=1.0, weight=True,
#                                          unit_center=None)
# yx = (gaussian_subcube.center[0], gaussian_subcube.center[1])

# Mask the region inside of r0 and outside of r1
cube_qso.mask_region(yx, args.r1, unit_center=None, unit_radius=None, inside=False)
# cube_qso.mask_region(yx, args.r0, unit_center=None, unit_radius=None, inside=True)

cube_qso.write('test.fits')
# Get the wavelength array.
wave = cube_qso[:, 0, 0].wave.coord(np.arange(0, cube_qso.shape[0], 1.0))


# First get the summed QSO spectrum
spec = cube_qso.sum(axis=(1, 2))
flux_initial = spec.data
flux_initial = flux_initial / np.nanmedian(flux_initial)  # median normalize
fig, ax = plt.subplots(4, figsize=(7, 7), sharex=True)
ax[0].plot(wave, flux_initial, drawstyle='steps-mid', color='black', label=r'$\rm initial\ spectrum$')

# Now we will get a robust spectrum.
# Mask the region outside of r1
# cube_qso.mask_region(yx, args.r0, unit_center=None, unit_radius=None, inside=True)
# cube_qso.mask_region(yx, args.r1, unit_center=None, unit_radius=None, inside=False)


# Start qso light subtraction
# cube_copy = cube.mask_region(yx, args.r, unit_center=None, unit_radius=None, inside=False)
image_mask = cube[200, :, :].mask_region(yx, args.r, unit_center=None, unit_radius=None, inside=False)

nY, nX = image_white.shape
fluxArray = np.zeros((len(wave), nY, nX))
ratioArray = np.zeros((len(wave), nY, nX))

# loop through each pixel
xArray = range(0, cube.shape[2])
yArray = range(0, cube.shape[1])
for x in xArray:
   for y in yArray:
      # print(x, y)
      if image_mask[y, x] is True:
         # print('Masked')
         pass
      else:
        # print('Not Masked')
        # extract the spectrum
        sp = cube[:, y, x]
        thisFlux = sp.data
        mask = thisFlux.mask
        index = np.where(mask == False)

        if len(index) == 0:
            pass
        else:
            thisMed = np.nanmedian(thisFlux[index])
            thisFlux = thisFlux / thisMed

            coeff = np.polyfit(wave, thisFlux, 1)
            slopes[i] = coeff[0]
            if thisMed > 0:

                ratio = median_filter(flux_initial / thisFlux, 101)

                thisFlux_fixed = thisFlux * ratio
                fluxArray[:, i] = thisFlux_fixed
                ratioArray[:, i] = ratio
                # print('Not Masked')



for sp in iter_spe(cube):

    if
    thisFlux = sp.data
    mask = thisFlux.mask
    index = np.where(mask == False)

    if len(index) == 0:
        pass
    else:
        thisMed = np.nanmedian(thisFlux[index])
        thisFlux = thisFlux / thisMed

        if args.ao:
            index = np.where((wave < 5800) | (wave > 5980) & np.isfinite(thisFlux) & (thisFlux != 0.0))
            coeff = np.polyfit(wave[index], thisFlux[index], 1)
        else:
            coeff = np.polyfit(wave, thisFlux, 1)
        slopes[i] = coeff[0]
        if thisMed > 0:

            ratio = median_filter(flux_initial / thisFlux, 101)

            thisFlux_fixed = thisFlux * ratio
            fluxArray[:, i] = thisFlux_fixed
            ratioArray[:, i] = ratio


            ax[1].plot(wave, thisFlux, drawstyle='steps-mid')
            ax[2].plot(wave, ratio, drawstyle='steps-mid')

            ax[3].plot(wave, thisFlux_fixed, drawstyle='steps-mid')

        else:
            fluxArray[:, i] = np.nan
            ratioArray[:, i] = np.nan
        i = i + 1

# Calculate the median
flux_median = np.nanmedian(fluxArray, 1)
ax[0].plot(wave, flux_median, drawstyle='steps-mid', color='red', linestyle=':', label=r'$\rm robust\ spectrum$')
ax[0].legend()
ax[0].minorticks_on()
ax[1].minorticks_on()
ax[3].set_xlabel(r'$\rm wavelength \ [\AA]$')
ax[3].set_ylabel(r'$\rm Normalized\ flux$')
ax[2].set_ylabel(r'$\rm smoothed\ ratio$')
ax[1].set_ylabel(r'$\rm Normalized\ flux$')
ax[0].set_ylabel(r'$\rm Normalized\ flux$')
ax[0].legend()
fig.tight_layout()
plt.savefig(args.f.replace('.fits', '_QSO.pdf'))

#
qso = Table()
qso['wave'] = wave
qso['flux'] = flux_median
qso.write(args.f.replace('.fits', '_QSO.fits'), overwrite=True)


# Subtract the model and write-out the result
print('Writing...')
print(np.shape(ratioArray.reshape(len(wave), nX, nY)))
cube_qso_light = cube_copy.clone(data_init=ratio_array.reshape, var_init=np.zeros)
cube[:, :, :] = cube[:, :, :] - cube_qso_light
cube.write(args.f.replace('.fits', '_subtracted.fits'))


print('Done')