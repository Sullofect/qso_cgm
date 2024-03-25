#!/usr/bin/env python

# import necessary classes
import os
import sys
import argparse
import numpy as np
from scipy.signal import savgol_filter

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

# Find the QSO center
# print('Fitting gaussian to quasar position in the white-light image...')
gaussian = image_white.gauss_fit(center=(args.y, args.x), circular=True, fwhm=3, weight=True, unit_center=None)

# extract the subcube near the quasar
yx = (gaussian.center[0], gaussian.center[1])
cube_qso = cube.subcube(yx, args.r1 + 1, unit_center=None, unit_size=None)
yx_cube = yx
print(yx_cube)

# Get an updated centroid for the quasar based on the smaller cube which is likely more outlier resistant.
image_white = cube_qso[200, :, :]
print(image_white.shape)
gaussian_subcube = image_white.gauss_fit(center=np.array(image_white.shape) / 2, circular=True, fwhm=1.0, weight=True,
                                         unit_center=None)
yx = (gaussian_subcube.center[0], gaussian_subcube.center[1])
print(yx)

# Mask the region inside of r0 and outside of r1
cube_qso.mask_region(yx, args.r1, unit_center=None, unit_radius=None, inside=False)
# cube_qso.mask_region(yx, args.r0, unit_center=None, unit_radius=None, inside=True)

cube_qso.write('test.fits')
# Get the wavelength array.
# wave = cube_qso[:, 0, 0].wave.coord(np.arange(0, cube_qso.shape[0], 1.0))
wave = cube.wave.coord()

def interpLine(wv, ratio, wv1, wv2, wv3, wv4):
    # interpolate through narrow lines
    # blue window [wv1, wv2], red window [wv3, wv4]
    mask_b = (wv>wv1)&(wv<wv2)
    mask_r = (wv>wv3)&(wv<wv4)
    mask_mid = (wv>wv2)&(wv<wv3)
    med_b = np.median(ratio[mask_b])
    med_r = np.median(ratio[mask_r])
    interp = interp1d([(wv1+wv2)/2,(wv3+wv4)/2],[med_b,med_r])
    return mask_mid, interp(wv[mask_mid])

# First get the summed QSO spectrum
spec = cube_qso.sum(axis=(1, 2))
flux_initial = spec.data
# flux_initial = cube.data[:, 234, 226]
# flux_initial = savgol_filter(flux_initial, window_length=7, polyorder=1)
# wv1, wv2, wv3, wv4 = 7930, 7935, 7965, 7970
# mask, interp = interpLine(wave, flux_initial, wv1, wv2, wv3, wv4)
# flux_initial[mask] = interp

flux_initial_med = flux_initial / np.nanmedian(flux_initial)  # median normalize
fig, ax = plt.subplots(4, figsize=(7, 7), sharex=True)
ax[0].plot(wave, flux_initial_med, drawstyle='steps-mid', color='black', label=r'$\rm initial\ spectrum$')

# Now we will get a robust spectrum.
# Mask the region outside of r1
# cube_qso.mask_region(yx, args.r0, unit_center=None, unit_radius=None, inside=True)
# cube_qso.mask_region(yx, args.r1, unit_center=None, unit_radius=None, inside=False)


# Start qso light subtraction
image_mask = cube[300, :, :]
image_mask.unmask()
image_mask.mask_region(yx_cube, args.r, unit_center=None, unit_radius=None, inside=False)
mask = image_mask.mask.data
nY, nX = cube.shape[1], cube.shape[2]
fluxArray = np.zeros((len(wave), nY, nX))
ratioArray = np.zeros((len(wave), nY, nX))

# loop through each pixel
xArray = range(0, cube.shape[2])
yArray = range(0, cube.shape[1])
for x in xArray:
    for y in yArray:
        if mask[y, x] == 0:
            # extract the spectrum
            print(x, y)
            sp = cube[:, y, x]
            thisFlux = sp.data
            mask_flux = thisFlux.mask
            index = np.where(mask_flux == False)
            thisMed = np.nanmedian(thisFlux[index])
            thisFlux_Med = thisFlux.data / thisMed

            ratio = median_filter(thisFlux / flux_initial, size=91)
            thisFlux_fixed = flux_initial * ratio
            fluxArray[:, y, x] = thisFlux_fixed
            ratioArray[:, y, x] = ratio

            if ((x < 227) * (x > 224)) * ((y < 238) * (y > 232)):
                ax[1].plot(wave, thisFlux_Med, drawstyle='steps-mid')
                ax[2].plot(wave, ratio / thisMed, drawstyle='steps-mid')
                ax[3].plot(wave, thisFlux - thisFlux_fixed, drawstyle='steps-mid')
        else:
            pass


# Calculate the median
flux_median = np.nanmedian(np.where(fluxArray != 0, fluxArray, np.nan), axis=(1, 2))
# ax[0].plot(wave, flux_median, drawstyle='steps-mid', color='red', linestyle=':', label=r'$\rm robust\ spectrum$')
ax[0].legend()
ax[0].minorticks_on()
ax[1].minorticks_on()
ax[3].set_xlabel(r'$\rm wavelength \ [\AA]$')
ax[3].set_ylabel(r'$\rm Normalized\ flux$')
ax[2].set_ylabel(r'$\rm smoothed\ ratio$')
ax[1].set_ylabel(r'$\rm Normalized\ flux$')
ax[0].set_ylabel(r'$\rm Normalized\ flux$')
fig.tight_layout()
plt.savefig(args.f.replace('.fits', '_QSO.pdf'))

#
qso = Table()
qso['wave'] = wave
qso['flux'] = flux_initial
qso.write(args.f.replace('.fits', '_QSO.fits'), overwrite=True)


# Subtract the model and write-out the result
print('Writing...')
cube.data -= fluxArray
cube.write(args.f.replace('.fits', '_subtracted_ZQL.fits'))

print('Done')