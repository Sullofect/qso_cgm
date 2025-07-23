#!/usr/bin/env python
import argparse

import numpy as np
from astropy.table import Table
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
import astropy.units as u
import sys
import time
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

parser = argparse.ArgumentParser(description='Subtract continuum with polynomial fitting')
parser.add_argument('-m', metavar='m', help='MUSE datacube filename (without .fits)', required=True, type=str)
parser.add_argument('-s', metavar='s', help='filename to save the subtracted cube to (without .fits)', required=True,
                    type=str)

parser.add_argument('-b0', metavar='s', help='lower bound of the blue continuum region', required=True, type=float)
parser.add_argument('-b1', metavar='s', help='upper bound of the blue continuum region', required=True, type=float)

parser.add_argument('-r0', metavar='s', help='lower bound of the red continuum region', required=True, type=float)
parser.add_argument('-r1', metavar='s', help='lower bound of the red continuum region', required=True, type=float)

parser.add_argument('-l0', metavar='s',
                    help='upper bound for the emission-line region to mask out when fitting continuum', required=False,
                    type=float, default=-999.0)
parser.add_argument('-l1', metavar='s',
                    help='upper bound for the emission-line region to mask out when fitting continuum', required=False,
                    type=float, default=-999.0)

parser.add_argument('-o', metavar='s', help='Polynomial order', required=False, type=int, default=3)

parser.add_argument('-ra', metavar='ra', help='ra center for subcube (decimal degrees)', required=False, type=float,
                    default=-999.0)
parser.add_argument('-dec', metavar='dec', help='dec center for subcube (decimal degrees)', required=False, type=float,
                    default=-999.0)
parser.add_argument('-radius', metavar='radius', help='dec center for subcube (arcseconds)', required=False, type=float,
                    default=-999.0)
parser.add_argument('-cpu', metavar='cpus to use', help='cpus to use', required=False, type=int, default=4)

args = parser.parse_args()

if args.m == args.s:
    print('Input and output cube have the same filename, aborting')
    sys.exit()

print('Reading in the data cube')
cube = Cube('{}.fits'.format(args.m))
print(cube)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(cube.data[0, :, :], origin='lower', aspect='auto', interpolation='nearest')
# plt.show()
# raise ValueError('Testing')

print('Selecting wavelength region)')
cube = cube.select_lambda(args.b0, args.r1)
print(cube)

if (args.ra > 0) & (args.dec > -90) & (args.radius > 0):
    print('Selecting subcub around ra={:0.5f}, dec={:0.5f} with radius r={:0.1f} arcsec'.format(args.ra, args.dec,
                                                                                                args.radius))
    if args.radius < 60:
        cube = cube.subcube((args.dec, args.ra), args.radius)
    print(cube)

# print('Creating empty continuum cube')
# continuum = cube.clone(data_init=np.empty, var_init=np.zeros)

cube = cube.select_lambda(args.b0, args.r1)
# Make a blue continuum and red continuum median and save
cube_blue = cube.select_lambda(args.b0, args.b1)
cube_red = cube.select_lambda(args.r0, args.r1)

# Computing median
image_blue = cube_blue.median(axis=0)
image_red = cube_red.median(axis=0)

# Convert to erg/s/cm^2/arcsec^2
pixel_size = cube_blue.wcs.get_step() * 3600
image_line = cube_blue.sum(axis=0)
image_blue = image_blue * 1e-20 / pixel_size[0] / pixel_size[1] * cube_blue.wave.get_step()

# Convert to erg/s/cm^2/arcsec^2
pixel_size = cube_red.wcs.get_step() * 3600
image_line = cube_red.sum(axis=0)
image_red = image_red * 1e-20 / pixel_size[0] / pixel_size[1] * cube_red.wave.get_step()

image_blue.write('{}_contblue.fits'.format(args.s))
image_red.write('{}_contred.fits'.format(args.s))

cube_red = 0
cube_blue = 0

print('Continuum image written')


# Define the continuum subtraction function
def spec_continuumsubtract(sp):
    if (args.l0 > 0) & (args.l1 > 0):
        sp.mask_region(args.l1, args.r0)

    if len(np.where(sp.mask == False)[0]) > 3:
        co = sp.poly_spec(3, weight=True)

        sp.unmask()
        co.unmask()

        sp[:] = sp[:] - co[:]
    return sp


print('Iterating through spaxels and performing continuum fit')
time0 = time.time()

# cube = cube.loop_spe_multiprocessing(f=spec_continuumsubtract, verbose=True, cpu=args.cpu)

for sp in iter_spe(cube):
  if (args.l0 > 0) & (args.l1 > 0):
     sp.mask_region(args.l0, args.l1)


  if len(np.where(sp.mask == False)[0]) > 3:
     co = sp.poly_spec(3, weight=True)

     i = 1

     sp.unmask()
     co.unmask()
     sp[:] = sp[:] - co[:]


time1 = time.time()
print('Continuum fit and subtraction took {:0.2f} seconds'.format(time1 - time0))

# Add continuum subtracted info to the fits header.
cube.primary_header['csub_orig'] = '{}.fits'.format(args.m)
cube.primary_header['csub_b0'] = args.b0
cube.primary_header.comments['csub_b0'] = 'lower bound for blue end of the continuum region (Angstrom)'
cube.primary_header['csub_b1'] = args.b1
cube.primary_header.comments['csub_b1'] = 'upper bound for blue end of the continuum region (Angstrom)'
cube.primary_header['csub_r0'] = args.r0
cube.primary_header.comments['csub_r0'] = 'lower bound for red end of the continuum region (Angstrom)'
cube.primary_header['csub_r1'] = args.r1
cube.primary_header.comments['csub_r1'] = 'upper bound for red end of the continuum region (Angstrom)'
cube.primary_header['csub_l0'] = args.l0
cube.primary_header.comments['csub_l0'] = 'blue end of the line region masked for continuum subtraction (Angstrom)'
cube.primary_header['csub_l1'] = args.l1
cube.primary_header.comments['csub_l1'] = 'red end of the line region masked for continuum subtraction (Angstrom)'
cube.primary_header['csub_ra'] = args.ra
cube.primary_header.comments[
    'csub_ra'] = 'RA coordinate center for subtraction subcube (decimal decrees; -999 if not used)'
cube.primary_header['csub_dec'] = args.dec
cube.primary_header.comments[
    'csub_dec'] = 'DEC coordinate center for subtraction subcute (decimal decrees; -999 if not used)'
cube.primary_header['csub_rad'] = args.radius
cube.primary_header.comments['csub_rad'] = 'size of the subtraction subcute around ra, dec(arcsec; -999 if not used)'

print('Writing subtracted cube')
cube.write('{}.fits'.format(args.s))

# Create a surface brightness map as well.
cube_line = cube.select_lambda(args.l0, args.l1)

# Sum the flux density to get flux assuming standard MUSE units of 1e-20 erg/s/cm^2/Ang in flux density
pixel_size = cube_line.wcs.get_step() * 3600
image_line = cube_line.sum(axis=0)

# Convert to erg/s/cm^2/arcsec^2
image_line = image_line * 1e-20 / pixel_size[0] / pixel_size[1] * cube.wave.get_step()

image_line.primary_header['BUNIT'] = 'erg/s/cm**2/arcsec**2'
image_line.write('{}_SB.fits'.format(args.s))
