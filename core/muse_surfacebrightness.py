#!/usr/bin/env python 
import argparse
import numpy as np
from astropy.table import Table
from mpdaf.obj import Cube, Image, WCS, WaveCoord, iter_spe
import astropy.units as u
import aplpy
from astropy.io import fits
import cosmography
from astropy.stats import mad_std
import coord
import sys
import os
#import time


parser = argparse.ArgumentParser(description='Making surface brightness image and profile')
parser.add_argument('-m', metavar='m', help='MUSE datacube root name', required=True, type=str)
parser.add_argument('-l', metavar='l', help='line name', required=True, type=str)
parser.add_argument('-ra', metavar='ra', help='ra center for subcube (decimal degrees)', required=True, type=float, default=-999.0)
parser.add_argument('-dec', metavar='dec', help='dec center for subcube (decimal degrees)', required=True, type=float, default=-999.0)
parser.add_argument('-radius', metavar='radius', help='dec center for subcube (arcseconds)', required=True, type=float, default=-999.0)
parser.add_argument('-z', metavar='z', help='redshift', required=True, type=float, default=-999.0)
parser.add_argument('-vmin', metavar='z', help='vmin', required=False, type=float, default=0.0)
parser.add_argument('-vmax', metavar='z', help='vmax', required=False, type=float, default=5e-17)
parser.add_argument('-fontsize', metavar='fontsize', help='fontsize', required=False, type=float, default=30.0)
parser.add_argument('-n', metavar='n', help='name', required=False, type=str, default='')
parser.add_argument('-contour', metavar='contour', help='contour SB level', required=False, type=float, default=0.0)
parser.add_argument('-smooth', metavar='smooth', help='smooth', required=False, type=float, default=2)
parser.add_argument('-kernel', metavar='smooth', help='smooth', required=False, type=str, default='box')
parser.add_argument('-unitswitch', metavar='unitswitch', help='unitswitch', required=False, type=int, default=1)




args = parser.parse_args()


# Create the file name
filename_SB = '{}_{}_SB.fits'.format(args.m, args.l)
filename_contred = '{}_{}_contred.fits'.format(args.m, args.l)
filename_contblue = '{}_{}_contred.fits'.format(args.m, args.l)

filename_SB_subtracted = '{}_subtracted_{}_SB.fits'.format(args.m, args.l)


print(filename_SB)
print(filename_contred)
print(filename_contblue)
print(filename_SB_subtracted)


continuum_red = Image(filename_contred)
continuum_blue = Image(filename_contred)
continuum = (continuum_blue + continuum_red)/2
print(continuum)


print('Finding continuum centroid')
gaussfit = continuum.gauss_fit(center=(args.dec, args.ra), fwhm=0.8)
print(gaussfit)
print(coord.coordtotheta(args.ra, args.dec, gaussfit.center[1], gaussfit.center[0]))
ra = gaussfit.center[1]
dec = gaussfit.center[0]


image = fits.getdata(filename_SB_subtracted, ext=1)
variance = fits.getdata(filename_SB_subtracted, ext=2)

header = fits.getheader(filename_SB_subtracted, ext=1)

# Calculate the surface brightness limit empirically.

print(args.n)
std = mad_std(image)
print('backgound standard deviation = {}'.format(std))
threeSigma_limit = std/np.sqrt(5*5)*3
print('three sigma SB limit = {}'.format(threeSigma_limit))

std_pipeline = np.nanmedian(np.sqrt(variance))
print('median error = {}'.format(std_pipeline))
threeSigma_limit = std_pipeline/np.sqrt(5*5)*3
print('three sigma SB limit pipeline = {}'.format(threeSigma_limit))


if args.unitswitch == True:
   image = image - np.nanmedian(image)
   
   image = image/1e-17
fits.writeto('temp.fits', image, overwrite=True)
fits.setval('temp.fits', 'CTYPE1', value=header['CTYPE1'])
fits.setval('temp.fits', 'CTYPE2', value=header['CTYPE2'])
fits.setval('temp.fits', 'EQUINOX', value=header['EQUINOX'])
fits.setval('temp.fits', 'CD1_1', value=header['CD1_1'])
fits.setval('temp.fits', 'CD2_1', value=header['CD2_1'])
fits.setval('temp.fits', 'CD1_2', value=header['CD1_2'])
fits.setval('temp.fits', 'CD2_2', value=header['CD2_2'])
fits.setval('temp.fits', 'CRPIX1', value=header['CRPIX1'])
fits.setval('temp.fits', 'CRPIX2', value=header['CRPIX2'])
fits.setval('temp.fits', 'CRVAL1', value=header['CRVAL1'])
fits.setval('temp.fits', 'CRVAL2', value=header['CRVAL2'])
fits.setval('temp.fits', 'LONPOLE', value=header['LONPOLE'])
fits.setval('temp.fits', 'LATPOLE', value=header['LATPOLE'])



# Create the surface brightness image
sb = aplpy.FITSFigure('temp.fits', north=True)

sb.recenter(ra, dec, args.radius/3600.0)


#sb.show_grayscale()
#sb.show_colorscale()
sb.show_colorscale(cmap='gist_heat_r',
                  stretch='linear', vmin=args.vmin, vmax=args.vmax)

sb.set_system_latex(True)
sb.add_colorbar(location='bottom')

sb.colorbar.set_axis_label_text(r'${\rm SB}\,\,{[\rm 10^{-17}\ erg\,cm^{-2}\,s^{-1}\,arcsec^{-2}]}$')
sb.colorbar.set_font(size=args.fontsize)
sb.colorbar.set_axis_label_font(size=args.fontsize)


sb.axis_labels.hide()
sb.tick_labels.hide()
sb.ticks.hide()
   
# Add scale bar
theta = cosmography.dtotheta(50, args.z)
sb.add_scalebar(theta/3600.0, linewidth=2)

sb.scalebar.set_corner('top left')
sb.scalebar.set_font_size(args.fontsize)
sb.scalebar.set_label(r"$\rm 50\ kpc =  {:0.0f}''$".format(theta))
sb.scalebar.set_color('black')


if args.contour > 0:
   sb.show_contour('temp.fits', levels=[args.contour], smooth=args.smooth, kernel=args.kernel, linewidths=2, colors='black')


if args.n != '':
   sb.add_label(0.98, 0.94, r'$\rm {}$'.format(args.n), size=args.fontsize, relative=True, horizontalalignment='right')

sb.show_markers(ra, dec, marker='*', s=1000, c='dodgerblue', edgecolor='black')


savename = filename_SB_subtracted.replace('.fits', '.pdf')
sb.save(savename)
os.system('open {}'.format(savename))



# Now read in the continuum image and narrow-band image






#annuli = Table()
#dtheta = 0.4
#thetaMin = 0.0
#thetaMax = 10.0
#r0 = np.arange(thetaMin, thetaMax, dtheta)
#r1 = r0 + dtheta
#r = (r0 + r1)/2
#
#annuli['r0'] = r0
#annuli['r1'] = r1
#annuli['r'] = r
#annuli['continuum_SB_mean'] = 0.0
#annuli['continuum_SB_median'] = 0.0
#annuli['continuum_SB_error'] = 0.0
#
#for annulus in annuli:
#   
#   #continuum.mask_ellipse(coord, annulus['r0'], 0, inside=True)
#   #continuum.mask_ellipse(coord, annulus['r1'], 0, inside=False)
#   
#
#   test = continuum_blue.ee(center=(args.dec, args.ra), radius=1.0)
#   print(test)
#   
#
#
#print(annuli)
   




