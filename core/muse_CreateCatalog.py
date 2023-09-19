import numpy as np
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import coord
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import cosmography
import redshift
import sys



def MakeCatalog(qso=None):
    path = '/Users/lzq/Dropbox/Data/MaskDesign/DES+LS+GAIA_{}.fits'.format(qso)
    data = fits.getdata(path, 1, ignore_missing_end=True)

    # Legacy survey
    flux_g = data['flux_g'] / data['mw_transmission_g']
    flux_r = data['flux_r'] / data['mw_transmission_r']
    flux_i = data['flux_i'] / data['mw_transmission_i']
    flux_z = data['flux_z'] / data['mw_transmission_z']

    fluxErr_g = 1 / np.sqrt(data['flux_ivar_g'])
    fluxErr_r = 1 / np.sqrt(data['flux_ivar_r'])
    fluxErr_i = 1 / np.sqrt(data['flux_ivar_i'])
    fluxErr_z = 1 / np.sqrt(data['flux_ivar_z'])

    mag_g = 22.5 - 2.5 * np.log10(flux_g)
    mag_r = 22.5 - 2.5 * np.log10(flux_r)
    mag_i = 22.5 - 2.5 * np.log10(flux_i)
    mag_z = 22.5 - 2.5 * np.log10(flux_z)

    isstar = np.zeros_like(mag_r)
    istar = np.where(~((data['type'] == 'PSF') * (mag_r < 21.5)), isstar, 1)


    #
    # redshift = {redshift: 0d, mag_r_01Lstar:0d, mag_z_01Lstar: 0d, theta_500kpc: 0d}
    zArray = fillarr(0.001, 0.05, 0.6)
    zArray = np.linspace(0.05, 0.6, 0.001)

    redshifts = replicate(redshift, n_elements(zArray))
    redshifts.redshift = zArray

    for i=0l, n_elements(redshifts)-1 do begin

    redshifts[i].theta_500kpc = rhototheta(500d, redshifts[i].redshift)

    redshifts[i].mag_r_01Lstar = absoluteMagToApparent(-19.0, 'Scd', 'SDSS_r', 'DECam_r', $redshifts[i].redshift)

    redshifts[i].mag_z_01Lstar = absoluteMagToApparent(-19.0, 'Scd', 'SDSS_r', 'DECam_z', $redshifts[i].redshift)


    redshift = {redshift: 0d, mag_r_01Lstar:0
    d, mag_z_01Lstar: 0
    d, theta_500kpc: 0
    d}
    zArray = fillarr(0.001, 0.05, 0.6)

    redshifts = replicate(redshift, n_elements(zArray))
    redshifts.redshift = zArray

    for i=0l, n_elements(redshifts)-1 do begin

    redshifts[i].theta_500kpc = rhototheta(500
    d, redshifts[i].redshift)

    redshifts[i].mag_r_01Lstar = absoluteMagToApparent(-19.0, 'Scd', 'SDSS_r', 'DECam_r', $
    redshifts[i].redshift)

    redshifts[i].mag_z_01Lstar = absoluteMagToApparent(-19.0, 'Scd', 'SDSS_r', 'DECam_z', $
    redshifts[i].redshift)

    endfor

