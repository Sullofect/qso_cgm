import muse_kc
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



def MakeCatalog(qso=None, z_qso=None, ra_qso=None, dec_qso=None):
    path = '/Users/lzq/Dropbox/Data/MaskDesign/DES+LS+GAIA_{}.fits'.format(qso)
    data = fits.getdata(path, 1, ignore_missing_end=True)


    # Calculate object theta
    ra_DES, dec_DES = data['ra_1'], data['dec_1']
    ra_LS, dec_LS = data['ra_2'], data['dec_2']
    ra_Gaia, dec_Gaia = data['ra_3'], data['dec_3']
    ra_object,

    c_qso = SkyCoord(ra_qso * u.deg, dec_qso * u.deg, frame='fk5')
    c_object = SkyCoord(ra_object * u.deg, dec_object * u.deg, frame='fk5')
    theta = c_qso.separation(c_object).arcsecond

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
    isstar = np.where(~((data['type'] == 'PSF') * (mag_r < 21.5)), isstar, 1)

    # Compute
    zArray = np.arange(0.05, z_qso + 0.05, 0.001)
    theta_500kpc = np.zeros_like(zArray)
    mag_r_01Lstar = np.zeros_like(zArray)
    mag_z_01Lstar = np.zeros_like(zArray)

    for i, i_val in enumerate(zArray):
        theta_500kpc[i] = 500 * cosmo.arcsec_per_kpc_proper(i_val).value  # pkpc
        mag_r_01Lstar[i] = muse_kc.abs2app(m_abs=-19.0, z=i_val, model='Scd', filter_e='SDSS_r', filter_o='DECam_r')
        mag_z_01Lstar[i] = muse_kc.abs2app(m_abs=-19.0, z=i_val, model='Scd', filter_e='SDSS_z', filter_o='DECam_z')

    # Give priority
    for i, i_val in enumerate(zArray):
        index = where(objects.isStar eq 0l and objects.theta lt redshifts[i].theta_500kpc $ AND objects.mag_r lt redshifts[i].mag_r_01Lstar
        AND objects.mag_r lt galdepth_r $ AND objects.mag_r gt 16.0)
        objects[index].candidate_01Lstar_r = 1l

        index = where(objects.isStar eq 0l and objects.theta lt redshifts[i].theta_500kpc $ AND objects.mag_z lt redshifts[i].mag_z_01Lstar
        AND objects.mag_z lt galdepth_z $ AND objects.mag_z gt 16.0)
        objects[index].candidate_01Lstar_r = 1l

    index = where(objects.candidate_01Lstar_r eq 1l or objects.candidate_01Lstar_z eq 1l)
    objects[index].candidate_01Lstar = 1l


    # Calculate the imaging depth
    galdepth_g = 22.5 - np.log10(1 / np.sqrt(data['galdepth_g']))
    galdepth_r = 22.5 - np.log10(1 / np.sqrt(data['galdepth_r']))
    galdepth_z = 22.5 - np.log10(1 / np.sqrt(data['galdepth_z']))



    # #  Calculate ra, dec as strings
    # for i=0l, n_elements(objects)-1 do begin
    #
    # radec_string = adstring(objects[i].ra, objects[i].dec, p=1)
    # ra_dec_string = strtrim(radec_string, 1)
    # radec_split = strsplit(ra_dec_string, '  ', / extract, / regex)
    # ra_string = radec_split[0]
    # dec_string = radec_split[1]
    #
    # ra_string = repstr(ra_string, ' ', ':')
    # dec_string = repstr(dec_string, ' ', ':')
    #
    # objects[i].ra_string = ra_string
    # objects[i].dec_string = dec_string
    #
    # ; Create
    # the
    # galaxy
    # target
    list


# Create galaxy catalog
index = np.where((isstar == 0) * ((mag_r < 23.5) | (mag_z < 22.5)) * (mag_r > 15) * objects.theta < 600)

targets = objects[index]
targets.priority = (targets.mag_r + targets.mag_z) / 2

index = where(~finite(targets.priority))
targets[index].priority = targets[index].mag_r

index = where(~finite(targets.priority))
targets[index].priority = targets[index].mag_z

index = where(~finite(targets.priority))
targets[index].priority = targets[index].mag_g

index = where(targets.candidate_01Lstar lt 1l)
targets[index].priority = targets[index].priority + 20 + targets[index].theta / 60.0

targets = targets[where(finite(targets.priority))]

index = sort(targets.priority)
targets = targets[index]

openw, lun, '../catalogs/Q0107_triplet_targets.dat', / get_lun
for i=0l, n_elements(targets)-1 do begin

printf, lun, '@' + targets[i].id + ' ' + targets[i].ra_string + ' ' + targets[i].dec_string + ' ' + number_formatter(
    targets[i].priority)

endfor
free_lun, lun

openw, lun, '../catalogs/Q0107_triplet_targets.reg', / get_lun
for i=0l, n_elements(targets)-1 do begin

printf, lun, 'fk5; circle(' + targets[i].ra_string + ', ' + targets[i].dec_string + ', 2")'

endfor
free_lun, lun

#  select alignment stars
alignments = objects[where(objects.isstar eq 1l AND objects.inGAIA eq 1l)]
alignments = alignments[where(alignments.mag_r ge 17.0 and alignments.mag_r lt19.5)]; alignments[(alignments['DES_R'] > 17.0) & (alignments['DES_R'] < 19.5)]
alignments = alignments[where(alignments.mag_g - alignments.mag_r ge 0.0 AND alignments.mag_g - alignments.mag_r le
0.7)];alignments[(alignments['DES_G'] - alignments['DES_R'] > 0.0) & (alignments['DES_G'] - alignments['DES_R'] < 0.7)]

# help, alignments

openw, lun, '../catalogs/Q0107_triplet_alignments.dat', / get_lun
for i=0l, n_elements(alignments)-1 do begin

printf, lun, '*' + alignments[i].id + ' ' + alignments[i].ra_string + ' ' + alignments[i].dec_string + ' ' + \
number_formatter(alignments[i].mag_r)

endfor
free_lun, lun

openw, lun, '../catalogs/Q0107_triplet_alignments.reg', / get_lun
for i=0l, n_elements(alignments)-1 do begin

printf, lun, 'fk5; circle(' + alignments[i].ra_string + ', ' + alignments[i].dec_string + ', 2") # text="' + alignments[i].id + '"'
endfor
free_lun, lun
