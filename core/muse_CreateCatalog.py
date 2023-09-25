import cosmography
import redshift
import sys
import coord
import muse_kc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def MakeCatalog(qso=None, z_qso=None, ra_qso=None, dec_qso=None, priority_cut=None):
    path = '/Users/lzq/Dropbox/Data/CGM/MaskDesign/DES+LS+GAIA_{}.fits'.format(qso)
    data = fits.getdata(path, 1, ignore_missing_end=True)

    # Calculate object theta
    ra_DES, dec_DES = data['ra_1'], data['dec_1']
    ra_LS, dec_LS = data['ra_2'], data['dec_2']
    ra_Gaia, dec_Gaia = data['ra_3'], data['dec_3']

    #
    ra_object, dec_object = np.copy(ra_Gaia), np.copy(dec_Gaia)
    ra_object, dec_object = np.where(~np.isnan(ra_object), ra_object, ra_LS), \
                            np.where(~np.isnan(dec_object), dec_object, dec_LS)
    ra_object, dec_object = np.where(~np.isnan(ra_object), ra_object, ra_DES), \
                            np.where(~np.isnan(dec_object), dec_object, dec_DES)
    c_qso = SkyCoord(ra_qso * u.deg, dec_qso * u.deg, frame='fk5')
    c_object = SkyCoord(ra_object * u.deg, dec_object * u.deg, frame='fk5')
    theta = c_qso.separation(c_object).arcsecond
    ID_ls, ID_DES = data['ls_id'], data['coadd_object_id']

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
    print(mag_r_01Lstar)

    # Calculate the imaging depth
    galdepth_g = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_g'])))
    galdepth_r = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_r'])))
    galdepth_z = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_z'])))
    print(galdepth_r)

    # Give priority
    candidate_01Lstar_r = np.zeros_like(isstar)
    candidate_01Lstar_z = np.zeros_like(isstar)
    candidate_01Lstar = np.zeros_like(isstar)
    for i, i_val in enumerate(zArray):
        index = np.where((isstar == 0) * (theta < theta_500kpc[i]) * (mag_r < mag_r_01Lstar[i])
                         * (mag_r < galdepth_r) * (mag_r > 16.0))
        candidate_01Lstar_r[index] = 1

        index = np.where((isstar == 0) * (theta < theta_500kpc[i]) * (mag_z < mag_z_01Lstar[i])
                         * (mag_z < galdepth_z) * (mag_z > 16.0))
        candidate_01Lstar_z[index] = 1

    index = np.where((candidate_01Lstar_r == 1) | (candidate_01Lstar_z == 1))
    candidate_01Lstar[index] = 1
    print(candidate_01Lstar[np.where(candidate_01Lstar == 1)])

    # Create galaxy catalog
    targets = np.arange(len(isstar))
    index = np.where((isstar == 0) * ((mag_r < 23.5) | (mag_z < 22.5)) * (mag_r > 15) * theta < 600)
    targets = targets[index]
    priority = (mag_r[index] + mag_z[index]) / 2
    # plt.figure()
    # plt.hist(priority, range=[0, 40])
    # plt.show()

    index_1 = np.where(~np.isfinite(priority))
    priority[index_1] = mag_r[index][index_1]

    index_2 = np.where(~np.isfinite(priority))
    priority[index_2] = mag_z[index][index_2]

    index_3 = np.where(~np.isfinite(priority))
    priority[index_3] = mag_g[index][index_3]

    index_4 = np.where(candidate_01Lstar[index] != 1)
    priority[index_4] = priority[index_4] + 20 + theta[index][index_4] / 60.0

    targets = targets[np.where(np.isfinite(priority))]
    priority = priority[np.where(np.isfinite(priority))]

    #
    index = np.argsort(priority)
    targets = targets[index]
    priority = priority[index]

    if priority_cut is not None:
        select_priority = np.where((priority > priority_cut[0]) * (priority < priority_cut[1]))
        targets = targets[select_priority]
        priority = priority[select_priority]

    coords_string = np.asarray(c_object.to_string(style='hmsdms', sep=':', precision=2))
    ra_string = np.asarray(c_object.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True))
    dec_string = np.asarray(c_object.dec.to_string(unit=u.degree, sep=':', precision=2, pad=True))

    path_gal_dat = '/Users/lzq/Dropbox/Data/CGM/MaskDesign/{}_galaxies.dat'.format(qso)
    path_gal_reg = '/Users/lzq/Dropbox/Data/CGM/MaskDesign/{}_galaxies.reg'.format(qso)
    path_star_dat = '/Users/lzq/Dropbox/Data/CGM/MaskDesign/{}_stars.dat'.format(qso)
    path_star_reg = '/Users/lzq/Dropbox/Data/CGM/MaskDesign/{}_stars.reg'.format(qso)

    # Save tiles
    path_tile_info = '/Users/lzq/Dropbox/Data/CGM/MaskDesign/{}_tiles.dat'.format(qso)
    np.savetxt(path_tile_info, np.unique(data['brickname']), fmt="%s")

    # Select galaxies
    ID_gal = list(map(''.join, zip(np.full_like(targets, '@', dtype=str), np.asarray(ID_ls[targets], dtype=str))))
    gal_cat = np.array([ID_gal, coords_string[targets], np.round(priority, decimals=2)]).T
    np.savetxt(path_gal_dat, gal_cat, fmt="%s")

    gal_reg = list(map(''.join, zip(np.full_like(targets, 'fk5; circle(', dtype='<U15'),
                                    ra_string[targets], np.full_like(targets, ', ', dtype=str),
                                    dec_string[targets], np.full_like(targets, ', ', dtype=str),
                                    np.full_like(targets, '2") #', dtype='<U15'),
                                    np.full_like(targets, ' color="red"', dtype='<U15'))))
    # gal_reg = list(map(''.join, zip(np.full_like(targets, 'fk5; circle(', dtype='<U15'),
    #                                 ra_string[targets], np.full_like(targets, ', ', dtype=str),
    #                                 dec_string[targets], np.full_like(targets, ', ', dtype=str),
    #                                 np.full_like(targets, '2") # text={', dtype='<U15'), ID_gal,
    #                                 np.full_like(targets, '} color="red"', dtype='<U15'))))
    np.savetxt(path_gal_reg, gal_reg, fmt="%s")

    # Select alignment stars
    alignments = np.arange(len(isstar))
    select_star = np.where((isstar == 1) * (~np.isnan(ra_Gaia)) * (mag_r >= 17.0) * (mag_r < 19.5)
                           * (mag_g - mag_r > 0.0) * (mag_g - mag_r < 0.7))
    alignments = alignments[select_star]

    # Region file
    ID_star = list(map(''.join, zip(np.full_like(alignments, '*', dtype=str), np.asarray(ID_ls[alignments], dtype=str))))
    star_cat = np.array([ID_star, coords_string[alignments], np.round(mag_r[alignments], decimals=2)]).T
    np.savetxt(path_star_dat, star_cat, fmt="%s")
    star_reg = list(map(''.join, zip(np.full_like(alignments, 'fk5; circle(', dtype='<U15'),
                                     ra_string[alignments], np.full_like(alignments, ', ', dtype=str),
                                     dec_string[alignments], np.full_like(alignments, ', ', dtype=str),
                                     np.full_like(alignments, '2") # ', dtype='<U15'),
                                     np.full_like(alignments, 'color="blue"', dtype='<U20'))))
    # star_reg = list(map(''.join, zip(np.full_like(alignments, 'fk5; circle(', dtype='<U15'),
    #                                  ra_string[alignments], np.full_like(alignments, ', ', dtype=str),
    #                                  dec_string[alignments], np.full_like(alignments, ', ', dtype=str),
    #                                  np.full_like(alignments, '2") # text={', dtype='<U15'), ID_star,
    #                                  np.full_like(alignments, '} color="blue"', dtype='<U20'))))
    np.savetxt(path_star_reg, star_reg, fmt="%s")


MakeCatalog(qso='HE0238-1904', z_qso=0.6282, ra_qso=40.13577, dec_qso=-18.86427, priority_cut=None)
