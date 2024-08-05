import cosmography
import redshift
import sys
import coord
import muse_kc
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from regions import Regions
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def MakeCatalog(qso=None, filename=None, z_qso=None, ra_qso=None, dec_qso=None, priority_cut=None):
    path = '../../MaskDesign/{}_mask/DES+LS+GAIA_{}.fits'.format(filename, qso)
    data = fits.getdata(path, 1, ignore_missing_end=True)

    # Give ID
    ID_ls, ID_DES = data['ls_id'], data['coadd_object_id']
    ID = np.where(ID_ls != -9223372036854775808, ID_ls, ID_DES)

    # Calculate object theta
    ra_DES, dec_DES = data['ra_1'], data['dec_1']
    ra_LS, dec_LS = data['ra_2'], data['dec_2']
    ra_Gaia, dec_Gaia = data['ra_3'], data['dec_3']
    ra_object, dec_object = np.copy(ra_Gaia), np.copy(dec_Gaia)
    ra_object, dec_object = np.where(~np.isnan(ra_object), ra_object, ra_LS), \
                            np.where(~np.isnan(dec_object), dec_object, dec_LS)
    ra_object, dec_object = np.where(~np.isnan(ra_object), ra_object, ra_DES), \
                            np.where(~np.isnan(dec_object), dec_object, dec_DES)
    c_qso = SkyCoord(ra_qso * u.deg, dec_qso * u.deg, frame='fk5')
    print(c_qso)
    c_object = SkyCoord(ra_object * u.deg, dec_object * u.deg, frame='fk5')
    theta = c_qso.separation(c_object).arcsecond

    # Calculate the imaging depth
    galdepth_g = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_g'])))
    galdepth_r = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_r'])))
    galdepth_i = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_i'])))
    galdepth_z = 22.5 - np.log10(1 / np.sqrt(np.nanmedian(data['galdepth_z'])))
    galdepth_g = np.where(~np.isnan(galdepth_g), galdepth_g, 99)
    galdepth_r = np.where(~np.isnan(galdepth_r), galdepth_r, 99)
    galdepth_i = np.where(~np.isnan(galdepth_i), galdepth_i, 99)
    galdepth_z = np.where(~np.isnan(galdepth_z), galdepth_z, 99)

    # Legacy survey
    flux_g = data['flux_g'] / data['mw_transmission_g']
    flux_r = data['flux_r'] / data['mw_transmission_r']
    flux_i = data['flux_i'] / data['mw_transmission_i']
    flux_z = data['flux_z'] / data['mw_transmission_z']

    mag_g_LS = 22.5 - 2.5 * np.log10(flux_g)
    mag_r_LS = 22.5 - 2.5 * np.log10(flux_r)
    mag_i_LS = 22.5 - 2.5 * np.log10(flux_i)
    mag_z_LS = 22.5 - 2.5 * np.log10(flux_z)

    # DES
    mag_g_DES = data['mag_auto_g_dered']
    mag_r_DES = data['mag_auto_r_dered']
    mag_i_DES = data['mag_auto_i_dered']
    mag_z_DES = data['mag_auto_z_dered']
    mag_y_DES = data['mag_auto_y_dered']

    mag_g = np.where(~np.isnan(mag_g_LS), mag_g_LS, mag_g_DES)
    mag_r = np.where(~np.isnan(mag_r_LS), mag_r_LS, mag_r_DES)
    mag_i = np.where(~np.isnan(mag_i_LS), mag_i_LS, mag_i_DES)
    mag_z = np.where(~np.isnan(mag_z_LS), mag_z_LS, mag_z_DES)

    # Perform star-galaxy separation.
    # extended_coadd = ((data['SPREAD_MODEL_I'] + 3 * data['SPREADERR_MODEL_I']) > 0.005) * 1 +\
    #                  ((data['SPREAD_MODEL_I'] + data['SPREADERR_MODEL_I']) > 0.003) * 1 +\
    #                  ((data['SPREAD_MODEL_I'] - data['SPREADERR_MODEL_I']) > 0.003) * 1
    # isstar_DES = ((data['EXTENDED_COADD'] <= 1) & (data['MAG_AUTO_I'] <= 22.0)) * 1

    # isstar
    type_LS = data['type']
    type_DES = data['extended_class_coadd']
    type_DES = np.where((type_DES != 0), type_DES, 'PSF')
    type = np.where(type_LS != ' ', np.copy(type_LS), type_DES)

    isstar = np.zeros_like(mag_r)
    isstar = np.where(~((type == 'PSF') * (mag_r < 21.5)), isstar, 1)

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

    # Create galaxy catalog
    targets = np.arange(len(isstar))
    index = np.where((isstar == 0) * ((mag_r < 23.5) | (mag_z < 22.5)) * (mag_r > 15) * (theta < 600))
    targets = targets[index]
    priority = (mag_r[index] + mag_z[index]) / 2

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

    path_gal_dat = '../../MaskDesign/{}_mask/{}_@.dat'.format(filename, filename)
    path_gal_reg = '../../MaskDesign/{}_mask/{}_@.reg'.format(filename, filename)
    path_star_dat = '../../MaskDesign/{}_mask/{}_*.dat'.format(filename, filename)
    path_star_reg = '../../MaskDesign/{}_mask/{}_*.reg'.format(filename, filename)

    # Save tiles
    path_tile_info = '../../MaskDesign/{}_mask/{}_tiles.dat'.format(filename, filename)
    np.savetxt(path_tile_info, np.unique(data['brickname']), fmt="%s")

    # Select galaxies
    ID_gal = list(map(''.join, zip(np.full_like(targets, '@', dtype=str), np.asarray(ID[targets], dtype=str))))
    gal_cat = np.array([ID_gal, coords_string[targets], np.round(priority, decimals=2)]).T
    np.savetxt(path_gal_dat, gal_cat, fmt="%s")

    gal_reg = np.array(['# Region file format: DS9 version 4.1',
                        'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
                        'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                        'fk5'])

    gals_reg_2 = list(map(''.join, zip(np.full_like(targets, 'circle(', dtype='<U15'),
                                       ra_string[targets], np.full_like(targets, ', ', dtype=str),
                                       dec_string[targets], np.full_like(targets, ', ', dtype=str),
                                       np.full_like(targets, '2") # text={', dtype='<U15'), ID_gal,
                                       np.full_like(targets, '}', dtype='<U15'))))

    gal_reg = np.hstack((gal_reg, gals_reg_2))
    np.savetxt(path_gal_reg, gal_reg, fmt="%s")

    # For debugging
    # ID_gal_all = list(map(''.join, zip(np.full_like(targets, '@', dtype=str), np.asarray(ID, dtype=str))))
    # path_gal_reg_all = '../../Data/CGM/MaskDesign/{}_galaxies_all.reg'.format(qso)
    # gal_reg_all = list(map(''.join, zip(np.full_like(ra_object, 'fk5; circle(', dtype='<U15'),
    #                                 ra_string, np.full_like(ra_object, ', ', dtype=str),
    #                                 dec_string, np.full_like(ra_object, ', ', dtype=str),
    #                                 np.full_like(ra_object, '2") # text={', dtype='<U15'), ID_gal_all,
    #                                 np.full_like(ra_object, '} color="red"', dtype='<U15'))))
    # np.savetxt(path_gal_reg_all, gal_reg_all, fmt="%s")

    # Select alignment stars
    alignments = np.arange(len(isstar))
    select_star = np.where((isstar == 1) * (~np.isnan(ra_Gaia)) * (mag_r >= 17.0) * (mag_r < 19.5)
                           * (mag_g - mag_r > 0.0) * (mag_g - mag_r < 0.7))
    alignments = alignments[select_star]

    # * .dat file
    ID_star = list(map(''.join, zip(np.full_like(alignments, '*', dtype=str), np.asarray(ID[alignments], dtype=str))))
    star_cat = np.array([ID_star, coords_string[alignments], np.round(mag_r[alignments], decimals=2)]).T
    np.savetxt(path_star_dat, star_cat, fmt="%s")

    # * .reg file
    star_reg = np.array(['# Region file format: DS9 version 4.1',
                         'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
                         'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                         'fk5'])
    star_reg_2 = list(map(''.join, zip(np.full_like(alignments, 'circle(', dtype='<U15'),
                                       ra_string[alignments], np.full_like(alignments, ', ', dtype=str),
                                       dec_string[alignments], np.full_like(alignments, ', ', dtype=str),
                                       np.full_like(alignments, '5") # text={', dtype='<U15'), ID_star,
                                       np.full_like(alignments, '}', dtype='<U20'))))
    star_reg = np.hstack((star_reg, star_reg_2))
    np.savetxt(path_star_reg, star_reg, fmt="%s")


# def RefineCatalog(filename=None, priority_cut=None):
#     # Visually inspected catalog (.reg)
#     path_ac = '../../MaskDesign/{}/{}_@_ac.reg'.format(filename, filename)
#     regions_ac = np.loadtxt(path_ac, dtype=str, skiprows=3, comments='%')
#
#     # Full catalog
#     path = '../../MaskDesign/{}/{}_@.reg'.format(filename, filename)
#     regions = np.loadtxt(path, dtype=str, skiprows=3, comments='%')
#
#     idx = np.in1d(regions[:, 2], regions_ac[:, 2])
#     print("find {} number of objects in the refinement".format(len(idx[idx == True])))
#
#     path_gal_dat = '../../MaskDesign/{}/{}_@.dat'.format(filename, filename)
#     data_gal_dat = np.loadtxt(path_gal_dat, dtype=str)[idx]
#
#     if priority_cut is not None:
#         priority = np.asarray(data_gal_dat[:, 3], dtype=float)
#         select_priority = np.where((priority > priority_cut[0]) * (priority < priority_cut[1]))
#         data_gal_dat = data_gal_dat[select_priority]
#         path_gal_check_dat = '../../MaskDesign/{}/{}_@_ac_hp.dat'.format(filename, filename)
#     else:
#         path_gal_check_dat = '../../MaskDesign/{}/{}_@_ac.dat'.format(filename, filename)
#
#     # Remove object that are observed already
#     np.savetxt(path_gal_check_dat, data_gal_dat, fmt="%s")


# Select Star
def ConvertReg2Dat(dir=None, filename_p=None, p_type=None, filename_f=None, f_type=None, priority_cut=None, mode='keep_same'):
    # Visually inspected catalog (.reg)
    path_p = '../../MaskDesign/{}_mask/{}.reg'.format(dir, filename_p)
    if p_type == 'Sean':
        regions_p = np.loadtxt(path_p, dtype=str, comments='%')
        regions_p_ID = regions_p[:, 8]
    else:
        regions_p = np.loadtxt(path_p, dtype=str, skiprows=3, comments='%')
        regions_p_ID = regions_p[:, 2]  # Make sure there is no color column in ds9 reg file
    print(regions_p_ID)

    # Full catalog
    path_f = '../../MaskDesign/{}_mask/{}.reg'.format(dir, filename_f)
    path_f_dat = '../../MaskDesign/{}_mask/{}.dat'.format(dir, filename_f)
    if f_type == 'Sean':
        regions_f = np.loadtxt(path_f, dtype=str, comments='%')
        regions_f_ID = regions_f[:, 8]
    else:
        regions_f = np.loadtxt(path_f, dtype=str, skiprows=3, comments='%')
        regions_f_ID = regions_f[:, 2]
    print(regions_f_ID)

    idx = np.in1d(regions_f_ID, regions_p_ID)
    if mode == 'keep_same':
        idx = idx
        print("find {} number of common objects".format(len(idx[idx == True])))
    elif mode == 'find_diff':
        idx = ~idx
        filename_p += '_d'
        path_p_d = '../../MaskDesign/{}_mask/{}.reg'.format(dir, filename_p)
        regions_p_d = regions_f[idx]
        print("find {} number of different objects".format(len(idx[idx == True])))

    data_p_dat = np.loadtxt(path_f_dat, dtype=str)[idx]

    if priority_cut is not None:
        priority = np.asarray(data_p_dat[:, 3], dtype=float)
        select_priority = np.where((priority > priority_cut[0]) * (priority < priority_cut[1]))
        data_p_dat = data_p_dat[select_priority]
        path_p_dat = '../../MaskDesign/{}_mask/{}_hp.dat'.format(dir, filename_p)
        if mode == 'find_diff':
            regions_p_d = regions_p_d[select_priority]
    else:
        path_p_dat = '../../MaskDesign/{}_mask/{}.dat'.format(dir, filename_p)

    # Remove object that are observed already
    np.savetxt(path_p_dat, data_p_dat, fmt="%s")
    print(regions_p_d[:, 0])
    if mode == 'find_diff':
        if f_type != 'Sean':
            rows3 = np.array(['# Region file format: DS9 version 4.1',
                              'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
                              'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                              'fk5'])
            regions_p_d = list(map(' '.join, zip(regions_p_d[:, 0], regions_p_d[:, 1], regions_p_d[:, 2])))
            regions_p_d = np.hstack((rows3, regions_p_d))
        np.savetxt(path_p_d, regions_p_d, fmt="%s")




# MakeCatalog(qso='HE0238-1904', filename='HE0238', z_qso=0.6282, ra_qso=40.13577, dec_qso=-18.86427, priority_cut=None)
# ConvertReg2Dat(dir='HE0238', filename_f='HE0238_@', filename_p='HE0238_@_ac', priority_cut=[0, 40])
# ConvertReg2Dat(dir='HE0238', filename_f='HE0238_@', filename_p='HE0238_@_ac', priority_cut=None)
# ConvertReg2Dat(dir='.', filename_f='HE0238_@_ac', filename_p='HE0238i1', mode='find_diff', p_type='Sean',
# priority_cut=[0, 40])
# ConvertReg2Dat(dir='.', filename_f='HE0238_@_ac', filename_p='HE0238i1', mode='find_diff', p_type='Sean',
# priority_cut=None)
# convert .reg to dat
# ConvertReg2Dat(dir='HE0238', filename_f='HE0238_*', filename_p='HE0238i2_*', mode='keep_same', p_type=None,
# priority_cut=None)

#
# ConvertReg2Dat(dir='.', filename_f='HE0238i1_d', filename_p='HE0238i2', mode='find_diff', f_type=None,
#                p_type='Sean', priority_cut=[0, 40])
# ConvertReg2Dat(dir='.', filename_f='HE0238i1_d', filename_p='HE0238i2', mode='find_diff', f_type=None,
#                p_type='Sean', priority_cut=None)

# ConvertReg2Dat(dir='HE0238', filename_f='HE0238_*', filename_p='HE0238i3_*', mode='keep_same', p_type=None,
#                priority_cut=None)


# For HE0439
# MakeCatalog(qso='HE0439-5254', filename='HE0439', z_qso=1.0530, ra_qso=70.05020, dec_qso=-52.80486, priority_cut=None)
# ConvertReg2Dat(dir='HE0439', filename_f='HE0439_@', filename_p='HE0439_@_ac', priority_cut=[0, 40])
# ConvertReg2Dat(dir='HE0439', filename_f='HE0439_@', filename_p='HE0439_@_ac', priority_cut=None)
# ConvertReg2Dat(dir='HE0439', filename_f='HE0439_*', filename_p='HE0439i1_*', mode='keep_same', p_type=None, priority_cut=None)

# i2
# ConvertReg2Dat(dir='HE0439', filename_f='HE0439_@_ac', filename_p='HE0439i1', mode='find_diff', p_type='Sean',
#                priority_cut=[0, 40])
# ConvertReg2Dat(dir='HE0439', filename_f='HE0439_@_ac', filename_p='HE0439i1', mode='find_diff', p_type='Sean',
#                priority_cut=None)
# ConvertReg2Dat(dir='HE0439', filename_f='HE0439_*', filename_p='HE0439i2_*', mode='keep_same', p_type=None, priority_cut=None)

# 3C57
# i1 48 hp objects, 121 total objects
# MakeCatalog(qso='3C57', filename='3C57', z_qso=0.6718, ra_qso=30.488247, dec_qso=-11.542621, priority_cut=None)
# ConvertReg2Dat(dir='3C57', filename_f='3C57_@', filename_p='3C57_@_ac', priority_cut=[0, 40])
# ConvertReg2Dat(dir='3C57', filename_f='3C57_@', filename_p='3C57_@_ac', priority_cut=None)
# ConvertReg2Dat(dir='3C57', filename_f='3C57_*', filename_p='3C57i1_*', mode='keep_same', p_type=None, priority_cut=None)

# i2 38 hp objects, 140 total objects
# ConvertReg2Dat(dir='3C57', filename_f='3C57_@_ac', filename_p='3C57i1', mode='find_diff', p_type='Sean',
#                priority_cut=[0, 40])
# ConvertReg2Dat(dir='3C57', filename_f='3C57_@_ac', filename_p='3C57i1', mode='find_diff', p_type='Sean',
#                priority_cut=None)
# ConvertReg2Dat(dir='3C57', filename_f='3C57_*', filename_p='3C57i2_*', mode='keep_same', p_type=None, priority_cut=None)

# i3 28 hp objects, 147 total objects
# ConvertReg2Dat(dir='3C57', filename_f='3C57i1_d', filename_p='3C57i2', mode='find_diff', p_type='Sean',
#                priority_cut=[0, 40])
# ConvertReg2Dat(dir='3C57', filename_f='3C57i1_d', filename_p='3C57i2', mode='find_diff', p_type='Sean',
#                priority_cut=None)
# ConvertReg2Dat(dir='3C57', filename_f='3C57_*', filename_p='3C57i3_*', mode='keep_same', p_type=None, priority_cut=None)

# Test
ConvertReg2Dat(dir='3C57', filename_f='3C57i2_d', filename_p='3C57i3', mode='find_diff', p_type='Sean',
               priority_cut=[0, 40])