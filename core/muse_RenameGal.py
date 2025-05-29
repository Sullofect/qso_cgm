import os
import numpy as np
from astropy import units as u
import astropy.io.fits as fits
from astropy.table import Table
from astropy.coordinates import FK5
from muse_compare_z import compare_z
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM


# Turn Galaxy label
def ReturnGalLabel(sort_row=False, mode='final', return_HST=False, return_bins=False, print_radec=False):
    # Load info
    ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                         cat_will='ESO_DEEP_offset_zapped_objects.fits')
    bins_final = ggp_info[0]
    row_final = ggp_info[1]
    ID_final = ggp_info[2]
    z_final = ggp_info[3]
    name_final = ggp_info[5]
    ql_final = ggp_info[6]
    ra_final = ggp_info[7]
    dec_final = ggp_info[8]

    col_ID = np.arange(len(row_final))
    select_array = np.sort(np.array([1, 4, 5, 6, 7, 13, 20, 27, 35, 36, 57, 62, 64, 68, 72, 78, 80, 81, 82, 83, 88, 92,
                                     93, 120, 129, 134, 140, 141, 149, 162, 164, 179, 181, 182]))  # No row=11
    select_gal = np.in1d(row_final, select_array)
    row_final = row_final[select_gal]
    ID_final = ID_final[select_gal]
    name_final = name_final[select_gal]
    z_final = z_final[select_gal]
    ra_final = ra_final[select_gal]
    dec_final = dec_final[select_gal]

    # Getting photometry zero point
    path_pho = '../../Data/CGM/config/gal_all/HE0238-1904_sex_gal_all.fits'
    data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)
    catalog = SkyCoord(data_pho['AlPHAWIN_J2000'], data_pho['DELTAWIN_J2000'], unit="deg")
    c = SkyCoord(ra_final, dec_final, unit="deg")
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)

    # Photometry
    data_pho_gal = data_pho[idx]
    ra_hst, dec_hst = data_pho_gal['AlPHAWIN_J2000'], data_pho_gal['DELTAWIN_J2000']
    galaxy_no_change = np.array([6, 7, 129, 140, 149, 181, 182])
    select_no_change = np.in1d(select_array, galaxy_no_change)
    ra_hst[select_no_change] = ra_final[select_no_change]
    dec_hst[select_no_change] = dec_final[select_no_change]

    if mode == 'initial':
        if return_HST:
            if return_bins:
                return row_final, ID_final, name_final, z_final, ra_hst, dec_hst, bins_final
            else:
                return row_final, ID_final, name_final, z_final, ra_hst, dec_hst
        else:
            return row_final, ID_final, name_final, z_final, ra_final, dec_final

    # Calculate the offset between MUSE and gaia
    ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
    ra_qso_gaia, dec_qso_gaia = 40.13576715640353, -18.86426977828008
    ra_qso_hst, dec_qso_hst = data_pho['AlPHAWIN_J2000'][1296], data_pho['DELTAWIN_J2000'][1296]
    ra_final = ra_final - (ra_qso_gaia - ra_qso_muse)
    dec_final = dec_final - (dec_qso_gaia - dec_qso_muse)
    ra_hst = ra_hst - (ra_qso_gaia - ra_qso_hst)
    dec_hst = dec_hst - (dec_qso_gaia - dec_qso_hst)

    # Change coordinate for galaxy with row = 1
    row_1_sort = np.where(row_final == 1)
    ra_final[row_1_sort] = 40.1440392
    dec_final[row_1_sort] = -18.8597159

    # z_qso = 0.6282144177077355
    # v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)

    # Report angular separation and ra dec
    skycoord_host = SkyCoord(ra_qso_gaia, dec_qso_gaia, unit='deg', frame=FK5)
    # print(skycoord_host.to_string('hmsdms', sep=':'))
    skycoord = SkyCoord(ra_hst, dec_hst, unit='deg', frame=FK5)
    # print(skycoord.to_string('hmsdms', sep=':'))
    sep_final = skycoord.separation(skycoord_host).arcsecond

    # Rename the galaxy
    sort_sep = np.argsort(sep_final)
    sep_final = sep_final[sort_sep]
    ra_final = ra_final[sort_sep]
    dec_final = dec_final[sort_sep]

    row_final = row_final[sort_sep]
    ID_final = ID_final[sort_sep]
    z_final = z_final[sort_sep]
    name_final = name_final[sort_sep]
    ID_sep_final = np.arange(len(sep_final)) + 1

    if print_radec:
        z = 0.6282144177077355
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_l = cosmo.angular_diameter_distance(z=z)
        ratio = (1 * u.radian).to(u.arcsec).value
        d_sep = (sep_final * d_l / ratio).to(u.kpc).value

        print(np.vstack((sep_final, d_sep, row_final, ID_sep_final,
                         skycoord[sort_sep].to_string('hmsdms', sep=':', precision=1))).T)

    if sort_row:
        sort_row = np.argsort(row_final)
        sep_final = sep_final[sort_row]
        ra_final = ra_final[sort_row]
        dec_final = dec_final[sort_row]
        row_final = row_final[sort_row]
        ID_final = ID_final[sort_row]
        z_final = z_final[sort_row]
        name_final = name_final[sort_row]
        ID_sep_final = ID_sep_final[sort_row]

    # print(sep_final.arcsecond)
    # print(row_final)
    # print(ID_sep_final)
    if mode == 'final':
        filename = '../../Dropbox/Data/CGM/GalaxyInfo/gal_info_re.fits'
        if os.path.isfile(filename) is not True:
            t = Table()
            t['ra'] = ra_final
            t['dec'] = dec_final
            t['row'] = row_final
            t['ID'] = ID_final
            t['z'] = z_final
            t['name'] = name_final
            t['G#'] = ID_sep_final
            t.write(filename, format='fits', overwrite=True)
        return ra_final, dec_final, row_final, ID_final, z_final, name_final, ID_sep_final

# ReturnGalLabel(print_radec=True)