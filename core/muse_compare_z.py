import os
import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
from matplotlib import rc
from astropy.io import ascii
from astropy.table import Table
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def compare_z(cat_sean=None, cat_will=None, z_qso=0.6282144177077355, name_qso='HE0238-1904'):
    path_s = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D', cat_sean)
    data_s = fits.getdata(path_s, 1, ignore_missing_end=True)
    path_w = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D', cat_will)
    data_w = fits.getdata(path_w, 1, ignore_missing_end=True)

    # Basic information in catalog
    ra_w, dec_w = data_w['ra'], data_w['dec']
    row_s, row_w = data_s['row'], data_w['row']
    ID_s, ID_w = data_s['id'], data_w['id']
    name_s, name_w = data_s['name'], data_w['name']
    ql_s, ql_w = data_s['quality'], data_w['quality']
    cl_s, cl_w = data_s['class'], data_w['class']
    z_s, z_w = data_s['redshift'], data_w['redshift']
    ct_s, ct_w = data_s['comment'], data_w['comment']
    cl_s_num, cl_w_num = np.zeros_like(cl_s), np.zeros_like(cl_w)

    # Array manipulation
    classes = ['galaxy', 'star', 'quasar', 'hizgal']
    for i in range(4):
        cl_s_num = np.where(cl_s != classes[i], cl_s_num, i)
        cl_w_num = np.where(cl_w != classes[i], cl_w_num, i)
    cl_s_num, cl_w_num = cl_s_num.astype(float), cl_w_num.astype(float)
    v_w = 3e5 * (z_w - z_qso) / (1 + z_qso)
    v_s = 3e5 * (z_s - z_qso) / (1 + z_qso)

    ql_mask = ma.masked_where(np.abs(ql_s - ql_w) == 0, row_s)
    row_ql_diff = row_s[~ql_mask.mask]
    name_ql_diff = name_s[~ql_mask.mask]
    ql_s_diff = ql_s[~ql_mask.mask]
    ql_w_diff = ql_w[~ql_mask.mask]

    cl_mask = ma.masked_where(np.abs(cl_s_num - cl_w_num) == 0, row_s)
    row_cl_diff = row_s[~cl_mask.mask]
    name_cl_diff = name_s[~cl_mask.mask]
    cl_s_diff = cl_s[~cl_mask.mask]
    cl_w_diff = cl_w[~cl_mask.mask]

    v_mask = ma.masked_where(np.abs(v_s - v_w) <= 20, row_s)
    row_z_diff = row_s[~v_mask.mask]
    ql_s_z_diff = ql_s[~v_mask.mask]
    ql_w_z_diff = ql_w[~v_mask.mask]
    name_z_diff = name_s[~v_mask.mask]
    z_s_diff = z_s[~v_mask.mask]
    z_w_diff = z_w[~v_mask.mask]
    v_s_diff = v_s[~v_mask.mask]
    v_w_diff = v_w[~v_mask.mask]

    Table_1 = Table()
    Table_1["Row"] = row_ql_diff
    Table_1['Name'] = name_ql_diff
    Table_1["Sean's quality"] = ql_s_diff
    Table_1["Will's quality"] = ql_w_diff
    ascii.write(Table_1, path_savetab + 'compare_quality.csv', format='ecsv', overwrite=True)

    Table_2 = Table()
    Table_2["Row"] = row_cl_diff
    Table_2['Name'] = name_cl_diff
    Table_2["Sean's class"] = cl_s_diff
    Table_2["Will's class"] = cl_w_diff
    ascii.write(Table_2, path_savetab + 'compare_class.csv', format='ecsv', overwrite=True)

    Table_3 = Table()
    Table_3["Row"] = row_z_diff
    Table_3['Name'] = name_z_diff
    Table_3["Sean's velocity"] = v_s_diff
    Table_3["Will's velocity"] = v_w_diff
    Table_3["Sean's Quality"] = ql_s_z_diff
    Table_3["Will's Quality"] = ql_w_z_diff
    Table_3["Sean's z"] = z_s_diff
    Table_3["Will's z"] = z_w_diff
    Table_3["Velocity diff"] = np.abs(v_s_diff - v_w_diff)
    ascii.write(Table_3, path_savetab + 'compare_velocity.csv', format='ecsv', overwrite=True)

    select_gal = np.where(cl_w == 'galaxy')
    row_gal = row_w[select_gal]
    ID_gal = ID_w[select_gal]
    z_gal = z_w[select_gal]
    name_gal = name_w[select_gal]
    ql_gal = ql_w[select_gal]
    ra_gal, dec_gal = ra_w[select_gal], dec_w[select_gal]

    select_qua = np.where((ql_gal == 1) | (ql_gal == 2))
    row_qua = row_gal[select_qua]
    ID_qua = ID_gal[select_qua]
    z_qua = z_gal[select_qua]
    v_qua = 3e5 * (z_qua - z_qso) / (1 + z_qso)
    name_qua = name_gal[select_qua]
    ql_qua = ql_gal[select_qua]
    ra_qua, dec_qua = ra_gal[select_qua], dec_gal[select_qua]

    bins_ggp = np.arange(-2000, 2200, 200)
    select_v = np.where((v_qua > bins_ggp[0]) * (v_qua < bins_ggp[-1]))
    row_ggp = row_qua[select_v]
    ID_ggp = ID_qua[select_v]
    z_ggp = z_qua[select_v]
    v_ggp = v_qua[select_v]
    name_ggp = name_qua[select_v]
    ql_ggp = ql_qua[select_v]
    ra_ggp, dec_ggp = ra_qua[select_v], dec_qua[select_v]
    output = np.array([bins_ggp, row_ggp, ID_ggp, z_ggp, v_ggp, name_ggp, ql_ggp, ra_ggp, dec_ggp], dtype=object)
    #
    filename = '/Users/lzq/Dropbox/Data/CGM/GalaxyInfo/gal_info.fits'
    if os.path.isfile(filename) is not True:
        t = Table()
        t['row'] = row_ggp
        t['ID'] = ID_ggp
        t['z'] = z_ggp
        t['v'] = v_ggp
        t['name'] = name_ggp
        t['ql'] = ql_ggp
        t['ra'] = ra_ggp
        t['dec'] = dec_ggp
        t.write(filename, format='fits', overwrite=True)
    return output
