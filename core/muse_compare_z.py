import os
import lmfit
import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import minimize
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
    return output

def PlotVelDis():
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
    z_final = z_final[select_gal]
    ra_final = ra_final[select_gal]
    dec_final = dec_final[select_gal]

    z_qso = 0.6282144177077355
    v_gal = 3e5 * (z_final - z_qso) / (1 + z_qso)

    # Determine whcih galaxy is below the line
    line = np.array([[40.1300330, 40.1417710], [-18.8587229, -18.8698312]])
    vector_ra = ra_final - line[0, 0]
    vector_dec = dec_final - line[1, 0]
    vector_line = np.array([line[0, 1] - line[0, 0], line[1, 1] - line[1, 0]])
    value = np.cross(np.vstack((vector_ra, vector_dec)).T,  vector_line)
    value_sort = value < 0
    row_above, row_below = row_final[value_sort], row_final[np.invert(value_sort)]
    v_above, v_below = v_gal[value_sort], v_gal[np.invert(value_sort)]

    # Plot the fitting
    # mu_all, scale_all = norm.fit(v_gal)
    # mu_above, scale_above = norm.fit(v_above)
    # mu_below, scale_below = norm.fit(v_below)

    # Normalization
    nums, v_edge = np.histogram(v_gal, bins=bins_final)
    normalization_all = np.sum(nums) * 200
    nums, v_edge = np.histogram(v_above, bins=bins_final)
    normalization_above = np.sum(nums) * 200
    nums, v_edge = np.histogram(v_below, bins=bins_final)
    normalization_below = np.sum(nums) * 200


    # Minimize likelihood
    def gauss(x, mu, sigma):
        return np.exp(- (x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)


    def loglikelihood(x_0, x):
        mu1, sigma1, mu2, sigma2, p1 = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
        return -1 * np.sum(np.log(p1 * gauss(x, mu1, sigma1) + (1 - p1) * gauss(x, mu2, sigma2)))

    # def loglikelihood(x_0, x):
    #     vals = x_0.valuesdict()
    #     mu1 = vals['mu1']
    #     sigma1 = vals['sigma1']
    #     mu2 = vals['mu2']
    #     sigma2 = vals['sigma2']
    #     p1 = vals['p1']
    #     return -1 * np.sum(np.log(p1 * gauss(x, mu1, sigma1) + (1 - p1) * gauss(x, mu2, sigma2)))


    guesses = [-80, 200, 500, 500, 0.3]
    bnds = ((-100, 0), (0, 500), (200, 1000), (0, 1000), (0, 1))
    result = minimize(loglikelihood, guesses, (v_gal), bounds=bnds, method="Powell")
    print(result.x)


    # Bootstrap
    result_array = np.zeros((50, 5))
    for i in range(len(result_array)):
        v_gal_i = np.random.choice(v_gal, replace=True, size=len(v_gal))
        result_i = minimize(loglikelihood, guesses, (v_gal_i), bounds=bnds, method="Powell")
        result_array[i, :] = result_i.x
    result_std = np.std(result_array, axis=0)


    # parameters_all = lmfit.Parameters()
    # parameters_all.add_many(('mu1', -80, True, -100, 0, None),
    #                         ('sigma1', 200, True, 0, 500, None),
    #                         ('mu2', 500, True, 200, 1000, None),
    #                         ('sigma2', 500, True, 0, 1000, None),
    #                         ('p1', 0.3, True, 0, 1, None))
    # out = lmfit.minimize(loglikelihood, parameters_all, args=(v_gal,), method='powell')
    # print(out.params.pretty_print())
    # print(lmfit.fit_report(out))

    # Plot
    rv = np.linspace(-2000, 2000, 1000)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.vlines(0, 0, 15, linestyles='--', color='k', label=r"$\mathrm{QSO's \; redshift}$")
    # plt.plot(rv, normalization_all * norm.pdf(rv, mu_all, scale_all), '-k', lw=1, alpha=1,
    #          label=r'$\mu = \, $' + str("{0:.0f}".format(mu_all)) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $' +
    #                str("{0:.0f}".format(scale_all)) + r'$\mathrm{\, km/s}$')
    # plt.plot(rv, normalization_above * norm.pdf(rv, mu_above, scale_above), '-r', lw=1, alpha=1,
    #          label=r'$\mu = \, $' + str("{0:.0f}".format(mu_above)) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $' +
    #                str("{0:.0f}".format(scale_above)) + r'$\mathrm{\, km/s}$')
    # plt.plot(rv, normalization_below * norm.pdf(rv, mu_below, scale_below), '-b', lw=1, alpha=1,
    #          label=r'$\mu = \, $' + str("{0:.0f}".format(mu_below)) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $' +
    #                str("{0:.0f}".format(scale_below)) + r'$\mathrm{\, km/s}$')
    plt.plot(rv, result.x[4] * normalization_all * norm.pdf(rv, result.x[0], result.x[1]), '-', c='purple', lw=1, alpha=1,
             label=r'$P_{1} = \,$' + str("{0:.2f}".format(result.x[4])) + r'$\pm$'
                   + str("{0:.2f}".format(result_std[4])) + '\n' + r'$\mu = \, $'
                   + str("{0:.0f}".format(result.x[0])) + r'$\pm$'
                   + str("{0:.0f}".format(result_std[0])) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $'
                   + str("{0:.0f}".format(result.x[1])) + r'$\pm$'
                   + str("{0:.0f}".format(result_std[1])) + r'$\mathrm{\, km/s}$', zorder=110)
    plt.plot(rv, (1 - result.x[4]) * normalization_all * norm.pdf(rv, result.x[2], result.x[3]), '-', c='orange', lw=1,
             alpha=1, label=r'$P_{2} = \,$' + str("{0:.2f}".format(1 - result.x[4])) + r'$\mp$'
                            + str("{0:.2f}".format(result_std[4])) + '\n' + r'$\mu = \, $'
                            + str("{0:.0f}".format(result.x[2])) + r'$\pm$'
                            + str("{0:.0f}".format(result_std[2])) + r'$\mathrm{\, km/s}$, ' + '\n' + r'$\sigma = \, $'
                            + str("{0:.0f}".format(result.x[3])) + r'$\pm$'
                            + str("{0:.0f}".format(result_std[3])) + r'$\mathrm{\, km/s}$', zorder=100)
    plt.plot(rv, result.x[4] * normalization_all * norm.pdf(rv, result.x[0], result.x[1]) +
             (1 - result.x[4]) * normalization_all * norm.pdf(rv, result.x[2], result.x[3]), '-k', lw=1, alpha=1)
    plt.hist(v_gal, bins=bins_final, color='k', histtype='step', label=r'$\mathrm{v_{all}}$')
    plt.hist(v_above, bins=bins_final, facecolor='orange', histtype='stepfilled', alpha=0.5, label=r'$\mathrm{v_{orange}}$')
    plt.hist(v_below, bins=bins_final, facecolor='purple', histtype='stepfilled', alpha=0.5, label=r'$\mathrm{v_{purple}}$')
    plt.xlim(-2000, 2000)
    plt.ylim(0, 12)
    plt.minorticks_on()
    plt.xlabel(r'$\Delta v [\mathrm{km \; s^{-1}}]$', size=20)
    plt.ylabel(r'$\mathrm{Numbers}$', size=20)
    plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                    labelsize=20)
    plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
    plt.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
    plt.savefig(path_savefig + 'galaxy_velocity_all', bbox_inches='tight')


# PlotVelDis()