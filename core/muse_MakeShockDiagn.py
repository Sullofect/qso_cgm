import os
import pymysql
import numpy as np
import pandas as pd
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.table import Table
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

# Load fitted result
path_fit_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                             'RegionLinesRatio', 'RegionLinesRatio_dered.fits')
data_fit_info = fits.getdata(path_fit_info, 1, ignore_missing_end=True)
flux_OII, flux_OIII, flux_Hbeta, flux_NeV = data_fit_info['flux_OII'], data_fit_info['flux_OIII5008'], \
                                            data_fit_info['flux_Hbeta'],data_fit_info['flux_NeV3346']
dflux_OII, dflux_OIII, dflux_Hbeta, dflux_NeV = data_fit_info['dflux_OII'], data_fit_info['dflux_OIII5008'], \
                                                data_fit_info['dflux_Hbeta'], data_fit_info['dflux_NeV3346']
region = data_fit_info['region']

# Change it to 3 sigma limit
# S3, S4
mask_S3, mask_S4, mask_B4 = region == 'S3', region == 'S4', region == 'B4_new'
flux_OII[mask_S3], flux_OII[mask_S4], flux_OII[mask_B4] = 0.3 + 2.9, 1.34 + 4.17, 0.31 + 0.99
flux_OIII[mask_S3], flux_OIII[mask_S4], flux_OIII[mask_B4] = 6.27 + 2.44, 3.39 + 3.14, 0.83 + 0.40
flux_Hbeta[mask_S3], flux_Hbeta[mask_S4], flux_Hbeta[mask_B4] = 0.27 + 0.73, 0.28 + 0.52, 0.24 + 0.24
flux_NeV[mask_S3], flux_NeV[mask_S4], flux_NeV[mask_B4] = 0.15 + 0.18, 0.18 + 0.27, 0.18 + 0.18

#
dflux_OII[mask_S3], dflux_OII[mask_S4], dflux_OII[mask_B4] = np.sqrt(0.1 ** 2 + 0.1 ** 2), \
                                                             np.sqrt(0.18 ** 2 + 0.20 ** 2), \
                                                             np.sqrt(0.11 ** 2 + 0.16 ** 2)
dflux_OIII[mask_S3], dflux_OIII[mask_S4], dflux_OIII[mask_B4] = np.sqrt(0.22 ** 2 + 0.22 ** 2), \
                                                                np.sqrt(0.10 ** 2 + 0.12 ** 2), \
                                                                np.sqrt(0.06 ** 2 + 0.11 ** 2)
dflux_Hbeta[mask_S3], dflux_Hbeta[mask_S4], dflux_Hbeta[mask_B4] = np.sqrt(0.09 ** 2 + 0.09 ** 2), \
                                                                   np.sqrt(0.08 ** 2 + 0.09 ** 2), \
                                                                   np.sqrt(0.08 ** 2 + 0.08 ** 2)
dflux_NeV[mask_S3], dflux_NeV[mask_S4], dflux_NeV[mask_B4] = np.sqrt(0.05 ** 2 + 0.06 ** 2), \
                                                             np.sqrt(0.06 ** 2 + 0.09 ** 2), \
                                                             np.sqrt(0.06 ** 2 + 0.06 ** 2)

#
# path_fit_info_S3S4B4 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'RegionLinesRatio',
#                                     'RegionLinesRatio_dered_S3S4.fits')
# data_fit_info_S3S4B4 = fits.getdata(path_fit_info, 1, ignore_missing_end=True)
# flux_OII_S3S4B4, flux_OIII_S3S4B4, flux_Hbeta_S3S4B4 = data_fit_info_S3S4B4['flux_OII'] \
#                                                        + data_fit_info_S3S4B4['flux_OII_wing'], \
#                                                        data_fit_info_S3S4B4['flux_OIII5008'] \
#                                                        + data_fit_info_S3S4B4['flux_OIII5008_wing'], \
#                                                        data_fit_info_S3S4B4['flux_Hbeta'] \
#                                                        + data_fit_info_S3S4B4['flux_Hbeta_wing']
# dflux_OII_S3S4B4, dflux_OIII_S3S4B4, dflux_Hbeta_S3S4B4 = np.sqrt(data_fit_info_S3S4B4['dflux_OII'] ** 2
#                                                                   + data_fit_info_S3S4B4['dflux_OII_wing'] ** 2), \
#                                                           np.sqrt(data_fit_info_S3S4B4['dflux_OIII'] ** 2
#                                                                   + data_fit_info_S3S4B4['dflux_OIII_wing'] ** 2),\
#                                                           np.sqrt(data_fit_info_S3S4B4['dflux_Hbeta'] ** 2
#                                                                   + data_fit_info_S3S4B4['dflux_Hbeta_wing'] ** 2),
# dflux_NeV_S3S4B4 = np.sqrt(data_fit_info_S3S4B4['dflux_NeV'] ** 2 + data_fit_info_S3S4B4['dflux_NeV_wing'] ** 2)

#
# mask_S3, mask_S4, mask_B4 = region == 'S3', region == 'S4', region == 'B4'
# flux_OII[mask_S3], flux_OII[mask_S4], flux_OII[mask_B4] = flux_OII_S3S4B4[0], flux_OII_S3S4B4[1], 0.31 + 0.99
# flux_OIII[mask_S3], flux_OIII[mask_S4], flux_OIII[mask_B4] = flux_OIII_S3S4B4[0], flux_OIII_S3S4B4[1], 0.83 + 0.4
# flux_Hbeta[mask_S3], flux_Hbeta[mask_S4], flux_Hbeta[mask_B4] = flux_Hbeta_S3S4B4[0], flux_Hbeta_S3S4B4[1], 0.48
# flux_NeV[mask_S3], flux_NeV[mask_S4], flux_NeV[mask_B4] = 3 * dflux_NeV_S3S4B4[0], 3 * dflux_NeV_S3S4B4[1], 0.36


# Fitted result
OIII_OII = flux_OIII / flux_OII
OIII_OII_log = np.log10(OIII_OII)
OIII_Hbeta = flux_OIII / flux_Hbeta
OIII_Hbeta_log = np.log10(OIII_Hbeta)
OIII_Hbeta_3sig = np.log10(flux_OIII / 3 / dflux_Hbeta)
NeV_OIII = flux_NeV / flux_OIII
NeV_OIII_log = np.log10(NeV_OIII)
NeV_OII = flux_NeV / flux_OII
NeV_OII_log = np.log10(NeV_OII)
NeV_OII_3sig = np.log10(3 * dflux_NeV / flux_OII)
#
OIII_OII_err = (flux_OIII / flux_OII) * \
               np.sqrt((dflux_OII / flux_OII) ** 2 + (dflux_OIII / flux_OIII) ** 2)
OIII_OII_err_log = OIII_OII_err / np.log(10) / OIII_OII
OIII_Hbeta_err = (flux_OIII / flux_Hbeta) * \
               np.sqrt((dflux_OIII / flux_OIII) ** 2 + (dflux_Hbeta / flux_Hbeta) ** 2)
OIII_Hbeta_err_log = OIII_Hbeta_err / np.log(10) / OIII_Hbeta
NeV_OIII_err = (flux_NeV / flux_OIII) * \
               np.sqrt((dflux_NeV / flux_NeV) ** 2 + (dflux_OIII / flux_OIII) ** 2)
NeV_OIII_err_log = NeV_OIII_err / np.log(10) / NeV_OIII
NeV_OII_err = (flux_NeV / flux_OII) * \
               np.sqrt((dflux_NeV / flux_NeV) ** 2 + (dflux_OII / flux_OII) ** 2)
NeV_OII_err_log = NeV_OII_err / np.log(10) / NeV_OII

# Comparison
path_info_n10 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy_grid',
                             'iteramodel_SP_solar_n10_grid1.txt')
path_info_n100 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy_grid',
                              'iteramodel_SP_solar_n100_grid1.txt')
path_info_n10_NeV = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy_grid',
                                 'iteramodel_SP_solar_n10_NeV_grid1.txt')
path_info_n100_NeV = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy_grid',
                                  'iteramodel_SP_solar_n100_NeV_grid1.txt')
data_n10 = np.loadtxt(path_info_n10, encoding="ISO-8859-1")
data_n100 = np.loadtxt(path_info_n100, encoding="ISO-8859-1")
data_n10_NeV = np.loadtxt(path_info_n10_NeV, encoding="ISO-8859-1")
data_n100_NeV = np.loadtxt(path_info_n100_NeV, encoding="ISO-8859-1")

#
shck_vel_n10, mag_fld_n10 = data_n10[:, 1], data_n10[:, 0]
shck_vel_n100, mag_fld_n100 = data_n100[:, 1], data_n100[:, 0]

#
OIII_Hbeta_n10 = np.log10(data_n10[:, 2])
OIII_Hbeta_n100 = np.log10(data_n100[:, 2])
OIII_OII_n10 = np.log10(data_n10[:, 4] / data_n10[:, 5])
OIII_OII_n100 = np.log10(data_n100[:, 4] / data_n100[:, 5])
NeV_OII_n10 = np.log10(data_n10_NeV[:, 4] / data_n10_NeV[:, 5])
NeV_OII_n100 = np.log10(data_n100_NeV[:, 4] / data_n100_NeV[:, 5])


#
# mask = [5::3. ]
size_n10 = (len(np.unique(mag_fld_n10)), len(np.unique(shck_vel_n10)))
size_n100 = (len(np.unique(mag_fld_n100)), len(np.unique(shck_vel_n100)))
shck_vel_n10_mat = np.asarray(shck_vel_n10).reshape(size_n10).T[0:-5:2, 3::1]
mag_fld_n10_mat = np.asarray(mag_fld_n10).reshape(size_n10).T[0:-5:2, 3::1]
OIII_Hbeta_n10_mat = np.asarray(OIII_Hbeta_n10).reshape(size_n10).T[0:-5:2, 3::1]
OIII_OII_n10_mat = np.asarray(OIII_OII_n10).reshape(size_n10).T[0:-5:2, 3::1]
NeV_OII_n10_mat = np.asarray(NeV_OII_n10).reshape(size_n10).T[0:-5:2, 3::1]

#
idx_100_200 = np.s_[:-30:2, 0::1]
idx_200_1000 = np.s_[-31::2, 0::1]
shck_vel_n100_mat = np.asarray(shck_vel_n100).reshape(size_n100).T
mag_fld_n100_mat = np.asarray(mag_fld_n100).reshape(size_n100).T
OIII_Hbeta_n100_mat = np.asarray(OIII_Hbeta_n100).reshape(size_n100).T
OIII_OII_n100_mat = np.asarray(OIII_OII_n100).reshape(size_n100).T
NeV_OII_n100_mat = np.asarray(NeV_OII_n100).reshape(size_n100).T
# print(shck_vel_n10_mat, mag_fld_n10_mat)
print(shck_vel_n100_mat[idx_100_200], mag_fld_n100_mat[idx_100_200])
print(shck_vel_n100_mat[idx_200_1000], mag_fld_n100_mat[idx_200_1000])

# SQL query
host = os.environ['MdB_HOST']
user = os.environ['MdB_USER']
passwd = os.environ['MdB_PASSWD']
port = os.environ['MdB_PORT']

# Connect to the database
# db = pymysql.connect(host=host, user=user, passwd=passwd, port=int(port), db='3MdBs')

# Figure
fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300, )
plt.subplots_adjust(wspace=0.3, hspace=None)
# for i_den in [100]:
#
#     code = """SELECT shock_params.shck_vel AS shck_vel, shock_params.mag_fld AS mag_fld,
#                          log10(emis_VI.OIII_5007/(emis_VI.OII_3726 + emis_VI.OII_3729)) AS OIII_OII,
#                          log10(emis_VI.OIII_5007/emis_VI.HI_4861) AS OIII_Hbeta
#                          FROM shock_params
#                          INNER JOIN emis_VI ON emis_VI.ModelID=shock_params.ModelID
#                          INNER JOIN abundances ON abundances.AbundID=shock_params.AbundID
#                          WHERE emis_VI.model_type='shock_plus_precursor'
#                          AND abundances.name='Allen2008_Solar'
#                          AND shock_params.ref='Allen08'
#                          AND shock_params.shck_vel>=100
#                          AND shock_params.shck_vel<=1000
#                          AND shock_params.preshck_dens=""" + str(i_den) + """\n ORDER BY shck_vel, mag_fld;"""
#
#     result = pd.read_sql(code, con=db)
#     result.to_csv('/Users/lzq/Dropbox/Data/CGM/ShockDiagnosis.txt')
#
#     # Resize data
#     size = (len(np.unique(result.shck_vel)), len(np.unique(result.mag_fld)))
#     shck_vel_mat = np.asarray(result.shck_vel).reshape(size)[::3, 3::2]
#     mag_fld_mat = np.asarray(result.mag_fld).reshape(size)[::3, 3::2]
#     OIII_OII_model = np.asarray(result.OIII_OII).reshape(size)[::3, 3::2]
#     OIII_Hbeta_model = np.asarray(result.OIII_Hbeta).reshape(size)[::3, 3::2]
#     print(shck_vel_mat, mag_fld_mat)

    # # Figure
    # ax.errorbar(OIII_Hbeta_log, OIII_OII_log, xerr=OIII_Hbeta_err_log, yerr=OIII_OII_err_log,
    #             fmt='.k', capsize=2, elinewidth=0.7, mfc='C1', ms=10, markeredgewidth=0.5)
    # for i in range(np.shape(shck_vel_mat)[0]):
    #     if i != np.shape(shck_vel_mat)[0] - 1:
    #         # ax.fill(np.hstack((OIII_Hbeta_model[i, :], OIII_Hbeta_model[i + 1, ::-1])),
    #         #         np.hstack((OIII_OII_model[i, :], OIII_OII_model[i + 1, ::-1])),
    #         #         color='C1', alpha=0.05 + i * 0.05)
    #         ax.fill(np.hstack((OIII_Hbeta_model_grid[i, :], OIII_Hbeta_model_grid[i + 1, ::-1])),
    #                 np.hstack((OIII_OII_model_grid[i, :], OIII_OII_model_grid[i + 1, ::-1])),
    #                 color='C2', alpha=0.05 + i * 0.05)
    # # ax.plot(OIII_Hbeta_model[:, :], OIII_OII_model[:, :], '-', lw=1, alpha=0.4, color='C1')
    # ax.plot(OIII_Hbeta_model_grid[:, :], OIII_OII_model_grid[:, :], '-', lw=1, alpha=0.4, color='C2')

# Figure
print(NeV_OII_err_log)
for l, l_val, in enumerate(NeV_OII_err_log):
    # ax[0].annotate(region[l], xy=(OIII_Hbeta_log[l], OIII_OII_log[l]), size=10, color='red',
    #                verticalalignment='top', horizontalalignment='right')
    # ax[1].annotate(region[l], xy=(OIII_OII_log[l], NeV_OII_3sig[l]), size=10, color='red',
    #                verticalalignment='top', horizontalalignment='right')
    if l == 11 or l == 12:
        ax[0].errorbar(OIII_Hbeta_3sig[l], OIII_OII_log[l], yerr=OIII_OII_err_log[l], xerr=0, fmt='.k', capsize=2,
                       elinewidth=0.7, mfc='C1', ms=10, markeredgewidth=0.5)
        ax[0].arrow(OIII_Hbeta_3sig[l], OIII_OII_log[l], dx=0.1, dy=0, facecolor='C1', width=0.001, head_width=0.01,
                    head_length=0.02)
    elif l == 13:
        ax[0].errorbar(OIII_Hbeta_log[l], OIII_OII_log[l], yerr=OIII_OII_err_log[l], xerr=0, fmt='D', color='k', capsize=2,
                       elinewidth=0.7, mfc='C1', ms=5, markeredgewidth=0.5)
        ax[0].arrow(OIII_Hbeta_log[l], OIII_OII_log[l], dx=0.1, dy=0, facecolor='C1', width=0.001, head_width=0.01,
                    head_length=0.02)
    elif l == 2 or l == 3:
        ax[0].errorbar(OIII_Hbeta_log[l], OIII_OII_log[l], xerr=OIII_Hbeta_err_log[l],
                       yerr=OIII_OII_err_log[l], fmt='D', color='k', capsize=2,
                       elinewidth=0.7, mfc='C1', ms=5, markeredgewidth=0.5)
    else:
        ax[0].errorbar(OIII_Hbeta_log[l], OIII_OII_log[l], xerr=OIII_Hbeta_err_log[l], yerr=OIII_OII_err_log[l],
                       fmt='.k', capsize=2, elinewidth=0.7, mfc='C1', ms=10, markeredgewidth=0.5)
    if l == 0 or l == 1 or l == 5:
        ax[1].errorbar(OIII_OII_log[l], NeV_OII_log[l], xerr=OIII_OII_err_log[l], yerr=NeV_OII_err_log[l], fmt='.k',
                       capsize=2, elinewidth=0.7, mfc='C1', ms=10, markeredgewidth=0.5)
    elif l == 2 or l == 3 or l == 13:
        ax[1].errorbar(OIII_OII_log[l], NeV_OII_3sig[l], xerr=OIII_OII_err_log[l], yerr=0, fmt='D', color='k', capsize=2,
                       elinewidth=0.7, mfc='C1', ms=5, markeredgewidth=0.5)
        ax[1].arrow(OIII_OII_log[l], NeV_OII_3sig[l], dy=-0.2, dx=0, facecolor='C1', width=0.001, head_width=0.01,
                    head_length=0.02)
    else:
        ax[1].errorbar(OIII_OII_log[l], NeV_OII_3sig[l], xerr=OIII_OII_err_log[l], yerr=0, fmt='.k', capsize=2,
                       elinewidth=0.7, mfc='C1', ms=10, markeredgewidth=0.5)
        ax[1].arrow(OIII_OII_log[l], NeV_OII_3sig[l], dy=-0.2, dx=0, facecolor='C1', width=0.001, head_width=0.01,
                    head_length=0.02)
# for i in range(np.shape(shck_vel_n10_mat)[0]):
#     if i != np.shape(shck_vel_n10_mat)[0] - 1:
#         ax[0].fill(np.hstack((OIII_Hbeta_n10_mat[i, :], OIII_Hbeta_n10_mat[i + 1, ::-1])),
#                    np.hstack((OIII_OII_n10_mat[i, :], OIII_OII_n10_mat[i + 1, ::-1])),
#                    color='C1', alpha=0.1 + i * 0.05)
#         ax[1].fill(np.hstack((OIII_OII_n10_mat[i, :], OIII_OII_n10_mat[i + 1, ::-1])),
#                    np.hstack((NeV_OII_n10_mat[i, :], NeV_OII_n10_mat[i + 1, ::-1])),
#                    color='C1', alpha=0.1 + i * 0.05)
# ax[0].plot(OIII_Hbeta_n10_mat[:, :], OIII_OII_n10_mat[:, :], '-', lw=1, alpha=0.4, color='C1')
# ax[1].plot(OIII_OII_n10_mat[:, :], NeV_OII_n10_mat[:, :], '-', lw=1, alpha=0.4, color='C1')
for i in range(np.shape(shck_vel_n100_mat[idx_100_200])[0]):
    if i != np.shape(shck_vel_n100_mat[idx_100_200])[0] - 1:
        ax[0].fill(np.hstack((OIII_Hbeta_n100_mat[idx_100_200][i, :], OIII_Hbeta_n100_mat[idx_100_200][i + 1, ::-1])),
                   np.hstack((OIII_OII_n100_mat[idx_100_200][i, :], OIII_OII_n100_mat[idx_100_200][i + 1, ::-1])),
                   color='C1', alpha=0.1 + i * 0.05)
        ax[1].fill(np.hstack((OIII_OII_n100_mat[idx_100_200][i, :], OIII_OII_n100_mat[idx_100_200][i + 1, ::-1])),
                   np.hstack((NeV_OII_n100_mat[idx_100_200][i, :], NeV_OII_n100_mat[idx_100_200][i + 1, ::-1])),
                   color='C1', alpha=0.1 + i * 0.05)
ax[0].plot(OIII_Hbeta_n100_mat[idx_100_200][:, :], OIII_OII_n100_mat[idx_100_200][:, :], '-', lw=1, alpha=0.4, color='C1')
ax[1].plot(OIII_OII_n100_mat[idx_100_200][:, :], NeV_OII_n100_mat[idx_100_200][:, :], '-', lw=1, alpha=0.4, color='C1')
for i in range(np.shape(shck_vel_n100_mat[idx_200_1000])[0]):
    if i != np.shape(shck_vel_n100_mat[idx_200_1000])[0] - 1:
        ax[0].fill(np.hstack((OIII_Hbeta_n100_mat[idx_200_1000][i, :], OIII_Hbeta_n100_mat[idx_200_1000][i + 1, ::-1])),
                   np.hstack((OIII_OII_n100_mat[idx_200_1000][i, :], OIII_OII_n100_mat[idx_200_1000][i + 1, ::-1])),
                   color='gray', alpha=0.07 + i * 0.05)
        ax[1].fill(np.hstack((OIII_OII_n100_mat[idx_200_1000][i, :], OIII_OII_n100_mat[idx_200_1000][i + 1, ::-1])),
                   np.hstack((NeV_OII_n100_mat[idx_200_1000][i, :], NeV_OII_n100_mat[idx_200_1000][i + 1, ::-1])),
                   color='gray', alpha=0.07 + i * 0.05)
ax[0].plot(OIII_Hbeta_n100_mat[idx_200_1000][:, :], OIII_OII_n100_mat[idx_200_1000][:, :], '-', lw=1, alpha=0.4, color='gray')
ax[1].plot(OIII_OII_n100_mat[idx_200_1000][:, :], NeV_OII_n100_mat[idx_200_1000][:, :], '-', lw=1, alpha=0.4, color='gray')
ax[0].set_xlim(0, 1.5)
ax[0].set_ylim(-0.8, 1)
ax[1].set_xlim(-0.8, 1)
ax[1].set_ylim(-3, 0)
ax[0].minorticks_on()
ax[1].minorticks_on()
ax[0].set_xlabel(r'$\mathrm{log([O \, III] / H\beta)}$', size=20)
ax[0].set_ylabel(r'$\mathrm{log([O \, III] / [O \, II])}$', size=20)
ax[1].set_xlabel(r'$\mathrm{log([O \, III] / [O \, II])}$', size=20)
ax[1].set_ylabel(r'$\mathrm{log([Ne \, V] / [O \, II])}$', size=20)
ax[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                  labelsize=20, size=5)
ax[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
ax[1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                  labelsize=20, size=5)
ax[1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/ShockDiagnosis.png', bbox_inches='tight')