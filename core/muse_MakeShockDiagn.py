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

# Fitted result
OIII_OII = flux_OIII / flux_OII
OIII_OII_log = np.log10(OIII_OII)
OIII_Hbeta = flux_OIII / flux_Hbeta
OIII_Hbeta_log = np.log10(OIII_Hbeta)
NeV_OIII = flux_NeV / flux_OIII
NeV_OIII_log = np.log10(NeV_OIII)

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

# SQL query
host = os.environ['MdB_HOST']
user = os.environ['MdB_USER']
passwd = os.environ['MdB_PASSWD']
port = os.environ['MdB_PORT']

# Comparison
path_grid_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                             'cloudy_grid', 'iteramodel_SP_solar_n10_grid1.txt')
data_grid = np.loadtxt(path_grid_info, encoding="ISO-8859-1")
shck_grid, mag_fld_grid = data_grid[:, 1], data_grid[:, 0]
OIII_Hbeta_grid = np.log10(data_grid[:, 2])
OIII_OII_grid = np.log10(data_grid[:, 4] / data_grid[:, 5])

size_grid = (len(np.unique(mag_fld_grid)), len(np.unique(shck_grid)))
shck_vel_mat = np.asarray(shck_grid).reshape(size_grid).T[::3, 3::2]
mag_fld_mat = np.asarray(mag_fld_grid).reshape(size_grid).T[::3, 3::2]
OIII_OII_model_grid = np.asarray(OIII_OII_grid).reshape(size_grid).T[::3, 3::2]
OIII_Hbeta_model_grid = np.asarray(OIII_Hbeta_grid).reshape(size_grid).T[::3, 3::2]
print(shck_vel_mat, mag_fld_mat)


# Connect to the database
db = pymysql.connect(host=host, user=user, passwd=passwd, port=int(port), db='3MdBs')

fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
for i_den in [100]:

    code = """SELECT shock_params.shck_vel AS shck_vel, shock_params.mag_fld AS mag_fld, 
                         log10(emis_VI.OIII_5007/(emis_VI.OII_3726 + emis_VI.OII_3729)) AS OIII_OII, 
                         log10(emis_VI.OIII_5007/emis_VI.HI_4861) AS OIII_Hbeta 
                         FROM shock_params 
                         INNER JOIN emis_VI ON emis_VI.ModelID=shock_params.ModelID
                         INNER JOIN abundances ON abundances.AbundID=shock_params.AbundID
                         WHERE emis_VI.model_type='shock_plus_precursor'
                         AND abundances.name='Allen2008_Solar'
                         AND shock_params.ref='Allen08'
                         AND shock_params.shck_vel>=100
                         AND shock_params.shck_vel<=1000
                         AND shock_params.preshck_dens=""" + str(i_den) + """\n ORDER BY shck_vel, mag_fld;"""

    result = pd.read_sql(code, con=db)
    result.to_csv('/Users/lzq/Dropbox/Data/CGM/ShockDiagnosis.txt')

    # Resize data
    size = (len(np.unique(result.shck_vel)), len(np.unique(result.mag_fld)))
    shck_vel_mat = np.asarray(result.shck_vel).reshape(size)[::3, 3::2]
    mag_fld_mat = np.asarray(result.mag_fld).reshape(size)[::3, 3::2]
    OIII_OII_model = np.asarray(result.OIII_OII).reshape(size)[::3, 3::2]
    OIII_Hbeta_model = np.asarray(result.OIII_Hbeta).reshape(size)[::3, 3::2]
    # print(result)
    print(shck_vel_mat, mag_fld_mat)

    # Figure
    ax.errorbar(OIII_Hbeta_log, OIII_OII_log, xerr=OIII_Hbeta_err_log, yerr=OIII_OII_err_log,
                fmt='.k', capsize=2, elinewidth=0.7, mfc='C1', ms=10, markeredgewidth=0.5)
    for i in range(np.shape(shck_vel_mat)[0]):
        if i != np.shape(shck_vel_mat)[0] - 1:
            ax.fill(np.hstack((OIII_Hbeta_model[i, :], OIII_Hbeta_model[i + 1, ::-1])),
                    np.hstack((OIII_OII_model[i, :], OIII_OII_model[i + 1, ::-1])),
                    color='C1', alpha=0.05 + i * 0.05)
            ax.fill(np.hstack((OIII_Hbeta_model_grid[i, :], OIII_Hbeta_model_grid[i + 1, ::-1])),
                    np.hstack((OIII_OII_model_grid[i, :], OIII_OII_model_grid[i + 1, ::-1])),
                    color='C2', alpha=0.05 + i * 0.05)
    ax.plot(OIII_Hbeta_model[:, :], OIII_OII_model[:, :], '-', lw=1, alpha=0.4, color='C1')
    ax.plot(OIII_Hbeta_model_grid[:, :], OIII_OII_model_grid[:, :], '-', lw=1, alpha=0.4, color='C2')
ax.set_xlim(0, 1.5)
ax.set_ylim(-0.8, 1)
ax.minorticks_on()
ax.set_xlabel(r'$\mathrm{log([O \, III] / H\beta)}$', size=20)
ax.set_ylabel(r'$\mathrm{log([O \, III] / [O \, II])}$', size=20)
ax.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
               labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
               size=3)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/ShockDiagnosis.png', bbox_inches='tight')