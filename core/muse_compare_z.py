import os
import numpy as np
import pandas as pd
import numpy.ma as ma
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rc

path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'

# InteractiveShell.ast_node_interactivity = 'all'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

path_s = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects_sean.fits')
data_s = fits.getdata(path_s, 1, ignore_missing_end=True)
path_w = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects.fits')
data_w = fits.getdata(path_w, 1, ignore_missing_end=True)

ra_w, dec_w = data_w['ra'], data_w['dec']

row_s = data_s['row']
row_w = data_w['row']

ID_s = data_s['id']
ID_w = data_w['id']

name_s = data_s['name']
name_w = data_w['name']

ql_s = data_s['quality']
ql_w = data_w['quality']

cl_s = data_s['class']
cl_w = data_w['class']

cl_s_num = np.zeros_like(cl_s)
cl_w_num = np.zeros_like(cl_w)
classes = ['galaxy', 'star', 'quasar', 'hizgal']
for i in range(4):
    cl_s_num = np.where(cl_s != classes[i], cl_s_num, i)
    cl_w_num = np.where(cl_w != classes[i], cl_w_num, i)

cl_s_num = cl_s_num.astype(float)
cl_w_num = cl_w_num.astype(float)

z_s = data_s['redshift']
z_w = data_w['redshift']
z_qso = 0.6282144177077355

v_w = 3e5 * (z_w - z_qso) / (1 + z_qso)
v_s = 3e5 * (z_s - z_qso) / (1 + z_qso)

ct_s = data_s['comment']
ct_w = data_w['comment']

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

# Table 1
columns_1 = [r"Row", r"Name", r"Sean's qua", r"Will's qua"]
ql_compare_1 = np.stack([row_ql_diff, name_ql_diff, ql_s_diff, ql_w_diff], axis=1)
Table_1 = pd.DataFrame(ql_compare_1, index=1 + np.arange(len(row_ql_diff)), columns=columns_1)
print(Table_1)

columns_2 = [r"Row", r"Name", r"Sean's class", r"Will's class"]
ql_compare_2 = np.stack([row_cl_diff, name_cl_diff, cl_s_diff, cl_w_diff], axis=1)
Table_2 = pd.DataFrame(ql_compare_2, index=1 + np.arange(len(row_cl_diff)), columns=columns_2)
print(Table_2)

columns_3 = [r"Row", r"Name", r"Sean's v", r"Will's v", r"qua", r"Will' z", r"Sean' z", r"$\Delta v$"]
ql_compare_3 = np.stack(
    [row_z_diff, name_z_diff, v_s_diff, v_w_diff, ql_s_z_diff, z_w_diff, z_s_diff, np.abs(v_s_diff - v_w_diff)], axis=1)
Table_3 = pd.DataFrame(ql_compare_3, index=1 + np.arange(len(row_z_diff)), columns=columns_3)
print(Table_3)

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

bins = np.arange(-2000, 2200, 200)
select_z = np.where((v_qua > bins[0]) * (v_qua < bins[-1]))
row_final = row_qua[select_z]
ID_final = ID_qua[select_z]
z_final = z_qua[select_z]
v_final = v_qua[select_z]
name_final = name_qua[select_z]
ql_final = ql_qua[select_z]
ra_final, dec_final = ra_qua[select_z], dec_qua[select_z]

# fit
select_v = np.where((v_final > -2000) * (v_final < 2000))
v_fit = v_final[select_v]
mu, scale = norm.fit(v_fit)

# Normalization
nums, v_edge = np.histogram(v_fit, bins=bins)
normalization = np.sum(nums) * 200

#
rv = np.linspace(-2000, 2000, 1000)
plt.figure(figsize=(8, 5), dpi=300)
plt.vlines(0, 0, 11, linestyles='--', color='k', label=r"$\mathrm{QSO's \; redshift}$")
plt.hist(v_final[np.where(ql_final == 1)], bins=bins, facecolor='blue', histtype='stepfilled', alpha=0.5,
         label=r'$ \mathrm{Quality} = 1$')
plt.hist(v_final[np.where(ql_final == 2)], bins=bins, facecolor='brown', histtype='stepfilled', alpha=0.5,
         label=r'$ \mathrm{Quality} = 2$')
plt.plot(rv, normalization * norm.pdf(rv, mu, scale), '-r', lw=2, alpha=0.5,
         label=r'$\mu = $ ' + str("{0:.0f}".format(mu)) + r'$\mathrm{km/s}$, ' + '\n' + r'$\sigma = $ ' +
               str("{0:.0f}".format(scale)) + r'$\mathrm{km/s}$')
plt.xlim(-2000, 2000)
plt.ylim(0, 11)
plt.minorticks_on()
plt.xlabel(r'$\Delta v [\mathrm{km \; s^{-1}}]$', size=20)
plt.ylabel(r'$\mathrm{Numbers}$', size=20)
plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5, labelsize=20)
plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
plt.legend(prop={'size': 17}, framealpha=0, loc=1, fontsize=15)
plt.savefig(path_savefig + 'galaxy_velocity', bbox_inches='tight')
