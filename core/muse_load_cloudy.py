import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import palettable.scientific.sequential as sequential_s
from matplotlib import rc
from matplotlib import cm
from PyAstronomy import pyasl
from astropy import units as u
# from muse_load_cloudy import format_cloudy
from matplotlib.colors import ListedColormap
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def load_cloudy(filename='alpha_2.0', path='/Users/lzq/Dropbox/Data/CGM/cloudy/'):
    # Ionization parameter
    u_10 = np.array([])
    u_40 = np.array([])
    f = open(path + filename + '.out', 'r')
    for i, d in enumerate(f.readlines()):
        d = d.split(' ')
        if len(d) > 15:
            if d[11].startswith('U(1.0----)'):
                u_10 = np.hstack((u_10, d[11].split(':')[1]))
                u_40 = np.hstack((u_40, d[14].split(':')[1]))
    # Load alpha
    alpha = np.ones(len(u_10)) * float(filename.split('_')[1])

    # H den Grid parameter
    grid = np.genfromtxt(path + filename + '.grd', delimiter=None)
    hden_grid = grid[:, 6]

    # Line profile
    line = np.genfromtxt(path + filename + '.lin', delimiter=None)
    OIII5008, OIII4960, OIIIboth = line[:, 2], line[:, 3], line[:, 4]
    OII3727, OII3730, Hbeta = line[:, 5], line[:, 6], line[:, 7]

    # Temperature profile
    temp = np.genfromtxt(path + filename + '.avr', delimiter=None)

    data = np.vstack((alpha, hden_grid, u_10, u_40, OIII5008, Hbeta, OIII5008, OII3727 + OII3730, temp))

    return data.astype(float).T


def format_cloudy(filename=None, path=None):
    for i in range(len(filename)):
        if i == 0:
            output = load_cloudy(filename[i], path=path)
        else:
            c_i = load_cloudy(filename[i], path=path)
            output = np.vstack((output, c_i))
    return output

filename = ['alpha_2.0', 'alpha_1.7', 'alpha_1.4', 'alpha_1.2']
output = format_cloudy(filename=filename, path='/Users/lzq/Dropbox/Data/CGM/cloudy/50kpc_z=1/')

# Load grid, plot line ratio density, ionization parameter !!!
alpha = output[:, 0]
hden = output[:, 1]
logu_10 = np.log10(output[:, 2])
logu_40 = np.log10(output[:, 3])
OIII_Hbeta = output[:, 4] / output[:, 5]
OIII_OII = output[:, 6] / output[:, 7]
temp = output[:, 8]
alpha_mat = alpha.reshape((len(filename), 21))
temp_mat = temp.reshape((len(filename), 21))
logu_10_mat = logu_10.reshape((len(filename), 21))
logu_40_mat = logu_40.reshape((len(filename), 21))
OIII_Hbeta_mat = OIII_Hbeta.reshape((len(filename), 21))
OIII_OII_mat = OIII_OII.reshape((len(filename), 21))

# print(output[:, 1])
# # print(alpha)
# # print(logu)
# print(OIII_Hbeta)
# print(alpha_mat)

plt.figure(figsize=(8, 5), dpi=300)
for i in range(len(filename)):
    plt.plot(hden[:21], temp_mat[i, :], '-', label=r'$\mathrm{Alpha = }$' + str(alpha_mat[i, 0]))
# plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                  labelsize=15)
# plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
plt.xlabel(r"$\mathrm{log_{10}[Hydrogen \, density]}$", size=15)
plt.ylabel(r"$\mathrm{Temperature \, [K]}$", size=15)
plt.legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_temp_grid', bbox_inches='tight')


# Plot line ratio
f, axarr = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
# axarr.plot(np.log10(OIII_Hbeta), np.log10(OIII_OII), '.')
x_1 = np.log10(OIII_Hbeta_mat)
y_1 = np.log10(OIII_OII_mat)
y_1_sort = np.sort(y_1, axis=1)
x_1_sort = np.take_along_axis(x_1, np.argsort(y_1, axis=1), axis=1)
axarr.plot(np.log10(OIII_Hbeta_mat), np.log10(OIII_OII_mat), '-', color='grey', lw=1, alpha=0.3)
axarr.fill(np.hstack((x_1_sort[0, :], x_1_sort[1, ::-1])), np.hstack((y_1_sort[0, :], y_1_sort[1, ::-1])),
           color='grey', alpha=0.2)
axarr.fill(np.hstack((x_1_sort[1, :], x_1_sort[2, ::-1])), np.hstack((y_1_sort[1, :], y_1_sort[2, ::-1])),
           color='grey', alpha=0.4)
axarr.fill(np.hstack((x_1_sort[2, 2:], x_1_sort[3, ::-1][2:])), np.hstack((y_1_sort[2, 2:], y_1_sort[3, ::-1][2:])),
           color='grey', alpha=0.6)
axarr.set_xlabel(r'$\mathrm{log([O \, III]\lambda5008 / H\beta)}$', size=20, y=0.02)
axarr.set_ylabel(r'$\mathrm{log([O \, III]\lambda5008  / [O \, II] \lambda \lambda 3727,29)}$', size=20, x=0.05)
axarr.minorticks_on()
axarr.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                  labelsize=20, size=5)
axarr.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                  size=3)
axarr.set_xlim(-0.6, 1.4)
axarr.set_ylim(-2, 1)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatio_region_test', bbox_inches='tight')


# Load the actual data
path_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow', 'CUBE_OII_line_offset.fits')
path_Hbeta = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow','CUBE_Hbeta_line_offset.fits')
path_OIII4960 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                             'CUBE_OIII_4960_line_offset.fits')
path_OIII5008 = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cube_narrow',
                             'CUBE_OIII_5008_line_offset.fits')
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')

# Sampled region
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

# Muse Cube
cube_OII = Cube(path_OII)
cube_Hbeta = Cube(path_Hbeta)
cube_OIII4960 = Cube(path_OIII4960)
cube_OIII5008 = Cube(path_OIII5008)
wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
wave_Hbeta_vac = pyasl.airtovac2(cube_Hbeta.wave.coord())
wave_OIII4960_vac = pyasl.airtovac2(cube_OIII4960.wave.coord())
wave_OIII5008_vac = pyasl.airtovac2(cube_OIII5008.wave.coord())
wave_vac_stack = np.hstack((wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac))
wave_vac_all = np.array([wave_OII_vac, wave_Hbeta_vac, wave_OIII4960_vac, wave_OIII5008_vac], dtype=object)

flux_OII, flux_Hbeta = cube_OII.data * 1e-3, cube_Hbeta.data * 1e-3
flux_OIII4960, flux_OIII5008 = cube_OIII4960.data * 1e-3, cube_OIII5008.data * 1e-3
flux_OII_err, flux_Hbeta_err = np.sqrt(cube_OII.var) * 1e-3, np.sqrt(cube_Hbeta.var) * 1e-3
flux_OIII4960_err = np.sqrt(cube_OIII4960.var) * 1e-3
flux_OIII5008_err = np.sqrt(cube_OIII5008.var) * 1e-3

# Direct integration for every pixel
line_OII = 1.25 * integrate.simps(flux_OII, axis=0)
line_Hbeta = 1.25 * integrate.simps(flux_Hbeta, axis=0)
line_OIII4960 = 1.25 * integrate.simps(flux_OIII4960, axis=0)
line_OIII5008 = 1.25 * integrate.simps(flux_OIII5008, axis=0)

# Compared with the fitted result
line = 'OOHbeta'
path_fit_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit_OOHbeta',
                             'fit' + line + '_info_aperture_1.0.fits')
path_fit_info_err = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit_OOHbeta',
                                 'fit' + line + '_info_err_aperture_1.0.fits')
fit_info = fits.getdata(path_fit_info, 0, ignore_missing_end=True)
fit_info_err = fits.getdata(path_fit_info_err, 0, ignore_missing_end=True)


[z_fit, r_fit, fit_success, sigma_fit, flux_fit_OII, flux_fit_Hbeta, flux_fit_OIII5008, a_fit_OII, a_fit_Hbeta,
 a_fit_OIII4960, a_fit_OIII5008, b_fit_OII, b_fit_Hbeta, b_fit_OIII4960, b_fit_OIII5008] = fit_info
[dz_fit, dr_fit, dsigma_fit, dflux_fit_OII, dflux_fit_Hbeta, dflux_fit_OIII5008, da_fit_OII, da_fit_Hbeta,
 da_fit_OIII4960, da_fit_OIII5008, db_fit_OII, db_fit_Hbeta, db_fit_OIII4960, db_fit_OIII5008] = fit_info_err

line_OII_fitted = 1.25 * flux_fit_OII
line_Hbeta_fitted = 1.25 * flux_fit_Hbeta
line_OIII4960_fitted = 1.25 * flux_fit_OIII5008 / 3
line_OIII5008_fitted = 1.25 * flux_fit_OIII5008


# Calculate line ratio in sample region

# Load background variance estimate
path_bg_wave = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'bg_subtraction', 'bg_wave_info.fits')
path_bg_std = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'bg_subtraction', 'bg_std_info.fits')
bg_wave = fits.getdata(path_bg_wave, 0, ignore_missing_end=True)
bg_std, bg_mad_std = fits.getdata(path_bg_std, 0, ignore_missing_end=True)[0], \
                     fits.getdata(path_bg_std, 0, ignore_missing_end=True)[1]
bg_median = fits.getdata(path_bg_std, 0, ignore_missing_end=True)[2]
Hbeta_intersect, bg_wave_ind, Hbeta_wave_ind = np.intersect1d(bg_wave, cube_Hbeta.wave.coord(), return_indices=True)
Hbeta_bg_std = bg_std[bg_wave_ind]
Hbeta_bg_mad_std = bg_mad_std[bg_wave_ind]
Hbeta_bg_median = bg_median[bg_wave_ind]
print(Hbeta_bg_std)
print(Hbeta_bg_mad_std)
print(np.sqrt(cube_Hbeta[:, 10, 10].var) * 1e-3)
# plt.figure()
# # plt.plot(bg_wave[bg_wave_ind], Hbeta_bg_std, '-b')
# # plt.plot(bg_wave[bg_wave_ind], np.sqrt(cube_Hbeta[:, 10, 10].var) * 1e-3, '-r')
# plt.plot(bg_wave[bg_wave_ind], Hbeta_bg_median, '-b')
# plt.plot(bg_wave[bg_wave_ind], cube_Hbeta[:, 20, 20].data * 1e-3, '-r')
# plt.plot(bg_wave[bg_wave_ind], cube_Hbeta[:, 20, 20].data * 1e-3 - Hbeta_bg_median, '-k')
# plt.show()

OIII_OII_array, OIII_OII_err_array = np.zeros(len(ra_array)), np.zeros(len(ra_array))
OIII_Hbeta_array, OIII_Hbeta_err_array = np.zeros(len(ra_array)), np.zeros(len(ra_array))
OIII_array, Hbeta_sigma_array = np.zeros(len(ra_array)), np.zeros(len(ra_array))
for i in range(len(ra_array)):
    spe_OII_i = cube_OII.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)  # Unit in arcsec
    spe_Hbeta_i = cube_Hbeta.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    spe_OIII4960_i = cube_OIII4960.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    spe_OIII5008_i = cube_OIII5008.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)

    # Load the data
    flux_OII_i, flux_OII_err_i = spe_OII_i.data * 1e-3, np.sqrt(spe_OII_i.var) * 1e-3
    flux_Hbeta_i, flux_Hbeta_err_i = spe_Hbeta_i.data * 1e-3, np.sqrt(spe_Hbeta_i.var) * 1e-3
    flux_OIII4960_i, flux_OIII4960_err_i = spe_OIII4960_i.data * 1e-3, np.sqrt(spe_OIII4960_i.var) * 1e-3
    flux_OIII5008_i, flux_OIII5008_err_i = spe_OIII5008_i.data * 1e-3, np.sqrt(spe_OIII5008_i.var) * 1e-3
    flux_all = np.hstack((flux_OII_i, flux_Hbeta_i, flux_OIII4960_i, flux_OIII5008_i))
    flux_err_all = np.hstack((flux_OII_err_i, flux_Hbeta_err_i, flux_OIII4960_err_i, flux_OIII5008_err_i))

    # redefine Hbeta error or subtracted by median
    flux_Hbeta_i = flux_Hbeta_i - Hbeta_bg_median
    # flux_Hbeta_err_i = np.sqrt(flux_Hbeta_err_i ** 2 + Hbeta_bg_mad_std ** 2)

    # Direct integrations
    line_OII_i = 1.25 * integrate.simps(flux_OII_i)
    line_Hbeta_i = 1.25 * integrate.simps(flux_Hbeta_i)
    line_OIII4960_i = 1.25 * integrate.simps(flux_OIII4960_i)
    line_OIII5008_i = 1.25 * integrate.simps(flux_OIII5008_i)
    OIII_array[i] = line_OIII5008_i

    # Error
    error_OII_i = np.sqrt(1.25 * integrate.simps(flux_OII_err_i ** 2))
    error_Hbeta_i = np.sqrt(1.25 * integrate.simps(flux_Hbeta_err_i ** 2))
    error_OIII4960_i = np.sqrt(1.25 * integrate.simps(flux_OIII4960_err_i ** 2))
    error_OIII5008_i = np.sqrt(1.25 * integrate.simps(flux_OIII5008_err_i ** 2))

    #
    Hbeta_sigma_array[i] = error_Hbeta_i
    OIII_OII_array[i] = line_OIII5008_i / line_OII_i
    OIII_OII_err_array[i] = (line_OIII5008_i / line_OII_i) *\
                             np.sqrt((error_OIII5008_i / line_OIII5008_i) ** 2 + (error_OII_i / line_OII_i) ** 2)
    OIII_Hbeta_array[i] = line_OIII5008_i / line_Hbeta_i
    OIII_Hbeta_err_array[i] = (line_OIII5008_i / line_Hbeta_i) *\
                              np.sqrt((error_OIII5008_i / line_OIII5008_i) ** 2 + (error_Hbeta_i / line_Hbeta_i) ** 2)
# Load fitted result
path_fit_info_sr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'line_profile_selected_region.fits')
data_fit_info_sr = fits.getdata(path_fit_info_sr, 0, ignore_missing_end=True)
flux_OII_sr, flux_Hbeta_sr, flux_OIII5008_sr = data_fit_info_sr[:, 0], data_fit_info_sr[:, 1], data_fit_info_sr[:, 2]
dflux_OII_sr, dflux_Hbeta_sr, dflux_OIII5008_sr = data_fit_info_sr[:, 3], data_fit_info_sr[:, 4], data_fit_info_sr[:, 5]

# Redefine Hbeta error or subtracted by median
flux_Hbeta_sr = flux_Hbeta_sr - (1.25 * integrate.simps(Hbeta_bg_median))
# dflux_Hbeta_sr = np.sqrt(dflux_Hbeta_sr ** 2 + (1.25 * integrate.simps(Hbeta_bg_mad_std ** 2)))

# Fitted result
OIII_Hbeta_fitted = flux_OIII5008_sr / flux_Hbeta_sr
OIII_Hbeta_err_fitted = (flux_OIII5008_sr / flux_Hbeta_sr) *\
                            np.sqrt((dflux_OIII5008_sr / flux_OIII5008_sr) ** 2 + (dflux_Hbeta_sr / flux_Hbeta_sr) ** 2)
OIII_Hbeta_err_fitted_log = OIII_Hbeta_err_fitted / np.log(10) / OIII_Hbeta_fitted

OIII_OII_fitted = flux_OIII5008_sr / flux_OII_sr
OIII_OII_err_fitted = (flux_OIII5008_sr / flux_OII_sr) *\
                            np.sqrt((dflux_OIII5008_sr / flux_OIII5008_sr) ** 2 + (dflux_OII_sr / flux_OII_sr) ** 2)
OIII_OII_err_fitted_log = OIII_OII_err_fitted / np.log(10) / OIII_OII_fitted

# Load grid, plot line ratio density, ionization parameter !!!
path_df = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy_grid',
                       'iteramodel_dustfreeAGN_Z1.0_n100.txt_grid1.txt')
path_dy = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'cloudy_grid',
                       'iteramodel_dustyAGN_Z1.0_n100.txt_grid1.txt')
grid_df = np.loadtxt(path_df, encoding="ISO-8859-1")
grid_dy = np.loadtxt(path_dy, encoding="ISO-8859-1")

alpha_df, alpha_dy = grid_df[:, 0], grid_dy[:, 0]
logu_df, logu_dy = grid_df[:, 1], grid_dy[:, 1]
OIII_Hbeta_df, OIII_Hbeta_dy = grid_df[:, 2], grid_dy[:, 2]
OIII_OII_df, OIII_OII_dy = grid_df[:, 4] / grid_df[:, 5], grid_dy[:, 4] / grid_df[:, 5]

logu_df_mat, logu_dy_mat = logu_df.reshape((4, 13)), logu_dy.reshape((4, 13))
OIII_Hbeta_df_mat, OIII_Hbeta_dy_mat = OIII_Hbeta_df.reshape((4, 13)), OIII_Hbeta_dy.reshape((4, 13))
OIII_OII_df_mat, OIII_OII_dy_mat = OIII_OII_df.reshape((4, 13)), OIII_OII_dy.reshape((4, 13))

fig, axarr = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)

# Fill between
x_1, x_2, x_3 = np.log10(OIII_Hbeta_df_mat), np.log10(OIII_Hbeta_dy_mat), np.log10(OIII_Hbeta_mat)
y_1, y_2, y_3 = np.log10(OIII_OII_df_mat), np.log10(OIII_OII_dy_mat), np.log10(OIII_OII_mat)
y_1_sort, y_2_sort, y_3_sort = np.sort(y_1, axis=1), np.sort(y_2, axis=1), np.sort(y_3, axis=1)
x_1_sort, x_2_sort = np.take_along_axis(x_1, np.argsort(y_1, axis=1), axis=1), \
                     np.take_along_axis(x_2, np.argsort(y_2, axis=1), axis=1)
x_3_sort = np.take_along_axis(x_3, np.argsort(y_3, axis=1), axis=1)
# axarr[0].fill(np.hstack((x_1_sort[0, :], x_1_sort[1, ::-1])), np.hstack((y_1_sort[0, :], y_1_sort[1, ::-1])),
#               color='grey', alpha=0.2)
# axarr[0].fill(np.hstack((x_1_sort[1, :], x_1_sort[2, ::-1])), np.hstack((y_1_sort[1, :], y_1_sort[2, ::-1])),
#               color='grey', alpha=0.4)
# axarr[0].fill(np.hstack((x_1_sort[2, 2:], x_1_sort[3, ::-1][2:])), np.hstack((y_1_sort[2, 2:], y_1_sort[3, ::-1][2:])),
#               color='grey', alpha=0.6)
axarr[0].fill(np.hstack((x_3_sort[0, :], x_3_sort[1, ::-1])), np.hstack((y_3_sort[0, :], y_3_sort[1, ::-1])),
              color='grey', alpha=0.2)
axarr[0].fill(np.hstack((x_3_sort[1, :], x_3_sort[2, ::-1])), np.hstack((y_3_sort[1, :], y_3_sort[2, ::-1])),
              color='grey', alpha=0.4)
axarr[0].fill(np.hstack((x_3_sort[2, 2:], x_3_sort[3, ::-1][2:])), np.hstack((y_3_sort[2, 2:], y_3_sort[3, ::-1][2:])),
              color='grey', alpha=0.6)
axarr[0].fill(np.hstack((x_2_sort[0, :], x_2_sort[1, ::-1])), np.hstack((y_2_sort[0, :], y_2_sort[1, ::-1])),
              color='red', alpha=0.2)
axarr[0].fill(np.hstack((x_2_sort[1, :], x_2_sort[2, ::-1])), np.hstack((y_2_sort[1, :], y_2_sort[2, ::-1])),
              color='red', alpha=0.4)
axarr[0].fill(np.hstack((x_2_sort[2, 2:], x_2_sort[3, ::-1][2:])), np.hstack((y_2_sort[2, 2:], y_2_sort[3, ::-1][2:])),
              color='red', alpha=0.6)

# Plot Data
OIII_Hbeta_err_array_log = OIII_Hbeta_err_array / np.log(10) / OIII_Hbeta_array
OIII_OII_err_array_log = OIII_OII_err_array / np.log(10) / OIII_OII_array
OIII_Hbeta_3sigma = OIII_array / (3 * Hbeta_sigma_array)
OIII_Hbeta_3sigma_fitted = flux_OIII5008_sr / (3 * dflux_Hbeta_sr)
# OIII_Hbeta_3sigma_err = (OIII_array / (3 * Hbeta_sigma_array)) - OIII_Hbeta_3sigma
# OIII_Hbeta_3sigma_err_log = OIII_Hbeta_3sigma_err / np.log(10) / OIII_Hbeta_3sigma
print(OIII_Hbeta_array)
for i, ival in enumerate(OIII_Hbeta_array):
    axarr[1].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_array)[i] + 0.1, np.log10(OIII_OII_array)[i] - 0.8),
                      size=10, color='red', verticalalignment='top', horizontalalignment='right')
    if ival <= 0:
        # axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_3sigma)[i] + 0.1, np.log10(OIII_OII_array)[i] - 0.8) ,
        #                   color='red', size=10, verticalalignment='top', horizontalalignment='right')
        # axarr[0].errorbar(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], fmt='.k', capsize=2,
        #                   elinewidth=1, mfc='red', xuplims=True, ms=10, yerr=OIII_OII_err_array_log[i],
        #                   xerr=[[0.0], [0]])
        # axarr[0].arrow(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], dx=-0.5, dy=0,
        #                facecolor='red', width=0.005, head_width=0.05, head_length=0.08)
        axarr[1].errorbar(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], fmt='.k', capsize=2,
                          elinewidth=1, mfc='red', xuplims=True, ms=10, yerr=OIII_OII_err_array_log[i],
                          xerr=[[0.0], [0]])
        axarr[1].arrow(np.log10(OIII_Hbeta_3sigma)[i], np.log10(OIII_OII_array)[i], dx=-0.5, dy=0,
                       facecolor='red', width=0.005, head_width=0.05, head_length=0.08)

for i, ival in enumerate(OIII_Hbeta_fitted):
    # axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_fitted)[i] + 0.1, np.log10(OIII_OII_fitted)[i] - 0.8),
    #                   size=10, color='orange', verticalalignment='top', horizontalalignment='right')
    if ival <= 0:
        # axarr[0].annotate(text_array[i], xy=(np.log10(OIII_Hbeta_3sigma_fitted)[i] + 0.1,
        #                                      np.log10(OIII_OII_fitted)[i] - 0.8),
        #                   color='orange', size=10, verticalalignment='top', horizontalalignment='right')
        # axarr[0].errorbar(np.log10(OIII_Hbeta_3sigma_fitted)[i], np.log10(OIII_OII_fitted)[i], fmt='.k', capsize=2,
        #                   elinewidth=1, mfc='orange', xuplims=True, ms=10, yerr=OIII_OII_err_fitted_log[i],
        #                   xerr=[[0.0], [0]])
        # axarr[0].arrow(np.log10(OIII_Hbeta_3sigma_fitted)[i], np.log10(OIII_OII_fitted)[i], dx=-0.5, dy=0,
        #                facecolor='orange', width=0.005, head_width=0.05, head_length=0.08)
        axarr[1].errorbar(np.log10(OIII_Hbeta_3sigma_fitted)[i], np.log10(OIII_OII_fitted)[i], fmt='.k', capsize=2,
                          elinewidth=1, mfc='orange', xuplims=True, ms=10, yerr=OIII_OII_err_fitted_log[i],
                          xerr=[[0.0], [0]])
        axarr[1].arrow(np.log10(OIII_Hbeta_3sigma_fitted)[i], np.log10(OIII_OII_fitted)[i], dx=-0.5, dy=0,
                       facecolor='orange', width=0.005, head_width=0.05, head_length=0.08)


axarr[0].errorbar(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), yerr=OIII_OII_err_array_log,
                  xerr=OIII_Hbeta_err_array_log, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
# axarr[0].errorbar(np.log10(OIII_Hbeta_fitted), np.log10(OIII_OII_fitted), yerr=OIII_OII_err_fitted_log,
#                   xerr=OIII_Hbeta_err_fitted_log, fmt='.k', capsize=2, elinewidth=1, mfc='orange', ms=10)


axarr[1].errorbar(np.log10(OIII_Hbeta_array), np.log10(OIII_OII_array), yerr=OIII_OII_err_array_log,
                  xerr=OIII_Hbeta_err_array_log, fmt='.k', capsize=2, elinewidth=1, mfc='red', ms=10)
axarr[1].errorbar(np.log10(OIII_Hbeta_fitted), np.log10(OIII_OII_fitted), yerr=OIII_OII_err_fitted_log,
                  xerr=OIII_Hbeta_err_fitted_log, fmt='.k', capsize=2, elinewidth=1, mfc='orange', ms=10)
# axarr[0].plot(np.log10(OIII_Hbeta_df), np.log10(OIII_OII_df), '.r', ms=3)
# axarr[0].plot(np.log10(OIII_Hbeta_df_mat), np.log10(OIII_OII_df_mat), '-', color='grey', lw=1, alpha=0.3)
axarr[0].plot(np.log10(OIII_Hbeta_mat), np.log10(OIII_OII_mat), '-', color='grey', lw=1, alpha=0.3)
# axarr[1].plot(np.log10(OIII_Hbeta_dy), np.log10(OIII_OII_dy), '.r', ms=3)
axarr[1].plot(np.log10(OIII_Hbeta_dy_mat), np.log10(OIII_OII_dy_mat), '-', color='grey', lw=1, alpha=0.3)

fig.supxlabel(r'$\mathrm{log([O \, III]\lambda5008 / H\beta)}$', size=20, y=0.02)
fig.supylabel(r'$\mathrm{log([O \, III]\lambda5008  / [O \, II] \lambda \lambda 3727,29)}$', size=20, x=0.05)
axarr[0].minorticks_on()
axarr[1].minorticks_on()
axarr[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                     labelsize=20, size=5)
axarr[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                     size=3)
axarr[1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                     labelsize=20, size=5)
axarr[1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                     size=3)
axarr[0].set_xlim(-0.6, 1.4)
axarr[0].set_ylim(-2, 1)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatio_region_cor.png', bbox_inches='tight')


# Plot lineratio separately
f, axarr = plt.subplots(2, 1, figsize=(4, 8), dpi=300, sharex=True)
# axarr.plot(np.log10(OIII_Hbeta), np.log10(OIII_OII), '.')
for i in range(len(filename)):
    # axarr[0].plot(hden[:21], np.log10(OIII_Hbeta_mat)[i, :], '-', lw=1,
    #               label=r'$\mathrm{Alpha = }$' + str(alpha_mat[i, 0]))
    # axarr[1].plot(hden[:21], np.log10(OIII_OII_mat)[i, :], '-', lw=1,
    #               label=r'$\mathrm{Alpha = }$' + str(alpha_mat[i, 0]))
    if i == 2:
        axarr[0].plot(logu_df_mat[i, :], np.log10(OIII_Hbeta_df_mat)[i, :], '--', color='C' + str(i), lw=1)
        axarr[0].plot(logu_dy_mat[i, :], np.log10(OIII_Hbeta_dy_mat)[i, :], '-.', color='C' + str(i), lw=1)
        axarr[0].plot(logu_10_mat[i, :], np.log10(OIII_Hbeta_mat)[i, :], '-', color='C' + str(i), lw=1,
                      label=r'$\mathrm{Alpha = }$' + str(alpha_mat[i, 0]))
        axarr[1].plot(logu_df_mat[i, :], np.log10(OIII_OII_df_mat)[i, :], '--', color='C' + str(i), lw=1)
        axarr[1].plot(logu_dy_mat[i, :], np.log10(OIII_OII_dy_mat)[i, :], '-.', color='C' + str(i), lw=1)
        axarr[1].plot(logu_10_mat[i, :], np.log10(OIII_OII_mat)[i, :], '-', color='C' + str(i), lw=1)

axarr[1].set_xlabel(r"$\mathrm{log_{10}[Hydrogen \, density]}$", size=15)
axarr[1].set_xlabel(r"$\mathrm{log_{10}[U]}$", size=15)
axarr[0].set_ylabel(r'$\mathrm{log([O \, III]\lambda5008 / H\beta)}$', size=15)
axarr[1].set_ylabel(r'$\mathrm{log([O \, III]\lambda5008  / [O \, II] \lambda \lambda 3727,29)}$', size=15, x=0.05)
axarr[0].minorticks_on()
axarr[1].minorticks_on()
axarr[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                  labelsize=20, size=5)
axarr[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                  size=3)
axarr[1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                  labelsize=20, size=5)
axarr[1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                  size=3)
# axarr.set_xlim(-0.6, 1.4)
# axarr.set_ylim(-2, 1)
axarr[0].legend(prop={'size': 15}, framealpha=0, loc=3, fontsize=15)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatio_region_test', bbox_inches='tight')