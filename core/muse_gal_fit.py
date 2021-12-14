import os
import sfdmap
import numpy as np
import astropy.io.fits as fits
import bagpipes as pipes
from matplotlib import rc
from matplotlib import cm
from PyAstronomy import pyasl
from matplotlib.colors import ListedColormap
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord
from astropy.table import Table
from mpdaf.obj import Cube
from mpdaf.obj import WCS as mpdaf_WCS
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

path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits')

cube = Cube(path)
hdul = fits.open(path)  # open a FITS file
hdr = hdul[1].header
wcs = mpdaf_WCS(hdr)

# # Calculate the white image
# image_white = cube.sum(axis=0)
# p, q = image_white.peak()['p'], image_white.peak()['q']
# p_q = wcs.sky2pix(np.vstack((dec_final, ra_final)).T, nearest=True)
# p_gal, q_gal = p_q.T[0], p_q.T[1]

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)

# plt.figure(figsize=(8, 5), dpi=300)
# plt.imshow(image_white.data, origin='lower', cmap=newcmp, norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar()
# # cbar.set_label(r'$\mathrm{Arcsinh}$')
# plt.contour(image_white.data, levels=[1e5, 1e6, 1e7, 1e8], colors=newcolors_red[200::30, :], linewidths=0.5, alpha=0.5,
#             norm=matplotlib.colors.LogNorm())
#
# plt.plot(q_gal, p_gal, 'o', color='brown', ms=7, alpha=0.4, markerfacecolor='None', markeredgecolor='red',
#          markeredgewidth=0.5)
# for i in range(len(row_final)):
#     plt.annotate(str(row_final[i]), (q_gal[i], p_gal[i]), fontsize=5)
#
# plt.axis('off')
# # plt.xlim(200, 250)
# # plt.ylim(200, 250)
# print(row_final)


# Getting photometry zero point
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)

path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'HE0238-1904_sex.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)

path_image = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'check.fits')
data_image = fits.getdata(path_image, 1, ignore_missing_end=True)

w_pho = WCS(fits.open(path_image)[1].header)
catalog = pixel_to_skycoord(data_pho['X_IMAGE'], data_pho['Y_IMAGE'], w_pho)
c = SkyCoord(ra_final, dec_final, unit="deg")
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# Hubble image
f_hb = fits.open(path_hb)
w_hb = WCS(f_hb[1].header)
x, y = skycoord_to_pixel(c, w_hb)

# Photometry
data_pho = data_pho[idx]
x_image = data_pho['X_IMAGE']
y_image = data_pho['Y_IMAGE']
mag_iso = data_pho['MAG_ISO']
dmag_iso = data_pho['MAGERR_ISO']
mag_isocor = data_pho['MAG_ISOCOR']
dmag_isocor = data_pho['MAGERR_ISOCOR']
mag_auto = data_pho['MAG_AUTO']
dmag_auto = data_pho['MAGERR_AUTO']

# chech consistency
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'HE0238-1904_drc_offset.fits')
data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)


dustmap_path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'PyQSOfit', 'sfddata')
m = sfdmap.SFDMap(dustmap_path)
ebv_array = m.ebv(ra_final, dec_final)
m_ex = 0.611 * 3.2 * ebv_array

mag_iso_dred = mag_iso - m_ex
mag_isocor_dred = mag_isocor - m_ex
mag_auto_dred = mag_auto - m_ex

dmag_iso_dred = dmag_iso
dmag_isocor_dred = dmag_isocor
dmag_auto_dred = dmag_auto


# gal_col = [r"Row", r"Ra", r"Dec", r"Mag_iso", r"Mag_isocor", r"Mag_auto"]
# gal_phot = np.stack([row_final, ra_final, dec_final, mag_iso_dred, mag_isocor_dred, mag_auto_dred], axis=1)
# Table_1 = pd.DataFrame(gal_phot, index=1 + np.arange(len(row_final)), columns=gal_col)
# Table_1

path_pho_des = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'des_dr2_galaxys_pho.fits')
data_pho_des = fits.getdata(path_pho_des, 1, ignore_missing_end=True)


row_des = data_pho_des['t1_id']
mag_g_dred = data_pho_des['mag_auto_g_dered']
mag_r_dred = data_pho_des['mag_auto_r_dered']
mag_i_dred = data_pho_des['mag_auto_i_dered']
mag_z_dred = data_pho_des['mag_auto_z_dered']
mag_Y_dred = data_pho_des['mag_auto_Y_dered']

dmag_g_dred = data_pho_des['magerr_auto_g']
dmag_r_dred = data_pho_des['magerr_auto_r']
dmag_i_dred = data_pho_des['magerr_auto_i']
dmag_z_dred = data_pho_des['magerr_auto_z']
dmag_Y_dred = data_pho_des['magerr_auto_Y']


# Combine photometry
col_ID = np.arange(len(row_final))
have_des_pho = np.in1d(row_final, row_des)
print(row_des)
print(row_final)
print(col_ID[have_des_pho])

mag_all = np.zeros((len(row_final), 6))
dmag_all = mag_all.copy()
mag_all[:, 0], dmag_all[:, 0] = mag_auto_dred, dmag_auto_dred
mag_all[col_ID[have_des_pho], 1:] = np.array([mag_g_dred, mag_r_dred, mag_i_dred, mag_z_dred, mag_Y_dred]).T
dmag_all[col_ID[have_des_pho], 1:] = np.array([dmag_g_dred, dmag_r_dred, dmag_i_dred, dmag_z_dred, dmag_Y_dred]).T


mag_all = np.where((mag_all != 0) * (mag_all != 99), mag_all, np.inf)
dmag_all = np.where((dmag_all != 0) * (dmag_all != 99), dmag_all, 0)

flux_all = 10 ** ((23.9 - mag_all) / 2.5) # microjanskys
flux_all_err = flux_all * np.log(10) * dmag_all / 2.5
flux_all_err = np.where(flux_all_err != 0, flux_all_err, 99)
print(np.array([flux_all[9], flux_all_err[9]]).T)



def bin(spectrum, binn):
    """ Bins up two or three column spectral data by a specified factor. """

    binn = int(binn)
    nbins = len(spectrum) / binn
    binspec = np.zeros((int(nbins), spectrum.shape[1]))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i * binn:(i + 1) * binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 1] = np.mean(spec_slice[:, 1])

        if spectrum.shape[1] == 3:
            binspec[i, 2] = (1. / float(binn)
                             * np.sqrt(np.sum(spec_slice[:, 2] ** 2)))

    return binspec


def load_data(row):
    row_sort = np.where(row_final == float(row))

    flux = flux_all[row_sort][0]
    flux_err = flux_all_err[row_sort][0]
    phot = np.array([flux, flux_err]).T

    path_spe = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                            row + '_' + str(ID_final[row_sort][0]) + '_' + name_final[row_sort][0] + '_spec1D.fits')
    spec = Table.read(path_spe)
    spec = spec[spec['mask'] == 1]

    wave = pyasl.vactoair2(spec['wave'])
    flux = spec['flux'] * 1e-20
    flux_err = spec['error'] * 1e-20

    spectrum = np.array([wave, flux, flux_err]).T
    return spectrum, phot
# bin(spectrum, 5), phot
# bin(spectrum, 10)
# phot
# bin(spectrum, 20), phot


print(row_des)
row_number = '82'
# filt_list=np.loadtxt("filters/filters_list.txt", dtype="str")
galaxy = pipes.galaxy(row_number, load_data, filt_list=np.loadtxt("filters/filters_list.txt", dtype="str"))
# galaxy.plot()

exp = {}                                  # Tau-model star-formation history component
exp["age"] = (0.01, 15.)                  # Vary age between 100 Myr and 15 Gyr. In practice
                                          # the code automatically limits this to the age of
                                          # the Universe at the observed redshift.

exp["tau"] = (0.01, 8.0)                   # Vary tau between 300 Myr and 10 Gyr
exp["massformed"] = (6.0, 13.0)             # vary log_10(M*/M_solar) between 1 and 15
exp["metallicity"] = (0.01, 2.5)         # vary Z between 0 and 2.5 Z_oldsolar

dust = {}                                 # Dust component
dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
dust["Av"] = (0., 2.0)                     # Vary Av between 0 and 2 magnitudes

fit_instructions = {}                     # The fit instructions dictionary
z_gal_i = z_final[np.where(row_final == float(row_number))]
fit_instructions["redshift"] = (z_gal_i - 0.01, z_gal_i + 0.01)
fit_instructions["redshift_prior"] = "Gaussian"
fit_instructions["redshift_prior_mu"] = z_gal_i
fit_instructions["redshift_prior_sigma"] = 0.01


fit_instructions["age_prior"] = 'log_10'
fit_instructions["tau_prior"] = 'log_10'
fit_instructions["massformed_prior"] = 'log_10'
fit_instructions["metallicity_prior"] = 'log_10'


fit_instructions["veldisp"] = (1., 1000.)   #km/s
fit_instructions["veldisp_prior"] = "log_10"

calib = {}
calib["type"] = "polynomial_bayesian"
calib["0"] = (0.1, 20) # Zero order is centred on 1, at which point there is no change to the spectrum.
fit_instructions["calib"] = calib

noise = {}
noise["type"] = "white_scaled"
noise["scaling"] = (1., 10.)
noise["scaling_prior"] = "log_10"
fit_instructions["noise"] = noise


fit_instructions["exponential"] = exp
fit_instructions["dust"] = dust


fit = pipes.fit(galaxy, fit_instructions)
fit.fit(verbose=True)

fig = fit.plot_spectrum_posterior(save=True, show=True)
fig = fit.plot_sfh_posterior(save=True, show=True)
fig = fit.plot_corner(save=True, show=True)