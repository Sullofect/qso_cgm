import os
import numpy as np
import astropy.io.fits as fits
import bagpipes as pipes
from muse_compare_z import compare_z
from muse_kc import app2abs
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table
import sys
sys.path.insert(1, '/Users/lzq/Dropbox/pyobs/')
from kcorrect import apptoabs
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def load_data(row):
    global ID_final, name_final
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


ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                     cat_will='ESO_DEEP_offset_zapped_objects.fits')
row_final = ggp_info[1]
ID_final = ggp_info[2]
z_final = ggp_info[3]
name_final = ggp_info[5]
ra_final = ggp_info[7]
dec_final = ggp_info[8]

print(ID_final)
print(z_final)
print(name_final)
print(ra_final)
print(dec_final)

# Getting photometry zero point
path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'gal_81',
                        'HE0238-1904_sex_gal_81.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)
path_image = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'gal_81',
                          'check_seg_gal_81.fits')

w_pho = WCS(fits.open(path_image)[2].header)
catalog = pixel_to_skycoord(data_pho['X_IMAGE'], data_pho['Y_IMAGE'], w_pho)
c = SkyCoord(ra_final, dec_final, unit="deg")
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# Photometry
data_pho = data_pho[idx]
number = data_pho['NUMBER']
x_image = data_pho['X_IMAGE']
y_image = data_pho['Y_IMAGE']
mag_iso = data_pho['MAG_ISO']
dmag_iso = data_pho['MAGERR_ISO']
mag_isocor = data_pho['MAG_ISOCOR']
dmag_isocor = data_pho['MAGERR_ISOCOR']
mag_auto = data_pho['MAG_AUTO']
dmag_auto = data_pho['MAGERR_AUTO']
print(number)
# One way to deal with extinction
m_ex = 0.049

mag_iso_dred = mag_iso - m_ex
mag_isocor_dred = mag_isocor - m_ex
mag_auto_dred = mag_auto - m_ex

dmag_iso_dred = dmag_iso
dmag_isocor_dred = dmag_isocor
dmag_auto_dred = dmag_auto

Table_pho = Table()
Table_pho["Row"] = row_final
Table_pho["Image Number"] = number
Table_pho['Ra'] = ra_final
Table_pho["Dec"] = dec_final
Table_pho["Mag_iso_dred"] = mag_iso_dred
Table_pho["Mag_isocor_dred"] = mag_isocor_dred
Table_pho["Mag_auto_dred"] = mag_auto_dred
ascii.write(Table_pho, path_savetab + 'check_photometry.csv', format='ecsv', overwrite=True)

path_pho_des = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'des_dr2_galaxys_pho_2.fits')
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
# print(row_des)
# print(row_final)
# print(col_ID[have_des_pho])

# determine if blended or not
mag_hst_dred = mag_auto_dred
dmag_hst_dred = dmag_auto_dred + 0.1

offset = mag_i_dred - mag_hst_dred[col_ID[have_des_pho]]
mag_g_dred -= offset
mag_r_dred -= offset
mag_i_dred -= offset
mag_z_dred -= offset
mag_Y_dred -= offset

print(mag_g_dred)
print(mag_r_dred)
print(mag_i_dred)
print(mag_z_dred)
print(mag_Y_dred)

mag_all = np.zeros((len(row_final), 6))
dmag_all = mag_all.copy()
mag_all[:, 0], dmag_all[:, 0] = mag_hst_dred, dmag_hst_dred
mag_all[col_ID[have_des_pho], 1:] = np.array([mag_g_dred, mag_r_dred, mag_i_dred, mag_z_dred, mag_Y_dred]).T
dmag_all[col_ID[have_des_pho], 1:] = np.array([dmag_g_dred, dmag_r_dred, dmag_i_dred, dmag_z_dred, dmag_Y_dred]).T

#
mag_all = np.where((mag_all != 0) * (mag_all != 99), mag_all, np.inf)
dmag_all = np.where((dmag_all != 0) * (dmag_all != 99), dmag_all, 0)

# remove invalid entry
# bad Y band
bad_Y = np.array([35, 93, 164])
mask_Y = np.in1d(row_final, bad_Y)
mag_all[col_ID[mask_Y], 5] = np.inf
dmag_all[col_ID[mask_Y], 5] = 0
# print(row_final)
# print(col_ID[mask_Y])
# print(mag_all)
# print(dmag_all)

flux_all = 10 ** ((23.9 - mag_all) / 2.5)  # microjanskys
flux_all_err = flux_all * np.log(10) * dmag_all / 2.5
flux_all_err = np.where(flux_all_err != 0, flux_all_err, 99)
#

print('M_abs is ', app2abs(m_app=mag_hst_dred[0], z=z_final[0], model='S0', filter_e='Bessell_B.dat',
                           filter_o='ACS_f814W.dat'))

print('M_abs_sean is ', apptoabs(mag_hst_dred[0], 'S0', 'Bessell_B', 'ACS_f814W', z_final[0]))


# Good: 1, 13, 35, 62, 78, 92, 164, 179: Done!
# Good rerun: 120, 134, 141
# Good but with iso: 4, 88, 162
# Good but with v_max=200: 20, 27, 93
# Good but with iso and v_max=200: 36,
# Good with more calibration: 57
# Good with more calibration and isor: 82
# Good but with Quasar light subtraction: 5, 6, 7, 83, 181 182
# bad: 64
# bad: 80 need Legacy Surveys g r z
# bad: 81 still blended
for i in [81]:
    row_number = str(i)
    galaxy = pipes.galaxy(row_number, load_data, filt_list=np.loadtxt("filters/filters_list.txt", dtype="str"))
    # galaxy.plot()

    exp = {}  # Tau-model star-formation history component
    exp["age"] = (0.01, 15.)  # Vary age between 100 Myr and 15 Gyr. In practice
    # the code automatically limits this to the age of
    # the Universe at the observed redshift.

    exp["tau"] = (0.01, 8.0)  # Vary tau between 300 Myr and 10 Gyr
    exp["massformed"] = (6.0, 13.0)  # vary log_10(M*/M_solar) between 1 and 15
    exp["metallicity"] = 1
    # (0.01, 1)  # vary Z between 0 and 2.5 Z_oldsolar

    dust = {}  # Dust component
    dust["type"] = "Calzetti"  # Define the shape of the attenuation curve
    dust["Av"] = (0., 2.0)  # Vary Av between 0 and 2 magnitudes

    fit_instructions = {}  # The fit instructions dictionary
    z_gal_i = z_final[np.where(row_final == float(row_number))]
    fit_instructions["redshift"] = (z_gal_i - 0.0001, z_gal_i + 0.0001)
    fit_instructions["redshift_prior"] = "Gaussian"
    fit_instructions["redshift_prior_mu"] = z_gal_i
    fit_instructions["redshift_prior_sigma"] = 0.0001

    fit_instructions["age_prior"] = 'log_10'
    fit_instructions["tau_prior"] = 'log_10'
    fit_instructions["massformed_prior"] = 'log_10'
    # fit_instructions["metallicity_prior"] = 'log_10'

    fit_instructions["veldisp"] = (50., 1000.)  # km/s
    fit_instructions["veldisp_prior"] = "log_10"

    calib = {}
    calib["type"] = "polynomial_bayesian"
    calib["0"] = (0.1, 20)  # Zero order is centred on 1, at which point there is no change to the spectrum.

    # calib["0"] = (0.5, 5)  # Zero order is centred on 1, at which point there is no change to the spectrum.
    # calib["0_prior"] = "Gaussian"
    # calib["0_prior_mu"] = 1.0
    # calib["0_prior_sigma"] = 0.25
    #
    # calib["1"] = (-0.5, 2)  # Subsequent orders are centred on zero.
    # calib["1_prior"] = "Gaussian"
    # calib["1_prior_mu"] = 0.
    # calib["1_prior_sigma"] = 0.25
    #
    # calib["2"] = (-0.5, 0.5)
    # calib["2_prior"] = "Gaussian"
    # calib["2_prior_mu"] = 0.
    # calib["2_prior_sigma"] = 0.25

    fit_instructions["calib"] = calib

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 10.)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    fit_instructions["exponential"] = exp
    fit_instructions["dust"] = dust

    fit = pipes.fit(galaxy, fit_instructions, run='Trial_5')
    fit.fit(verbose=True)

    fit.plot_spectrum_posterior(save=True, show=True)
    fit.plot_sfh_posterior(save=True, show=True)
    fit.plot_corner(save=True, show=True)
