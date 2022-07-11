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
from astropy.coordinates import SkyCoord
from astropy.table import Table

# Compare with Sean's kcorrection
# import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/Users/lzq/Dropbox/pyobs/')
from kcorrect import apptoabs

path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def load_data(row):
    global ID_final, row_final, name_final, flux_all, flux_all_err, qls, spectrum_exists
    row_sort = np.where(row_final == float(row))

    flux = flux_all[row_sort][0]
    flux_err = flux_all_err[row_sort][0]
    phot = np.array([flux, flux_err]).T
    if qls is True:
        path_spe = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting',
                                'Subtracted_ESO_DEEP_offset_zapped_spec1D', row + '_' + str(ID_final[row_sort][0])
                                + '_' + name_final[row_sort][0] + '_spec1D.fits')
    elif qls is False:
        path_spe = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting',
                                'ESO_DEEP_offset_zapped_spec1D', row + '_' + str(ID_final[row_sort][0])
                                + '_' + name_final[row_sort][0] + '_spec1D.fits')
    spec = Table.read(path_spe)
    spec = spec[spec['mask'] == 1]

    wave = pyasl.vactoair2(spec['wave'])  # Bagpipe want air wavelength
    flux = spec['flux'] * 1e-20
    flux_err = spec['error'] * 1e-20

    spectrum = np.array([wave, flux, flux_err]).T

    if spectrum_exists is False:
        return phot
    elif spectrum_exists is True:
        return spectrum, phot


# determine if blended or not
def gal_fit(gal_num=None, run_name='Trial_5', flux_hst='auto', dflux_sys=0.1, cal='0', v_min=50, v_max=1000,
            prior='log'):
    global mag_i_dred, mag_g_dred, mag_r_dred, mag_z_dred, mag_Y_dred, \
           dmag_i_dred, dmag_g_dred, dmag_r_dred, dmag_z_dred, dmag_Y_dred, \
           mag_iso_dred, dmag_iso_dred, mag_auto_dred,dmag_auto_dred, \
           row_final, col_ID, z_final, ID_final, name_final, flux_all, flux_all_err, qls, spectrum_exists

    # Combine photometry
    col_ID = np.arange(len(row_final))
    have_des_pho = np.in1d(row_final, row_des)

    if flux_hst == 'iso':
        mag_hst_dred = mag_iso_dred
        dmag_hst_dred = dmag_iso_dred + dflux_sys  # Add systematic error: 0.1 mag
    else:
        mag_hst_dred = mag_auto_dred
        dmag_hst_dred = dmag_auto_dred + dflux_sys  # Add systematic error: 0.1 mag

    offset = mag_i_dred - mag_hst_dred[col_ID[have_des_pho]]
    mag_g_dred -= offset
    mag_r_dred -= offset
    mag_i_dred -= offset
    mag_z_dred -= offset
    mag_Y_dred -= offset

    # print(mag_g_dred)
    # print(mag_r_dred)
    # print(mag_i_dred)
    # print(mag_z_dred)
    # print(mag_Y_dred)

    mag_all = np.zeros((len(row_final), 6))
    dmag_all = mag_all.copy()
    mag_all[:, 0], dmag_all[:, 0] = mag_hst_dred, dmag_hst_dred
    mag_all[col_ID[have_des_pho], 1:] = np.array([mag_g_dred, mag_r_dred, mag_i_dred, mag_z_dred, mag_Y_dred]).T
    dmag_all[col_ID[have_des_pho], 1:] = np.array([dmag_g_dred + dflux_sys, dmag_r_dred + dflux_sys,
                                                   dmag_i_dred + dflux_sys, dmag_z_dred + dflux_sys,
                                                   dmag_Y_dred + dflux_sys]).T  # Add systematic error: 0.1 mag

    # Remove invalid DES photometry
    mag_all = np.where((mag_all != 0) * (mag_all != 99), mag_all, np.inf)
    dmag_all = np.where((dmag_all != 0) * (dmag_all != 99), dmag_all, 0)

    # remove invalid entry bad Y band
    bad_Y = np.array([35, 93, 164])
    mask_Y = np.in1d(row_final, bad_Y)
    mag_all[col_ID[mask_Y], 5] = np.inf
    dmag_all[col_ID[mask_Y], 5] = 0

    # Remove invalid bad g band
    bad_g = np.array([80])
    mask_g = np.in1d(row_final, bad_g)
    mag_all[col_ID[mask_g], 1] = np.inf
    dmag_all[col_ID[mask_g], 1] = 0

    flux_all = 10 ** ((23.9 - mag_all) / 2.5)  # microjanskys
    flux_all_err = flux_all * np.log(10) * dmag_all / 2.5
    flux_all_err = np.where(flux_all_err != 0, flux_all_err, 99)

    # Deterime the absolute magnitude
    # for i in gal_num:
    #     gal_num_sort = np.where(row_final == i)
    #     print(str(i), 'M_S0 is', str(app2abs(m_app=mag_hst_dred[gal_num_sort], z=z_final[gal_num_sort],
    #                                           model='S0', filter_e='Bessell_B', filter_o='ACS_f814W')))
    #     print(str(i), 'M_Scd is', str(app2abs(m_app=mag_hst_dred[gal_num_sort], z=z_final[gal_num_sort],
    #                                            model='Scd', filter_e='Bessell_B', filter_o='ACS_f814W')))
    #     print(str(i), 'M_irr is', str(app2abs(m_app=mag_hst_dred[gal_num_sort], z=z_final[gal_num_sort],
    #                                            model='irregular', filter_e='Bessell_B', filter_o='ACS_f814W')))
    for i in gal_num:
        row_number = str(i)
        galaxy = pipes.galaxy(row_number, load_data, spectrum_exists=spectrum_exists,
                              filt_list=np.loadtxt("filters/filters_list.txt", dtype="str"))
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

        # Velocity
        fit_instructions["veldisp"] = (v_min, v_max)  # km/s

        if prior == 'log':
            fit_instructions["age_prior"] = 'log_10'
            fit_instructions["tau_prior"] = 'log_10'
            fit_instructions["massformed_prior"] = 'log_10'
            # fit_instructions["metallicity_prior"] = 'log_10'
            fit_instructions["veldisp_prior"] = 'log_10'
        elif prior == 'uniform':
            fit_instructions["age_prior"] = 'uniform'
            fit_instructions["tau_prior"] = 'uniform'
            fit_instructions["massformed_prior"] = 'uniform'
            # fit_instructions["metallicity_prior"] = 'uniform'
            fit_instructions["veldisp_prior"] = 'uniform'


        calib = {}
        calib["type"] = "polynomial_bayesian"
        if cal == '0':
            calib["0"] = (0.1, 20)  # Zero order is centred on 1, at which point there is no change to the spectrum.
        else:
            calib["0"] = (0.5, 5)  # Zero order is centred on 1, at which point there is no change to the spectrum.
            calib["0_prior"] = "Gaussian"
            calib["0_prior_mu"] = 1.0
            calib["0_prior_sigma"] = 0.25

            calib["1"] = (-0.5, 2)  # Subsequent orders are centred on zero.
            calib["1_prior"] = "Gaussian"
            calib["1_prior_mu"] = 0.
            calib["1_prior_sigma"] = 0.25

            calib["2"] = (-0.5, 0.5)
            calib["2_prior"] = "Gaussian"
            calib["2_prior_mu"] = 0.
            calib["2_prior_sigma"] = 0.25

        fit_instructions["calib"] = calib

        noise = {}
        noise["type"] = "white_scaled"
        noise["scaling"] = (1., 10.)
        if prior == 'log':
            noise["scaling_prior"] = "log_10"
        elif prior == 'uniform':
            noise["scaling_prior"] = "uniform"
        fit_instructions["noise"] = noise

        fit_instructions["exponential"] = exp
        fit_instructions["dust"] = dust

        fit = pipes.fit(galaxy, fit_instructions, run=run_name)
        fit.fit(verbose=True)
        if spectrum_exists is True:
            fit.plot_spectrum_posterior(save=True, show=True)
        fit.plot_sfh_posterior(save=True, show=True)
        fit.plot_corner(save=True, show=True)
        print(str(i), np.percentile(fit.posterior.samples["stellar_mass"], [16, 50, 84]))
        print(str(i), np.percentile(np.log10(fit.posterior.samples["mass_weighted_age"]), [16, 50, 84]))


# Load data
ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                     cat_will='ESO_DEEP_offset_zapped_objects.fits')
row_final, ID_final, z_final = ggp_info[1], ggp_info[2], ggp_info[3]
name_final, ql_final, ra_final, dec_final = ggp_info[5], ggp_info[6], ggp_info[7], ggp_info[8]

# print(ID_final)
# print(row_final)
# print(z_final)
# print(name_final)
# print(ra_final)
# print(dec_final)
# print(ql_final)

# Getting photometry zero point
path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'gal_all',
                        'HE0238-1904_sex_gal_all.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)
catalog = SkyCoord(data_pho['AlPHAWIN_J2000'], data_pho['DELTAWIN_J2000'], unit="deg")
c = SkyCoord(ra_final, dec_final, unit="deg")
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# Photometry
data_pho = data_pho[idx]
number, x_image, y_image = data_pho['NUMBER'], data_pho['X_IMAGE'], data_pho['Y_IMAGE']
mag_iso, dmag_iso = data_pho['MAG_ISO'], data_pho['MAGERR_ISO']
mag_isocor, dmag_isocor = data_pho['MAG_ISOCOR'], data_pho['MAGERR_ISOCOR']
mag_auto, dmag_auto = data_pho['MAG_AUTO'], data_pho['MAGERR_AUTO']

# Add num=181 diffraction subtracted photometry
ds_sort = np.where(row_final == 181)
mag_iso[ds_sort] = 25.937 - 2.5 * np.log10(12.636047 - 6.021952)  # 181 total=12.636047, diffraction=6.021952
mag_auto[ds_sort] = 25.937 - 2.5 * np.log10(12.636047 - 6.021952)

# Extinction
m_ex = 0.049
mag_iso_dred = mag_iso - m_ex
mag_isocor_dred = mag_isocor - m_ex
mag_auto_dred = mag_auto - m_ex

dmag_iso_dred = dmag_iso
dmag_isocor_dred = dmag_isocor
dmag_auto_dred = dmag_auto

# Check photometry
# Table_pho = Table()
# Table_pho["Row"] = row_final
# Table_pho["Image Number"] = number
# Table_pho['Ra'] = ra_final
# Table_pho["Dec"] = dec_final
# Table_pho["Mag_iso_dred"] = mag_iso_dred
# Table_pho["Mag_isocor_dred"] = mag_isocor_dred
# Table_pho["Mag_auto_dred"] = mag_auto_dred
# ascii.write(Table_pho, path_savetab + 'check_photometry.csv', format='ecsv', overwrite=True)


# Load DES data
path_pho_des = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'des_dr2_galaxys_pho_final.fits')
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

# Summary
# Good: 1, 13, 35, 62, 78, 92, 164, 179: Done!
# Good rerun: 120, 134, 141: Done!
# Good but with iso: 4, 88, 162 : Done!
# Good but with v_max=200: 20, 27, 93 : Done!
# Good but with iso and v_max=200: 36 : Done!
# Good with more calibration: 57: Done!
# Good with more calibration and isor: 82: Done!
# Good but with Quasar light subtraction: 5, 6, 7, 83, 181 182 : Done!
# bad: 64: dont need separate script!: Done!
# bad: 80 need Legacy Surveys g r z: dont need separate script!: Done!
# bad: 81 still blended: dont need separate script!: Done!

# Trial 5: Both info
# qls, spectrum_exists = False, True
# gal_fit(gal_num=[1, 13, 35, 62, 78, 92, 120, 134, 141, 164, 179],
#         run_name='Trial_5', flux_hst='auto', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[4, 88, 162], run_name='Trial_5', flux_hst='iso', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[20, 27, 93], run_name='Trial_5', flux_hst='auto', cal='0', v_min=50, v_max=200)
# gal_fit(gal_num=[36], run_name='Trial_5', flux_hst='iso', cal='0', v_min=50, v_max=200)
# gal_fit(gal_num=[57], run_name='Trial_5', flux_hst='auto', cal='2', v_min=50, v_max=1000)
# gal_fit(gal_num=[64], run_name='Trial_5', flux_hst='auto', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[80, 81], run_name='Trial_5', flux_hst='iso', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[82], run_name='Trial_5', flux_hst='iso', cal='2', v_min=50, v_max=1000)
# qls = True
# gal_fit(gal_num=[5, 7, 83, 181, 182], run_name='Trial_5', flux_hst='auto', cal='0', v_min=50, v_max=1000)


# Trial 6: Photometry only for 1, 4, 13, 35, 36, 57, 62, 64, 80, 82, 88, 93, 120, 134, 141, 164
# qls, spectrum_exists = False, False
# gal_fit(gal_num=[1, 13, 35, 62, 120, 134, 141, 164], run_name='Trial_6', flux_hst='auto', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[4, 88], run_name='Trial_6', flux_hst='iso', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[93], run_name='Trial_6', flux_hst='auto', cal='0', v_min=50, v_max=200)
# gal_fit(gal_num=[36], run_name='Trial_6', flux_hst='iso', cal='0', v_min=50, v_max=200)
# gal_fit(gal_num=[57], run_name='Trial_6', flux_hst='auto', cal='2', v_min=50, v_max=1000)
# gal_fit(gal_num=[64], run_name='Trial_6', flux_hst='auto', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[80], run_name='Trial_6', flux_hst='iso', cal='0', v_min=50, v_max=1000)
# gal_fit(gal_num=[82], run_name='Trial_6', flux_hst='iso', cal='2', v_min=50, v_max=1000)

# Trial 7: Uniform prior
# qls, spectrum_exists = False, True
# gal_fit(gal_num=[1, 13, 35, 62, 78, 92, 120, 134, 141, 164, 179],
#         run_name='Trial_7', flux_hst='auto', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[4, 88, 162], run_name='Trial_7', flux_hst='iso', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[20, 27, 93], run_name='Trial_7', flux_hst='auto', cal='0', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[36], run_name='Trial_7', flux_hst='iso', cal='0', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[57], run_name='Trial_7', flux_hst='auto', cal='2', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[64], run_name='Trial_7', flux_hst='auto', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[80, 81], run_name='Trial_7', flux_hst='iso', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[82], run_name='Trial_7', flux_hst='iso', cal='2', v_min=50, v_max=1000, prior='uniform')
# qls = True
# gal_fit(gal_num=[5, 7, 83, 181, 182], run_name='Trial_7', flux_hst='auto', cal='0',
#         v_min=50, v_max=1000, prior='uniform')

# Trial 8: Inflate all errors + Uniform prior + calibration
qls, spectrum_exists = False, True
# gal_fit(gal_num=[1, 13, 35, 62, 78, 92, 120, 134, 141, 164, 179], run_name='Trial_8', flux_hst='auto', cal='2',
#         v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[4, 88, 162], run_name='Trial_8', flux_hst='iso', cal='2', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[20, 27, 93], run_name='Trial_8', flux_hst='auto', cal='2', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[36], run_name='Trial_8', flux_hst='iso', cal='2', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[57], run_name='Trial_8', flux_hst='auto', cal='2', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[64], run_name='Trial_8', flux_hst='auto', cal='2', v_min=50, v_max=1000, prior='uniform')
gal_fit(gal_num=[80, 81], run_name='Trial_8', flux_hst='iso', cal='2', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[82], run_name='Trial_8', flux_hst='iso', cal='2', v_min=50, v_max=1000, prior='uniform')
# qls = True
# gal_fit(gal_num=[5, 7, 83, 181, 182], run_name='Trial_8', flux_hst='auto', cal='2', v_min=50, v_max=1000,
#         prior='uniform')

# Trial 9: Inflate all errors + Uniform prior
# qls, spectrum_exists = False, True
# gal_fit(gal_num=[1, 13, 35, 62, 78, 92, 120, 134, 141, 164, 179],
#         run_name='Trial_9', flux_hst='auto', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[4, 88, 162], run_name='Trial_9', flux_hst='iso', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[20, 27, 93], run_name='Trial_9', flux_hst='auto', cal='0', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[36], run_name='Trial_9', flux_hst='iso', cal='0', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[57], run_name='Trial_9', flux_hst='auto', cal='2', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[64], run_name='Trial_9', flux_hst='auto', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[80, 81], run_name='Trial_9', flux_hst='iso', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[82], run_name='Trial_9', flux_hst='iso', cal='2', v_min=50, v_max=1000, prior='uniform')
# qls = True
# gal_fit(gal_num=[5, 7, 83, 181, 182], run_name='Trial_9', flux_hst='auto', cal='0', v_min=50, v_max=1000,
#         prior='uniform')

# Trial 10: Inflate all errors + Uniform prior + photomtery only
# qls, spectrum_exists = False, False
# gal_fit(gal_num=[1, 13, 35, 62, 120, 134, 141, 164],
#         run_name='Trial_10', flux_hst='auto', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[4, 88], run_name='Trial_10', flux_hst='iso', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[93], run_name='Trial_10', flux_hst='auto', cal='0', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[36], run_name='Trial_10', flux_hst='iso', cal='0', v_min=50, v_max=200, prior='uniform')
# gal_fit(gal_num=[57], run_name='Trial_10', flux_hst='auto', cal='2', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[64], run_name='Trial_10', flux_hst='auto', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[80], run_name='Trial_10', flux_hst='iso', cal='0', v_min=50, v_max=1000, prior='uniform')
# gal_fit(gal_num=[82], run_name='Trial_10', flux_hst='iso', cal='2', v_min=50, v_max=1000, prior='uniform')