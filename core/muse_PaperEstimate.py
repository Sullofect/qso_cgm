import os
import muse_kc
import pyneb as pn
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from muse_RenameGal import ReturnGalLabel
from regions import RectangleSkyRegion, RectanglePixelRegion
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path_savetab = '/Users/lzq/Dropbox/Data/CGM_tables/'
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


# Completness
path_s = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects_sean.fits')
data_s = fits.getdata(path_s, 1, ignore_missing_end=True)
path_w = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'redshifting', 'ESO_DEEP_offset_zapped_spec1D',
                      'ESO_DEEP_offset_zapped_objects.fits')
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

# Only need
sort = ql_w != 0
ra_w, dec_w = ra_w[sort], dec_w[sort]
z_w = z_w[sort]
row_w = row_w[sort]

# Getting photometry zero point
path_pho = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'config', 'gal_all',
                        'HE0238-1904_sex_gal_all.fits')
data_pho = fits.getdata(path_pho, 1, ignore_missing_end=True)
path_hb = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'HE0238-1904_drc_offset.fits')
wcs_hst = WCS(fits.open(path_hb)[1].header)

# match two catalog
catalog = SkyCoord(data_pho['AlPHAWIN_J2000'], data_pho['DELTAWIN_J2000'], unit="deg")
c = SkyCoord(ra_w, dec_w, unit="deg")
test = SkyCoord(40.1345022, -18.8509434, unit="deg")
idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# Photometry
data_pho_red = data_pho[idx]
mag_auto, dmag_auto = data_pho['MAG_AUTO'], data_pho['MAGERR_AUTO']
center_sky = SkyCoord(40.1359, -18.8643, unit='deg', frame='fk5')
region_sky = RectangleSkyRegion(center=center_sky, width=65 / 3600 * u.deg, height=65 / 3600 * u.deg, angle=60 * u.deg)
mask = region_sky.contains(catalog, wcs_hst)
print(region_sky.contains(test, wcs_hst))
#
number_red, x_image_red, y_image_red = data_pho_red['NUMBER'], data_pho_red['X_IMAGE'], data_pho_red['Y_IMAGE']
mag_iso_red, dmag_iso_red = data_pho_red['MAG_ISO'], data_pho_red['MAGERR_ISO']
mag_isocor_red, dmag_isocor_red = data_pho_red['MAG_ISOCOR'], data_pho_red['MAGERR_ISOCOR']
mag_auto_red, dmag_auto_red = data_pho_red['MAG_AUTO'], data_pho_red['MAGERR_AUTO']
# print(row_w[(mag_auto_red < 23) * (mag_auto_red > 22)])
# print(mag_auto_red[(mag_auto_red < 23) * (mag_auto_red > 22)])
# print(mag_auto[mask][(mag_auto[mask] < 23) * (mag_auto[mask] > 22)])

#
bins = np.linspace(18, 30, 13)
plt.figure(figsize=(5, 5))
plt.hist(mag_auto_red, bins=bins, color='red', histtype='step', alpha=0.5, label='Redshift survey')
plt.hist(mag_auto[mask], bins=bins, color='blue', histtype='step', alpha=0.5, label='Entire catalog')
plt.minorticks_on()
plt.xlabel('Redshift')
plt.ylabel('Number')
plt.legend()
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/Mag_vs_redshift.png', bbox_inches='tight')


# Estimate Gas table
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

area = (radius_array * 100 / 15) ** 2 * np.pi
# print('area =', area)

z_qso = 0.6282144177077355

# Dered ratio
lineRatio = Table.read('/Users/lzq/Dropbox/Data/CGM/RegionLinesRatio/'
                       'RegionLinesRatio_dered.fits')
v_gas = 3e5 * (lineRatio['z'] - z_qso) / (1 + z_qso)
dv_gas = np.sqrt((3e5 * (lineRatio['dz']) / (1 + z_qso)) ** 2 + 3 ** 2) # Weilbacher+2020
dsigma_gas = np.sqrt(lineRatio['dsigma'] ** 2 + 4 ** 2)  # Kamann+2016
lineRatio_S3S4 = Table.read('/Users/lzq/Dropbox/Data/CGM/RegionLinesRatio/'
                            'RegionLinesRatio_S3S4_dered.fits')
v_gas_S3S4 = 3e5 * (lineRatio_S3S4['z'] - z_qso) / (1 + z_qso)
dv_gas_S3S4 = np.sqrt((3e5 * (lineRatio_S3S4['dz']) / (1 + z_qso)) ** 2 + 3 ** 2)
dsigma_S3S4 = np.sqrt(lineRatio_S3S4['dsigma'] ** 2 + 4 ** 2)
v_gas_S3S4_wing = 3e5 * (lineRatio_S3S4['z'] + lineRatio_S3S4['dz_wing'] - z_qso) / (1 + z_qso)
dv_gas_S3S4_wing = np.sqrt((3e5 * (lineRatio_S3S4['ddz_wing']) / (1 + z_qso)) ** 2 + 3 ** 2)
dsigma_S3S4_wing = np.sqrt(lineRatio_S3S4['dsigma_wing'] ** 2 + 4 ** 2)
print('flux in [O II]', np.round(lineRatio['flux_OII'], 2))
print('flux in Hbeta', np.round(lineRatio['flux_Hbeta'], 2))
print('flux in [O III]]', np.round(lineRatio['flux_OIII5008'], 2))
print('flux in [Ne V]]', np.round(lineRatio['flux_NeV3346'], 2))
print('flux in [O III]]', np.round(lineRatio['flux_OIII4364'], 2))
print('flux in [He II]]', np.round(lineRatio['flux_HeII4687'], 2))

#
print('dflux in [O II]', np.round(lineRatio['dflux_OII'], 2))
print('dflux in Hbeta', np.round(lineRatio['dflux_Hbeta'], 2))
print('dflux in [O III]]', np.round(lineRatio['dflux_OIII5008'], 2))
print('dflux in [Ne V]]', np.round(lineRatio['dflux_NeV3346'], 2))
print('dflux in [O III]]', np.round(lineRatio['dflux_OIII4364'], 2))
print('dflux in [He II]]', np.round(lineRatio['dflux_HeII4687'], 2))
print('LOS velocity is', np.round(v_gas, 0), np.round(dv_gas, 0))
print('LOS velocity dispersion is', np.round(lineRatio['sigma'], 0), np.round(dsigma_gas, 0))

# S3 S4 Account for the wing
print('flux in [O II] S3 S4', np.round(lineRatio_S3S4['flux_OII'], 2))
print('flux in Hbeta S3 S4', np.round(lineRatio_S3S4['flux_Hbeta'], 2))
print('flux in [O III]] S3 S4', np.round(lineRatio_S3S4['flux_OIII5008'], 2))
print('flux in [Ne V]] S3 S4', np.round(lineRatio_S3S4['flux_NeV3346'], 2))
print('flux in [O III]] S3 S4', np.round(lineRatio_S3S4['flux_OIII4364'], 2))
print('flux in [He II]] S3 S4', np.round(lineRatio_S3S4['flux_HeII4687'], 2))

print('dflux in [O II] S3 S4', np.round(lineRatio_S3S4['dflux_OII'], 2))
print('dflux in Hbeta S3 S4', np.round(lineRatio_S3S4['dflux_Hbeta'], 2))
print('dflux in [O III]] S3 S4', np.round(lineRatio_S3S4['dflux_OIII5008'], 2))
print('dflux in [Ne V]] S3 S4', np.round(lineRatio_S3S4['dflux_NeV3346'], 2))
print('dflux in [O III]] S3 S4', np.round(lineRatio_S3S4['dflux_OIII4364'], 2))
print('dflux in [He II]] S3 S4', np.round(lineRatio_S3S4['dflux_HeII4687'], 2))

print('flux in [O II] S3 S4 wing', np.round(lineRatio_S3S4['flux_OII_wing'], 2))
print('flux in Hbeta S3 S4 wing', np.round(lineRatio_S3S4['flux_Hbeta_wing'], 2))
print('flux in [O III]] S3 S4 wing', np.round(lineRatio_S3S4['flux_OIII5008_wing'], 2))
print('flux in [Ne V]] S3 S4 wing', np.round(lineRatio_S3S4['flux_NeV3346_wing'], 2))
print('flux in [O III]] S3 S4 wing', np.round(lineRatio_S3S4['flux_OIII4364_wing'], 2))
print('flux in [He II]] S3 S4 wing', np.round(lineRatio_S3S4['flux_HeII4687_wing'], 2))

print('dflux in [O II] S3 S4 wing', np.round(lineRatio_S3S4['dflux_OII_wing'], 2))
print('dflux in Hbeta S3 S4 wing', np.round(lineRatio_S3S4['dflux_Hbeta_wing'], 2))
print('dflux in [O III]] S3 S4 wing', np.round(lineRatio_S3S4['dflux_OIII5008_wing'], 2))
print('dflux in [Ne V]] S3 S4 wing', np.round(lineRatio_S3S4['dflux_NeV3346_wing'], 2))
print('dflux in [O III]] S3 S4 wing', np.round(lineRatio_S3S4['dflux_OIII4364_wing'], 2))
print('dflux in [He II]] S3 S4 wing', np.round(lineRatio_S3S4['dflux_HeII4687_wing'], 2))

print('LOS velocity of S3 S4 is', np.round(v_gas_S3S4, 0), np.round(dv_gas_S3S4, 0))
print('LOS velocity of S3 S4 wing is', np.round(v_gas_S3S4_wing, 0), np.round(dv_gas_S3S4_wing, 0))
print('LOS velocity dispersion of S3 S4 is', np.round(lineRatio_S3S4['sigma'], 0), np.round(dsigma_S3S4, 0))
print('LOS velocity dispersion of S3 S4 wing is', np.round(lineRatio_S3S4['sigma_wing'], 0), np.round(dsigma_S3S4_wing, 0))

# For B4
v_gas_B4 = 3e5 * (0.628376 - z_qso) / (1 + z_qso)
v_gas_B4_wing = 3e5 * (0.628376 - 0.000612 - z_qso) / (1 + z_qso)
dv_gas_B4 = np.sqrt((3e5 * (0.000013) / (1 + z_qso)) ** 2 + 3 ** 2)
dv_gas_B4_wing = np.sqrt((3e5 * (0.000225) / (1 + z_qso)) ** 2 + 3 ** 2)
sigma_B4 = 30.152064
sigma_B4_wing = 200.846869
dsigma_B4 = np.sqrt(4.576155 ** 2 + 4 ** 2)
dsigma_B4_wing = np.sqrt(35.392421 ** 2 + 4 ** 2)
print('LOS velocity of B4', np.round(v_gas_B4, 0), np.round(dv_gas_B4, 0))
print('LOS velocity of B4 wing', np.round(v_gas_B4_wing, 0), np.round(dv_gas_B4_wing, 0))
print('LOS velocity dispersion of B4', np.round(sigma_B4, 0), np.round(dsigma_B4, 0))
print('LOS velocity dispersion of B4 wing', np.round(sigma_B4_wing, 0), np.round(dsigma_B4_wing, 0))

# Ratio between Hgamma, Hdelta
print('Balmer line Hgam / Hbeta ratio', np.round(lineRatio['flux_Hgam'] / lineRatio['flux_Hbeta'], 2))
print('Balmer line Hdel / Hbeta ratio', np.round(lineRatio['flux_Hdel'] / lineRatio['flux_Hbeta'], 2))
print('Balmer line Hdel / Hgam ratio', np.round(lineRatio['flux_Hdel'] / lineRatio['flux_Hgam'], 2))

# Virial theorem
G = 6.67e-8
R_red, v_red = 200 * 3.086e21, 506 * 1e5
R_blue, v_blue = 100 * 3.086e21, 91 * 1e5
M_red = 2 * R_red * v_red ** 2 / G / 2e33
M_blue = 2 * R_blue * v_blue ** 2 / G / 2e33
print(np.format_float_scientific(M_red), np.format_float_scientific(M_blue))

# Munari et al. 2013
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h_z = cosmo.H(z_qso).value / 100
sigma_red, sigma_blue = 506, 91
M_200_red = (sigma_red / 1100) ** 3 * 1e15 / h_z
M_200_red_test = (370 / 1100) ** 3 * 1e15 / h_z
M_200_blue = (sigma_blue / 1100) ** 3 * 1e15 / h_z
M_200_blue_test = (50 / 1100) ** 3 * 1e15 / h_z
print('Halo mass are', np.format_float_scientific(M_200_red), np.format_float_scientific(M_200_blue))
print('Halo mass test are', np.format_float_scientific(M_200_red_test),
      np.format_float_scientific(M_200_blue_test))

# Estimate typical L star is like at z=0.63
m_abs_Lstar = muse_kc.abs2app(m_abs=-21.5, z=0.63, model='Scd', filter_e='Bessell_B', filter_o='ACS_f814W')
print('Apparent magnitude of L* is', m_abs_Lstar)

# Use Chen+2019
area_arcsec = np.pi * radius_array ** 2  # in arcsec ** 2
SB_Halpha = 3 * lineRatio['flux_Hbeta'] * 1e-17 / area_arcsec
C = 1  # Clumping factor
l = 30  # in kpc
n_e = np.sqrt(SB_Halpha * (1 + z_qso) ** 4 / C / l / 1.7e-15)
print(np.round(n_e, 2))


# Compute logU
r = 10 ** 23.15
n = 10 ** -0.8
c = 3e10
Q = 10 ** 56.9741 + 10 ** 57.1016 + 10 ** 57.4155 + 10 ** 57.4155
logU = np.log10(Q / 4 / np.pi / r ** 2 / n / c)

# load the region
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

# Calculate the distance to a specific region
z = 0.6282144177077355
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.angular_diameter_distance(z=z)
ratio = (1 * u.radian).to(u.arcsec).value
arcsec_15 = (15 * d_l / ratio).to(u.kpc).value
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_s2, dec_s2 = 40.1364401, -18.8655766
c_qso = SkyCoord(ra_qso_muse, dec_qso_muse, frame='icrs', unit='deg')
c_s2 = SkyCoord(ra_array, dec_array, frame='icrs', unit='deg')
ang_sep = c_s2.separation(c_qso).to(u.arcsec).value
distance = np.log10((ang_sep * d_l / ratio).to(u.cm).value)
print((ang_sep * d_l / ratio).to(u.kpc).value)

region = ['S1', 'S2', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'B1', 'B2_new', 'B3_new']
den_array = [1.59, 1.68, 4.18, 2.7, 3.03, 3.45, 4.13, 3.6, 2.07, 2.92, 1.87]
alpha_array = [-0.77, -0.79, -0.29, -0.59, -0.9, -1, -1.02, -1.07, -0.73, -0.56, -0.67]
z_array = [-0.16, 0.14, 0.14, -0.18, -0.09, -0.19, 0.04, -0.13, -0.46, -0.56, -0.4]

def CreateGrid(z_array, alpha_array, den_array, L_qso=46.54, region=None, trial=None):
    global text_array
    command_array = np.array([])
    for i in range(len(z_array)):
        dis = np.around(distance[text_array == region[i]][0], decimals=2)
        os.makedirs('/Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU', exist_ok=True)
        os.popen('cp /Users/lzq/Dropbox/Data/CGM/cloudy/S1_t1/linelist.dat'
                 ' /Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU/linelist.dat')
        lines = np.array(['Table power law spectral index ' + str(alpha_array[i]) + ', low=0.37, high=73.5 ',
                          'nuL(nu) = ' + str(L_qso) + ' at 1.0 Ryd',
                          'hden ' + str(den_array[i]),
                          'metals ' + str(z_array[i]) + ' log',
                          'radius ' + str(dis),
                          'iterative to convergence'])
        np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU/'
                   + region[i] + '.in', lines, fmt="%s")

        # Command
        command = np.array(['$cloudy -r ' + region[i]])
        command_array = np.hstack((command_array, command))
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/ComputeLogU/command.txt', command_array, fmt="%s")

# CreateGrid(z_array, alpha_array, den_array, region=region)


# Estimate mass to light ratio
ID_ste = np.array([2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 19, 21, 22, 23, 24,
                   25, 26, 27, 28, 29, 30, 32, 33, 34])
M_ste = 10 ** np.array([9.5, 8.3, 7.4, 10.1, 9.3, 10.4, 9.7, 11.5, 10.6, 10.0, 9.0, 10.1, 9.5, 10.3, 10.3, 10.1, 9.5,
                        9.8, 8.8, 9.5, 9.9, 9.5, 8.3, 8.9, 9.0, 11.2, 10.8])
M_B = np.array([-18.5, -17.3, -17, -19.4, -18.0, -20, -18.1, -21.9, -20.4, -19.2, -17, -19.1, -18, -19.7,
                -19.7, -18.9, -17.6, -18.5, -16.3, -18.1, -18.8, -19.1, -17.8, -17.6, -17.1, -21.2, -22.1])
L_B = 10 ** (0.4 * (5.33 - M_B))
M_L = M_ste / L_B
print('G1', np.log10(np.median(M_L) * 10 ** (0.4 * (5.33 + 17.5))))
print('G3', np.log10(np.median(M_L) * 10 ** (0.4 * (5.33 + 18.3))))

# # test
# m_abs = muse_kc.app2abs(m_app=23.8368, z=0.6280, model='irregular', filter_e='Bessell_B', filter_o='ACS_f814W')
m_abs = muse_kc.app2abs(m_app=22.3, z=0.6341, model='S0', filter_e='Bessell_B', filter_o='ACS_f814W')
print('Apparent magnitude of L* is', m_abs)


#
M_BH = 9.4
print((M_BH - 8.95) / 1.40 + 11)  # Different IMF
print((M_BH - 9 - np.log10(0.49)) / 1.16 + 11)
print((M_BH - 9 - 0.3 - np.log10(0.49)) / 1.16 + 11)  # 0.3 dex error on M_BH


# Dynamical mass 3C57
G = 6.67e-8
R_3C57, v_3C57 = 5 / 7 * 50 * 3.086e21, 250 * 1e5
M_3C57 = R_3C57 * v_3C57 ** 2 / G / 2e33
print(np.format_float_scientific(M_3C57))

