# %matplotlib inline
import os
import glob
import sys
import lmfit
import warnings
import numpy as np
import matplotlib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from mpdaf.drs import PixTable
from matplotlib import rc
from PyAstronomy import pyasl
from astropy.io import fits
warnings.filterwarnings("ignore")
sys.path.append('/Users/lzq/Dropbox/PyQSOFit')

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'
path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits')
cube = Cube(path)

# Calculate the white image
image_white = cube.sum(axis=0)
p, q = image_white.peak()['p'], image_white.peak()['q']

# plt.figure(figsize=(8, 5), dpi=300)
# plt.imshow(np.arcsinh(image_white.data), origin='lower', vmin=9, cmap='hot')
# cbar = plt.colorbar()
# cbar.set_label(r'$\mathrm{Arcsinh}$')
# # plt.axis('off')
# plt.savefig(path_savefig + 'qso_image.png', bbox_inches='tight')

# EE Curve
wave = pyasl.airtovac2(cube.wave.coord())
EE_ape = cube.subcube_circle_aperture((p, q), 10, unit_center=None) # Arcsec
source_100 = EE_ape[350, :, :]  # change to wavelength instead of index
radius_100, ee_100 = source_100.eer_curve(unit_center=None)
source_200 = EE_ape[3000, :, :]
radius_200, ee_200 = source_200.eer_curve(unit_center=None)

# Determine the seeing
ee_max_100 = np.max(ee_100)
ee_max_200 = np.max(ee_200)
f_100 = interp1d(ee_100, radius_100)
f_200 = interp1d(ee_200, radius_200)
print(f_100(ee_max_100 * 0.5))
print(f_200(ee_max_200 * 0.5))

# Make the EE plot
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(radius_100, ee_100, label=r'$\lambda = $' + str(wave[350]))
plt.plot(radius_200, ee_200, label=r'$\lambda = $' + str(wave[3000]))
plt.vlines(f_100(ee_max_100), ymin=0, ymax=1)
plt.vlines(f_200(ee_max_200), ymin=0, ymax=1)
plt.xlabel(r'$\mathrm{Arcseconds}$', size=20)
plt.ylabel(r'$\mathrm{Ratio}$', size=20)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
plt.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                size=3)
plt.savefig(path_savefig + 'ee_curve', bbox_inches='tight')


# Old version
# # Spectrum
# cube_ape = cube.aperture((p, q), 2, unit_center=None, unit_radius=None, is_sum=True)
#
# wave_vac = pyasl.airtovac2(cube_ape.wave.coord())
# # convert air wavelength to vacuum 2. measure the redshift (OII doublet emission line)
# # 3. luminosity and the line width of Hbeta
# flux = cube_ape.data * 1e-3
# flux_err = np.sqrt(cube_ape.var) * 1e-3
#
# print((np.sqrt(cube_ape.var) / cube_ape.data).max())
#
# # Fitting the redshift
# wave_OII3727_vac = 3727.092
# wave_OII3729_vac = 3729.875
# c_kms = 2.998e5
#
# OII_region = np.where((wave_vac > 6050) * (wave_vac < 6100))
# wave_OII_vac = wave_vac[OII_region]
# flux_OII = flux[OII_region]
#
#
# # Define a function with the MUSE resolution
# def getFWHM_MUSE(wave):
#     return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04)
#
#
# # Define a function with the MUSE resolution
# def getSigma_MUSE(wave):
#     return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355
#
#
# # Define the model fit function
# def model(wave_vac, z, sigma_kms, fluxOII, rOII3729_3727, a, b):
#     wave_OII3727_obs = wave_OII3727_vac * (1 + z)
#     wave_OII3729_obs = wave_OII3729_vac * (1 + z)
#
#     sigma_OII3727_A = np.sqrt((sigma_kms / c_kms * wave_OII3727_obs) ** 2 + (getSigma_MUSE(wave_OII3727_obs)) ** 2)
#     sigma_OII3729_A = np.sqrt((sigma_kms / c_kms * wave_OII3729_obs) ** 2 + (getSigma_MUSE(wave_OII3729_obs)) ** 2)
#
#     # sigma_OII3727_A = sigma_kms / c_kms * wave_OII3727_obs
#     # sigma_OII3729_A = sigma_kms / c_kms * wave_OII3729_obs
#
#     fluxOII3727 = fluxOII / (1 + rOII3729_3727)
#     fluxOII3729 = fluxOII / (1 + 1.0 / rOII3729_3727)
#
#     peakOII3727 = fluxOII3727 / np.sqrt(2 * sigma_OII3727_A ** 2 * np.pi)
#     peakOII3729 = fluxOII3729 / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)
#
#     OII3727_gaussian = peakOII3727 * np.exp(-(wave_vac - wave_OII3727_obs) ** 2 / 2 / sigma_OII3727_A ** 2)
#     OII3729_gaussian = peakOII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)
#
#     return OII3727_gaussian + OII3729_gaussian + a * wave_vac + b
#
#
# redshift_guess = 0.63
# sigma_kms_guess = 150.0
# flux_OII_guess = 42
# rOII3729_3727_guess = 100
#
# parameters = lmfit.Parameters()
# parameters.add_many(('z', redshift_guess, True, None, None, None),
#                     ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
#                     ('fluxOII', flux_OII_guess, True, None, None, None),
#                     ('rOII3729_3727', rOII3729_3727_guess, True, 0, 3, None),
#                     ('a', 0.0, True, None, None, None),
#                     ('b', 100, True, None, None, None))
# spec_model = lmfit.Model(model, missing='drop')
# result = spec_model.fit(flux_OII, wave_vac=wave_OII_vac, params=parameters)
# # print('Success = {}'.format(result.success))
# # print(result.fit_report())
# z = result.best_values['z']
#
#
# def rest2obs(x):
#     return x * (1 + z)
#
#
# def obs2rest(x):
#     return x / (1 + z)
#
#
# # Plot the Spectrum
# fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
# ax.plot(wave_vac, flux, color='k')
# ax.plot(wave_vac, flux_err, color='lightgrey')
# ax.set_xlim(5200, 9300)
# ax.set_ylim(0, 100)
# ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; (\AA)}$', size=20)
# ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20)
#
# secax = ax.secondary_xaxis('top', functions=(obs2rest, rest2obs))
# secax.minorticks_on()
# secax.set_xlabel(r'$\mathrm{Rest \mathrm{-} frame \; Wavelength \; (\AA)}$', size=20)
# secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
# secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
# ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
#
# axins = ax.inset_axes([0.71, 0.61, 0.27, 0.3])
# axins.plot(wave_OII_vac, flux_OII, color='black', drawstyle='steps-mid')
# axins.plot(wave_OII_vac, result.best_fit, color='red')
# axins.set_xlim(6060, 6085)
# axins.set_ylim(55, 59)
# axins.set_ylabel(r'${f}_{\lambda}$', size=15)
# axins.text(6062, 58, r'$\mathrm{[O\;II]}$', size=13)
# axins.minorticks_on()
# axins.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on',
#                   labelsize=13, size=5)
# axins.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on',
#                   size=3)
#
# secaxins = axins.secondary_xaxis('top', functions=(obs2rest, rest2obs))
# secaxins.minorticks_on()
# secaxins.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=13)
# secaxins.tick_params(axis='x', which='minor', direction='in', top='on', size=3)
# plt.savefig('/Users/lzq/Dropbox/qso_cgm/qso_spectrum_new',  bbox_inches='tight')
#
#
#
# # Iron and continuum fitting
# cont_region_1 = np.where((wave_vac > 4435 * (1 + z)) * (wave_vac < 4700 * (1 + z)))
# wave_contr1_vac = wave_vac[cont_region_1]
# flux_contr1 = flux[cont_region_1]
#
# cont_region_2 = np.where((wave_vac > 5100 * (1 + z)) * (wave_vac < 5535 * (1 + z)))
# wave_contr2_vac = wave_vac[cont_region_2]
# flux_contr2 = flux[cont_region_2]
#
# wave_cont_vac = np.hstack((wave_contr1_vac, wave_contr1_vac))
# flux_cont = np.hstack((flux_contr1, flux_contr2))
#
# path = '/Users/lzq/Dropbox/Data/CGM/'
# fe_uv = np.genfromtxt(path + 'fe_uv.txt')
# fe_op = np.genfromtxt(path + 'fe_optical.txt')
#
# def Fe_flux_balmer(xval, p1, p2):
#     "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
#     yval = np.zeros_like(xval)
#
#     wave_Fe_balmer = 10 ** fe_op[:, 0]
#     flux_Fe_balmer = fe_op[:, 1] / 1e-17
#     ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.), True, False)
#     wave_Fe_balmer = wave_Fe_balmer[ind]
#     flux_Fe_balmer = flux_Fe_balmer[ind]
#     Fe_FWHM = p2
#
#
#     xval_new = xval / (1.0 + z)
#     ind = np.where((xval_new > 3686.) & (xval_new < 7484.), True, False)
#     if np.sum(ind) > 100:
#         if Fe_FWHM < 900.0:
#             sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
#         else:
#             sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
#
#         # Get sigma in pixel space
#         sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
#
#         khalfsz = np.round(4 * sig_pix + 1, 0)
#         xx = np.arange(0, khalfsz * 2, 1) - khalfsz
#         kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
#         kernel = kernel / np.sum(kernel)
#
#         flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel, 'same')
#
#         tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv)
#
#         yval[ind] = p1 * interpolate.splev(xval_new[ind], tck)
#     return yval
#
#
# def continuum(xval, p1, p2, p3, p4):
#     return Fe_flux_balmer(xval, p1, p2) + p3 * wave_vac ** p4
#
#
# p1_guess = 0.63
# p2_guess = 1200
# p3_guess = 5
# p4_guess = 1
#
# parameters = lmfit.Parameters()
# parameters.add_many(('p1', p1_guess, True, None, None, None),
#                     ('p2', p2_guess, True, 0, 1200.0, None),
#                     ('p3', p3_guess, True, None, None, None),
#                     ('p4', p4_guess, True, 0, 3, None))
# spec_model = lmfit.Model(model, missing='drop')
# result = spec_model.fit(flux_cont, wave_vac=wave_cont_vac_vac, params=parameters)
# # print('Success = {}'.format(result.success))
# print(result.fit_report())
#
# p1 = result.best_values['p1']
#
#
# # def Fe_pa
#
#
#
#
# # class QsoFit:
# #     def __init__(self):
# #
# #
# #     def Fit(self, ):
# #         return
#
#
#
#
#     # def continuum(self, A, power):
#     #
#     # def Fe_pa
#
#
# """
# Continuum components described by 14 parameters
#  pp[0]: norm_factor for the MgII Fe_template
#  pp[1]: FWHM for the MgII Fe_template
#  pp[2]: small shift of wavelength for the MgII Fe template
#  pp[3:5]: same as pp[0:2] but for the Hbeta/Halpha Fe template
#  pp[6]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
#  pp[7]: slope for the power-law continuum
#  pp[8:10]: norm, Te and Tau_e for the Balmer continuum at <3646 A
#  pp[11:13]: polynomial for the continuum
# """
#
#
#
# # QSO_fit
#
# from PyQSOFit import QSOFit
#
# path = '../qso_cgm/'
#
# newdata = np.rec.array([(6564.61, 'Ha', 6400., 6800., 'Ha_br', 3, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05), \
#                         (6564.61, 'Ha', 6400., 6800., 'Ha_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
#                         (6549.85, 'Ha', 6400., 6800., 'NII6549', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 1, 0.001), \
#                         (6585.28, 'Ha', 6400., 6800., 'NII6585', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 1, 0.003), \
#                         (6718.29, 'Ha', 6400., 6800., 'SII6718', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 2, 0.001), \
#                         (6732.67, 'Ha', 6400., 6800., 'SII6732', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 2, 0.001), \
#                         (4862.68, 'Hb', 4640., 5100., 'Hb_br', 1, 5e-3, 0.004, 0.05, 0.01, 0, 0, 0, 0.01), \
#                         (4862.68, 'Hb', 4640., 5100., 'Hb_na', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
#                         (4960.30, 'Hb', 4640., 5100., 'OIII4959c', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
#                         (5008.24, 'Hb', 4640., 5100., 'OIII5007c', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 1, 1, 0, 0.004), \
#                         # (4960.30,'Hb',4640.,5100.,'OIII4959w',1,3e-3,2.3e-4,0.004,0.01,2,2,0,0.001),\
#                         # (5008.24,'Hb',4640.,5100.,'OIII5007w',1,3e-3,2.3e-4,0.004,0.01,2,2,0,0.002),\
#                         # (4687.02,'Hb',4640.,5100.,'HeII4687_br',1,5e-3,0.004,0.05,0.005,0,0,0,0.001),\
#                         # (4687.02,'Hb',4640.,5100.,'HeII4687_na',1,1e-3,2.3e-4,0.0017,0.005,1,1,0,0.001),\
#
#                         # (3934.78,'CaII',3900.,3960.,'CaII3934',2,1e-3,3.333e-4,0.0017,0.01,99,0,0,-0.001),\
#
#                         # (3728.48,'OII',3650.,3800.,'OII3728',1,1e-3,3.333e-4,0.0017,0.01,1,1,0,0.001),\
#
#                         # (3426.84,'NeV',3380.,3480.,'NeV3426',1,1e-3,3.333e-4,0.0017,0.01,0,0,0,0.001),\
#                         # (3426.84,'NeV',3380.,3480.,'NeV3426_br',1,5e-3,0.0025,0.02,0.01,0,0,0,0.001),\
#
#                         (2798.75, 'MgII', 2700., 2900., 'MgII_br', 1, 5e-3, 0.004, 0.05, 0.0017, 0, 0, 0, 0.05), \
#                         (2798.75, 'MgII', 2700., 2900., 'MgII_na', 2, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
#                         (1908.73, 'CIII', 1700., 1970., 'CIII_br', 2, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01), \
#                         # (1908.73,'CIII',1700.,1970.,'CIII_na',1,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),\
#                         # (1892.03,'CIII',1700.,1970.,'SiIII1892',1,2e-3,0.001,0.015,0.003,1,1,0,0.005),\
#                         # (1857.40,'CIII',1700.,1970.,'AlIII1857',1,2e-3,0.001,0.015,0.003,1,1,0,0.005),\
#                         # (1816.98,'CIII',1700.,1970.,'SiII1816',1,2e-3,0.001,0.015,0.01,1,1,0,0.0002),\
#                         # (1786.7,'CIII',1700.,1970.,'FeII1787',1,2e-3,0.001,0.015,0.01,1,1,0,0.0002),\
#                         # (1750.26,'CIII',1700.,1970.,'NIII1750',1,2e-3,0.001,0.015,0.01,1,1,0,0.001),\
#                         # (1718.55,'CIII',1700.,1900.,'NIV1718',1,2e-3,0.001,0.015,0.01,1,1,0,0.001),\
#
#                         (1549.06, 'CIV', 1500., 1700., 'CIV_br', 1, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05), \
#                         (1549.06, 'CIV', 1500., 1700., 'CIV_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
#                         # (1640.42,'CIV',1500.,1700.,'HeII1640',1,1e-3,5e-4,0.0017,0.008,1,1,0,0.002),\
#                         # (1663.48,'CIV',1500.,1700.,'OIII1663',1,1e-3,5e-4,0.0017,0.008,1,1,0,0.002),\
#                         # (1640.42,'CIV',1500.,1700.,'HeII1640_br',1,5e-3,0.0025,0.02,0.008,1,1,0,0.002),\
#                         # (1663.48,'CIV',1500.,1700.,'OIII1663_br',1,5e-3,0.0025,0.02,0.008,1,1,0,0.002),\
#
#                         # (1402.06,'SiIV',1290.,1450.,'SiIV_OIV1',1,5e-3,0.002,0.05,0.015,1,1,0,0.05),\
#                         # (1396.76,'SiIV',1290.,1450.,'SiIV_OIV2',1,5e-3,0.002,0.05,0.015,1,1,0,0.05),\
#                         # (1335.30,'SiIV',1290.,1450.,'CII1335',1,2e-3,0.001,0.015,0.01,1,1,0,0.001),\
#                         # (1304.35,'SiIV',1290.,1450.,'OI1304',1,2e-3,0.001,0.015,0.01,1,1,0,0.001),\
#
#                         (1215.67, 'Lya', 1150., 1290., 'Lya_br', 1, 5e-3, 0.004, 0.05, 0.02, 0, 0, 0, 0.05), \
#                         (1215.67, 'Lya', 1150., 1290., 'Lya_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0, 0, 0.002) \
#                         ], \
#                        formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,\
#                      float32,float32,float32,float32,float32', \
#                        names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue')
# # ------header-----------------
# hdr = fits.Header()
# hdr['lambda'] = 'Vacuum Wavelength in Ang'
# hdr['minwav'] = 'Lower complex fitting wavelength range'
# hdr['maxwav'] = 'Upper complex fitting wavelength range'
# hdr['ngauss'] = 'Number of Gaussians for the line'
# hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
# hdr['minsig'] = 'Lower range of line sigma [lnlambda]'
# hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'
# hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
# hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
# hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
# hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
# hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
# # ------save line info-----------
# hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
# hdu.writeto(path + 'qsopar.fits', overwrite=True)
#
#
# path1 = '../PyQSOFit/'               # the path of the source code file and qsopar.fits
# path2 = '../qso_cgm/data/result/' # path of fitting results
# path3 = '../qso_cgm/data/QA_result/'   # path of figure
# path4 = '../PyQSOFit/sfddata/'             # path of dusp reddening map
#
# #Requried
# # an important note that all the data input must be finite, especically for the error !!!
# lam_qsofit = cube_ape.wave.coord()       # OBS wavelength [A]
# flux_qsofit = flux.data     # OBS flux [erg/s/cm^2/A]
# flux_err_qsofit = flux_err.data # 1 sigma error
#
# # get data prepared
# q = QSOFit(lam_qsofit, flux_qsofit, flux_err_qsofit, z, ra=40.1359, dec=-18.8643, plateid=0, mjd=0, fiberid=0,
#            path=path1)
#
# start = timeit.default_timer()
# # do the fitting
# q.Fit(name='HE0238-1904', nsmooth=1, and_or_mask=False, deredden=True, reject_badpix=False, wave_range=None,
#       wave_mask=np.array([[4650, 4750]]), decomposition_host=True, Mi=None, npca_gal=5, npca_qso=20, Fe_uv_op=True,
#       Fe_flux_range=np.array([4435, 4685]), poly=True, BC=False, rej_abs=False, initial_guess=None, MC=False,
#       n_trails=5, linefit=True, tie_lambda=True, tie_width=True, tie_flux_1=True, tie_flux_2=True, save_result=True,
#       plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True, dustmap_path=path4, save_fig_path=path3,
#       save_fits_path=path2, save_fits_name='qso_spe')
#
# end = timeit.default_timer()
#
# print('Fitting finished in : ' + str(np.round(end - start)) + 's')
#
# # grey shade on the top is the continuum windiows used to fit.
#




