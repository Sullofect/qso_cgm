import os
import sys
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from PyAstronomy import pyasl
sys.path.append('../../PyQSOFit')
from PyQSOFit import QSOFit
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import UnivariateSpline
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=3, visible=True)
rc('ytick.minor', size=3, visible=True)
wave_MgII2796_vac = 2796.35
wave_MgII2803_vac = 2803.53
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
wave_OII3728_vac = (wave_OII3727_vac + wave_OII3729_vac) / 2
c_kms = 2.998e5

def rest2obs(x):
    return x * (1 + z)

def obs2rest(x):
    return x / (1 + z)

def model_cont(wave_vac, a, b):
    return a * wave_vac + b

# Define the model fit function
def model_OII(wave_vac, z, sigma_kms, flux_OII, r_OII3729_3727, a, b):
    wave_OII3727_obs = wave_OII3727_vac * (1 + z)
    wave_OII3729_obs = wave_OII3729_vac * (1 + z)

    sigma_OII3727_A = (sigma_kms / c_kms * wave_OII3727_obs)
    sigma_OII3729_A = (sigma_kms / c_kms * wave_OII3729_obs)

    flux_OII3727 = flux_OII / (1 + r_OII3729_3727)
    flux_OII3729 = flux_OII / (1 + 1.0 / r_OII3729_3727)

    peak_OII3727 = flux_OII3727 / np.sqrt(2 * sigma_OII3727_A ** 2 * np.pi)
    peak_OII3729 = flux_OII3729 / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)

    OII3727_gaussian = peak_OII3727 * np.exp(-(wave_vac - wave_OII3727_obs) ** 2 / 2 / sigma_OII3727_A ** 2)
    OII3729_gaussian = peak_OII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)

    return OII3727_gaussian + OII3729_gaussian + a * wave_vac + b

# Path to the spectra
path_spectra = '../../Proposal/HST+JWST/cycle33/spec-7235-56603-0078.fits'
data = fits.open(path_spectra)
wave, flux, flux_err = 10 ** data[1].data['loglam'], data[1].data['flux'], 1 / np.sqrt(data[1].data['ivar'])
model = data[1].data['model']

# OII and Fit OII
# OII_region = np.where((wave > 6600) * (wave < 6675))
OII_region = np.where((wave > 6500) * (wave < 6775))
wave_OII = wave[OII_region]
flux_OII = flux[OII_region]

redshift_guess = 0.78071
sigma_kms_guess = 150.0
flux_OII_guess = 42
r_OII3729_3727_guess = 100

parameters = lmfit.Parameters()
parameters.add_many(('z', redshift_guess, True, None, None, None),
                    ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
                    ('flux_OII', flux_OII_guess, True, None, None, None),
                    ('r_OII3729_3727', r_OII3729_3727_guess, True, 0, 3, None),
                    ('a', 0.0, True, None, None, None),
                    ('b', 100, True, None, None, None))
spec_model = lmfit.Model(model_OII, missing='drop')
result_OII = spec_model.fit(flux_OII, wave_vac=wave_OII, params=parameters)
z = result_OII.best_values['z']
print('redshift is ', z)

# Fit Hbeta
# Fit the Spectrum
path = '../../PyQSOFit/'
path1 = path             # the path of the source code file and qsopar.fits
path2 = '../../qso_cgm/data/result/' # path of fitting results
path3 = '../../qso_cgm/data/QA_result/'   # path of figure
path4 = '../../PyQSOFit/sfddata/'             # path of dust reddening map

newdata = np.rec.array([(6564.61, 'Ha', 6400., 6800., 'Ha_br', 3, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05), \
                        # (6564.61, 'Ha', 6400., 6800., 'Ha_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002),\
                        # (6549.85, 'Ha', 6400., 6800., 'NII6549', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 1, 0.001),\
                        # (6585.28, 'Ha', 6400., 6800., 'NII6585', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 1, 0.003),\
                        # (6718.29, 'Ha', 6400., 6800., 'SII6718', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 2, 0.001),\
                        # (6732.67, 'Ha', 6400., 6800., 'SII6732', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 1, 1, 2, 0.001),\

                        (4862.68, 'Hb', 4640., 5100., 'Hb_br', 3, 5e-3, 0.004, 0.05, 0.01, 0, 0, 0, 0.01), \
                        (4862.68, 'Hb', 4640., 5100., 'Hb_na', 2, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002), \
                        (4960.30, 'Hb', 4640., 5100., 'OIII4960c', 2, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002), \
                        (5008.24, 'Hb', 4640., 5100., 'OIII5008c', 2, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.004), \
                        # (4960.30, 'Hb', 4640., 5100., 'OIII4959w', 1,3e-3, 2.3e-4, 0.004, 0.01, 2, 2, 0, 0.001),\
                        # (5008.24, 'Hb', 4640., 5100., 'OIII5007w', 1,3e-3, 2.3e-4, 0.004, 0.01, 2, 2, 0, 0.002),\
                        # (4687.02, 'Hb', 4640., 5100., 'HeII4687_br', 1,5e-3, 0.004, 0.05, 0.005, 0, 0, 0, 0.001),\
                        # (4687.02, 'Hb', 4640., 5100., 'HeII4687_na', 1,1e-3, 2.3e-4, 0.0017, 0.005, 1, 1, 0, 0.001),\

                        # (3934.78, 'CaII', 3900., 3960., 'CaII3934', 2, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001),\
                        #
                        # (3728.48, 'OII', 3650., 3800., 'OII3728', 1, 1e-3, 3.333e-4, 0.0017, 0.01, 1, 1, 0, 0.001),\
                        #
                        # (3426.84, 'NeV', 3380., 3480., 'NeV3426', 1, 1e-3, 3.333e-4, 0.0017, 0.01, 0, 0, 0, 0.001),\
                        # (3426.84, 'NeV', 3380., 3480., 'NeV3426_br', 1, 5e-3, 0.0025, 0.02, 0.01, 0, 0, 0, 0.001),\


                        (2798.75, 'MgII', 2700., 2900., 'MgII_br', 1, 5e-3, 0.004, 0.05, 0.0017, 0, 0, 0, 0.05), \
                        (2798.75, 'MgII', 2700., 2900., 'MgII_na', 2, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \

                        (1908.73, 'CIII', 1700., 1970., 'CIII_br', 2, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01), \
                        # (1908.73, 'CIII', 1700., 1970., 'CIII_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
                        # (1892.03, 'CIII', 1700., 1970., 'SiIII1892', 1, 2e-3, 0.001, 0.015, 0.003, 1, 1, 0, 0.005), \
                        # (1857.40, 'CIII', 1700., 1970., 'AlIII1857', 1, 2e-3, 0.001, 0.015, 0.003, 1, 1, 0, 0.005), \
                        # (1816.98, 'CIII', 1700., 1970., 'SiII1816', 1, 2e-3, 0.001, 0.015, 0.01, 1, 1, 0, 0.0002), \
                        # (1786.7, 'CIII', 1700., 1970., 'FeII1787', 1, 2e-3, 0.001, 0.015, 0.01, 1, 1, 0, 0.0002), \
                        # (1750.26, 'CIII', 1700., 1970., 'NIII1750', 1, 2e-3, 0.001, 0.015, 0.01, 1, 1, 0, 0.001), \
                        # (1718.55, 'CIII', 1700., 1900., 'NIV1718', 1, 2e-3, 0.001, 0.015, 0.01, 1, 1, 0, 0.001),\

                        (1549.06, 'CIV', 1500., 1700., 'CIV_br', 1, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05), \
                        (1549.06, 'CIV', 1500., 1700., 'CIV_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 1, 1, 0, 0.002), \
                        (1640.42, 'CIV', 1500., 1700., 'HeII1640', 1, 1e-3, 5e-4, 0.0017, 0.008, 1, 1, 0, 0.002), \
                        (1663.48, 'CIV', 1500., 1700., 'OIII1663', 1, 1e-3, 5e-4, 0.0017, 0.008, 1, 1, 0, 0.002), \
                        (1640.42, 'CIV', 1500., 1700., 'HeII1640_br', 1, 5e-3, 0.0025, 0.02, 0.008, 1, 1, 0, 0.002), \
                        (1663.48, 'CIV', 1500., 1700., 'OIII1663_br', 1, 5e-3, 0.0025, 0.02, 0.008, 1, 1, 0, 0.002), \

                        # (1402.06, 'SiIV', 1290., 1450., 'SiIV_OIV1', 1, 5e-3, 0.002, 0.05, 0.015, 1, 1, 0, 0.05), \
                        # (1396.76, 'SiIV', 1290., 1450., 'SiIV_OIV2', 1, 5e-3, 0.002, 0.05, 0.015, 1, 1, 0, 0.05), \
                        # (1335.30, 'SiIV', 1290., 1450., 'CII1335', 1, 2e-3, 0.001, 0.015, 0.01, 1, 1, 0, 0.001), \
                        # (1304.35, 'SiIV', 1290., 1450., 'OI1304', 1, 2e-3, 0.001, 0.015, 0.01, 1, 1, 0, 0.001), \

                        (1215.67, 'Lya', 1150., 1290., 'Lya_br', 1, 5e-3, 0.004, 0.05, 0.02, 0, 0, 0, 0.05), \
                        (1215.67, 'Lya', 1150., 1290., 'Lya_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0, 0, 0.002) \
                        ], \
                       formats='float32, a20, float32, float32, a20, float32, float32, float32, float32, float32, '
                               'float32, float32, float32, float32', \
                       names='lambda, compname, minwav, maxwav, linename, ngauss, inisig, minsig, maxsig, voff, '
                             'vindex, windex, findex, fvalue')

# ------header-----------------
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'
hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'

# ------save line info-----------
hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
hdu.writeto(path + 'qsopar.fits', overwrite=True)

#Requried
# an important note that all the data input must be finite, especically for the error !!!
lam_qsofit = wave       # OBS wavelength [A]
flux_qsofit = flux     # OBS flux [erg/s/cm^2/A]
flux_err_qsofit = flux_err # 1 sigma error

# get data prepared
q = QSOFit(lam_qsofit, flux_qsofit, flux_err_qsofit, z, ra=38.342344816, dec=-4.918564492, plateid=0, mjd=0, fiberid=0,
           path=path1)

# do the fitting
q.Fit(name='J0233−0455', nsmooth=1, and_or_mask=False, deredden=True, reject_badpix=False, wave_range=None,
      wave_mask=None, decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, Fe_flux_range=np.array([4435, 4685]), poly=True, BC=False, rej_abs=False,
      initial_guess=None, MC=False, n_trails=5, linefit=True, tie_lambda=True, tie_width=True,
      tie_flux_1=True, tie_flux_2=True, save_result=True, plot_fig=True, save_fig=True,
      plot_line_name=True, plot_legend=True, dustmap_path=path4, save_fig_path=path3,
      save_fits_path=path2, save_fits_name='J0233−0455_fittings')


# Plot the Spectrum
fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
ax.plot(wave, flux, lw=1, color='k', drawstyle='steps-mid', label=r'$\mathrm{Data}$')
ax.plot(wave, flux_err, lw=1, color='lightgrey', label=r'$\mathrm{err}$')
ax.set_xlim(4500, 9200)
ax.set_ylim(-10, 95)
ax.minorticks_on()
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; [10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}}]$', size=20)
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.annotate(text=r'$\mathrm{Mg \, II}$', xy=(5200, 77), xycoords='data', size=20)
ax.annotate(text=r'$\mathrm{[O \, II]}$', xy=(6800, 77), xycoords='data', size=20)
# ax.annotate(text=r'$\mathrm{[O \, III]}$', xy=(8600, 77), xycoords='data', size=20)
ax.fill_between([6630, 6650], [-10, -10], [95, 95], color='lightgrey', alpha=0.5)
ax.fill_between([4900, 5100], [-10, -10], [95, 95], color='lightgrey', alpha=0.5)
ax.fill_between([8550, 9000], [-10, -10], [95, 95], color='lightgrey', alpha=0.7)

# Second axis
secax = ax.secondary_xaxis('top', functions=(obs2rest, rest2obs))
secax.minorticks_on()
secax.set_xlabel(r'$\mathrm{Rest \mbox{-} frame \; Wavelength \; [\AA]}$', size=20)
secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)

# Second plot
# axins = ax.inset_axes([0.07, 0.1, 0.2, 0.3])
# axins.plot(wave_OII, flux_OII, color='black', drawstyle='steps-mid')
# axins.plot(wave_OII, result_OII.best_fit, color='red')
# axins.set_xlim(6600, 6680)
# axins.set_ylim(27, 35)
# axins.minorticks_on()
# axins.set_ylabel(r'${f}_{\lambda}$', size=15)
# axins.text(6652, 32.5, r'$\mathrm{[O\;II]}$', size=13)
# axins.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=13, size=5)
# axins.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)

# Third plot
Hbeta_region = np.where((wave > 8550) * (wave < 9000))
wave_Hbeta = wave[Hbeta_region]
flux_Hbeta = flux[Hbeta_region]
model_Hbeta = model[Hbeta_region]

axins = ax.inset_axes([0.79, 0.67, 0.2, 0.3])
axins.plot(wave_Hbeta, flux_Hbeta, lw=1, color='black', drawstyle='steps-mid')
axins.plot(q.wave * (1 + z), q.Manygauss(np.log(q.wave), q.gauss_result) / (1 + z) + q.f_conti_model / (1 + z), 'red',
           label=r'$\mathrm{Broad} + \mathrm{Narrow}$',)
axins.set_xlim(8500, 9050)
axins.set_ylim(10, 60)
axins.minorticks_on()
axins.set_ylabel(r'${f}_{\lambda}$', size=15)
axins.text(8600, 50, r'$\mathrm{H\beta}$', size=13)
axins.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=13, size=5)
axins.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Proposal/HST+JWST/cycle33/Spectra_QSO', bbox_inches='tight')

# Calcualte BH mass
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.luminosity_distance(z).to(u.cm).value
# fwhm, sigma, ew, peak, area = q.line_prop('Hb', q.line_result[21:30], 'broad')
fwhm = 6402
L_5100 = 4 * np.pi * d_l ** 2 * float(q.conti_result[13]) * (5100.0/3000.0) ** float(q.conti_result[14])
waveL_5100 = 10 ** 46.21
L_bol = 10.33 * waveL_5100  # bolometric correction should be 10.33? # Shen Y2011-06 eq.5 9.26?
log_M_BH = np.log10((fwhm / 1000) ** 2 * (waveL_5100 / 1e44) ** 0.5) + 6.91
print('Monochromatic luminosity is', waveL_5100, 'erg/s')
print('Bolometric Luminosity is', L_bol, 'erg/s')
print('logM_Blackhole is ', log_M_BH, 'solar mass')

# Plot the MgII emission line absorption line
MgII_region = np.where((wave > 4910) * (wave < 5050))
wave_MgII = wave[MgII_region]
vel_MgII = (wave_MgII - wave_MgII2796_vac * (1 + z)) / (wave_MgII2796_vac * (1 + z)) * c_kms
flux_MgII = flux[MgII_region]
vel_OII = (wave_OII - wave_OII3728_vac * (1 + z)) / (wave_OII3728_vac * (1 + z)) * c_kms
# model_MgII = model[MgII_region]

MgII_cont = np.where(((vel_MgII > 500) * (vel_MgII < 700))    |
                     ((vel_MgII > 1100) * (vel_MgII < 1350)) |
                     ((vel_MgII > 1840) * (vel_MgII < 2300)))
wave_MgII_cont = wave_MgII[MgII_cont]
vel_MgII_cont = vel_MgII[MgII_cont]
flux_MgII_cont = flux_MgII[MgII_cont]

MgII_plot = np.where((vel_MgII > 500) * (vel_MgII < 2300))
wave_MgII_plot = wave_MgII[MgII_plot]
vel_MgII_plot = vel_MgII[MgII_plot]
flux_MgII_plot = flux_MgII[MgII_plot]

# Fit the continuum
parameters = lmfit.Parameters()
parameters.add_many(('a', 0.0, True, None, None, None), ('b', 100, True, None, None, None))
spec_model = lmfit.Model(model_cont, missing='drop')
result_MgII = spec_model.fit(flux_MgII_cont, wave_vac=wave_MgII_cont, params=parameters)
flux_MgII_plot /= model_cont(wave_MgII_plot, result_MgII.best_values['a'], result_MgII.best_values['b'])

#
MgII_spl = np.where((vel_MgII > -2300) * (vel_MgII < 500))
wave_MgII_spl = wave_MgII[MgII_spl]
vel_MgII_spl = vel_MgII[MgII_spl]
flux_MgII_spl = flux_MgII[MgII_spl]

# Fit with spline
spl = UnivariateSpline(vel_MgII_spl, flux_MgII_spl, s=50)
flux_MgII_smooth = spl(vel_MgII_spl)

# Make it connected
vel_MgII_spl = np.hstack((vel_MgII_spl, vel_MgII_plot[0]))
flux_MgII_spl = np.hstack((flux_MgII_spl, flux_MgII_plot[0]))
flux_MgII_smooth = np.hstack((flux_MgII_smooth, 1))
flux_MgII_spl_plot = flux_MgII_spl / flux_MgII_smooth

# plt.figure()
# plt.plot(vel_MgII, flux_MgII, 'k', label='Data')
# plt.plot(vel_MgII_spl, flux_MgII_smooth, 'r', label='Spline')
# plt.savefig('../../Proposal/HST+JWST/cycle33/test.png', bbox_inches='tight')

# Make OII plot
flux_OII -= model_cont(wave_OII, result_OII.best_values['a'], result_OII.best_values['b'])
OII_fit = result_OII.best_fit - model_cont(wave_OII, result_OII.best_values['a'], result_OII.best_values['b'])

fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True, dpi=300)
fig.subplots_adjust(hspace=0.0)
ax[0].plot(vel_OII, flux_OII, color='black', drawstyle='steps-mid')
ax[0].plot(vel_OII, OII_fit, color='red')
ax[1].plot(vel_MgII_spl, flux_MgII_spl_plot, color='k', drawstyle='steps-mid')
ax[1].plot(vel_MgII_plot, flux_MgII_plot, 'k', drawstyle='steps-mid')
ax[1].set_xlabel(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$', size=20)
fig.supylabel(r'$\mathrm{Normalized \; Flux}$', size=20, x=-0.02)
ax[0].set_title(r'$\rm [O \, II]$', size=20, x=0.15, y=0.50)
ax[1].set_title(r'$\rm Mg \, II$', size=20, x=0.15, y=0.50)
ax[0].axvline(0, ls='--', color='grey', lw=1)
# ax[0].axhline(1, ls='--', color='grey', lw=1)
ax[1].axvline(0, ls='--', color='grey', lw=1)
ax[1].axhline(1, ls='--', color='grey', lw=1)
ax[0].set_xlim(-2300, 2300)
ax[0].set_ylim(-1, 6)
ax[0].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax[0].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax[1].tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax[1].tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Proposal/HST+JWST/cycle33/MgII_QSO.png', bbox_inches='tight')


# Exposure time and orbit calculation
# STIS+COS
z_qso = z
z_abs_1, z_abs_2 = (4994.591 - wave_MgII2796_vac) / wave_MgII2796_vac, \
                   (5007.258 - wave_MgII2803_vac) / wave_MgII2803_vac
print('the absorber redshift is', z_abs_1, z_abs_2)
# STIS with E230M 2415
Lya, NV_1, NV_2 = 1215.67, 1238.82, 1242.80  # 14868 38688 47343
CIV_1, CIV_2 = 1548.195, 1550.770  # 13959 22428
# if orbit = 4, science exp = (50 - 6 - 6 - 7.5 - 4) * 60 + (50 - 4) * 60 * 3 = 9870
# if orbit = 5, science exp = (50 - 6 - 6 - 7.5 - 4) * 60 + (50 - 4) * 60 * 4 = 12630
# if orbit = 6, science exp = (50 - 6 - 6 - 7.5 - 4) * 60 + (50 - 4) * 60 * 5 = 15390

# COS with G185M 1941 A
Lyb, OVI_1, CII, OVI_2 = 1025.72, 1031.92, 1036.33, 1037.61 # 7,437 7,235, 8,448, 9,367
# 1 min instrument change
# if orbit = 3, science exp = (50 - 6.5 - 3 - 5) * 60 * 1 + (50 - 4 - 2) * 60 * 2  = 7410

# COS with G160M 1600 A
LL = 912  # 1919 with S/N = 2 per resolution element
# if orbit = 1, science exp = (50 - 4 - 2 - 1) * 60 = 2580

# STIS orbit


