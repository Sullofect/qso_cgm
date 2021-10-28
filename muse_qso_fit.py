import os
import sys
import glob
import lmfit
import warnings
import matplotlib
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import rc
from mpdaf.obj import Cube
sys.path.append('/Users/lzq/Dropbox/PyQSOFit')
from PyQSOFit import QSOFit
from PyAstronomy import pyasl
from mpdaf.drs import PixTable
from astropy.cosmology import FlatLambdaCDM
warnings.filterwarnings("ignore")
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def rest2obs(x):
    return x * (1 + z)


def obs2rest(x):
    return x / (1 + z)


# Define a function with the MUSE resolution
def getFWHM_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04)


# Define a function with the MUSE resolution
def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355


# Define the model fit function
def model(wave_vac, z, sigma_kms, fluxOII, rOII3729_3727, a, b):
    wave_OII3727_obs = wave_OII3727_vac * (1 + z)
    wave_OII3729_obs = wave_OII3729_vac * (1 + z)

    sigma_OII3727_A = np.sqrt((sigma_kms / c_kms * wave_OII3727_obs) ** 2 + (getSigma_MUSE(wave_OII3727_obs)) ** 2)
    sigma_OII3729_A = np.sqrt((sigma_kms / c_kms * wave_OII3729_obs) ** 2 + (getSigma_MUSE(wave_OII3729_obs)) ** 2)

    fluxOII3727 = fluxOII / (1 + rOII3729_3727)
    fluxOII3729 = fluxOII / (1 + 1.0 / rOII3729_3727)

    peakOII3727 = fluxOII3727 / np.sqrt(2 * sigma_OII3727_A ** 2 * np.pi)
    peakOII3729 = fluxOII3729 / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)

    OII3727_gaussian = peakOII3727 * np.exp(-(wave_vac - wave_OII3727_obs) ** 2 / 2 / sigma_OII3727_A ** 2)
    OII3729_gaussian = peakOII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)

    return OII3727_gaussian + OII3729_gaussian + a * wave_vac + b


# Take Muse data
path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits')
cube = Cube(path)

# Calculate the white image
image_white = cube.sum(axis=0)
p, q = image_white.peak()['p'], image_white.peak()['q']

# Spectrum
cube_ape = cube.aperture((p, q), 2, unit_center=None, unit_radius=None, is_sum=True)
wave_vac = pyasl.airtovac2(cube_ape.wave.coord())  # convert air wavelength to vacuum
flux = cube_ape.data * 1e-3
flux_err = np.sqrt(cube_ape.var) * 1e-3

# Find the redshift
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
c_kms = 2.998e5

OII_region = np.where((wave_vac > 6050) * (wave_vac < 6100))
wave_OII_vac = wave_vac[OII_region]
flux_OII = flux[OII_region]

redshift_guess = 0.63
sigma_kms_guess = 150.0
flux_OII_guess = 42
rOII3729_3727_guess = 100

parameters = lmfit.Parameters()
parameters.add_many(('z', redshift_guess, True, None, None, None),
                    ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
                    ('fluxOII', flux_OII_guess, True, None, None, None),
                    ('rOII3729_3727', rOII3729_3727_guess, True, 0, 3, None),
                    ('a', 0.0, True, None, None, None),
                    ('b', 100, True, None, None, None))
spec_model = lmfit.Model(model, missing='drop')
result = spec_model.fit(flux_OII, wave_vac=wave_OII_vac, params=parameters)
# print('Success = {}'.format(result.success))
# print(result.fit_report())
z = result.best_values['z']

# Plot the Spectrum
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
ax.plot(wave_vac, flux, color='k')
ax.plot(wave_vac, flux_err, color='lightgrey')
ax.set_xlim(5200, 9300)
ax.set_ylim(0, 100)
ax.minorticks_on()
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; (\AA)}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20)
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)

# Second axis
secax = ax.secondary_xaxis('top', functions=(obs2rest, rest2obs))
secax.minorticks_on()
secax.set_xlabel(r'$\mathrm{Rest \mbox{-} frame \; Wavelength \; (\AA)}$', size=20)
secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)

# Second axis
axins = ax.inset_axes([0.71, 0.61, 0.27, 0.3])
axins.plot(wave_OII_vac, flux_OII, color='black', drawstyle='steps-mid')
axins.plot(wave_OII_vac, result.best_fit, color='red')
axins.set_xlim(6060, 6085)
axins.set_ylim(55, 59)
axins.minorticks_on()
axins.set_ylabel(r'${f}_{\lambda}$', size=15)
axins.text(6062, 58, r'$\mathrm{[O\;II]}$', size=13)
axins.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=13, size=5)
axins.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
secaxins = axins.secondary_xaxis('top', functions=(obs2rest, rest2obs))
secaxins.minorticks_on()
secaxins.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=13)
secaxins.tick_params(axis='x', which='minor', direction='in', top='on', size=3)
fig.savefig('/Users/lzq/Dropbox/qso_cgm/qso_spec_fit_z', bbox_inches='tight')

# Fit the Spectrum
path = '../PyQSOFit/'
path1 = path             # the path of the source code file and qsopar.fits
path2 = '../qso_cgm/data/result/' # path of fitting results
path3 = '../qso_cgm/data/QA_result/'   # path of figure
path4 = '../PyQSOFit/sfddata/'             # path of dust reddening map

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
lam_qsofit = wave_vac       # OBS wavelength [A]
flux_qsofit = flux     # OBS flux [erg/s/cm^2/A]
flux_err_qsofit = flux_err # 1 sigma error

# get data prepared
q = QSOFit(lam_qsofit, flux_qsofit, flux_err_qsofit, z, ra=40.1359, dec=-18.8643, plateid=0, mjd=0, fiberid=0,
           path=path1)

# do the fitting
q.Fit(name='HE0238-1904', nsmooth=1, and_or_mask=False, deredden=True, reject_badpix=False, wave_range=None,
      wave_mask=np.array([[4650, 4750]]), decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, Fe_flux_range=np.array([4435, 4685]), poly=True, BC=False, rej_abs=False,
      initial_guess=None, MC=False, n_trails=5, linefit=True, tie_lambda=True, tie_width=True,
      tie_flux_1=True, tie_flux_2=True, save_result=True, plot_fig=True, save_fig=True,
      plot_line_name=True, plot_legend=True, dustmap_path=path4, save_fig_path=path3,
      save_fits_path=path2, save_fits_name='qso_fittings')


# Plot the Spectrum
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
ax.plot(q.wave * (1 + z), q.flux / (1 + z), color='k', label=r'$\mathrm{Data}$')
ax.plot(q.wave * (1 + z), q.err / (1 + z), color='lightgrey', label=r'$\mathrm{err}$')
ax.plot(q.wave * (1 + z), q.Manygauss(np.log(q.wave), q.gauss_result) / (1 + z) + q.f_conti_model / (1 + z),
        'b', label='Line')
ax.plot(q.wave * (1 + z), q.f_conti_model / (1 + z), 'c', label=r'$\mathrm{Iron}$')
ax.plot(q.wave * (1 + z), q.PL_poly_BC / (1 + z), 'orange', label=r'$\mathrm{Continuum}$')
#ax.set_xlim(5200, 9300)
#ax.set_ylim(0, 100)
ax.minorticks_on()
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; (\AA)}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20)
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)

secax = ax.secondary_xaxis('top', functions=(obs2rest, rest2obs))
secax.minorticks_on()
secax.set_xlabel(r'$\mathrm{Rest \mbox{-} frame \; Wavelength \; (\AA)}$', size=20)
secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)
ax.legend(prop={'size': 17}, framealpha=0, loc=3, fontsize=15)
fig.savefig('/Users/lzq/Dropbox/qso_cgm/qso_spec_fit_lines', bbox_inches='tight')

# Plot the Hbeta broad Spectrum
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
ax.plot(q.wave * (1 + z), q.Manygauss(np.log(q.wave), q.gauss_result) / (1 + z), 'b', label=r'$\mathrm{Broad} + \mathrm{Narrow}$', lw=2)
ax.plot(q.wave * (1 + z), q.Manygauss(np.log(q.wave), q.gauss_result[0:9]) / (1 + z), 'r', label=r'$\mathrm{Broad}$', lw=2)

ax.set_xlim(7500, 8500)
ax.minorticks_on()
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; (\AA)}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; (10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}})$', size=20)
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)

secax = ax.secondary_xaxis('top', functions=(obs2rest, rest2obs))
secax.minorticks_on()
secax.set_xlabel(r'$\mathrm{Rest \mbox{-} frame \; Wavelength \; (\AA)}$', size=20)
secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)
ax.legend(prop={'size': 17}, framealpha=0, loc=2, fontsize=15)
fig.savefig('/Users/lzq/Dropbox/qso_cgm/qso_spec_Hbeta_broad', bbox_inches='tight')

# Calcualte BH mass
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.luminosity_distance(z).to(u.cm).value
fwhm, sigma, ew, peak, area = q.line_prop(q.linelist[1][0], q.line_result[6:15], 'broad')
L_5100 = 4 * np.pi * d_l ** 2 * float(q.conti_result[13]) * (5100.0/3000.0) ** float(q.conti_result[14])
waveL_5100 = 5100 * L_5100
L_bol = 9.26 * waveL_5100 * 1e-17
log_M_BH = np.log10((fwhm/1000) ** 2 * (waveL_5100 * 1e-17/1e44) ** 0.5) + 6.91
print('logM_Blackhole is ', log_M_BH)
print('Bolometric Luminosity is', L_bol)
